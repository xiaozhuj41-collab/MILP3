"""SVGP 单点时间序列预测器实现。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import gpytorch
import torch
from torch import Tensor

from ..interfaces import BasePredictor
from ..schemas.svgp import ForecastResult
from .dataset import SlidingWindowTensors, build_sliding_windows, nc_to_tensor
from .model import SVGPTimeSeriesModel, build_svgp_likelihood

if TYPE_CHECKING:
    from config.models import AlgoParams, PredictionConfig

torch.set_default_dtype(torch.float64)
LOGGER = logging.getLogger(__name__)


@dataclass
class StandardScaler:
    """Tensor 标准化缩放器。

    Attributes:
        mean: 均值张量。
        std: 标准差张量。
        eps: 防止除零的极小值（来自配置）。
    """

    mean: Tensor | None = None
    std: Tensor | None = None
    eps: float = 1e-6

    def fit(self, value: Tensor) -> None:
        """根据输入张量估计均值和标准差。

        Args:
            value: 待拟合张量。
        """

        self.mean = value.mean(dim=0, keepdim=True)
        self.std = value.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)

    def transform(self, value: Tensor) -> Tensor:
        """标准化输入张量。

        Args:
            value: 待标准化张量。

        Returns:
            标准化后的张量。

        Raises:
            RuntimeError: 当缩放器尚未拟合时抛出。
        """

        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def inverse_transform(self, value: Tensor) -> Tensor:
        """反标准化张量。

        Args:
            value: 标准化空间中的张量。

        Returns:
            原始尺度张量。

        Raises:
            RuntimeError: 当缩放器尚未拟合时抛出。
        """

        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        return value * self.std.to(value.device) + self.mean.to(value.device)

    def inverse_transform_variance(self, variance: Tensor) -> Tensor:
        """将标准化空间中的方差还原到原始尺度。

        Args:
            variance: 标准化空间中的方差。

        Returns:
            原始尺度方差。

        Raises:
            RuntimeError: 当缩放器尚未拟合时抛出。
        """

        if self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse variance transform")
        return variance * torch.square(self.std.to(variance.device))


class SVGPModelPredictor(BasePredictor):
    """SVGP 单点气象时间序列预测器。

    该类封装模型加载、特征标准化和前向推理，并严格返回 ForecastResult。
    输入支持 .nc 文件路径、SlidingWindowTensors、Tensor 或包含 features/x 的字典。
    所有训练无关超参必须由 ``PredictionConfig`` / ``AlgoParams`` 注入。
    """

    def __init__(
        self,
        prediction: "PredictionConfig",
        *,
        runtime_device: str,
        model_path: Path | str | None = None,
    ) -> None:
        """初始化 SVGP 预测器。

        Args:
            prediction: 预测模块分层配置。
            runtime_device: 运行时设备字符串，通常取自 ``global.device``。
            model_path: 可选模型检查点路径。
        """

        from config.models import PredictionConfig as PredictionConfigModel  # noqa: PLC0415

        if not isinstance(prediction, PredictionConfigModel):
            raise TypeError("prediction must be a PredictionConfig instance")

        torch.set_default_dtype(torch.float64)
        self._prediction = prediction
        self._svgp = prediction.svgp
        temporal = prediction.temporal

        self.input_window = temporal.lookback_hours
        self.forecast_horizon = temporal.forecast_hours
        self.num_features = self._svgp.num_features
        self.input_dim = self.input_window * self.num_features

        self.device = torch.device(runtime_device)
        self.num_inducing_points = self._svgp.num_inducing_points
        self._checkpoint_loaded = False

        self._svgp_jitter = float(self._svgp.jitter)

        LOGGER.info("SVGP predictor device selected: %s", self.device)

        self.feature_scaler = StandardScaler(eps=float(self._svgp.standard_scaler_eps))
        self.target_scaler = StandardScaler(eps=float(self._svgp.standard_scaler_eps))

        self.model = SVGPTimeSeriesModel(
            input_dim=self.input_dim,
            forecast_horizon=self.forecast_horizon,
            num_inducing_points=self.num_inducing_points,
            device=self.device,
            whiten=bool(self._svgp.whiten),
            kernel_params=self._svgp.kernel_params,
            inducing_noise_scale=float(self._svgp.inducing_noise_scale),
            inducing_default_low=float(self._svgp.inducing_default_low),
            inducing_default_high=float(self._svgp.inducing_default_high),
            variational_jitter=float(self._svgp.jitter),
        ).to(self.device)

        self.likelihood = build_svgp_likelihood(
            forecast_horizon=self.forecast_horizon,
            likelihood_noise_floor=self._svgp_jitter,
        ).to(self.device)

        if model_path is not None:
            self.load(model_path)

        self.model.eval()
        self.likelihood.eval()

    @classmethod
    def from_algo_params(
        cls,
        params: "AlgoParams",
        *,
        model_path: Path | str | None = None,
    ) -> SVGPModelPredictor:
        """基于根算法配置构造预测器。

        Args:
            params: 根算法配置。
            model_path: 可选检查点路径。

        Returns:
            配置驱动的预测器实例。
        """

        return cls(prediction=params.prediction, runtime_device=params.global_.device, model_path=model_path)

    def cache_key_fields(self) -> dict[str, object]:
        """供流水线缓存阶段的稳定键片段。"""

        from core.cache.hasher import StableHasher  # noqa: PLC0415

        return {
            "prediction_config_hash": StableHasher.hash(self._prediction.model_dump(mode="json")),
            "device": str(self.device),
        }

    def load(self, model_path: Path | str) -> None:
        """加载 SVGP 模型检查点。

        Args:
            model_path: 模型检查点路径。

        Raises:
            FileNotFoundError: 当检查点不存在时抛出。
            RuntimeError: 当检查点加载失败时抛出。
        """

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SVGP checkpoint does not exist: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        try:
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            if "likelihood_state_dict" in checkpoint:
                self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

            self._load_scaler_from_checkpoint(checkpoint)
            self._checkpoint_loaded = True
        except (KeyError, RuntimeError, TypeError) as exc:
            raise RuntimeError(f"Failed to load SVGP checkpoint: {path}") from exc

    def predict(self, input_data: Any) -> ForecastResult:
        """执行单点气象序列预测。

        Args:
            input_data: 预测输入，可为 .nc 路径、SlidingWindowTensors、Tensor 或字典。

        Returns:
            ForecastResult，包含未来 ``forecast_horizon`` 个时间点的均值与方差。

        Raises:
            ValueError: 当输入维度不符合滑动窗口要求时抛出。
        """

        torch.set_default_dtype(torch.float64)
        started_at = time.perf_counter()
        LOGGER.info("SVGP prediction started")
        x_window, y_window, forecast_time = self._prepare_input(input_data)

        if x_window.ndim != 3:
            raise ValueError("predict input must have shape [batch_size, seq_len, num_features]")

        batch_size, seq_len, num_features = x_window.shape
        if seq_len != self.input_window or num_features != self.num_features:
            raise ValueError("input tensor shape does not match predictor configuration")

        x_window = x_window.to(self.device, dtype=torch.float64)
        LOGGER.info(
            "SVGP input prepared | windows=%d | seq_len=%d | features=%d | device=%s",
            batch_size,
            seq_len,
            num_features,
            self.device,
        )

        x_flat = x_window.reshape(batch_size, seq_len * num_features)
        self._fit_feature_scaler_if_needed(x_flat)
        x_scaled = self.feature_scaler.transform(x_flat)
        LOGGER.info("SVGP X standardized | shape=%s", tuple(x_scaled.shape))

        if y_window is not None:
            y_window = y_window.to(self.device, dtype=torch.float64)
            self._fit_target_scaler_if_needed(y_window)
            LOGGER.info("SVGP Y standardized | shape=%s", tuple(y_window.shape))

        if not self._checkpoint_loaded:
            self._rebuild_model_with_train_x(x_scaled)

        inference_started_at = time.perf_counter()
        LOGGER.info("SVGP forward inference started")
        # jitter 与 likelihood 噪声下界对齐，保证 Cholesky / 共轭梯度数值稳定
        jitter_ctx = gpytorch.settings.cholesky_jitter(self._svgp_jitter)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), jitter_ctx:
            distribution = self.likelihood(self.model(x_scaled))
        LOGGER.info(
            "SVGP forward inference finished | elapsed=%.3fs",
            time.perf_counter() - inference_started_at,
        )

        mean = distribution.mean
        variance = distribution.variance.clamp_min(0.0)

        mean_latest = self._select_latest_prediction(mean)
        variance_latest = self._select_latest_prediction(variance)

        if self.target_scaler.mean is not None and self.target_scaler.std is not None:
            mean_latest = self.target_scaler.inverse_transform(mean_latest)
            variance_latest = self.target_scaler.inverse_transform_variance(variance_latest)

        mean_latest = mean_latest.reshape(-1)
        variance_latest = variance_latest.reshape(-1)

        result = ForecastResult(
            time=forecast_time,
            mean=[float(value) for value in mean_latest.detach().cpu().tolist()],
            variance=[float(value) for value in variance_latest.detach().cpu().tolist()],
        )
        LOGGER.info("SVGP prediction finished | elapsed=%.3fs", time.perf_counter() - started_at)
        return result

    def _prepare_input(self, input_data: Any) -> tuple[Tensor, Tensor | None, list[datetime]]:
        """将不同输入格式统一为预测张量。

        Args:
            input_data: 原始预测输入。

        Returns:
            二元组，包含 ``X`` Tensor 与未来预测时间列表。
        """

        if isinstance(input_data, (str, Path)):
            windows = nc_to_tensor(
                nc_path=input_data,
                input_window=self.input_window,
                forecast_horizon=self.forecast_horizon,
            )
            return windows.x.clone(), windows.y.clone(), windows.target_time[-1]

        if isinstance(input_data, SlidingWindowTensors):
            return input_data.x.clone(), input_data.y.clone(), input_data.target_time[-1]

        if isinstance(input_data, Tensor):
            x_window = self._normalize_tensor_input(input_data)
            forecast_time = self._build_default_forecast_time(last_time=None)
            return x_window, None, forecast_time

        if isinstance(input_data, dict):
            return self._prepare_dict_input(input_data)

        raise ValueError(f"Unsupported SVGP predictor input type: {type(input_data)!r}")

    def _prepare_dict_input(self, input_data: dict[str, Any]) -> tuple[Tensor, Tensor | None, list[datetime]]:
        """解析字典形式的预测输入。

        Args:
            input_data: 输入字典，支持 path、x、features、time、latitude、longitude。

        Returns:
            二元组，包含 ``X`` Tensor 与未来预测时间列表。
        """

        if "path" in input_data:
            windows = nc_to_tensor(
                nc_path=input_data["path"],
                latitude=input_data.get("latitude"),
                longitude=input_data.get("longitude"),
                input_window=self.input_window,
                forecast_horizon=self.forecast_horizon,
            )
            return windows.x.clone(), windows.y.clone(), windows.target_time[-1]

        if "x" in input_data:
            x_window = self._normalize_tensor_input(torch.as_tensor(input_data["x"], dtype=torch.float64))
            y_window = None
            if "y" in input_data:
                y_window = torch.as_tensor(input_data["y"], dtype=torch.float64)
            time_values = input_data.get("time")
            forecast_time = self._resolve_forecast_time(time_values)
            return x_window, y_window, forecast_time

        if "features" in input_data:
            features = torch.as_tensor(input_data["features"], dtype=torch.float64)
            time_values = input_data.get("time")
            if time_values is None:
                time_values = self._build_history_time(features.shape[0])

            windows = build_sliding_windows(
                features=features,
                time=list(time_values),
                input_window=self.input_window,
                forecast_horizon=self.forecast_horizon,
            )
            return windows.x.clone(), windows.y.clone(), windows.target_time[-1]

        raise ValueError("dict input must contain one of: path, x, features")

    def _normalize_tensor_input(self, value: Tensor) -> Tensor:
        """统一 Tensor 输入维度。

        Args:
            value: 输入张量，支持 ``[seq_len, num_features]`` 或 ``[batch_size, seq_len, num_features]``。

        Returns:
            三维输入张量，Shape 为 ``[batch_size, seq_len, num_features]``。
        """

        tensor = value.to(dtype=torch.float64)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 3:
            return tensor
        raise ValueError("tensor input must have shape [seq_len, num_features] or [batch, seq_len, num_features]")

    def _select_latest_prediction(self, value: Tensor) -> Tensor:
        """选择最新窗口对应的多步预测。

        Args:
            value: 模型输出张量，期望 Shape 为 ``[batch_size, forecast_horizon]``。

        Returns:
            最新窗口预测，Shape 为 ``[forecast_horizon]``。
        """

        if value.ndim == 1:
            return value
        return value[-1]

    def _fit_feature_scaler_if_needed(self, x_flat: Tensor) -> None:
        """在没有外部缩放参数时使用当前输入拟合特征缩放器。

        Args:
            x_flat: 展平后的输入特征，Shape 为 ``[batch_size, input_dim]``。
        """

        if self.feature_scaler.mean is None or self.feature_scaler.std is None:
            self.feature_scaler.fit(x_flat)

    def _fit_target_scaler_if_needed(self, y: Tensor) -> None:
        """在没有外部目标缩放参数时拟合目标缩放器。

        Args:
            y: 目标矩阵，Shape 为 ``[num_windows, forecast_horizon]`` 或 ``[forecast_horizon]``。
        """

        if self.target_scaler.mean is None or self.target_scaler.std is None:
            if y.ndim == 1:
                y = y.unsqueeze(0)
            self.target_scaler.fit(y)

    def _rebuild_model_with_train_x(self, train_x: Tensor) -> None:
        """使用去重后的标准化训练输入重建诱导点。

        Args:
            train_x: 标准化后的输入矩阵，Shape 为 ``[num_windows, input_dim]``。
        """

        inducing_points = SVGPTimeSeriesModel.build_inducing_points_from_train_x(
            train_x=train_x.detach(),
            max_inducing_points=self.num_inducing_points,
            noise_scale=float(self._svgp.inducing_noise_scale),
        )
        LOGGER.info(
            "SVGP inducing points rebuilt | requested_max=%d | actual=%d | device=%s",
            self.num_inducing_points,
            inducing_points.size(0),
            self.device,
        )
        self.model = SVGPTimeSeriesModel(
            input_dim=self.input_dim,
            forecast_horizon=self.forecast_horizon,
            num_inducing_points=inducing_points.size(0),
            inducing_points=inducing_points,
            device=self.device,
            whiten=bool(self._svgp.whiten),
            kernel_params=self._svgp.kernel_params,
            inducing_noise_scale=float(self._svgp.inducing_noise_scale),
            inducing_default_low=float(self._svgp.inducing_default_low),
            inducing_default_high=float(self._svgp.inducing_default_high),
            variational_jitter=float(self._svgp.jitter),
        ).to(self.device)
        self.likelihood = build_svgp_likelihood(
            forecast_horizon=self.forecast_horizon,
            likelihood_noise_floor=self._svgp_jitter,
        ).to(self.device)
        self.model.eval()
        self.likelihood.eval()

    def _load_scaler_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """从检查点加载特征与目标缩放参数。

        Args:
            checkpoint: torch.load 得到的检查点字典。
        """

        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            self.feature_scaler.mean = torch.as_tensor(
                checkpoint["feature_mean"],
                dtype=torch.float64,
                device=self.device,
            )
            self.feature_scaler.std = torch.as_tensor(
                checkpoint["feature_std"],
                dtype=torch.float64,
                device=self.device,
            )

        if "target_mean" in checkpoint and "target_std" in checkpoint:
            self.target_scaler.mean = torch.as_tensor(
                checkpoint["target_mean"],
                dtype=torch.float64,
                device=self.device,
            )
            self.target_scaler.std = torch.as_tensor(
                checkpoint["target_std"],
                dtype=torch.float64,
                device=self.device,
            )

    def _resolve_forecast_time(self, time_values: Sequence[datetime] | None) -> list[datetime]:
        """解析或生成预测时间。

        Args:
            time_values: 历史输入时间序列。

        Returns:
            未来预测时间列表。
        """

        if time_values:
            return self._build_default_forecast_time(last_time=time_values[-1])
        return self._build_default_forecast_time(last_time=None)

    def _build_history_time(self, time_steps: int) -> list[datetime]:
        """构造默认历史时间序列。

        Args:
            time_steps: 历史序列长度。

        Returns:
            按小时递增的历史时间列表。
        """

        end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=time_steps - 1)
        return [start_time + timedelta(hours=offset) for offset in range(time_steps)]

    def _build_default_forecast_time(self, last_time: datetime | None) -> list[datetime]:
        """构造默认未来预测时间。

        Args:
            last_time: 历史序列最后一个时间点；为空时使用当前整点。

        Returns:
            未来 ``forecast_horizon`` 个小时的时间列表。
        """

        base_time = last_time or datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return [
            base_time + timedelta(hours=offset)
            for offset in range(1, self.forecast_horizon + 1)
        ]
