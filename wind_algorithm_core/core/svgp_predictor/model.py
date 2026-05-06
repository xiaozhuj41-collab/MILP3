"""单点时间序列 SVGP 模型定义。"""

from __future__ import annotations

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from config.models import SVGPKernelParamsConfig

from .kernel import build_weather_kernel

torch.set_default_dtype(torch.float64)


def build_svgp_likelihood(
    *,
    forecast_horizon: int,
    likelihood_noise_floor: float,
) -> gpytorch.likelihoods.MultitaskGaussianLikelihood:
    """构建与多步 SVGP 输出匹配的高斯似然。

    Args:
        forecast_horizon: 未来预测步数。
        likelihood_noise_floor: 噪声下界，与配置 ``prediction.svgp.jitter`` 对齐以增强数值稳定。

    Returns:
        多任务高斯似然。
    """

    return gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=forecast_horizon,
        noise_constraint=gpytorch.constraints.GreaterThan(likelihood_noise_floor),
    )


class SVGPTimeSeriesModel(gpytorch.models.ApproximateGP):
    """面向单点气象时间序列的稀疏变分 GP 模型。

    模型输入为展平后的滑动窗口特征：
    原始 ``X`` Shape 为 ``[batch_size, seq_len, num_features]``，
    展平后 Shape 为 ``[batch_size, seq_len * num_features]``。
    多任务维度对应 ``forecast_horizon``，即一次输出未来多个小时的预测。
    """

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int = 24,
        num_inducing_points: int = 100,
        inducing_points: Tensor | None = None,
        train_x: Tensor | None = None,
        *,
        device: torch.device,
        whiten: bool = True,
        kernel_params: SVGPKernelParamsConfig | None = None,
        inducing_noise_scale: float = 1e-5,
        inducing_default_low: float = -2.0,
        inducing_default_high: float = 2.0,
        variational_jitter: float = 1e-6,
    ) -> None:
        """初始化 SVGP 时间序列模型。

        Args:
            input_dim: 展平后的输入特征维度，即 ``seq_len * num_features``。
            forecast_horizon: 未来预测步数，也是多任务输出数量。
            num_inducing_points: 诱导点数量，单点时间序列默认 100。
            inducing_points: 可选自定义诱导点，Shape 为 ``[num_inducing_points, input_dim]``。
            train_x: 可选标准化训练输入，Shape 为 ``[num_windows, input_dim]``。
            device: 张量与诱导点所在设备，与配置 ``global.device`` 一致。
            whiten: 白化变分时用 ``VariationalStrategy``，否则用 ``UnwhitenedVariationalStrategy``。
            kernel_params: 核函数显式超参。
            inducing_noise_scale: 诱导点去重后施加的微量扰动尺度。
            inducing_default_low: 无训练数据时一维锚点网格下界。
            inducing_default_high: 无训练数据时一维锚点网格上界。
            variational_jitter: 变分策略对角 ``jitter_val``，与 ``svgp.jitter`` 对齐。
        """

        torch.set_default_dtype(torch.float64)
        if inducing_points is None:
            if train_x is not None:
                inducing_points = self.build_inducing_points_from_train_x(
                    train_x=train_x,
                    max_inducing_points=num_inducing_points,
                    noise_scale=inducing_noise_scale,
                )
            else:
                inducing_points = self._build_default_inducing_points(
                    num_inducing_points=num_inducing_points,
                    input_dim=input_dim,
                    low=inducing_default_low,
                    high=inducing_default_high,
                )

        inducing_points = inducing_points.to(device=device, dtype=torch.float64)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=torch.Size([forecast_horizon]),
        )

        if whiten:
            base_variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
                jitter_val=float(variational_jitter),
            )
        else:
            base_variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
                jitter_val=float(variational_jitter),
            )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_variational_strategy,
            num_tasks=forecast_horizon,
        )

        super().__init__(variational_strategy)

        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([forecast_horizon])
        )
        self.covar_module = build_weather_kernel(
            input_dim=input_dim,
            batch_shape=torch.Size([forecast_horizon]),
            kernel_params=kernel_params,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        """执行 GP 前向计算。

        Args:
            x: 展平后的输入特征，Shape 为 ``[batch_size, input_dim]``。

        Returns:
            多任务变分高斯分布，任务维度对应 ``forecast_horizon``。

        Raises:
            ValueError: 当输入不是二维特征矩阵时抛出。
        """

        if x.ndim != 2:
            raise ValueError("SVGPTimeSeriesModel input must have shape [batch_size, input_dim]")

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def _build_default_inducing_points(
        num_inducing_points: int,
        input_dim: int,
        *,
        low: float,
        high: float,
    ) -> Tensor:
        """构建默认诱导点。

        Args:
            num_inducing_points: 诱导点数量。
            input_dim: 输入特征维度。
            low: 标准化空间一维网格下界。
            high: 标准化空间一维网格上界。

        Returns:
            默认诱导点张量，Shape 为 ``[num_inducing_points, input_dim]``。
        """

        base = torch.linspace(low, high, num_inducing_points, dtype=torch.float64)
        return base.unsqueeze(-1).repeat(1, input_dim)

    @staticmethod
    def build_inducing_points_from_train_x(
        train_x: Tensor,
        max_inducing_points: int = 100,
        noise_scale: float = 1e-5,
    ) -> Tensor:
        """基于去重后的训练输入初始化诱导点。

        Args:
            train_x: 标准化训练输入，Shape 为 ``[num_windows, input_dim]``。
            max_inducing_points: 最大诱导点数量，默认 100。
            noise_scale: 添加到诱导点上的微小噪声尺度，避免完全重合。

        Returns:
            诱导点张量，Shape 为 ``[min(max, unique_count), input_dim]``。

        Raises:
            ValueError: 当 ``train_x`` 不是二维矩阵时抛出。
        """

        if train_x.ndim != 2:
            raise ValueError("train_x must have shape [num_windows, input_dim]")

        unique_train_x = torch.unique(train_x.to(dtype=torch.float64), dim=0)
        if unique_train_x.size(0) == 0:
            raise ValueError("train_x must contain at least one unique row")

        num_inducing_points = min(max_inducing_points, unique_train_x.size(0))
        if unique_train_x.size(0) > num_inducing_points:
            indices = torch.linspace(
                0,
                unique_train_x.size(0) - 1,
                steps=num_inducing_points,
                device=unique_train_x.device,
            ).round().long()
            inducing_points = unique_train_x.index_select(dim=0, index=indices)
        else:
            inducing_points = unique_train_x

        noise = torch.randn_like(inducing_points) * noise_scale
        return inducing_points + noise
