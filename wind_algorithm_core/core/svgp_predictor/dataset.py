"""单点 ERA5 时间序列数据集构建工具。"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch import Tensor

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class SlidingWindowTensors:
    """滑动窗口张量结果。

    Attributes:
        x: 输入窗口张量，Shape 为 [num_windows, input_window, num_features]。
        y: 预测目标张量，Shape 为 [num_windows, forecast_horizon]。
        target_time: 每个预测目标对应的时间序列，长度为 num_windows。
    """

    x: Tensor
    y: Tensor
    target_time: list[list[datetime]]


@dataclass(frozen=True)
class Era5PointSeries:
    """单点 ERA5 气象时间序列。

    Attributes:
        time: ERA5 时间坐标。
        features: 特征矩阵，Shape 为 [time_steps, num_features]。
        feature_names: 特征名称，当前默认为 wind_speed 与 swh。
    """

    time: list[datetime]
    features: np.ndarray
    feature_names: tuple[str, ...]


def load_era5_single_point(
    nc_path: Path | str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> Era5PointSeries:
    """读取单点 ERA5 .nc 文件并提取风速与浪高。

    Args:
        nc_path: ERA5 NetCDF 文件路径。
        latitude: 可选目标纬度；提供时按 nearest 选择最近网格点。
        longitude: 可选目标经度；提供时按 nearest 选择最近网格点。

    Returns:
        单点 ERA5 时间序列，特征包含 wind_speed 与 swh。

    Raises:
        FileNotFoundError: 当 .nc 文件不存在时抛出。
        KeyError: 当必要变量 u10、v10 或 swh 缺失时抛出。
        ValueError: 当提取结果无法形成二维时间序列时抛出。
    """

    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 file does not exist: {path}")

    with xr.open_dataset(path) as dataset:
        selected = _select_single_point(dataset, latitude=latitude, longitude=longitude)

        for variable_name in ("u10", "v10", "swh"):
            if variable_name not in selected:
                raise KeyError(f"Missing ERA5 variable: {variable_name}")

        selected = _fill_missing_values(selected)

        u10 = _to_1d_array(selected["u10"], variable_name="u10")
        v10 = _to_1d_array(selected["v10"], variable_name="v10")
        swh = _to_1d_array(selected["swh"], variable_name="swh")

        # 风速标量 = sqrt(u10^2 + v10^2)，Shape 为 [time_steps]。
        wind_speed = np.sqrt(np.square(u10) + np.square(v10)).astype(np.float64)

        # 特征矩阵 Shape 为 [time_steps, num_features]，当前 num_features = 2。
        features = np.stack([wind_speed, swh.astype(np.float64)], axis=-1)
        if features.ndim != 2:
            raise ValueError("ERA5 features must be a 2D time series matrix")

        # 进入 PyTorch Tensor 前做最后一道防线，确保 NaN/Inf 不会泄漏到模型。
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        time_values = _extract_time(selected)
        if len(time_values) != features.shape[0]:
            raise ValueError("Time coordinate length must match feature time dimension")

        return Era5PointSeries(
            time=time_values,
            features=features,
            feature_names=("wind_speed", "swh"),
        )


def build_sliding_windows(
    features: np.ndarray | Tensor,
    time: Sequence[datetime],
    input_window: int = 72,
    forecast_horizon: int = 24,
    target_index: int = 0,
) -> SlidingWindowTensors:
    """将单点时间序列转换为监督学习滑动窗口。

    Args:
        features: 原始特征矩阵，Shape 为 [time_steps, num_features]。
        time: 原始时间坐标，长度为 time_steps。
        input_window: 输入历史窗口长度，例如过去 72 小时。
        forecast_horizon: 输出预测窗口长度，例如未来 24 小时。
        target_index: 预测目标在特征矩阵中的列索引，默认预测 wind_speed。

    Returns:
        滑动窗口张量对象。

    Raises:
        ValueError: 当时间长度不足或维度不符合要求时抛出。
    """

    torch.set_default_dtype(torch.float64)
    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    if feature_tensor.ndim != 2:
        raise ValueError("features must have shape [time_steps, num_features]")

    time_steps, num_features = feature_tensor.shape
    if len(time) != time_steps:
        raise ValueError("time length must match feature time dimension")

    if target_index < 0 or target_index >= num_features:
        raise ValueError("target_index is out of feature range")

    min_required_steps = input_window + forecast_horizon
    if time_steps < min_required_steps:
        raise ValueError(
            "time series is too short for the requested input window and forecast horizon"
        )

    x_windows: list[Tensor] = []
    y_windows: list[Tensor] = []
    target_times: list[list[datetime]] = []

    for start in range(0, time_steps - min_required_steps + 1):
        input_end = start + input_window
        target_end = input_end + forecast_horizon

        # X Shape: [input_window, num_features]，例如 [72, 2]。
        x_window = feature_tensor[start:input_end, :]

        # Y Shape: [forecast_horizon]，例如 [24]，默认目标为未来风速。
        y_window = feature_tensor[input_end:target_end, target_index]

        x_windows.append(x_window)
        y_windows.append(y_window)
        target_times.append(list(time[input_end:target_end]))

    # X Shape: [num_windows, input_window, num_features]。
    x = torch.stack(x_windows, dim=0)

    # Y Shape: [num_windows, forecast_horizon]。
    y = torch.stack(y_windows, dim=0)

    return SlidingWindowTensors(x=x, y=y, target_time=target_times)


def nc_to_tensor(
    nc_path: Path | str,
    latitude: float | None = None,
    longitude: float | None = None,
    input_window: int = 72,
    forecast_horizon: int = 24,
    target_index: int = 0,
) -> SlidingWindowTensors:
    """从单点 ERA5 .nc 文件直接构建滑动窗口 Tensor。

    Args:
        nc_path: ERA5 NetCDF 文件路径。
        latitude: 可选目标纬度；提供时按 nearest 选择最近网格点。
        longitude: 可选目标经度；提供时按 nearest 选择最近网格点。
        input_window: 输入历史窗口长度。
        forecast_horizon: 预测目标窗口长度。
        target_index: 预测目标列索引。

    Returns:
        包含 X、Y 与目标时间的滑动窗口张量对象。
    """

    series = load_era5_single_point(
        nc_path=nc_path,
        latitude=latitude,
        longitude=longitude,
    )
    return build_sliding_windows(
        features=series.features,
        time=series.time,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_index=target_index,
    )


def _select_single_point(
    dataset: xr.Dataset,
    latitude: float | None,
    longitude: float | None,
) -> xr.Dataset:
    """从 ERA5 数据集中选择单点序列。

    Args:
        dataset: 原始 ERA5 数据集。
        latitude: 可选目标纬度。
        longitude: 可选目标经度。

    Returns:
        单点 ERA5 数据集。
    """

    lat_name = _find_coord_name(dataset, candidates=("latitude", "lat"))
    lon_name = _find_coord_name(dataset, candidates=("longitude", "lon"))

    if latitude is not None and longitude is not None and lat_name and lon_name:
        return dataset.sel({lat_name: latitude, lon_name: longitude}, method="nearest")

    return dataset.squeeze(drop=True)


def _find_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    """查找可能的坐标名称。

    Args:
        dataset: ERA5 数据集。
        candidates: 候选坐标名。

    Returns:
        命中的坐标名；未命中时返回 None。
    """

    for name in candidates:
        if name in dataset.coords:
            return name
    return None


def _to_1d_array(data_array: xr.DataArray, variable_name: str) -> np.ndarray:
    """将 ERA5 变量转换为一维时间序列数组。

    Args:
        data_array: ERA5 变量。
        variable_name: 变量名称，用于错误信息。

    Returns:
        一维 float64 数组。

    Raises:
        ValueError: 当变量无法压缩成一维时间序列时抛出。
    """

    values = np.asarray(data_array.squeeze().values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{variable_name} must be a single-point 1D time series")
    return values


def _fill_missing_values(dataset: xr.Dataset) -> xr.Dataset:
    """填充 ERA5 单点时间序列中的缺失值。

    Args:
        dataset: 已选中单点的 ERA5 数据集。

    Returns:
        缺失值已清洗的数据集。
    """

    try:
        return dataset.ffill("time").bfill("time").fillna(0.0)
    except (ImportError, ModuleNotFoundError, ValueError):
        return dataset.fillna(0.0)


def _extract_time(dataset: xr.Dataset) -> list[datetime]:
    """提取 ERA5 时间坐标。

    Args:
        dataset: 单点 ERA5 数据集。

    Returns:
        Python datetime 时间列表。

    Raises:
        KeyError: 当 time 坐标缺失时抛出。
    """

    if "time" not in dataset.coords:
        raise KeyError("Missing ERA5 time coordinate")

    return [
        value.astype("datetime64[ms]").astype(datetime)
        for value in np.asarray(dataset.coords["time"].values)
    ]
