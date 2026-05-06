"""基于 SVGP 预测结果的蒙特卡洛场景采样。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..schemas.svgp import ForecastResult


@dataclass(frozen=True)
class MonteCarloSamples:
    """蒙特卡洛采样结果。

    Attributes:
        trajectories: 采样轨迹数组，Shape 为 ``[num_samples, horizon, num_variables]``。
        variable_names: 变量名称，默认包含 wind_speed 与 swh。
    """

    trajectories: np.ndarray
    variable_names: tuple[str, ...]


def monte_carlo_sample(
    forecast: ForecastResult,
    *,
    num_samples: int,
    random_seed: int,
    variable_scale_step: float,
    variable_names: Sequence[str] = ("wind_speed", "swh"),
) -> MonteCarloSamples:
    """基于预测均值和标准差生成未来气象轨迹。

    Args:
        forecast: SVGP 输出的预测结果。
        num_samples: 蒙特卡洛采样条数。
        random_seed: NumPy ``Generator`` 专用种子（与 ``scenario.random_state`` 对齐）。
        variable_scale_step: 次级变量缩放增量，对应 ``scenario.mc_variable_jitter_scale``。
        variable_names: 采样变量名称。

    Returns:
        蒙特卡洛采样结果，轨迹 Shape 为 ``[num_samples, horizon, num_variables]``.

    Raises:
        ValueError: 当采样数量或变量数量非法时抛出。
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    resolved_variable_names = tuple(variable_names)
    if not resolved_variable_names:
        raise ValueError("variable_names must not be empty")

    mean = np.asarray(forecast.mean, dtype=np.float64)
    variance = np.asarray(forecast.variance, dtype=np.float64)
    std = np.sqrt(np.maximum(variance, 0.0))

    generator = np.random.default_rng(random_seed)
    variable_samples: list[np.ndarray] = []

    for variable_index, _ in enumerate(resolved_variable_names):
        samples = generator.normal(
            loc=mean,
            scale=std,
            size=(num_samples, mean.shape[0]),
        )

        samples = np.clip(samples, a_min=0.0, a_max=None)

        if variable_index > 0:
            samples = samples * (1.0 + variable_scale_step * variable_index)

        variable_samples.append(samples)

    trajectories = np.stack(variable_samples, axis=-1)

    return MonteCarloSamples(
        trajectories=trajectories.astype(np.float64),
        variable_names=resolved_variable_names,
    )
