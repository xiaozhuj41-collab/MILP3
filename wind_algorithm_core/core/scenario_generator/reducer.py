"""随机场景削减逻辑。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducedScenario:
    """削减后的典型场景。

    Attributes:
        series: 展平后的典型场景序列。
        weight: 该场景发生概率。
    """

    series: list[float]
    weight: float


@dataclass(frozen=True)
class ScenarioReductionResult:
    """场景削减结果。

    Attributes:
        scenarios: 典型场景列表。
    """

    scenarios: list[ReducedScenario]


def k_means_reduce(
    trajectories: np.ndarray,
    num_scenarios: int = 10,
    *,
    random_state: int | None = 42,
    n_init: int = 10,
) -> ScenarioReductionResult:
    """使用 K-Means 将蒙特卡洛轨迹削减为典型场景。

    Args:
        trajectories: 原始采样轨迹，Shape 为 ``[num_samples, horizon, num_variables]``。
        num_scenarios: 目标典型场景数量，默认 10。
        random_state: K-Means 随机种子。
        n_init: ``sklearn.cluster.KMeans`` 的 ``n_init``。

    Returns:
        场景削减结果，权重之和严格为 1.0。

    Raises:
        ValueError: 当输入轨迹维度或场景数量非法时抛出。
    """

    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [num_samples, horizon, num_variables]")

    num_samples, horizon, num_variables = trajectories.shape
    if num_samples <= 0:
        raise ValueError("trajectories must contain at least one sample")

    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    resolved_num_scenarios = min(num_scenarios, num_samples)

    flattened = trajectories.reshape(num_samples, horizon * num_variables)

    kmeans = KMeans(
        n_clusters=resolved_num_scenarios,
        random_state=random_state,
        n_init=n_init,
    )
    kmeans.fit(flattened)
    labels = kmeans.predict(flattened)

    cluster_sizes = np.bincount(labels, minlength=resolved_num_scenarios).astype(np.float64)
    weights = _normalize_weights(cluster_sizes)

    scenarios = [
        ReducedScenario(
            series=[float(value) for value in kmeans.cluster_centers_[index].tolist()],
            weight=float(weights[index]),
        )
        for index in range(resolved_num_scenarios)
        if weights[index] > 0.0
    ]

    scenarios = _force_weight_sum_to_one(scenarios)
    return ScenarioReductionResult(scenarios=scenarios)


def kmedoids_reduce(
    trajectories: np.ndarray,
    num_scenarios: int = 10,
    *,
    random_state: int | None = 42,
    metric: str = "euclidean",
) -> ScenarioReductionResult:
    """使用 K-Medoids（PAM）削减蒙特卡洛轨迹。

    Args:
        trajectories: 原始采样轨迹，Shape 为 ``[num_samples, horizon, num_variables]``。
        num_scenarios: 目标典型场景数量。
        random_state: sklearn-extra ``KMedoids.random_state``。
        metric: 距离度量名称。

    Returns:
        场景削减结果。

    Raises:
        ImportError: 未安装 ``scikit-learn-extra`` 时抛出。
        ValueError: 输入非法时抛出。
    """

    try:
        from sklearn_extra.cluster import KMedoids  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - 环境缺省依赖
        raise ImportError(
            "reduction_method=kmedoids 需要可选依赖 scikit-learn-extra（pip install scikit-learn-extra）"
        ) from exc

    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [num_samples, horizon, num_variables]")

    num_samples, horizon, num_variables = trajectories.shape
    if num_samples <= 0:
        raise ValueError("trajectories must contain at least one sample")

    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    resolved_num_scenarios = min(num_scenarios, num_samples)
    flattened = trajectories.reshape(num_samples, horizon * num_variables)

    LOGGER.info(
        "KMedoids reduction | samples=%d | clusters=%d | metric=%s | random_state=%s",
        num_samples,
        resolved_num_scenarios,
        metric,
        random_state,
    )

    kmedoids = KMedoids(
        n_clusters=resolved_num_scenarios,
        metric=metric,
        random_state=random_state,
        method="pam",
    )
    kmedoids.fit(flattened)
    labels = kmedoids.predict(flattened)

    cluster_sizes = np.bincount(labels, minlength=resolved_num_scenarios).astype(np.float64)
    weights = _normalize_weights(cluster_sizes)

    centers = getattr(kmedoids, "cluster_centers_", None)
    if centers is None:
        raise RuntimeError("KMedoids did not expose cluster_centers_; check scikit-learn-extra version")

    scenarios = [
        ReducedScenario(
            series=[float(value) for value in centers[index].tolist()],
            weight=float(weights[index]),
        )
        for index in range(resolved_num_scenarios)
        if weights[index] > 0.0
    ]

    scenarios = _force_weight_sum_to_one(scenarios)
    return ScenarioReductionResult(scenarios=scenarios)


def _normalize_weights(cluster_sizes: np.ndarray) -> np.ndarray:
    """根据聚类簇样本数量计算概率权重。

    Args:
        cluster_sizes: 每个聚类簇中的样本数量。

    Returns:
        权重数组，元素和为 1.0。

    Raises:
        ValueError: 当总样本数量为 0 时抛出。
    """

    total = float(cluster_sizes.sum())
    if total <= 0.0:
        raise ValueError("cluster size total must be positive")
    return cluster_sizes / total


def _force_weight_sum_to_one(scenarios: list[ReducedScenario]) -> list[ReducedScenario]:
    """修正浮点误差，确保场景权重严格求和为 1.0。

    Args:
        scenarios: 待修正的典型场景列表。

    Returns:
        权重已修正的典型场景列表。

    Raises:
        ValueError: 当场景列表为空时抛出。
    """

    if not scenarios:
        raise ValueError("reduced scenarios must not be empty")

    if len(scenarios) == 1:
        only = scenarios[0]
        return [ReducedScenario(series=only.series, weight=1.0)]

    fixed = list(scenarios[:-1])
    prefix_sum = sum(scenario.weight for scenario in fixed)
    last_weight = max(0.0, 1.0 - prefix_sum)
    last = scenarios[-1]
    fixed.append(ReducedScenario(series=last.series, weight=last_weight))
    return fixed
