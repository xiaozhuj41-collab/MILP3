"""风场节点距离矩阵的持久化缓存。"""

from __future__ import annotations

from typing import Any

from config.models import DistanceConfig

from ..cache.cache_manager import CacheManager
from ..schemas.milp import TaskNode
from .cost_calculator import build_distance_matrix


def cached_build_distance_matrix(
    tasks: list[TaskNode],
    farm_distance: DistanceConfig,
    *,
    algo_config_fingerprint: str,
    cache_manager: CacheManager | None,
) -> dict[tuple[int, int], float]:
    """在满足配置时读取/写入缓存的 Haversine 距离矩阵。

    Args:
        tasks: MILP 任务节点序列。
        farm_distance: ``farm_layout.distance`` Pydantic 模型。
        algo_config_fingerprint: 全局配置哈希，用于参数变动时失效。
        cache_manager: 缓存门面；若为 ``None`` 则跳过缓存。

    Returns:
        距离矩阵，键为 ``(i, j)``。
    """

    earth_radius_km = float(farm_distance.earth_radius_km)

    if not farm_distance.cache or cache_manager is None:
        return build_distance_matrix(tasks, earth_radius_km=earth_radius_km)

    sorted_nodes_meta: list[list[Any]] = [
        [task.id, round(task.loc.lat, 8), round(task.loc.lon, 8)] for task in tasks
    ]

    cache_key_payload: dict[str, Any] = {
        "component": "distance_matrix",
        "algo_config_fingerprint": algo_config_fingerprint,
        "metric": farm_distance.metric,
        "earth_radius_km": earth_radius_km,
        "nodes_fp": sorted_nodes_meta,
    }

    try:
        cached = cache_manager.get(cache_key_payload)
    except Exception:  # noqa: BLE001 - 缓存不可用时不阻断主求解
        cached = None

    if cached is not None and isinstance(cached, dict):
        restored: dict[tuple[int, int], float] = {}
        for raw_key, value in cached.items():
            if "_" in raw_key:
                ia, jb = raw_key.split("_", 1)
                restored[(int(ia), int(jb))] = float(value)
        if restored:
            return restored

    dense = build_distance_matrix(tasks, earth_radius_km=earth_radius_km)
    serializable = {f"{i}_{j}": float(value) for (i, j), value in dense.items()}

    try:
        cache_manager.set(cache_key_payload, serializable)
    except Exception:  # noqa: BLE001
        pass

    return dense
