"""海上风电运维 VRP 成本计算函数。"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from ..schemas.milp import TaskNode

DEFAULT_EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    origin: TaskNode,
    destination: TaskNode,
    *,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
) -> float:
    """计算两个任务节点之间的大圆距离。

    Args:
        origin: 起点任务节点。
        destination: 终点任务节点。
        earth_radius_km: 地球半径，单位为千米。

    Returns:
        两点间海面近似距离，单位为千米。
    """

    lat1 = radians(origin.loc.lat)
    lon1 = radians(origin.loc.lon)
    lat2 = radians(destination.loc.lat)
    lon2 = radians(destination.loc.lon)

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2.0) ** 2
    return 2.0 * earth_radius_km * asin(sqrt(a))


def estimate_travel_time_hours(
    origin: TaskNode,
    destination: TaskNode,
    vessel_speed_kmh: float,
    *,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
) -> float:
    """估算船舶航行时间。

    Args:
        origin: 起点任务节点。
        destination: 终点任务节点。
        vessel_speed_kmh: 船舶平均航速，单位为千米/小时。
        earth_radius_km: 地球半径。

    Returns:
        航行时间，单位为小时。

    Raises:
        ValueError: 当船速不为正数时抛出。
    """

    if vessel_speed_kmh <= 0.0:
        raise ValueError("vessel_speed_kmh must be positive")
    distance = haversine_distance_km(origin, destination, earth_radius_km=earth_radius_km)
    return distance / vessel_speed_kmh


def calculate_sailing_cost(
    origin: TaskNode,
    destination: TaskNode,
    cost_per_km: float,
    *,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
) -> float:
    """计算船舶从一个节点航行到另一个节点的成本。

    Args:
        origin: 起点任务节点。
        destination: 终点任务节点。
        cost_per_km: 单位距离航行成本。
        earth_radius_km: 地球半径。

    Returns:
        航行成本。
    """

    distance = haversine_distance_km(origin, destination, earth_radius_km=earth_radius_km)
    return distance * cost_per_km


def calculate_downtime_penalty(
    task: TaskNode,
    arrival_time_hours: float,
    penalty_per_hour: float,
) -> float:
    """计算风机停机惩罚成本。

    停机惩罚用到达并开始维修前的等待时间近似表示；若任务配置了时间窗，
    则只惩罚晚于时间窗开始的延误小时数，否则以到达时间作为停机小时数。

    Args:
        task: 风机运维任务节点。
        arrival_time_hours: 船舶到达该节点的时间，单位为小时。
        penalty_per_hour: 单位停机小时惩罚成本。

    Returns:
        停机惩罚成本。
    """

    if task.time_window is None:
        downtime_hours = max(0.0, arrival_time_hours)
    else:
        horizon_start = task.time_window.start
        downtime_hours = max(
            0.0,
            arrival_time_hours
            - (task.time_window.start - horizon_start).total_seconds() / 3600.0,
        )
    return downtime_hours * penalty_per_hour


def build_distance_matrix(
    tasks: list[TaskNode],
    *,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
) -> dict[tuple[int, int], float]:
    """构建节点间距离矩阵。

    Args:
        tasks: 任务节点列表，第 0 个节点视为港口/起终点。
        earth_radius_km: 地球半径。

    Returns:
        以 ``(i, j)`` 为键的距离矩阵，单位为千米。
    """

    return {
        (i, j): haversine_distance_km(origin, destination, earth_radius_km=earth_radius_km)
        for i, origin in enumerate(tasks)
        for j, destination in enumerate(tasks)
        if i != j
    }


def build_travel_time_matrix(
    tasks: list[TaskNode],
    vessel_speed_kmh: float,
    *,
    earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM,
) -> dict[tuple[int, int], float]:
    """构建节点间航行时间矩阵。

    Args:
        tasks: 任务节点列表，第 0 个节点视为港口/起终点。
        vessel_speed_kmh: 船舶平均航速，单位为千米/小时。
        earth_radius_km: 地球半径。

    Returns:
        以 ``(i, j)`` 为键的航行时间矩阵，单位为小时。
    """

    return {
        (i, j): estimate_travel_time_hours(
            origin,
            destination,
            vessel_speed_kmh,
            earth_radius_km=earth_radius_km,
        )
        for i, origin in enumerate(tasks)
        for j, destination in enumerate(tasks)
        if i != j
    }
