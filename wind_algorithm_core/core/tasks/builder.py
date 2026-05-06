"""将 ``farm_layout`` 与 ``tasks`` YAML 段落合成为 MILP 所需 ``TaskNode`` 序列。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.schemas.base import Coordinate
from core.schemas.milp import TaskNode

if TYPE_CHECKING:
    from config.models import AlgoParams, TurbineLayoutConfig


def build_task_nodes_for_milp(params: "AlgoParams") -> list[TaskNode]:
    """装配「母港 + 待服务风机任务」列表，供 VRP/MILP 使用。

    母港取自 ``farm_layout.depot``；风机坐标取自 ``farm_layout.turbines``；
    服务时长 / 优先级 / 截止期限仅取自 ``tasks`` 段落，不从拓扑读出。

    Args:
        params: 根算法配置。

    Returns:
        节点列表：索引 ``0`` 为母港，其余顺序与 YAML ``tasks`` 列表一致。

    Raises:
        ValueError: 当任务引用未知风机或任务列表为空时抛出。
    """

    from config.models import AlgoParams as AlgoParamsModel  # noqa: PLC0415

    if not isinstance(params, AlgoParamsModel):
        raise TypeError("params must be an AlgoParams instance")

    layout = params.farm_layout
    turbine_map: dict[str, TurbineLayoutConfig] = {item.id: item for item in layout.turbines}

    if not params.tasks:
        raise ValueError("AlgoParams.tasks must contain at least one maintenance task")

    mobilization = float(max(params.business.vessel.mobilization_time_h, 1e-6))
    depot = layout.depot
    depot_node = TaskNode(
        id=depot.id,
        loc=Coordinate(lat=depot.lat, lon=depot.lon),
        duration=mobilization,
        demand=0.0,
        priority=0.0,
        deadline_h=None,
        time_window=None,
    )

    nodes: list[TaskNode] = [depot_node]

    for spec in params.tasks:
        if spec.turbine_id not in turbine_map:
            msg = (
                "Task turbine_id references unknown turbine layout id: "
                f"{spec.turbine_id!r}. Defined ids: {sorted(turbine_map)!r}"
            )
            raise ValueError(msg)

        turbine = turbine_map[spec.turbine_id]
        nodes.append(
            TaskNode(
                id=spec.turbine_id,
                loc=Coordinate(lat=turbine.lat, lon=turbine.lon),
                duration=float(spec.duration_h),
                deadline_h=float(spec.deadline_h) if spec.deadline_h is not None else None,
                priority=float(spec.priority),
                demand=0.0,
                time_window=None,
            )
        )

    return nodes
