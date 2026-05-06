"""MILP 运筹优化数据契约。"""

from typing import Any

from pydantic import BaseModel, Field

from .base import Coordinate, TimeWindow


class TaskNode(BaseModel):
    """运维任务节点。

    Attributes:
        id: 任务节点唯一标识。
        loc: 任务节点地理坐标。
        duration: 任务服务时长。
        time_window: 可选时间窗约束。
        demand: 任务资源需求量。
        priority: 任务优先级，用于加权停机惩罚（母港常为 0）。
        deadline_h: 相对规划零点的小时粒度截止时刻；为空表示不设截止。
    """

    id: str = Field(min_length=1)
    loc: Coordinate
    duration: float = Field(gt=0.0)
    time_window: TimeWindow | None = None
    demand: float = Field(default=0.0, ge=0.0)
    priority: float = Field(default=0.0, ge=0.0)
    deadline_h: float | None = Field(default=None, ge=0.0)


class RoutingSolution(BaseModel):
    """路径优化求解结果。

    Attributes:
        routes: 路径集合，每条路径由任务节点 ID 顺序组成。
        total_cost: 解的总成本。
        metadata: 求解器返回的附加信息。
    """

    routes: list[list[str]]
    total_cost: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
