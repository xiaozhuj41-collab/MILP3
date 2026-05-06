"""海上风电运维 VRP 的 PuLP 模型与决策变量定义。"""

from dataclasses import dataclass

import pulp

from ..schemas.milp import TaskNode


@dataclass
class VRPDecisionVariables:
    """VRP 决策变量集合。

    Attributes:
        x: 船舶节点流转变量 x[i,j]，二进制，表示是否从节点 i 航行到节点 j。
        t: 节点到达/开始作业时间变量 t[i]，连续，单位为小时。
        u: MTZ 顺序变量 u[i]，用于消除子回路。
        service_slot: 作业时段选择变量 y[i,h]，二进制，表示节点 i 是否在小时 h 开始作业。
        sailing_slot: 航行时段选择变量 z[i,j,h]，二进制，表示弧 (i,j) 是否在小时 h 启航。
    """

    x: dict[tuple[int, int], pulp.LpVariable]
    t: dict[int, pulp.LpVariable]
    u: dict[int, pulp.LpVariable]
    service_slot: dict[tuple[int, int], pulp.LpVariable]
    sailing_slot: dict[tuple[int, int, int], pulp.LpVariable]


@dataclass
class VRPModelBundle:
    """VRP 模型构建结果。

    Attributes:
        model: PuLP 最小化模型。
        variables: 模型决策变量集合。
        node_indices: 所有节点索引，第 0 个节点代表港口/起终点。
        task_indices: 风机任务节点索引，不包含港口。
        time_slots: 离散小时索引，用于表达天气硬约束。
    """

    model: pulp.LpProblem
    variables: VRPDecisionVariables
    node_indices: list[int]
    task_indices: list[int]
    time_slots: list[int]


def create_vrp_model(
    tasks: list[TaskNode],
    planning_horizon_hours: int,
    model_name: str = "offshore_wind_maintenance_vrp",
) -> VRPModelBundle:
    """初始化海上风电运维 VRP MILP 模型。

    数学含义：
    - 节点 0 表示运维母港/码头，所有船舶路径从 0 出发并最终回到 0。
    - x[i,j] = 1 表示船舶从节点 i 直接航行到节点 j，否则为 0。
    - t[i] 表示船舶到达节点 i 并开始作业的时间。
    - u[i] 是 MTZ 顺序变量，用于禁止不经过港口的风机子回路。
    - y[i,h] = 1 表示节点 i 在离散小时 h 开始作业，用于将连续到达时间
      与气象禁航/禁作业时段建立线性约束。
    - z[i,j,h] = 1 表示船舶在小时 h 从 i 启航去 j，用于表达气象禁航约束。

    Args:
        tasks: 任务节点列表，必须至少包含港口节点与一个风机节点。
        planning_horizon_hours: 规划时域长度，单位为小时。
        model_name: PuLP 模型名称。

    Returns:
        VRP 模型与变量集合。

    Raises:
        ValueError: 当节点数量或规划时域非法时抛出。
    """

    if len(tasks) < 2:
        raise ValueError("VRP requires at least one depot and one task node")
    if planning_horizon_hours <= 0:
        raise ValueError("planning_horizon_hours must be positive")

    model = pulp.LpProblem(model_name, pulp.LpMinimize)
    node_indices = list(range(len(tasks)))
    task_indices = node_indices[1:]
    time_slots = list(range(planning_horizon_hours))

    # 核心流转变量：x_{i,j} ∈ {0,1}，表示船舶是否选择弧 (i,j)。
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i in node_indices
        for j in node_indices
        if i != j
    }

    # 到达时间变量：t_i ≥ 0，表示船舶到达节点 i 并开始作业的小时数。
    t = {
        i: pulp.LpVariable(
            f"t_{i}",
            lowBound=0,
            upBound=planning_horizon_hours,
            cat=pulp.LpContinuous,
        )
        for i in node_indices
    }

    # MTZ 顺序变量：u_i 表示节点 i 在路径中的访问顺序。
    u = {
        i: pulp.LpVariable(
            f"u_{i}",
            lowBound=0,
            upBound=len(task_indices),
            cat=pulp.LpContinuous,
        )
        for i in node_indices
    }

    # 作业时段选择变量：y_{i,h} ∈ {0,1}，用于表达“第 i 个风机是否在小时 h 作业”。
    service_slot = {
        (i, h): pulp.LpVariable(f"y_{i}_{h}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i in task_indices
        for h in time_slots
    }

    # 航行时段选择变量：z_{i,j,h} ∈ {0,1}，用于表达“船舶是否在小时 h 从 i 启航去 j”。
    sailing_slot = {
        (i, j, h): pulp.LpVariable(f"z_{i}_{j}_{h}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i, j in x
        for h in time_slots
    }

    return VRPModelBundle(
        model=model,
        variables=VRPDecisionVariables(
            x=x,
            t=t,
            u=u,
            service_slot=service_slot,
            sailing_slot=sailing_slot,
        ),
        node_indices=node_indices,
        task_indices=task_indices,
        time_slots=time_slots,
    )
