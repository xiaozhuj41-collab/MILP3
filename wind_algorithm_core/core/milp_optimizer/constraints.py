"""海上风电运维 VRP 约束集合。"""

import logging
from math import ceil

import pulp

from ..schemas.milp import TaskNode
from ..schemas.scenario import ScenarioSet
from .vrp_model import VRPModelBundle

LOGGER = logging.getLogger(__name__)


def add_vrp_flow_constraints(bundle: VRPModelBundle) -> None:
    """添加 VRP 基础流平衡约束。

    数学含义：
    - 对每个风机节点 k，约束 sum_i x[i,k] = 1，表示每台风机必须且只能被访问一次。
    - 对每个风机节点 k，约束 sum_j x[k,j] = 1，表示船舶完成该节点作业后必须离开。
    - 对港口节点 0，约束 sum_j x[0,j] = 1 且 sum_i x[i,0] = 1，
      表示单艘船从港口出发并最终回到港口。

    Args:
        bundle: VRP 模型与变量集合。
    """

    model = bundle.model
    x = bundle.variables.x
    depot = 0

    for node in bundle.task_indices:
        model += (
            pulp.lpSum(x[i, node] for i in bundle.node_indices if i != node) == 1,
            f"visit_once_in_{node}",
        )
        model += (
            pulp.lpSum(x[node, j] for j in bundle.node_indices if j != node) == 1,
            f"visit_once_out_{node}",
        )

    model += (
        pulp.lpSum(x[depot, j] for j in bundle.task_indices) == 1,
        "depot_depart_once",
    )
    model += (
        pulp.lpSum(x[i, depot] for i in bundle.task_indices) == 1,
        "depot_return_once",
    )


def add_mtz_constraints(bundle: VRPModelBundle) -> None:
    """添加 MTZ 子回路消除约束。

    数学公式：
    对任意两个不同风机节点 i, j：
        u_i - u_j + n * x_{i,j} <= n - 1

    物理含义：
    如果船舶从 i 直接到 j，即 x_{i,j}=1，则必须有 u_j >= u_i + 1，
    表示 j 在访问顺序上晚于 i。该约束排除了不经过港口的风机闭环子回路。

    Args:
        bundle: VRP 模型与变量集合。
    """

    model = bundle.model
    x = bundle.variables.x
    u = bundle.variables.u
    depot = 0
    n = len(bundle.task_indices)

    model += (u[depot] == 0, "depot_order_zero")
    for node in bundle.task_indices:
        model += (u[node] >= 1, f"order_lower_{node}")
        model += (u[node] <= n, f"order_upper_{node}")

    for i in bundle.task_indices:
        for j in bundle.task_indices:
            if i == j:
                continue
            model += (
                u[i] - u[j] + n * x[i, j] <= n - 1,
                f"mtz_no_subtour_{i}_{j}",
            )


def add_time_propagation_constraints(
    bundle: VRPModelBundle,
    tasks: list[TaskNode],
    travel_time: dict[tuple[int, int], float],
    big_m: float,
) -> None:
    """添加路径时间递推约束。

    数学公式：
    对任意弧 (i,j)：
        t_j >= t_i + service_i + travel_{i,j} - M * (1 - x_{i,j})

    物理含义：
    如果船舶选择从 i 到 j，则到达 j 的时间必须晚于完成 i 节点作业并航行到 j
    的时间；如果未选择该弧，Big-M 项会放松该约束。

    Args:
        bundle: VRP 模型与变量集合。
        tasks: 任务节点列表。
        travel_time: 航行时间矩阵。
        big_m: Big-M 常数。
    """

    model = bundle.model
    variables = bundle.variables
    depot = 0

    model += (variables.t[depot] == 0, "depot_start_time_zero")

    for i, j in variables.x:
        if j == depot:
            continue
        service_duration = 0.0 if i == depot else tasks[i].duration
        model += (
            variables.t[j]
            >= variables.t[i]
            + service_duration
            + travel_time[i, j]
            - big_m * (1 - variables.x[i, j]),
            f"time_propagation_{i}_{j}",
        )


def add_time_window_constraints(bundle: VRPModelBundle, tasks: list[TaskNode]) -> None:
    """添加任务时间窗约束。

    物理含义：
    如果风机节点配置了最早/最晚可作业时间，则船舶到达并开始作业的时间必须落在
    该时间窗内。这里以所有任务时间窗的最早起点作为规划时域零点。

    Args:
        bundle: VRP 模型与变量集合。
        tasks: 任务节点列表。
    """

    windows = [task.time_window for task in tasks[1:] if task.time_window is not None]
    if not windows:
        return

    horizon_start = min(window.start for window in windows if window is not None)
    for index, task in enumerate(tasks):
        if index == 0 or task.time_window is None:
            continue

        lower = (task.time_window.start - horizon_start).total_seconds() / 3600.0
        upper = (task.time_window.end - horizon_start).total_seconds() / 3600.0
        bundle.model += (bundle.variables.t[index] >= lower, f"time_window_start_{index}")
        bundle.model += (bundle.variables.t[index] <= upper, f"time_window_end_{index}")


def add_deadline_constraints(bundle: VRPModelBundle, tasks: list[TaskNode]) -> None:
    """添加 ``TaskNode.deadline_h`` 代表的相对截止时刻约束。

    Args:
        bundle: VRP 模型与变量集合。
        tasks: 任务节点列表，索引 ``0`` 为母港。
    """

    for index, task in enumerate(tasks):
        if index == 0 or task.deadline_h is None:
            continue
        bundle.model += (
            bundle.variables.t[index] <= float(task.deadline_h),
            f"task_deadline_hours_{index}",
        )


def add_service_slot_constraints(bundle: VRPModelBundle) -> None:
    """添加作业时段选择与连续时间变量联动约束。

    数学公式：
        sum_h y_{i,h} = 1
        t_i = sum_h h * y_{i,h}

    物理含义：
    每台风机必须选择且只能选择一个开始作业小时；连续变量 t_i 被锚定到该小时，
    从而可以用 y_{i,h}=0 线性表达“某个小时禁止作业”。

    Args:
        bundle: VRP 模型与变量集合。
    """

    model = bundle.model
    variables = bundle.variables

    for node in bundle.task_indices:
        model += (
            pulp.lpSum(variables.service_slot[node, hour] for hour in bundle.time_slots) == 1,
            f"service_slot_choose_once_{node}",
        )
        model += (
            variables.t[node]
            == pulp.lpSum(hour * variables.service_slot[node, hour] for hour in bundle.time_slots),
            f"service_time_link_{node}",
        )


def add_sailing_slot_constraints(
    bundle: VRPModelBundle,
    tasks: list[TaskNode],
    big_m: float,
) -> None:
    """添加航行时段选择与路径变量联动约束。

    数学公式：
        sum_h z_{i,j,h} = x_{i,j}
        depart_{i,j} = sum_h h * z_{i,j,h}
        depart_{i,j} >= t_i + service_i - M * (1 - x_{i,j})

    物理含义：
    如果船舶选择弧 (i,j)，则必须选择一个启航小时；启航小时必须晚于完成 i 节点
    作业的时间。后续天气约束可通过 z_{i,j,h}=0 禁止在恶劣天气小时启航。

    Args:
        bundle: VRP 模型与变量集合。
        tasks: 任务节点列表。
        big_m: Big-M 常数。
    """

    model = bundle.model
    variables = bundle.variables
    depot = 0

    for i, j in variables.x:
        model += (
            pulp.lpSum(variables.sailing_slot[i, j, hour] for hour in bundle.time_slots)
            == variables.x[i, j],
            f"sailing_slot_link_{i}_{j}",
        )

        departure_time = pulp.lpSum(
            hour * variables.sailing_slot[i, j, hour] for hour in bundle.time_slots
        )
        service_duration = 0.0 if i == depot else tasks[i].duration
        model += (
            departure_time
            >= variables.t[i] + service_duration - big_m * (1 - variables.x[i, j]),
            f"sailing_depart_after_service_{i}_{j}",
        )


def add_weather_safety_constraints(
    bundle: VRPModelBundle,
    tasks: list[TaskNode],
    scenarios: ScenarioSet,
    max_wind_speed: float,
    max_wave_height: float,
) -> list[int]:
    """添加风浪安全硬约束。

    场景序列约定：
    ScenarioSet 中每条 WeatherScenario.series 由场景生成器展平得到，前半段为
    wind_speed，后半段为 swh。若某小时在任一典型场景下风速或浪高超过阈值，
    则该小时被视为全局不安全小时。

    数学含义：
    - 禁作业：若小时 h 不安全，则 y_{i,h}=0，表示任何风机不能在 h 开始作业。
    - 禁航行：若小时 h 不安全，则 z_{i,j,h}=0，表示船舶不能在 h 从任意节点启航。
    - 作业跨时段安全：若节点 i 的作业持续 d_i 小时，则起始小时 h 到
      h+ceil(d_i)-1 都必须安全，否则 y_{i,h}=0。

    Args:
        bundle: VRP 模型与变量集合。
        tasks: 任务节点列表。
        scenarios: 典型气象场景集合。
        max_wind_speed: 安全作业最大风速阈值。
        max_wave_height: 安全作业最大浪高阈值。

    Returns:
        不安全小时索引列表。

    Note:
        计算完 ``unsafe_hours`` 后会输出诊断日志，便于排查因天气硬约束导致无可行解的情况。
    """

    planning_hours = len(bundle.time_slots)
    unsafe_hours = detect_unsafe_hours(
        scenarios=scenarios,
        max_wind_speed=max_wind_speed,
        max_wave_height=max_wave_height,
        planning_horizon_hours=planning_hours,
    )
    LOGGER.info(
        "[weather-hard] planning_hours=%d unsafe_hour_count=%d unsafe_hours=%s",
        planning_hours,
        len(unsafe_hours),
        unsafe_hours,
    )
    unsafe_set = set(unsafe_hours)

    for hour in unsafe_hours:
        for node in bundle.task_indices:
            bundle.model += (
                bundle.variables.service_slot[node, hour] == 0,
                f"weather_no_service_{node}_{hour}",
            )

        for i, j in bundle.variables.x:
            bundle.model += (
                bundle.variables.sailing_slot[i, j, hour] == 0,
                f"weather_no_sailing_{i}_{j}_{hour}",
            )

    for node in bundle.task_indices:
        duration_slots = max(1, ceil(tasks[node].duration))
        for hour in bundle.time_slots:
            if hour + duration_slots > len(bundle.time_slots):
                bundle.model += (
                    bundle.variables.service_slot[node, hour] == 0,
                    f"weather_no_overrun_service_{node}_{hour}",
                )
                continue

            occupied_hours = range(hour, min(hour + duration_slots, len(bundle.time_slots)))
            if any(occupied in unsafe_set for occupied in occupied_hours):
                bundle.model += (
                    bundle.variables.service_slot[node, hour] == 0,
                    f"weather_no_overlap_service_{node}_{hour}",
                )

    return unsafe_hours


def compute_expected_soft_weather_penalty_profile(
    scenarios: ScenarioSet,
    *,
    soft_max_wind_speed: float,
    soft_max_wave_height: float,
    wind_penalty: float,
    wave_penalty: float,
    planning_horizon_hours: int,
) -> list[float]:
    """按典型场景加权期望，计算在每个小时起始作业的软惩罚单价。

    对任意典型场景 ``s`` 与起始小时 ``h``，超限部分为
    ``max(0, wind_s(h)-wind_soft)`` 与 ``max(0, swh_s(h)-wave_soft)``，
    再乘以 ``scenario.weight`` 累加。

    Args:
        scenarios: 典型气象场景集合。
        soft_max_wind_speed: 软风速上限，超出部分按 ``wind_penalty`` 计费。
        soft_max_wave_height: 软浪高上限，超出部分按 ``wave_penalty`` 计费。
        wind_penalty: 风速每超出 1 m/s 的罚金系数。
        wave_penalty: 浪高每超出 1 m 的罚金系数。
        planning_horizon_hours: MILP 规划槽位数。

    Returns:
        长度等于规划时域的罚金列表 ``penalty[h]``。
    """

    penalties = [0.0] * planning_horizon_hours
    for scenario in scenarios.scenarios:
        series = scenario.series
        if len(series) < 2:
            continue

        half = len(series) // 2
        wind_series = series[:half]
        wave_series = series[half : half * 2]
        horizon = min(planning_horizon_hours, len(wind_series), len(wave_series))
        weight = float(scenario.weight)

        for hour in range(horizon):
            excess_wind = max(0.0, float(wind_series[hour]) - soft_max_wind_speed)
            excess_wave = max(0.0, float(wave_series[hour]) - soft_max_wave_height)
            penalties[hour] += weight * (wind_penalty * excess_wind + wave_penalty * excess_wave)

    return penalties


def detect_unsafe_hours(
    scenarios: ScenarioSet,
    max_wind_speed: float,
    max_wave_height: float,
    planning_horizon_hours: int,
) -> list[int]:
    """从 ScenarioSet 中识别不安全小时。

    Args:
        scenarios: 典型气象场景集合。
        max_wind_speed: 最大安全风速阈值。
        max_wave_height: 最大安全浪高阈值。
        planning_horizon_hours: 规划时域长度。

    Returns:
        不安全小时索引列表。
    """

    unsafe_hours: set[int] = set()

    for scenario in scenarios.scenarios:
        series = scenario.series
        if len(series) < 2:
            continue

        half = len(series) // 2
        wind_series = series[:half]
        wave_series = series[half : half * 2]
        horizon = min(planning_horizon_hours, len(wind_series), len(wave_series))

        for hour in range(horizon):
            if wind_series[hour] > max_wind_speed or wave_series[hour] > max_wave_height:
                unsafe_hours.add(hour)

    return sorted(unsafe_hours)
