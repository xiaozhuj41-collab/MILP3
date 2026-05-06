"""PuLP VRP 优化器实现。"""

from __future__ import annotations

from typing import Any, Sequence

import pulp

from config.loader import algo_params_fingerprint
from config.models import AlgoParams

from ..cache.cache_manager import CacheManager
from ..interfaces import BaseOptimizer
from ..schemas.milp import RoutingSolution, TaskNode
from ..schemas.scenario import ScenarioSet
from .constraints import (
    add_deadline_constraints,
    add_mtz_constraints,
    add_sailing_slot_constraints,
    add_service_slot_constraints,
    add_time_propagation_constraints,
    add_time_window_constraints,
    add_vrp_flow_constraints,
    add_weather_safety_constraints,
    compute_expected_soft_weather_penalty_profile,
)
from .cost_calculator import build_travel_time_matrix
from .distance_matrix_cache import cached_build_distance_matrix
from .vrp_model import VRPModelBundle, create_vrp_model


class InfeasibleError(RuntimeError):
    """MILP 模型无可行解异常。"""


class PuLPOptimizer(BaseOptimizer):
    """基于 PuLP/CBC 的海上风电运维 VRP 优化器。

    经济系数、规划时域、求解器与风浪约束均从 ``AlgoParams`` 注入；
    Haversine 距离矩阵可按配置写入 ``CacheManager``。
    """

    def __init__(
        self,
        params: AlgoParams,
        *,
        distance_cache_manager: CacheManager | None = None,
    ) -> None:
        """初始化优化器。

        Args:
            params: 根算法配置。
            distance_cache_manager: 与流水线共享的距离矩阵缓存。
        """

        self._params = params
        self._distance_cache_manager = distance_cache_manager
        self._algo_digest = algo_params_fingerprint(params)

    @classmethod
    def from_algo_params(
        cls,
        params: AlgoParams,
        *,
        distance_cache_manager: CacheManager | None = None,
    ) -> PuLPOptimizer:
        """工厂方法，便于与流水线装配。"""

        return cls(params=params, distance_cache_manager=distance_cache_manager)

    def cache_key_fields(self) -> dict[str, Any]:
        """供 ``WorkflowPipeline`` 写入缓存键的稳定字段。"""

        wc = self._params.optimization.weather_constraints
        return {
            "algo_digest": self._algo_digest,
            "horizon_hours": self._params.optimization.planning.horizon_hours,
            "mip_gap": self._params.optimization.solver_config.mip_gap,
            "solver": self._params.optimization.solver,
            "hard_wind": wc.hard.max_wind_speed,
            "hard_wave": wc.hard.max_wave_height,
        }

    def solve(self, tasks: Sequence[TaskNode], scenarios: ScenarioSet) -> RoutingSolution:
        """求解海上风电运维 VRP。"""

        task_list = list(tasks)
        if len(task_list) < 2:
            raise ValueError("PuLPOptimizer requires depot plus at least one task")

        planning_horizon = int(self._params.optimization.planning.horizon_hours)
        bundle = create_vrp_model(tasks=task_list, planning_horizon_hours=planning_horizon)

        distance = cached_build_distance_matrix(
            task_list,
            self._params.farm_layout.distance,
            algo_config_fingerprint=self._algo_digest,
            cache_manager=self._distance_cache_manager,
        )
        vessel_speed = float(self._params.business.vessel.speed_kmh)
        travel_time = build_travel_time_matrix(
            task_list,
            vessel_speed,
            earth_radius_km=float(self._params.farm_layout.distance.earth_radius_km),
        )
        buf = float(self._params.optimization.planning.big_m_buffer_hours)
        big_m = float(planning_horizon + sum(task.duration for task in task_list) + buf)

        soft = self._params.optimization.weather_constraints.soft
        soft_penalty_profile = compute_expected_soft_weather_penalty_profile(
            scenarios,
            soft_max_wind_speed=float(soft.max_wind_speed),
            soft_max_wave_height=float(soft.max_wave_height),
            wind_penalty=float(soft.wind_penalty),
            wave_penalty=float(soft.wave_penalty),
            planning_horizon_hours=planning_horizon,
        )

        self._set_objective(bundle=bundle, tasks=task_list, distance=distance, soft_penalty_profile=soft_penalty_profile)

        hard = self._params.optimization.weather_constraints.hard
        self._add_constraints(
            bundle=bundle,
            tasks=task_list,
            scenarios=scenarios,
            travel_time=travel_time,
            big_m=big_m,
            max_wind_speed=float(hard.max_wind_speed),
            max_wave_height=float(hard.max_wave_height),
        )

        solver_cfg = self._params.optimization.solver_config
        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=int(solver_cfg.time_limit_seconds),
            gapRel=float(solver_cfg.mip_gap),
            threads=int(solver_cfg.threads),
        )
        status_code = bundle.model.solve(solver)
        status = pulp.LpStatus[status_code]

        if status != "Optimal":
            raise InfeasibleError(f"MILP solver did not find an optimal solution: {status}")

        route = self._extract_route(bundle=bundle, tasks=task_list)
        total_cost = float(pulp.value(bundle.model.objective) or 0.0)
        return RoutingSolution(
            routes=[route],
            total_cost=max(0.0, total_cost),
            metadata={
                "solver": "PULP_CBC_CMD",
                "status": status,
                "planning_horizon_hours": planning_horizon,
                "objective": total_cost,
                "algo_digest": self._algo_digest,
            },
        )

    def _set_objective(
        self,
        bundle: VRPModelBundle,
        tasks: list[TaskNode],
        distance: dict[tuple[int, int], float],
        soft_penalty_profile: list[float],
    ) -> None:
        """航行成本 + 优先级加权停机惩罚 + 软风浪期望罚金。"""

        x = bundle.variables.x
        y = bundle.variables.service_slot

        vessel = self._params.business.vessel
        scheduling = self._params.business.scheduling

        sailing_cost = pulp.lpSum(
            distance[i, j] * float(vessel.fuel_cost_per_km) * x[i, j] for i, j in x
        )

        downtime_terms = []
        for node in bundle.task_indices:
            priority_weight = 1.0 + float(scheduling.priority_weight_scale) * float(tasks[node].priority)
            downtime_terms.append(
                float(scheduling.downtime_penalty_per_hour) * priority_weight * bundle.variables.t[node]
            )
        downtime_penalty = pulp.lpSum(downtime_terms)

        weather_soft_penalty = pulp.lpSum(
            y[node, hour] * soft_penalty_profile[hour]
            for node in bundle.task_indices
            for hour in bundle.time_slots
        )

        bundle.model += sailing_cost + downtime_penalty + weather_soft_penalty

    def _add_constraints(
        self,
        bundle: VRPModelBundle,
        tasks: list[TaskNode],
        scenarios: ScenarioSet,
        travel_time: dict[tuple[int, int], float],
        big_m: float,
        *,
        max_wind_speed: float,
        max_wave_height: float,
    ) -> None:
        """组装全部线性约束（含截止时间）。"""

        add_vrp_flow_constraints(bundle)
        add_mtz_constraints(bundle)
        add_time_propagation_constraints(
            bundle=bundle,
            tasks=tasks,
            travel_time=travel_time,
            big_m=big_m,
        )
        add_time_window_constraints(bundle=bundle, tasks=tasks)
        add_deadline_constraints(bundle=bundle, tasks=tasks)
        add_service_slot_constraints(bundle)
        add_sailing_slot_constraints(bundle=bundle, tasks=tasks, big_m=big_m)
        add_weather_safety_constraints(
            bundle=bundle,
            tasks=tasks,
            scenarios=scenarios,
            max_wind_speed=max_wind_speed,
            max_wave_height=max_wave_height,
        )

    def _extract_route(self, bundle: VRPModelBundle, tasks: list[TaskNode]) -> list[str]:
        """从流变量回溯路径。"""

        x = bundle.variables.x
        current = 0
        visited = {current}
        route_indices = [current]

        while True:
            next_nodes = [
                j
                for i, j in x
                if i == current and (pulp.value(x[i, j]) or 0.0) > 0.5
            ]
            if not next_nodes:
                break

            next_node = next_nodes[0]
            route_indices.append(next_node)
            if next_node == 0:
                break

            if next_node in visited:
                break

            visited.add(next_node)
            current = next_node

        return [tasks[index].id for index in route_indices]
