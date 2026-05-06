"""随机场景生成器实现。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..interfaces import BaseGenerator
from ..schemas.scenario import ScenarioSet, WeatherScenario
from ..schemas.svgp import ForecastResult
from .reducer import k_means_reduce, kmedoids_reduce
from .sampler import monte_carlo_sample

if TYPE_CHECKING:
    from config.models import AlgoParams, ScenarioConfig

LOGGER = logging.getLogger(__name__)


class KMedoidsGenerator(BaseGenerator):
    """随机场景生成器。

    支持 ``kmedoids`` 与 ``kmeans`` 两种削减策略，随机性由 ``ScenarioConfig.random_state``
    统一约束，确保蒙特卡洛抽样与聚类可复现。
    """

    def __init__(self, scenario: "ScenarioConfig") -> None:
        """初始化场景生成器。

        Args:
            scenario: ``scenario`` YAML 段落对应的强类型配置。
        """

        from config.models import ScenarioConfig as ScenarioConfigModel  # noqa: PLC0415

        if not isinstance(scenario, ScenarioConfigModel):
            raise TypeError("scenario must be a ScenarioConfig instance")

        self._scenario = scenario

    @classmethod
    def from_algo_params(cls, params: "AlgoParams") -> KMedoidsGenerator:
        """基于根配置构造生成器。

        Args:
            params: 根算法配置。

        Returns:
            生成器实例。
        """

        return cls(scenario=params.scenario)

    def cache_key_fields(self) -> dict[str, object]:
        """供流水线缓存阶段的稳定键片段。"""

        from core.cache.hasher import StableHasher  # noqa: PLC0415

        return {
            "scenario_config_hash": StableHasher.hash(self._scenario.model_dump(mode="json")),
        }

    def generate(self, forecast: ForecastResult) -> ScenarioSet:
        """根据 SVGP 预测结果生成典型随机场景集合。

        Args:
            forecast: SVGP 输出的预测结果。

        Returns:
            场景集合，所有场景权重之和严格为 1.0。
        """

        scenario_cfg = self._scenario

        samples = monte_carlo_sample(
            forecast=forecast,
            num_samples=scenario_cfg.num_mc_samples,
            random_seed=scenario_cfg.random_state,
            variable_scale_step=float(scenario_cfg.mc_variable_jitter_scale),
        )

        reduction_method = scenario_cfg.reduction_method.strip().lower()
        if reduction_method == "kmedoids":
            reduced = kmedoids_reduce(
                trajectories=samples.trajectories,
                num_scenarios=scenario_cfg.num_clusters,
                random_state=scenario_cfg.random_state,
            )
        elif reduction_method in {"kmeans", "k_means"}:
            reduced = k_means_reduce(
                trajectories=samples.trajectories,
                num_scenarios=scenario_cfg.num_clusters,
                random_state=scenario_cfg.random_state,
                n_init=scenario_cfg.kmeans_n_init,
            )
        else:
            raise ValueError(
                f"Unsupported scenario.reduction_method: {scenario_cfg.reduction_method!r}; "
                "expected one of: kmedoids, kmeans"
            )

        scenarios = [
            WeatherScenario(series=scenario.series, weight=scenario.weight)
            for scenario in reduced.scenarios
        ]

        scenarios = self._normalize_scenario_weights(scenarios)
        LOGGER.info(
            "Scenario reduction finished | method=%s | scenarios=%d",
            reduction_method,
            len(scenarios),
        )
        return ScenarioSet(scenarios=scenarios)

    def _normalize_scenario_weights(
        self,
        scenarios: list[WeatherScenario],
    ) -> list[WeatherScenario]:
        """归一化并修正场景概率权重。

        Args:
            scenarios: 待归一化场景列表。

        Returns:
            权重严格求和为 1.0 的场景列表。

        Raises:
            ValueError: 当场景列表为空或权重总和非法时抛出。
        """

        if not scenarios:
            raise ValueError("scenarios must not be empty")

        total_weight = sum(scenario.weight for scenario in scenarios)
        if total_weight <= 0.0:
            raise ValueError("scenario weights must have positive total mass")

        normalized = [
            WeatherScenario(
                series=scenario.series,
                weight=scenario.weight / total_weight,
            )
            for scenario in scenarios
        ]

        if len(normalized) == 1:
            only = normalized[0]
            return [WeatherScenario(series=only.series, weight=1.0)]

        prefix = normalized[:-1]
        last = normalized[-1]
        last_weight = max(0.0, 1.0 - sum(scenario.weight for scenario in prefix))
        return [
            *prefix,
            WeatherScenario(series=last.series, weight=last_weight),
        ]
