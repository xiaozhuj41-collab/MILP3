"""端到端算法工作流流水线。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .cache.cache_manager import CacheManager, CacheManagerError
from .interfaces import BaseGenerator, BaseOptimizer, BasePredictor
from .schemas.milp import RoutingSolution, TaskNode
from .schemas.scenario import ScenarioSet
from .schemas.svgp import ForecastResult

T = TypeVar("T")


class WorkflowPipeline:
    """海上智能运维端到端工作流。

    流水线按「SVGP 预测 → 场景生成 → MILP 优化」执行，并结合
    ``CacheManager``、``experiment.cache`` 与 ``algo_params`` 摘要控制命中行为。
    """

    def __init__(
        self,
        predictor: BasePredictor,
        generator: BaseGenerator,
        optimizer: BaseOptimizer,
        cache_manager: CacheManager | None = None,
        logger: logging.Logger | None = None,
        *,
        cache_enabled: bool = True,
        algo_params_fingerprint: str | None = None,
    ) -> None:
        """初始化工作流流水线。

        Args:
            predictor: 气象预测器实例。
            generator: 场景生成器实例。
            optimizer: 运筹优化器实例。
            cache_manager: 可选缓存门面；默认为项目缓存目录。
            logger: 可选 logger。
            cache_enabled: 是否启用磁盘缓存读写。
            algo_params_fingerprint: 完整 YAML 配置的稳定哈希（``hash_params=true`` 时传入）。
        """

        self.predictor = predictor
        self.generator = generator
        self.optimizer = optimizer
        self.cache_manager = cache_manager or CacheManager()
        self.logger = logger or logging.getLogger(__name__)
        self._cache_enabled = cache_enabled
        self._algo_params_fingerprint = algo_params_fingerprint

    def run(
        self,
        nc_file_path: Path | str,
        task_nodes: list[TaskNode],
        latitude: float | None = None,
        longitude: float | None = None,
        force_refresh: bool = False,
    ) -> RoutingSolution:
        """运行端到端预测、场景生成与路径优化流程。"""

        path = Path(nc_file_path)

        forecast = self._run_cached_stage(
            stage_name="prediction",
            key_params={
                "stage": "prediction",
                "predictor": self.predictor.__class__.__qualname__,
                "predictor_key": self._component_cache_stub(self.predictor),
                "input_file": self._file_fingerprint(path),
                "latitude": latitude,
                "longitude": longitude,
                **_maybe_digest(self._algo_params_fingerprint),
            },
            compute=lambda: self.predictor.predict(
                {
                    "path": path,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            ),
            expected_type=ForecastResult,
            force_refresh=force_refresh,
        )

        scenario_set = self._run_cached_stage(
            stage_name="scenario_generation",
            key_params={
                "stage": "scenario_generation",
                "generator": self.generator.__class__.__qualname__,
                "generator_key": self._component_cache_stub(self.generator),
                "forecast": forecast,
                **_maybe_digest(self._algo_params_fingerprint),
            },
            compute=lambda: self.generator.generate(forecast),
            expected_type=ScenarioSet,
            force_refresh=force_refresh,
        )

        solution = self._run_cached_stage(
            stage_name="optimization",
            key_params={
                "stage": "optimization",
                "optimizer": self.optimizer.__class__.__qualname__,
                "optimizer_key": self._component_cache_stub(self.optimizer),
                "tasks": task_nodes,
                "scenarios": scenario_set,
                **_maybe_digest(self._algo_params_fingerprint),
            },
            compute=lambda: self.optimizer.solve(task_nodes, scenario_set),
            expected_type=RoutingSolution,
            force_refresh=force_refresh,
        )

        self.logger.info(
            "Workflow finished | total_cost=%.4f | routes=%s",
            solution.total_cost,
            solution.routes,
        )
        return solution

    def _run_cached_stage(
        self,
        stage_name: str,
        key_params: dict[str, Any],
        compute: Callable[[], T],
        expected_type: type[T],
        force_refresh: bool,
    ) -> T:
        """执行带缓存的单阶段逻辑。"""

        start_time = time.perf_counter()

        if not self._cache_enabled:
            result = compute()
            if not isinstance(result, expected_type):
                raise TypeError(f"{stage_name} result must be {expected_type.__name__}")
            self.logger.info(
                "%s | cache bypassed | elapsed=%.3fs",
                stage_name,
                time.perf_counter() - start_time,
            )
            return result

        if not force_refresh:
            cached = self._safe_cache_get(stage_name=stage_name, key_params=key_params)
            if cached is not None:
                if not isinstance(cached, expected_type):
                    raise TypeError(
                        f"Cached value for {stage_name} must be {expected_type.__name__}"
                    )
                elapsed = time.perf_counter() - start_time
                self.logger.info(
                    "%s | Cache Hit | elapsed=%.3fs",
                    stage_name,
                    elapsed,
                )
                return cached

        self.logger.info("%s | Cache Miss | stage started", stage_name)
        result = compute()
        if not isinstance(result, expected_type):
            raise TypeError(f"{stage_name} result must be {expected_type.__name__}")

        self._safe_cache_set(stage_name=stage_name, key_params=key_params, value=result)
        elapsed = time.perf_counter() - start_time
        self.logger.info("%s | computed and cached | elapsed=%.3fs", stage_name, elapsed)
        return result

    def _component_cache_stub(self, instance: Any) -> dict[str, Any]:
        """提取算法实例用于缓存哈希的稳定字段。"""

        getter = getattr(instance, "cache_key_fields", None)
        if callable(getter):
            return dict(getter())

        simple_types = (str, int, float, bool, type(None))
        return {
            key: value
            for key, value in vars(instance).items()
            if isinstance(value, simple_types)
        }

    def _safe_cache_get(self, stage_name: str, key_params: dict[str, Any]) -> Any | None:
        """安全读取缓存。"""

        try:
            return self.cache_manager.get(key_params)
        except CacheManagerError as exc:
            self.logger.warning("%s | cache read failed: %s", stage_name, exc)
            return None

    def _safe_cache_set(self, stage_name: str, key_params: dict[str, Any], value: Any) -> None:
        """安全写入缓存。"""

        try:
            cache_path = self.cache_manager.set(key_params, value)
            self.logger.info("%s | cache saved: %s", stage_name, cache_path)
        except CacheManagerError as exc:
            self.logger.warning("%s | cache write failed: %s", stage_name, exc)

    def _file_fingerprint(self, path: Path) -> dict[str, Any]:
        """构建输入文件指纹。"""

        resolved = path.resolve()
        stat = resolved.stat()
        return {
            "path": resolved.as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }


def _maybe_digest(digest: str | None) -> dict[str, str]:
    """将配置摘要按需并入缓存键。"""

    if digest is None:
        return {}
    return {"algo_params_fingerprint": digest}
