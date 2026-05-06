"""算法分层配置的 Pydantic 契约定义。

所有模型均允许未知字段向后扩展（``extra=\"allow\"``），避免新增 YAML 键时校验失败。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetaConfig(BaseModel):
    """实验元信息与追溯字段。"""

    model_config = ConfigDict(extra="allow")

    config_version: str = "v0"
    experiment_name: str = "default"
    author: str | None = None
    created_at: str | None = None


class GlobalConfig(BaseModel):
    """全局运行时：随机种子、设备、日志级别等。"""

    model_config = ConfigDict(extra="allow")

    seed: int = Field(default=42, ge=0)
    timezone: str = "UTC"
    log_level: str = "INFO"
    device: str = "cpu"


class TemporalConfig(BaseModel):
    """预测时间粒度与窗口长度。"""

    model_config = ConfigDict(extra="allow")

    lookback_hours: int = 72
    forecast_hours: int = 24
    freq: str = "1H"


class SVGPKernelParamsConfig(BaseModel):
    """气象核函数的显式超参（避免代码内魔术数）。"""

    model_config = ConfigDict(extra="allow")

    matern_nu: float = Field(default=1.5, gt=0.0)
    daily_period_hours: float = Field(default=24.0, gt=0.0)
    yearly_period_hours: float = Field(default=8760.0, gt=0.0)


class SVGPConfig(BaseModel):
    """SVGP 预测子配置。"""

    model_config = ConfigDict(extra="allow")

    num_inducing_points: int = Field(default=100, gt=0)
    kernel: str = "matern_periodic_bundle"
    mean_function: str = "zero"
    likelihood: str = "gaussian"
    learning_rate: float = Field(default=0.01, gt=0.0)
    batch_size: int = Field(default=1024, gt=0)
    num_epochs: int = Field(default=50, ge=0)
    jitter: float = Field(default=1e-6, gt=0.0)
    whiten: bool = True
    num_features: int = Field(default=2, gt=0)
    standard_scaler_eps: float = Field(default=1e-6, gt=0.0)
    inducing_noise_scale: float = Field(default=1e-5, ge=0.0)
    inducing_default_low: float = -2.0
    inducing_default_high: float = 2.0
    kernel_params: SVGPKernelParamsConfig = Field(default_factory=SVGPKernelParamsConfig)


class PredictionConfig(BaseModel):
    """预测模块顶层。"""

    model_config = ConfigDict(extra="allow")

    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    svgp: SVGPConfig = Field(default_factory=SVGPConfig)


class ScenarioConfig(BaseModel):
    """场景生成模块。"""

    model_config = ConfigDict(extra="allow")

    num_mc_samples: int = Field(default=1000, gt=0)
    num_clusters: int = Field(default=10, gt=0)
    sampling_method: str = "gp_posterior"
    reduction_method: str = "kmedoids"
    random_state: int = Field(default=42, ge=0)
    kmeans_n_init: int = Field(default=10, ge=1)
    mc_variable_jitter_scale: float = Field(default=0.05, ge=0.0)


class SolverConfigSection(BaseModel):
    """MIP 求解器数值控制。"""

    model_config = ConfigDict(extra="allow")

    time_limit_seconds: int = Field(default=300, gt=0)
    mip_gap: float = Field(default=0.01, ge=0.0)
    threads: int = Field(default=4, ge=1)


class WeatherHardConfig(BaseModel):
    """风浪硬阈值：超过任一典型场景在该小时的值即禁止启航/在该小时起始作业。"""

    model_config = ConfigDict(extra="allow")

    max_wind_speed: float = Field(gt=0.0)
    max_wave_height: float = Field(gt=0.0)


class WeatherSoftConfig(BaseModel):
    """风浪软超限：仍以硬约束定义的禁航为保护，软段按期望加权惩罚进入目标函数。"""

    model_config = ConfigDict(extra="allow")

    max_wind_speed: float = Field(default=12.0, ge=0.0)
    max_wave_height: float = Field(default=1.5, ge=0.0)
    wind_penalty: float = Field(default=0.0, ge=0.0)
    wave_penalty: float = Field(default=0.0, ge=0.0)


class WeatherConstraintsConfig(BaseModel):
    """风浪约束：硬禁航 + 软惩罚。"""

    model_config = ConfigDict(extra="allow")

    hard: WeatherHardConfig
    soft: WeatherSoftConfig = Field(default_factory=WeatherSoftConfig)


class PlanningConfig(BaseModel):
    """规划时域离散化。"""

    model_config = ConfigDict(extra="allow")

    horizon_hours: int = Field(default=72, gt=0)
    time_step_hours: int = Field(default=1, gt=0)
    big_m_buffer_hours: float = Field(default=24.0, ge=0.0)


class OptimizationConfig(BaseModel):
    """MILP 优化模块。"""

    model_config = ConfigDict(extra="allow")

    solver: str = "pulp_cbc"
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    solver_config: SolverConfigSection = Field(default_factory=SolverConfigSection)
    weather_constraints: WeatherConstraintsConfig


class VesselConfig(BaseModel):
    """船舶业务参数。"""

    model_config = ConfigDict(extra="allow")

    speed_kmh: float = Field(default=25.0, gt=0.0)
    mobilization_time_h: float = Field(default=0.0, ge=0.0)
    hourly_cost: float = Field(default=0.0, ge=0.0)
    fuel_cost_per_km: float = Field(default=50.0, ge=0.0)
    operability_limits: dict[str, Any] = Field(default_factory=dict)


class SchedulingBusinessConfig(BaseModel):
    """与调度目标直接相关的经济权重。"""

    model_config = ConfigDict(extra="allow")

    downtime_penalty_per_hour: float = Field(default=1000.0, ge=0.0)
    priority_weight_scale: float = Field(default=1.0, ge=0.0)


class BusinessConfig(BaseModel):
    """业务与成本顶层。"""

    model_config = ConfigDict(extra="allow")

    vessel: VesselConfig = Field(default_factory=VesselConfig)
    scheduling: SchedulingBusinessConfig = Field(default_factory=SchedulingBusinessConfig)


class DepotConfig(BaseModel):
    """母港拓扑。"""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    lat: float
    lon: float


class TurbineLayoutConfig(BaseModel):
    """风机地理坐标（不含作业时长等业务字段）。"""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    lat: float
    lon: float
    failure_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="单日故障发生概率；未设置时使用 fault_simulation.default_failure_prob",
    )


class DistanceConfig(BaseModel):
    """距离度量与缓存策略。"""

    model_config = ConfigDict(extra="allow")

    metric: str = "haversine"
    cache: bool = True
    earth_radius_km: float = Field(default=6371.0, gt=0.0)


class FarmLayoutConfig(BaseModel):
    """风场拓扑。"""

    model_config = ConfigDict(extra="allow")

    coordinate_system: str = "WGS84"
    depot: DepotConfig
    turbines: list[TurbineLayoutConfig]
    distance: DistanceConfig = Field(default_factory=DistanceConfig)


class TaskSchedulingSpec(BaseModel):
    """单条可调度运维任务（与风机拓扑解耦）。"""

    model_config = ConfigDict(extra="allow")

    turbine_id: str = Field(min_length=1)
    duration_h: float = Field(gt=0.0)
    priority: float = Field(default=1.0, ge=0.0)
    deadline_h: float | None = Field(default=None, ge=0.0)


class Task(TaskSchedulingSpec):
    """动态或静态单条运维任务（与 ``TaskSchedulingSpec`` 严格同构，供 MILP builder 使用）。"""


class FaultTypeConfig(BaseModel):
    """故障类型：抽样权重、优先级与持续时长区间。"""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, description="类型标识，用于审计与状态追溯")
    prob: float = Field(gt=0.0, description="random.choices 权重，不必归一化为 1")
    priority: float = Field(ge=0.0)
    duration_range: tuple[float, float] = Field(
        description="持续时长闭区间 [小时]；左端点须为正",
    )

    def model_post_init(self, __context: Any) -> None:
        """校验时长区间合法。"""

        low, high = self.duration_range
        if low <= 0.0 or high <= 0.0:
            raise ValueError("duration_range endpoints must be positive")
        if low > high:
            raise ValueError("duration_range: low must be <= high")


class FaultPersistenceConfig(BaseModel):
    """动态故障状态持久化。"""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    state_file: str = Field(
        default="data/05_dynamic_faults/state.json",
        min_length=1,
        description="相对 project_root 或绝对路径",
    )


class FaultSimulationConfig(BaseModel):
    """工业级随机故障模拟（可复现、可持久化）。"""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    default_failure_prob: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="风机未配置 failure_rate 时的默认日故障概率",
    )
    fault_types: list[FaultTypeConfig] = Field(default_factory=list)
    persistence: FaultPersistenceConfig = Field(default_factory=FaultPersistenceConfig)


class ExperimentCacheConfig(BaseModel):
    """持久化缓存策略。"""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    hash_params: bool = True
    cache_dir: str = "./data/04_cache_store"


class ExperimentLoggingConfig(BaseModel):
    """实验落盘日志选项。"""

    model_config = ConfigDict(extra="allow")

    save_results: bool = True
    save_intermediate: bool = False


class ExperimentSection(BaseModel):
    """实验与再现性。"""

    model_config = ConfigDict(extra="allow")

    cache: ExperimentCacheConfig = Field(default_factory=ExperimentCacheConfig)
    logging: ExperimentLoggingConfig = Field(default_factory=ExperimentLoggingConfig)


class EnvConfig(BaseModel):
    """运行环境档位。"""

    model_config = ConfigDict(extra="allow")

    mode: str = "dev"


class AlgoParams(BaseModel):
    """根配置：与 ``algo_params.yaml`` 结构一一对应。"""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    meta: MetaConfig
    global_: GlobalConfig = Field(alias="global")
    prediction: PredictionConfig
    scenario: ScenarioConfig
    optimization: OptimizationConfig
    business: BusinessConfig
    farm_layout: FarmLayoutConfig
    tasks: list[TaskSchedulingSpec]
    fault_simulation: FaultSimulationConfig = Field(default_factory=FaultSimulationConfig)
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    env: EnvConfig = Field(default_factory=EnvConfig)


def default_algo_params_path() -> Path:
    """返回包内默认 YAML 路径。"""

    return Path(__file__).resolve().parent / "algo_params.yaml"
