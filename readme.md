# MILP3 — 海上智能运维算法核心说明

本项目主体代码位于 **`wind_algorithm_core`**：面向离岸风电运维的 **气象预测 → 随机场景削减 → MILP 路径调度** 一体化流水线。模块间交互严格通过 **`core/schemas`** 中的 Pydantic 契约传递，便于替换实现与做回归测试。

---

## 一、项目在解决什么问题？

1. **预测**：在历史与再分析资料（如 ERA5）上，用 **稀疏变分高斯过程（SVGP）** 得到未来时段风速、浪高等量的 **均值与不确定性**。
2. **场景**：依据预测分布做 **蒙特卡洛抽样**，再通过 **聚类削减**（K-Medoids 或 K-Means）得到有限条 **典型气象轨迹**及概率权重。
3. **决策**：在满足 **航程时间、离散作业/启航时段、风浪硬禁区、任务截止与优先级** 等约束下，用 **混合整数线性规划（MILP）** 求解 **单船巡检/维修路径**，最小化 **航行成本 + 加权停机惩罚 + 软风浪期望罚金**。

整体适合作为 **baseline 研究代码库** 或接入更大「智能运维」系统中的算法核心。

---

## 二、技术栈与约束

| 类别 | 主要依赖 |
|------|-----------|
| 配置与契约 | Python 3.10+，`pydantic` / `pydantic-settings` |
| 预测 | `torch`，`gpytorch` |
| 数据与时序网格 | `xarray`，NetCDF |
| 场景 | `numpy`，`scikit-learn`，可选 **`scikit-learn-extra`**（K-Medoids） |
| 优化 | `pulp`（默认驱动 **COIN CBC**） |
| 缓存 | 自研 **`core/cache`**（基于稳定哈希的磁盘存储） |

**工程约定**：面向接口编程（`interfaces.py`）；业务参数集中在 **`config/algo_params.yaml`**；所有随机源头通过 **`global.seed`** 及场景 **`random_state`** 等可配置项约束，便于 **复现实验 + 缓存键随配置失效**。

---

## 三、仓库与目录解构

```
MILP3/
├── .cursorrules                 # Cursor / 协作规范（算法架构与注释语言等）
├── wenjianjiegou.md             # wind_algorithm_core 树状文件说明（精简版索引）
├── readme.md                     # 本文件：中文项目综述
└── wind_algorithm_core/         # 可独立运行的算法包目录
```

更细的 **逐文件用途** 见 **`wenjianjiegou.md`**（与源码目录一一对应）。下面按 **逻辑分层**说明，便于理解调用关系而非死记路径。

---

## 四、配置体系（请先读这一节）

### 4.1 单一事实来源：`algo_params.yaml`

配置分为多层（均在同一 YAML 内），加载后校验为 **`AlgoParams`**（`config/models.py`）：

| 层级 | 作用举例 |
|------|-----------|
| `meta` | `config_version`，`experiment_name`（日志与追溯） |
| `global` | `seed`，`device`（cpu/cuda），`log_level` |
| `prediction.temporal` | 回看窗口、预报步数 |
| `prediction.svgp` | 诱导点数、`jitter`、`whiten`、核周期、缩放器 `eps`、`num_features` 等 |
| `scenario` | MC 样本数、`num_clusters`、`random_state`、`reduction_method`、`mc_variable_jitter_scale` |
| `optimization` | 规划_horizon，`solver_config`（时限、mip_gap、线程），风浪 **hard / soft** |
| `business` | 船舶航速、`fuel_cost_per_km`、`scheduling` 停机惩罚与优先级系数 |
| `farm_layout` | 母港、风机经纬度、`distance`（度量、缓存、地球半径_km） |
| `tasks` | 每条任务：`turbine_id`，`duration_h`，`priority`，`deadline_h`（不从拓扑读时长） |
| `experiment.cache` | 是否启用缓存、`hash_params`（是否把整个 YAML 指纹并入缓存键）、`cache_dir` |

**说明**：YAML 中超集字段可被 Pydantic 保留（模型 `extra="allow"`），便于你在不改动校验逻辑的情况下扩展试验参数。

### 4.2 加载与可选路径覆盖

- **`load_algo_params(path)`**：读 YAML → `AlgoParams`。
- **`algo_params_fingerprint(params)`**：整块配置的稳定哈希 → 流水线在 `hash_params: true` 时写入缓存键。
- **`apply_global_random_seed(seed)`**：对齐 `numpy` / `torch` / 标准库 `random`。
- **`ALGO_CONFIG_YAML`**：通过 **`config/settings.py`** 里 `AlgoPathsSettings` 覆盖默认 YAML 路径。

---

## 五、核心业务模块如何协作？

### 5.1 数据契约：`core/schemas`

- **`ForecastResult`**：时间与逐点均值、方差，供采样器消费。
- **`ScenarioSet`**：多条 **`WeatherScenario`**（`series` 展平：`[wind…][wave…]` 约定）、权重和为 1。
- **`TaskNode` / `RoutingSolution`**：MILP 节点与求解输出路径。

**禁止**在此目录写求解或训练逻辑——只保留 **数据结构 + 校验**。

### 5.2 接口：`core/interfaces.py`

- **`BasePredictor.predict` → `ForecastResult`**
- **`BaseGenerator.generate` → `ScenarioSet`**
- **`BaseOptimizer.solve` → `RoutingSolution`**

新的预测器/生成器/优化器应实现上述协议，即可被 **`WorkflowPipeline`** 编排。

### 5.3 三大算法模块

1. **`svgp_predictor`**：`SVGPModelPredictor`，推荐 **`SVGPModelPredictor.from_algo_params(params)`**。内部使用配置的 **`device`、`jitter`、cholesky 上下文、likelihood 噪声下界**，以及 **`whiten`** 在变分策略上的切换。
2. **`scenario_generator`**：`KMedoidsGenerator.from_algo_params`；蒙特卡洛与 **KMeans / KMedoids** 均使用配置的 **`scenario.random_state`**（及采样器自带的 `Generator` 种子）。
3. **`milp_optimizer`**：`PuLPOptimizer.from_algo_params`；读取 **航程、风浪 hard 禁航槽位、风浪 soft 的期望罚金**进目标，`solver_config` 映射至 **PuLP CBC**。

### 5.4 调度与缓存：`core/pipeline.py`

- 顺序：**预测 → 场景 → 优化**。
- 支持 **`cache_enabled`**；若 **`algo_params_fingerprint`** 传入，则每阶段缓存键附带 **整配置摘要**，改 YAML 即 **自动作废**旧缓存。
- 组件可实现 **`cache_key_fields()`**，在指纹之外补充本模块的配置片段哈希（预测/场景/优化器已实现）。

### 5.5 任务装配：`core/tasks/builder.py`

根据 **`farm_layout.depot`、`farm_layout.turbines`、`tasks:`** 列表生成 **`TaskNode` 序列**（索引 0 为港），保证 MILP **只调度「任务表」**，避免把维护时长耦合进单纯拓扑字段。

---

## 六、数据目录约定

| 路径 | 用途 |
|------|------|
| `data/01_raw_era5/` | 放置分时段 ERA5 `.nc`，`run_experiment` 会搜索第一个时间段子目录 |
| `data/04_cache_store/`（默认，`experiment.cache.cache_dir`） | `.json` 等缓存工件；文件名即内容哈希 |

实验脚本可把切片后的 ERA5 **写到同一缓存目录**下，便于与缓存策略统一备份。

---

## 七、如何运行

1. （建议）创建虚拟环境，`pip install -r wind_algorithm_core/requirements.txt`。  
   若使用 **K-Medoids**，需安装 **`scikit-learn-extra`**。
2. 将原始 ERA5 放入 **`wind_algorithm_core/data/01_raw_era5/<时段>/`**。
3. 按需修改 **`wind_algorithm_core/config/algo_params.yaml`**。
4. 在 **`wind_algorithm_core`** 目录下执行：

```bash
python run_experiment.py
```

入口会：**加载 YAML → 日志打印 `config_version` / `experiment_name` → 设种子 → 装配 Pipeline → NetCDF 切片 → 端到端求解**。

---

## 八、扩展与注意事项

1. **换求解器**：当前实现与 **PuLP + CBC** 绑定；若在 YAML 中写 `gurobi`/`cplex` 等，需自行实现对应 **`BaseOptimizer`** 并在入口处替换注入。
2. **`core/tasks/simulator.py`**：为 **动态故障/任务生成草稿**，与主线 **`AlgoParams`/`TaskSchedulingSpec`** 可能尚未完全一致；主线实验以 **静态 `tasks:` YAML + builder** 为准。
3. **生产阈值**：YAML 里的风浪 hard/soft、惩罚系数常为 **研究与打通链路** 用途，上线前应 **按海区规范与海工标准**单独标定。
4. **更完整的逐文件注释**：见 **`wenjianjiegou.md`**。

---

## 九、与设计规范的关系

若 **`wind_algorithm_core` 旁的 `.cursorrules`** 仍存在，请以其中 **「schemas 是唯一跨模块数据结构、中文注释风格」** 等团队规范为准；本 README 偏重 **架构解构与上手路径**，不与具体业务案例绑定。
