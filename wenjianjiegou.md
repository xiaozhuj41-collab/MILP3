# wind_algorithm_core — 文件结构说明

> 与仓库内实际目录对齐；`core/schemas` 仅存放契约，不含业务算法逻辑。

```text
wind_algorithm_core/
├── data/
│   ├── 01_raw_era5/                 # 原始 ERA5 NetCDF（oper / wave 等）
│   ├── 02_svgp_tensors/             # （可选）预处理的张量/训练数据占位
│   ├── 03_milp_scenarios/           # （可选）场景树序列化占位
│   └── 04_cache_store/              # 流水线各阶段 JSON 等持久化缓存（键为参数 SHA256）
│
├── config/                          # 标准配置层（YAML + Pydantic）
│   ├── __init__.py                  # 导出 load_algo_params / algo_params_fingerprint / apply_global_random_seed
│   ├── algo_params.yaml             # 单一事实来源：prediction / scenario / optimization / business / topology / tasks …
│   ├── models.py                    # AlgoParams 及子节 Pydantic 模型（extra="allow" 预留扩展）
│   ├── loader.py                    # YAML 加载 + 配置整体指纹（参与缓存失效）
│   ├── reproducibility.py           # random / numpy / torch 统一种子
│   └── settings.py                  # AlgoPathsSettings；环境变量 ALGO_CONFIG_YAML 可覆盖 YAML 路径
│
├── core/
│   ├── schemas/                     # 领域数据契约（严禁写算法）
│   │   ├── __init__.py
│   │   ├── base.py                  # Coordinate, TimeWindow …
│   │   ├── svgp.py                  # ForecastResult(time, mean, variance)
│   │   ├── scenario.py              # WeatherScenario, ScenarioSet（权重和约束）
│   │   └── milp.py                  # TaskNode(id, loc, duration, priority, deadline_h …), RoutingSolution
│   │
│   ├── interfaces.py                # BasePredictor / BaseGenerator / BaseOptimizer 协议定义
│   │
│   ├── cache/                       # 持久化缓存子系统
│   │   ├── hasher.py                # StableHasher：Pydantic/字典等的稳定 SHA256
│   │   ├── disk_store.py            # 按哈希落盘文件
│   │   ├── cache_manager.py         # get/set 统一入口
│   │   ├── serializer.py            # 值序列化为 json/msgpack 等
│   │   └── policies.py              # LRU 等容量策略
│   │
│   ├── svgp_predictor/              # SVGP 预测（PyTorch / GPyTorch）
│   │   ├── dataset.py               # nc_to_tensor、滑动窗口张量构造
│   │   ├── kernel.py                # 气象组合核（Matern + Periodic，超参来自配置）
│   │   ├── model.py                 # ApproximateGP、变分策略（whiten 可切换）、似然噪声下界
│   │   └── predictor.py             # SVGPModelPredictor(BasePredictor)；from_algo_params
│   │
│   ├── scenario_generator/          # 场景生成（NumPy / scikit-learn[-extra]）
│   │   ├── sampler.py               # monte_carlo_sample（显式 random_seed）
│   │   ├── reducer.py               # k_means_reduce / kmedoids_reduce（削减 + 权重）
│   │   └── generator.py             # KMedoidsGenerator(BaseGenerator)；from_algo_params
│   │
│   ├── milp_optimizer/              # MILP / VRP（PuLP / CBC）
│   │   ├── vrp_model.py             # PuLP 决策变量与模型骨架（x, t, y, z …）
│   │   ├── constraints.py           # 流守恒、MTZ、时间传递、时段、风浪硬约束、截止时间、软期望罚金权重计算
│   │   ├── cost_calculator.py       # Haversine / 航程时间 / 距离矩阵构造（可调地球半径）
│   │   ├── distance_matrix_cache.py # farm_layout.distance.cache 时读写 CacheManager
│   │   └── solver.py                # PuLPOptimizer(BaseOptimizer)，solver_config + 风浪 hard/soft
│   │
│   ├── tasks/                       # 「拓扑与调度任务」装配
│   │   ├── __init__.py
│   │   ├── builder.py               # build_task_nodes_for_milp：depot + tasks YAML → list[TaskNode]
│   │   └── simulator.py             # （草稿/占位）动态故障仿真；尚未与当前 AlgoParams 完全对齐时可忽略运行路径
│   │
│   └── pipeline.py                  # WorkflowPipeline：三阶段编排 + algo_params_fingerprint + cache_enabled
│
├── tests/                           # 单元测试占位（按需扩充）
├── requirements.txt                # pydantic / torch / gpytorch / pulp / xarray / PyYAML / scikit-learn-extra …
└── run_experiment.py                 # 入口：加载配置 → 种子 → ERA5 切片 → Pipeline.run
```

## 与设计文档的差异提示

| 条目 | 说明 |
|------|------|
| 配置读取 | 以 `algo_params.yaml` + `config/models.py` 为准，不再是仅 `settings.py` 平铺字段 |
| 场景削减 | `reducer.py` 同时提供 **KMeans** 与 **KMedoids**（`scikit-learn-extra`），由 YAML `scenario.reduction_method` 选择 |
| MILP 求解器文案 | YAML 可有 `solver: pulp_cbc` 等标识；当前实现绑定 **PuLP CBC CLI** |
| tasks | 运维时长仅从 `tasks:` 段落读取，风机坐标来自 `farm_layout.turbines` |
