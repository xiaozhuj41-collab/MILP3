"""端到端实验入口脚本。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from config import algo_params_fingerprint, apply_global_random_seed, load_algo_params
from config.models import AlgoParams
from config.settings import AlgoPathsSettings
from core.cache.cache_manager import CacheManager
from core.pipeline import WorkflowPipeline
from core.tasks import build_task_nodes_for_milp

LOGGER = logging.getLogger("wind_algorithm_core.experiment")


def main() -> int:
    """运行端到端算法实验。

    Returns:
        进程退出码。0 表示成功，1 表示失败。
    """

    started_at = time.perf_counter()

    try:
        project_root = Path(__file__).resolve().parent

        paths_cfg = AlgoPathsSettings()
        params = load_algo_params(paths_cfg.config_yaml)
        _configure_logging(params.global_.log_level)

        apply_global_random_seed(params.global_.seed)

        cache_dir_setting = Path(params.experiment.cache.cache_dir)
        cache_root = cache_dir_setting if cache_dir_setting.is_absolute() else project_root / cache_dir_setting
        cache_mgr = CacheManager(cache_dir=cache_root)
        algo_digest = algo_params_fingerprint(params) if params.experiment.cache.hash_params else None

        LOGGER.info(
            "Experiment bootstrap | config_version=%s | experiment_name=%s | yaml=%s | seed=%s",
            params.meta.config_version,
            params.meta.experiment_name,
            paths_cfg.config_yaml.resolve(),
            params.global_.seed,
        )

        pipeline = _build_pipeline_from_params(
            params=params,
            cache_manager=cache_mgr,
            algo_params_fingerprint=algo_digest,
        )

        raw_period_dir = _find_first_period_dir(project_root / "data" / "01_raw_era5")
        sliced_nc_path, point = _slice_first_100_hours(
            raw_period_dir=raw_period_dir,
            output_path=cache_root / "experiment_era5_100h.nc",
        )

        task_nodes = build_task_nodes_for_milp(params)

        LOGGER.info("Start workflow | input=%s", sliced_nc_path)
        solution = pipeline.run(
            nc_file_path=sliced_nc_path,
            task_nodes=task_nodes,
            latitude=point["latitude"],
            longitude=point["longitude"],
        )

        LOGGER.info("Final RoutingSolution")
        LOGGER.info("  total_cost: %.4f", solution.total_cost)
        for route_index, route in enumerate(solution.routes, start=1):
            LOGGER.info("  route_%d: %s", route_index, " -> ".join(route))

        elapsed = time.perf_counter() - started_at
        LOGGER.info("Experiment finished successfully | elapsed=%.3fs", elapsed)
        return 0
    except Exception as exc:  # noqa: BLE001 - 实验入口需要兜底打印清晰错误。
        LOGGER.exception("Experiment failed: %s", exc)
        return 1


def _configure_logging(level_name: str) -> None:
    """根据配置初始化根日志 Handler 与日志级别。"""

    resolved = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _build_pipeline_from_params(
    *,
    params: AlgoParams,
    cache_manager: CacheManager,
    algo_params_fingerprint: str | None,
) -> WorkflowPipeline:
    """由 ``AlgoParams`` 装配端到端流水线。"""

    from core.milp_optimizer import PuLPOptimizer  # noqa: PLC0415
    from core.scenario_generator import KMedoidsGenerator  # noqa: PLC0415
    from core.svgp_predictor import SVGPModelPredictor  # noqa: PLC0415

    if not isinstance(params, AlgoParams):
        raise TypeError("params must be AlgoParams")

    predictor = SVGPModelPredictor.from_algo_params(params)
    generator = KMedoidsGenerator.from_algo_params(params)
    optimizer = PuLPOptimizer.from_algo_params(params, distance_cache_manager=cache_manager)

    LOGGER.info("Pipeline stage 1 ready: SVGP prediction (config-driven)")
    LOGGER.info("Pipeline stage 2 ready: Monte Carlo scenario generation (config-driven)")
    LOGGER.info("Pipeline stage 3 ready: MILP optimization (config-driven)")
    return WorkflowPipeline(
        predictor=predictor,
        generator=generator,
        optimizer=optimizer,
        cache_manager=cache_manager,
        logger=LOGGER,
        cache_enabled=bool(params.experiment.cache.enabled),
        algo_params_fingerprint=algo_params_fingerprint,
    )


def _find_first_period_dir(raw_data_dir: Path) -> Path:
    """查找原始 ERA5 目录下第一个时间段子目录。

    Args:
        raw_data_dir: 原始 ERA5 数据目录。

    Returns:
        第一个时间段子目录路径，例如 ``2025_1_6``。

    Raises:
        FileNotFoundError: 当目录不存在或没有时间段子目录时抛出。
    """

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"ERA5 raw data directory does not exist: {raw_data_dir}")

    period_dirs = sorted(path for path in raw_data_dir.iterdir() if path.is_dir())
    if not period_dirs:
        raise FileNotFoundError(f"No period subdirectory found under: {raw_data_dir}")

    LOGGER.info("Found ERA5 period directory: %s", period_dirs[0])
    return period_dirs[0]


def _slice_first_100_hours(raw_period_dir: Path, output_path: Path) -> tuple[Path, dict[str, float]]:
    """合并 oper/wave 文件后切片前 100 个小时并保存为实验文件。

    Args:
        raw_period_dir: 原始 ERA5 时间段目录，内部包含 oper 与 wave 文件。
        output_path: 切片后的实验 NetCDF 输出路径。

    Returns:
        二元组，包含切片文件路径和单点经纬度。

    Raises:
        KeyError: 当数据集中缺少时间坐标时抛出。
    """

    import xarray as xr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    oper_files, wave_files = _find_oper_wave_files(raw_period_dir)
    oper_dataset = _open_and_combine_nc_files(oper_files)
    wave_dataset = _open_and_combine_nc_files(wave_files)

    try:
        dataset = xr.merge(
            [oper_dataset, wave_dataset],
            compat="override",
            join="inner",
            combine_attrs="override",
        )
        dataset = _normalize_time_coordinate(dataset)
        _assert_required_variables(dataset)
        time_dim = _resolve_time_dim(dataset)
        point = _resolve_reference_point(dataset)

        LOGGER.info(
            "Merged ERA5 files | period=%s | oper_files=%d | wave_files=%d",
            raw_period_dir,
            len(oper_files),
            len(wave_files),
        )
        # 真实数据切片：仅取最开始 2000 个小时，降低初次端到端实验耗时。
        sliced = dataset.isel({time_dim: slice(0, 2000)})
        sliced.load()
        sliced.to_netcdf(output_path)
    finally:
        oper_dataset.close()
        wave_dataset.close()

    LOGGER.info(
        "Sliced ERA5 data | output=%s | latitude=%.4f | longitude=%.4f",
        output_path,
        point["latitude"],
        point["longitude"],
    )
    return output_path, point


def _find_oper_wave_files(raw_period_dir: Path) -> tuple[list[Path], list[Path]]:
    """在时间段目录中查找 oper 与 wave NetCDF 文件。

    Args:
        raw_period_dir: 原始 ERA5 时间段目录。

    Returns:
        二元组，分别为 oper 文件列表与 wave 文件列表。

    Raises:
        FileNotFoundError: 当任一类文件缺失时抛出。
    """

    nc_files = sorted(raw_period_dir.glob("*.nc"))
    oper_files = [path for path in nc_files if "oper" in path.name.lower()]
    wave_files = [path for path in nc_files if "wave" in path.name.lower()]

    if not oper_files:
        raise FileNotFoundError(f"No oper .nc file found under: {raw_period_dir}")
    if not wave_files:
        raise FileNotFoundError(f"No wave .nc file found under: {raw_period_dir}")

    LOGGER.info("Found oper files: %s", [path.name for path in oper_files])
    LOGGER.info("Found wave files: %s", [path.name for path in wave_files])
    return oper_files, wave_files


def _open_and_combine_nc_files(nc_files: list[Path]) -> Any:
    """读取并合并多个 NetCDF 文件。

    这里采用逐个 open_dataset + load + combine_by_coords 的方式，避免
    open_mfdataset 在未安装 dask 的环境中失败。

    Args:
        nc_files: 待合并的 NetCDF 文件列表。

    Returns:
        合并后的 xarray Dataset。
    """

    import xarray as xr

    datasets = []
    for nc_file in nc_files:
        with xr.open_dataset(nc_file) as dataset:
            datasets.append(_normalize_time_coordinate(dataset).load())

    if len(datasets) == 1:
        return datasets[0]

    return xr.combine_by_coords(
        datasets,
        combine_attrs="override",
    )


def _assert_required_variables(dataset: Any) -> None:
    """确认合并后的 Dataset 同时包含风速与浪高变量。

    Args:
        dataset: 合并后的 xarray Dataset。

    Raises:
        KeyError: 当 u10、v10 或 swh 任一变量缺失时抛出。
    """

    required_variables = ("u10", "v10", "swh")
    missing = [name for name in required_variables if name not in dataset]
    if missing:
        raise KeyError(
            "Merged ERA5 dataset is missing required variables: "
            + ", ".join(missing)
        )


def _normalize_time_coordinate(dataset: Any) -> Any:
    """统一 ERA5 时间坐标名称为 time。

    Args:
        dataset: xarray Dataset。

    Returns:
        时间坐标已标准化的数据集。
    """

    if "time" in dataset.coords or "time" in dataset.dims:
        return dataset
    if "valid_time" in dataset.coords or "valid_time" in dataset.dims:
        return dataset.rename({"valid_time": "time"})
    return dataset


def _resolve_time_dim(dataset: Any) -> str:
    """解析时间维度名称。

    Args:
        dataset: xarray Dataset。

    Returns:
        时间维度名称。

    Raises:
        KeyError: 当缺少时间维度时抛出。
    """

    if "time" in dataset.dims:
        return "time"
    if "time" in dataset.coords:
        return "time"
    raise KeyError("ERA5 dataset must contain a time dimension")


def _resolve_reference_point(dataset: Any) -> dict[str, float]:
    """从 ERA5 数据集中解析一个单点经纬度。

    Args:
        dataset: xarray Dataset。

    Returns:
        包含 latitude 和 longitude 的字典。
    """

    lat_name = _first_existing_name(dataset, ("latitude", "lat"))
    lon_name = _first_existing_name(dataset, ("longitude", "lon"))

    if lat_name is None or lon_name is None:
        return {"latitude": 21.5, "longitude": 110.5}

    latitude = float(dataset[lat_name].values.reshape(-1)[0])
    longitude = float(dataset[lon_name].values.reshape(-1)[0])
    return {"latitude": latitude, "longitude": longitude}


def _first_existing_name(dataset: Any, candidates: tuple[str, ...]) -> str | None:
    """查找第一个存在的坐标或维度名称。

    Args:
        dataset: xarray Dataset。
        candidates: 候选名称。

    Returns:
        命中的名称；未命中时返回 None。
    """

    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    return None


if __name__ == "__main__":
    sys.exit(main())
