"""YAML 算法配置加载与摘要。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.cache.hasher import StableHasher

from .models import AlgoParams, default_algo_params_path


def load_algo_params(yaml_path: Path | str | None = None) -> AlgoParams:
    """从 YAML 文件加载并校验为 ``AlgoParams``。

    Args:
        yaml_path: 配置文件路径；默认使用 ``config/algo_params.yaml``。

    Returns:
        经验证的配置对象。

    Raises:
        FileNotFoundError: 当 YAML 不存在时抛出。
        ValidationError: 当结构与契约不兼容时抛出。
    """

    path = Path(yaml_path) if yaml_path is not None else default_algo_params_path()
    if not path.is_file():
        raise FileNotFoundError(f"Algorithm config YAML not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise TypeError(f"Algorithm config root must be a mapping, got {type(raw)!r}")

    return AlgoParams.model_validate(raw)


def algo_params_fingerprint(params: AlgoParams) -> str:
    """对完整算法配置生成稳定哈希，用于缓存失效。

    Args:
        params: 根配置对象。

    Returns:
        SHA256 十六进制摘要。
    """

    return StableHasher.hash(params.model_dump(mode="json"))
