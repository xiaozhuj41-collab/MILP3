"""运行时可覆盖的配置路径等小设置。

算法超参请以 ``algo_params.yaml`` + :func:`config.load_algo_params` 为唯一事实来源。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .loader import load_algo_params
from .models import AlgoParams, default_algo_params_path


class AlgoPathsSettings(BaseSettings):
    """环境与 ``.env`` 中的可选路径覆盖。

    Attributes:
        config_yaml: 算法 YAML 路径，可通过环境变量 ``ALGO_CONFIG_YAML`` 覆盖。
    """

    config_yaml: Path = Field(default_factory=default_algo_params_path)

    model_config = SettingsConfigDict(
        env_prefix="ALGO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_resolved_algo_params() -> tuple[AlgoPathsSettings, AlgoParams]:
    """读取路径设置并加载校验后的 YAML 配置。

    Returns:
        二元组：(路径设置, 根算法配置)。
    """

    paths = AlgoPathsSettings()
    return paths, load_algo_params(paths.config_yaml)
