"""全局随机数可复现性工具。"""

from __future__ import annotations

import random

import numpy as np
import torch


def apply_global_random_seed(seed: int) -> None:
    """为 numpy / torch / Python ``random`` 注入统一种子。

    Args:
        seed: 非负整数种子，与配置 ``global.seed`` 对齐。
    """

    if seed < 0:
        raise ValueError("random seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
