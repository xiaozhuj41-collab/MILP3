
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from config.models import AlgoParams, Task  # 你的 Pydantic 模型


# =========================
# 内部工具函数
# =========================

def _sample_fault_type(fault_types: List[Dict]) -> Dict:
    """按概率分布采样故障类型"""
    probs = [ft["prob"] for ft in fault_types]
    return random.choices(fault_types, weights=probs, k=1)[0]


def _sample_duration(duration_range):
    return round(random.uniform(duration_range[0], duration_range[1]), 2)


def _load_previous_state(state_file: Path) -> Dict:
    if not state_file.exists():
        return {}

    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_file: Path, state: Dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# =========================
# 核心函数
# =========================

def generate_daily_faults(
    params: AlgoParams,
) -> List[Task]:
    """
    生成每日动态故障任务（工业级实现）

    特性：
    - 支持设备异质性（failure_rate）
    - 支持故障类型分布
    - 支持故障持续（state persistence）
    - 完全受全局随机种子控制
    """

    fault_cfg = params.fault_simulation

    if not fault_cfg.enabled:
        return []

    default_prob = fault_cfg.default_failure_prob
    fault_types = fault_cfg.fault_types

    # =========================
    # 状态管理（故障持续）
    # =========================
    state_file = Path(fault_cfg.persistence.state_file)
    prev_state = _load_previous_state(state_file)

    new_state = {}
    tasks: List[Task] = []

    # =========================
    # 遍历风机
    # =========================
    for turbine in params.farm_layout.turbines:

        tid = turbine.id

        # ---------- 情况1：已有未完成故障 ----------
        if tid in prev_state:
            task_data = prev_state[tid]

            # 继续存在（简单模型：一定延续一天）
            new_state[tid] = task_data

            task = Task(**task_data)
            tasks.append(task)
            continue

        # ---------- 情况2：新故障生成 ----------
        prob = getattr(turbine, "failure_rate", default_prob)

        if random.random() < prob:

            fault = _sample_fault_type(fault_types)

            duration = _sample_duration(fault["duration_range"])

            task_data = {
                "turbine_id": tid,
                "duration_h": duration,
                "priority": fault["priority"],
            }

            task = Task(**task_data)

            tasks.append(task)
            new_state[tid] = task_data

    # =========================
    # 保存状态（用于下一天）
    # =========================
    if fault_cfg.persistence.enabled:
        _save_state(state_file, new_state)

    # =========================
    # 审计日志（每日快照）
    # =========================
    output_dir = Path("data/05_dynamic_faults")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"daily_tasks_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([t.model_dump() for t in tasks], f, indent=2, ensure_ascii=False)

    return tasks