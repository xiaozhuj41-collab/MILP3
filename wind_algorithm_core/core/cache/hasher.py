"""缓存键稳定哈希生成器。"""

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class StableHasher:
    """为缓存键参数生成稳定的 SHA256 哈希值。

    该类会将 Pydantic 模型、字典、列表、元组以及基础数据类型转换为
    可稳定 JSON 序列化的结构，并对字典键进行排序，保证相同语义输入
    在不同运行中得到一致哈希。
    """

    @classmethod
    def hash(cls, value: Any) -> str:
        """生成稳定 SHA256 哈希值。

        Args:
            value: 待哈希的缓存键参数。

        Returns:
            64 位十六进制 SHA256 哈希字符串。

        Raises:
            TypeError: 当输入对象无法转换为稳定结构时抛出。
        """

        normalized = cls._normalize(value)
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        """将对象转换为稳定、可 JSON 序列化的结构。

        Args:
            value: 待规范化的任意对象。

        Returns:
            可稳定 JSON 序列化的对象。

        Raises:
            TypeError: 当遇到不支持的对象类型时抛出。
        """

        if isinstance(value, BaseModel):
            return {
                "__type__": value.__class__.__qualname__,
                "data": cls._normalize(value.model_dump(mode="json")),
            }

        if isinstance(value, dict):
            return {
                str(key): cls._normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }

        if isinstance(value, (list, tuple)):
            return [cls._normalize(item) for item in value]

        if isinstance(value, set):
            normalized_items = [cls._normalize(item) for item in value]
            return sorted(
                normalized_items,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, Decimal):
            return str(value)

        if isinstance(value, Path):
            return value.as_posix()

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        raise TypeError(f"Unsupported cache key type: {type(value)!r}")


def generate_hash(value: Any) -> str:
    """生成缓存键参数的稳定 SHA256 哈希值。

    Args:
        value: 待哈希的缓存键参数。

    Returns:
        64 位十六进制 SHA256 哈希字符串。
    """

    return StableHasher.hash(value)
