"""持久化缓存统一门面。"""

from pathlib import Path
from typing import Any

from .disk_store import CacheIOError, DiskStore
from .hasher import StableHasher
from .policies import CachePolicyError, SizeBasedLRU
from .serializer import CacheSerializer


class CacheManagerError(RuntimeError):
    """缓存门面执行异常。"""


class CacheManager:
    """持久化缓存统一入口。

    对外仅暴露 get 与 set 方法，内部负责哈希、序列化、磁盘读写以及
    基于目录体积的 LRU 淘汰。

    Attributes:
        store: 底层磁盘存储。
        serializer: 缓存序列化门面。
        policy: 缓存淘汰策略。
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        """初始化缓存管理器。

        Args:
            cache_dir: 缓存目录，默认使用项目数据缓存目录。
            max_bytes: 缓存目录最大体积，默认 2GB。
        """

        self.store = DiskStore(cache_dir=cache_dir)
        self.serializer = CacheSerializer()
        self.policy = SizeBasedLRU(cache_dir=self.store.cache_dir, max_bytes=max_bytes)

    def get(self, key_params: dict[str, Any]) -> Any | None:
        """读取缓存对象。

        Args:
            key_params: 用于生成缓存键的参数字典。

        Returns:
            命中时返回反序列化后的缓存对象；未命中时返回 None。

        Raises:
            CacheManagerError: 当哈希、文件读取或反序列化失败时抛出。
        """

        try:
            cache_hash = StableHasher.hash(key_params)
            path = self.store.find(cache_hash)
            if path is None:
                return None

            data = self.store.read_path(path)
            if data is None:
                return None

            return self.serializer.loads(data=data, extension=path.suffix)
        except (CacheIOError, TypeError, ValueError, OSError) as exc:
            raise CacheManagerError("Failed to get cached value") from exc

    def set(self, key_params: dict[str, Any], value: Any) -> Path:
        """写入缓存对象。

        Args:
            key_params: 用于生成缓存键的参数字典。
            value: 待缓存对象。

        Returns:
            写入后的缓存文件路径。

        Raises:
            CacheManagerError: 当哈希、序列化、淘汰或文件写入失败时抛出。
        """

        try:
            cache_hash = StableHasher.hash(key_params)
            payload = self.serializer.dumps(value)
            self.policy.ensure_capacity(incoming_bytes=len(payload.data))
            return self.store.write(
                cache_hash=cache_hash,
                extension=payload.extension,
                data=payload.data,
            )
        except (CacheIOError, CachePolicyError, TypeError, ValueError, OSError) as exc:
            raise CacheManagerError("Failed to set cached value") from exc
