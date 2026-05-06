"""缓存淘汰策略。"""

from pathlib import Path


class CachePolicyError(RuntimeError):
    """缓存策略执行异常。"""


class SizeBasedLRU:
    """基于目录体积的 LRU 淘汰策略。

    当缓存目录总大小加上待写入文件大小超过阈值时，按照文件访问时间
    从旧到新删除缓存文件，直到释放出足够空间。

    Attributes:
        cache_dir: 缓存目录。
        max_bytes: 缓存目录允许的最大体积，默认 2GB。
    """

    def __init__(self, cache_dir: Path | str, max_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        """初始化 LRU 淘汰策略。

        Args:
            cache_dir: 缓存目录。
            max_bytes: 缓存目录最大体积，单位为字节。
        """

        self.cache_dir = Path(cache_dir)
        self.max_bytes = max_bytes

    def ensure_capacity(self, incoming_bytes: int) -> None:
        """确保缓存目录有足够空间写入新文件。

        Args:
            incoming_bytes: 即将写入的新文件大小，单位为字节。

        Raises:
            CachePolicyError: 当目录统计或文件删除失败时抛出。
        """

        if incoming_bytes > self.max_bytes:
            # 单个文件已超过阈值时，仍清理现有缓存，但无法保证写入后不超标。
            self._evict_until(0)
            return

        target_size = self.max_bytes - incoming_bytes
        self._evict_until(target_size)

    def current_size(self) -> int:
        """计算当前缓存目录总体积。

        Returns:
            缓存目录下普通文件总大小，单位为字节。

        Raises:
            CachePolicyError: 当文件状态读取失败时抛出。
        """

        total_size = 0
        try:
            for path in self._cache_files():
                total_size += path.stat().st_size
        except OSError as exc:
            raise CachePolicyError(f"Failed to calculate cache size: {self.cache_dir}") from exc
        return total_size

    def _evict_until(self, target_size: int) -> None:
        """删除最旧文件直到缓存目录体积不超过目标值。

        Args:
            target_size: 淘汰后的目标目录体积。

        Raises:
            CachePolicyError: 当文件状态读取或删除失败时抛出。
        """

        try:
            files = sorted(self._cache_files(), key=lambda path: path.stat().st_atime)
            total_size = sum(path.stat().st_size for path in files)

            for path in files:
                if total_size <= target_size:
                    break

                file_size = path.stat().st_size
                path.unlink()
                total_size -= file_size
        except OSError as exc:
            raise CachePolicyError(f"Failed to evict cache files: {self.cache_dir}") from exc

    def _cache_files(self) -> tuple[Path, ...]:
        """获取缓存目录下的普通文件。

        Returns:
            缓存文件路径元组。

        Raises:
            OSError: 当目录遍历失败时由调用方捕获。
        """

        if not self.cache_dir.exists():
            return ()
        return tuple(path for path in self.cache_dir.iterdir() if path.is_file())
