"""缓存底层磁盘读写。"""

from pathlib import Path
from typing import Iterable


class CacheIOError(RuntimeError):
    """缓存文件读写异常。"""


class DiskStore:
    """基于本地文件系统的缓存存储。

    Attributes:
        cache_dir: 缓存文件所在目录。
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """初始化磁盘缓存存储。

        Args:
            cache_dir: 缓存目录，默认指向项目数据缓存目录。

        Raises:
            CacheIOError: 当缓存目录创建失败时抛出。
        """

        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else project_root / "data/04_cache_store"
        self._ensure_cache_dir()

    def write(self, cache_hash: str, extension: str, data: bytes) -> Path:
        """写入缓存文件。

        Args:
            cache_hash: 缓存键对应的哈希值。
            extension: 文件扩展名。
            data: 待写入的二进制数据。

        Returns:
            写入后的缓存文件路径。

        Raises:
            CacheIOError: 当文件写入失败时抛出。
        """

        path = self.build_path(cache_hash, extension)
        try:
            self._ensure_cache_dir()
            path.write_bytes(data)
        except OSError as exc:
            raise CacheIOError(f"Failed to write cache file: {path}") from exc
        return path

    def read(self, cache_hash: str, extension: str) -> bytes | None:
        """读取缓存文件。

        Args:
            cache_hash: 缓存键对应的哈希值。
            extension: 文件扩展名。

        Returns:
            缓存二进制数据；文件不存在时返回 None。

        Raises:
            CacheIOError: 当文件读取失败时抛出。
        """

        path = self.build_path(cache_hash, extension)
        return self.read_path(path)

    def read_path(self, path: Path) -> bytes | None:
        """按完整路径读取缓存文件。

        Args:
            path: 缓存文件完整路径。

        Returns:
            缓存二进制数据；文件不存在时返回 None。

        Raises:
            CacheIOError: 当文件读取失败时抛出。
        """

        if not path.exists():
            return None

        try:
            data = path.read_bytes()
            # 更新访问时间，供 LRU 策略识别最近使用文件。
            path.touch(exist_ok=True)
            return data
        except OSError as exc:
            raise CacheIOError(f"Failed to read cache file: {path}") from exc

    def find(self, cache_hash: str) -> Path | None:
        """按哈希查找缓存文件。

        Args:
            cache_hash: 缓存键对应的哈希值。

        Returns:
            匹配的缓存文件路径；不存在时返回 None。

        Raises:
            CacheIOError: 当目录遍历失败时抛出。
        """

        try:
            matches = sorted(self.cache_dir.glob(f"{cache_hash}.*"))
        except OSError as exc:
            raise CacheIOError(f"Failed to list cache directory: {self.cache_dir}") from exc

        if not matches:
            return None
        return matches[0]

    def iter_files(self) -> Iterable[Path]:
        """遍历缓存目录下的普通文件。

        Returns:
            缓存文件路径迭代器。

        Raises:
            CacheIOError: 当目录遍历失败时抛出。
        """

        try:
            return tuple(path for path in self.cache_dir.iterdir() if path.is_file())
        except OSError as exc:
            raise CacheIOError(f"Failed to iterate cache directory: {self.cache_dir}") from exc

    def build_path(self, cache_hash: str, extension: str) -> Path:
        """构造缓存文件路径。

        Args:
            cache_hash: 缓存键对应的哈希值。
            extension: 文件扩展名。

        Returns:
            缓存文件路径。
        """

        normalized_extension = extension.lstrip(".")
        return self.cache_dir / f"{cache_hash}.{normalized_extension}"

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在。

        Raises:
            CacheIOError: 当目录创建失败时抛出。
        """

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheIOError(f"Failed to create cache directory: {self.cache_dir}") from exc
