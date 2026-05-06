"""缓存对象序列化策略。"""

import importlib
import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel


class SerializationFormat(str, Enum):
    """缓存序列化格式枚举。"""

    JSON = "json"
    PICKLE = "pkl"
    TORCH = "pt"


class SerializedPayload(BaseModel):
    """序列化后的缓存载荷。

    Attributes:
        data: 已序列化的二进制数据。
        extension: 建议使用的文件扩展名。
    """

    data: bytes
    extension: str


class SerializerStrategy(ABC):
    """序列化策略抽象基类。"""

    extension: str

    @abstractmethod
    def dumps(self, value: Any) -> bytes:
        """将对象序列化为二进制数据。

        Args:
            value: 待序列化对象。

        Returns:
            序列化后的二进制数据。
        """

    @abstractmethod
    def loads(self, data: bytes) -> Any:
        """从二进制数据反序列化对象。

        Args:
            data: 序列化后的二进制数据。

        Returns:
            反序列化后的对象。
        """


class PydanticJSONSerializer(SerializerStrategy):
    """Pydantic 模型 JSON 序列化策略。"""

    extension = SerializationFormat.JSON.value

    def dumps(self, value: BaseModel) -> bytes:
        """将 Pydantic 模型序列化为 JSON 字节。

        Args:
            value: Pydantic 模型实例。

        Returns:
            JSON UTF-8 字节数据。
        """

        payload = {
            "__serializer__": "pydantic",
            "module": value.__class__.__module__,
            "class": value.__class__.__qualname__,
            "data": value.model_dump(mode="json"),
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def loads(self, data: bytes) -> BaseModel | dict[str, Any]:
        """将 JSON 字节反序列化为 Pydantic 模型。

        Args:
            data: JSON UTF-8 字节数据。

        Returns:
            成功定位模型类时返回 Pydantic 模型实例，否则返回原始字典载荷。
        """

        payload = json.loads(data.decode("utf-8"))
        model_class = self._resolve_model_class(
            module_name=payload["module"],
            class_name=payload["class"],
        )
        if model_class is None:
            return payload
        return model_class.model_validate(payload["data"])

    def _resolve_model_class(self, module_name: str, class_name: str) -> type[BaseModel] | None:
        """根据模块名和类名解析 Pydantic 模型类。

        Args:
            module_name: 模型所在模块名。
            class_name: 模型类限定名。

        Returns:
            解析成功时返回模型类，否则返回 None。
        """

        try:
            module = importlib.import_module(module_name)
            resolved: Any = module
            for part in class_name.split("."):
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError):
            return None

        if isinstance(resolved, type) and issubclass(resolved, BaseModel):
            return resolved
        return None


class PickleSerializer(SerializerStrategy):
    """普通 Python 对象 Pickle 序列化策略。"""

    extension = SerializationFormat.PICKLE.value

    def dumps(self, value: Any) -> bytes:
        """将普通对象序列化为 Pickle 字节。

        Args:
            value: 待序列化对象。

        Returns:
            Pickle 二进制数据。
        """

        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> Any:
        """从 Pickle 字节反序列化对象。

        Args:
            data: Pickle 二进制数据。

        Returns:
            反序列化后的 Python 对象。
        """

        return pickle.loads(data)


class TorchSerializerPlaceholder(SerializerStrategy):
    """PyTorch Tensor 序列化预留策略。

    当前项目尚未引入 torch 依赖，因此该策略仅保留接口占位。
    """

    extension = SerializationFormat.TORCH.value

    def dumps(self, value: Any) -> bytes:
        """预留 torch.save 序列化接口。

        Args:
            value: 未来的 Tensor 或 PyTorch 对象。

        Raises:
            NotImplementedError: 当前未引入 torch，暂不支持该格式。
        """

        raise NotImplementedError("Torch serialization is reserved but not enabled")

    def loads(self, data: bytes) -> Any:
        """预留 torch.load 反序列化接口。

        Args:
            data: 未来的 PyTorch 序列化数据。

        Raises:
            NotImplementedError: 当前未引入 torch，暂不支持该格式。
        """

        raise NotImplementedError("Torch deserialization is reserved but not enabled")


class CacheSerializer:
    """缓存序列化门面。"""

    def __init__(self) -> None:
        """初始化序列化策略注册表。"""

        self._json_serializer = PydanticJSONSerializer()
        self._pickle_serializer = PickleSerializer()
        self._strategies: dict[str, SerializerStrategy] = {
            self._json_serializer.extension: self._json_serializer,
            self._pickle_serializer.extension: self._pickle_serializer,
            TorchSerializerPlaceholder.extension: TorchSerializerPlaceholder(),
        }

    def dumps(self, value: Any) -> SerializedPayload:
        """根据对象类型选择序列化策略。

        Args:
            value: 待缓存对象。

        Returns:
            序列化载荷，包含二进制数据与扩展名。
        """

        if isinstance(value, BaseModel):
            return SerializedPayload(
                data=self._json_serializer.dumps(value),
                extension=self._json_serializer.extension,
            )

        return SerializedPayload(
            data=self._pickle_serializer.dumps(value),
            extension=self._pickle_serializer.extension,
        )

    def loads(self, data: bytes, extension: str) -> Any:
        """根据文件扩展名选择反序列化策略。

        Args:
            data: 已序列化的二进制数据。
            extension: 文件扩展名。

        Returns:
            反序列化后的对象。

        Raises:
            ValueError: 当扩展名没有对应策略时抛出。
        """

        normalized_extension = extension.lstrip(".")
        strategy = self._strategies.get(normalized_extension)
        if strategy is None:
            raise ValueError(f"Unsupported cache serialization extension: {extension}")
        return strategy.loads(data)
