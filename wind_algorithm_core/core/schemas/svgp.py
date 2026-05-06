"""SVGP 预测结果数据契约。"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ForecastResult(BaseModel):
    """气象预测结果。

    Attributes:
        time: 预测时间序列。
        mean: 每个时间点对应的预测均值。
        variance: 每个时间点对应的预测方差。
    """

    time: list[datetime] = Field(min_length=1)
    mean: list[float] = Field(min_length=1)
    variance: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_series_shape(self) -> "ForecastResult":
        """校验时间、均值、方差序列长度一致且方差非负。

        Returns:
            校验后的预测结果对象。

        Raises:
            ValueError: 当序列长度不一致或方差为负数时抛出。
        """

        expected_length = len(self.time)
        if len(self.mean) != expected_length or len(self.variance) != expected_length:
            raise ValueError("time, mean and variance must have the same length")

        if any(value < 0.0 for value in self.variance):
            raise ValueError("variance values must be non-negative")

        return self
