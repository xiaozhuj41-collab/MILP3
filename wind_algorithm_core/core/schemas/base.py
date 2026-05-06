"""领域基础数据结构。"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class Coordinate(BaseModel):
    """地理坐标。

    Attributes:
        lat: 纬度，取值范围为 [-90, 90]。
        lon: 经度，取值范围为 [-180, 180]。
    """

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class TimeWindow(BaseModel):
    """时间窗约束。

    Attributes:
        start: 时间窗开始时间。
        end: 时间窗结束时间。
    """

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_time_order(self) -> "TimeWindow":
        """校验时间窗结束时间必须晚于开始时间。

        Returns:
            校验后的时间窗对象。

        Raises:
            ValueError: 当结束时间早于或等于开始时间时抛出。
        """

        if self.end <= self.start:
            raise ValueError("TimeWindow.end must be later than TimeWindow.start")
        return self
