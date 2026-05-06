"""随机气象场景数据契约。"""

from pydantic import BaseModel, Field, model_validator


class WeatherScenario(BaseModel):
    """单条气象场景轨迹。

    Attributes:
        series: 场景对应的时间序列数值。
        weight: 该场景在场景集中的概率权重。
    """

    series: list[float] = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)


class ScenarioSet(BaseModel):
    """气象场景集合。

    Attributes:
        scenarios: 削减后的气象场景列表。
    """

    scenarios: list[WeatherScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "ScenarioSet":
        """校验场景概率权重之和为 1。

        Returns:
            校验后的场景集合对象。

        Raises:
            ValueError: 当场景权重之和不为 1 时抛出。
        """

        total_weight = sum(scenario.weight for scenario in self.scenarios)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("scenario weights must sum to 1")
        return self
