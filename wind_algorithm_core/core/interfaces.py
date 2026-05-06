"""核心模块接口协议定义。"""

from typing import Any, Protocol, Sequence

from .schemas import ForecastResult, RoutingSolution, ScenarioSet, TaskNode


class BasePredictor(Protocol):
    """气象预测器接口。

    所有预测模块实现都应遵循该协议，输出统一的 ForecastResult。
    """

    def predict(self, input_data: Any) -> ForecastResult:
        """执行气象预测。

        Args:
            input_data: 预测器输入数据，具体类型由预测器实现决定。

        Returns:
            标准化气象预测结果。
        """

        ...


class BaseGenerator(Protocol):
    """随机场景生成器接口。

    所有场景生成模块实现都应遵循该协议，输入预测结果并输出场景集合。
    """

    def generate(self, forecast: ForecastResult) -> ScenarioSet:
        """根据预测结果生成随机场景集合。

        Args:
            forecast: 标准化气象预测结果。

        Returns:
            标准化气象场景集合。
        """

        ...


class BaseOptimizer(Protocol):
    """运筹优化器接口。

    所有优化模块实现都应遵循该协议，输入任务节点与气象场景集合并输出路径解。
    """

    def solve(
        self,
        tasks: Sequence[TaskNode],
        scenarios: ScenarioSet,
    ) -> RoutingSolution:
        """求解运维路径优化问题。

        Args:
            tasks: 待优化的运维任务节点序列。
            scenarios: 随机场景集合。

        Returns:
            标准化路径优化结果。
        """

        ...
