"""SVGP 气象时间序列核函数定义。"""

from __future__ import annotations

import gpytorch
from gpytorch.kernels import Kernel
from torch import Size

from config.models import SVGPKernelParamsConfig


def build_weather_kernel(
    input_dim: int,
    *,
    batch_shape: Size | None = None,
    kernel_params: SVGPKernelParamsConfig | None = None,
) -> Kernel:
    """构建单点气象时间序列组合核函数。

    核函数由 MaternKernel 与 PeriodicKernel 组合而成：
    MaternKernel 用于捕捉风速、浪高序列中的突变和非平滑变化；
    日周期与年周期 PeriodicKernel 用于表达气象序列常见周期性。

    Args:
        input_dim: 输入特征维度。若滑动窗口被展平，则为 ``seq_len * num_features``。
        batch_shape: 可选批形状，多任务 SVGP 会传入 ``[forecast_horizon]``。
        kernel_params: 显式周期与 Matern ``nu``；缺省使用配置模型中的默认值。

    Returns:
        适用于 GPyTorch ApproximateGP 的组合核函数。
    """

    params = kernel_params or SVGPKernelParamsConfig()
    resolved_batch_shape = batch_shape or Size()

    matern_kernel = gpytorch.kernels.MaternKernel(
        nu=params.matern_nu,
        ard_num_dims=input_dim,
        batch_shape=resolved_batch_shape,
    )

    daily_periodic_kernel = gpytorch.kernels.PeriodicKernel(
        ard_num_dims=input_dim,
        batch_shape=resolved_batch_shape,
    )
    daily_periodic_kernel.period_length = params.daily_period_hours

    yearly_periodic_kernel = gpytorch.kernels.PeriodicKernel(
        ard_num_dims=input_dim,
        batch_shape=resolved_batch_shape,
    )
    yearly_periodic_kernel.period_length = params.yearly_period_hours

    return gpytorch.kernels.ScaleKernel(
        matern_kernel + daily_periodic_kernel + yearly_periodic_kernel,
        batch_shape=resolved_batch_shape,
    )
