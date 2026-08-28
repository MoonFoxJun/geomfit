"""Solver module of the functional solver package."""

# 求解器子包对外暴露的五个求解器：
# - FunctionalSolver：集大成的主求解器，同时支持基函数路径与核路径；
# - GramSolver：专门基于 Gram 矩阵/法方程的求解器；
# - KernelSolver：核方法（表示定理 / 核岭回归）求解器；
# - GradientSolver：梯度下降族（GD / 动量 / Adam）求解器；
# - AdaptiveSolver：贪心自适应选基（前向选择 / OMP / 交叉验证）求解器。
from .functional_solver import FunctionalSolver
from .gram_solver import GramSolver
from .kernel_solver import KernelSolver
from .gradient_solver import GradientSolver
from .adaptive import AdaptiveSolver

# 显式声明 __all__，控制 from geomfit.solver import * 时导出哪些名字
__all__ = [
    "FunctionalSolver",
    "GramSolver",
    "KernelSolver",
    "GradientSolver",
    "AdaptiveSolver",
]
