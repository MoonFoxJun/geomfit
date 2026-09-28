"""Functional Solver - a unified framework for functional approximation."""

# 包级导出：把最常用的类直接暴露到 geomfit 命名空间，方便用户
# 用 from geomfit import FunctionalSolver 这样的方式导入，而不用记住
# 每个类在子包里的完整路径。
from .core.data_container import MultiDimData
from .core.basis_container import BasisSet, BasisInfo
from .basis.factory import BasisFactory
from .basis.additive import AdditiveBasis
from .kernel.base import Kernel
from .kernel.rbf import RBFKernel
from .kernel.polynomial import PolynomialKernel
from .kernel.mahalanobis import MahalanobisKernel
from .inner_product.base import InnerProduct
# 主求解器：封装了"基函数路径"与"核路径"两套算法，见 solver/functional_solver.py
from .solver.functional_solver import FunctionalSolver
# 网格数据上的可分离（三明治）求解器：把 P×P 联合求解拆成各维小求解
from .solver.grid_solver import GridSolver

# 包的版本号，供 setup.py / 用户代码查询当前安装版本
__version__ = "0.1.0"
