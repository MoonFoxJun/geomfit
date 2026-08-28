"""Functional Solver - a unified framework for functional approximation."""

from .core.data_container import MultiDimData
from .core.basis_container import BasisSet, BasisInfo
from .basis.factory import BasisFactory
from .basis.additive import AdditiveBasis
from .kernel.base import Kernel
from .kernel.rbf import RBFKernel
from .kernel.polynomial import PolynomialKernel
from .kernel.mahalanobis import MahalanobisKernel
from .inner_product.base import InnerProduct
from .solver.functional_solver import FunctionalSolver

__version__ = "0.1.0"
