"""Basis-function families and the basis factory for the functional solver."""

# 基函数包的统一出口：BasisFactory 提供预定义的常见基（含张量积构造器），
# 各类 Basis 类（Polynomial/Fourier/Wavelet/RBF/Custom/Additive）提供各自的构造方法
from .factory import BasisFactory
from .polynomial import PolynomialBasis
from .fourier import FourierBasis
from .wavelet import WaveletBasis
from .rbf import RBFBasis
from .custom import CustomBasis
from .additive import AdditiveBasis

__all__ = [
    "BasisFactory",
    "PolynomialBasis",
    "FourierBasis",
    "WaveletBasis",
    "RBFBasis",
    "CustomBasis",
    "AdditiveBasis",
]
