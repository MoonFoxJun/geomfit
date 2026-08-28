"""Basis containers: basis metadata and design-matrix assembly for the functional solver."""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

from .data_container import MultiDimData


@dataclass
class BasisInfo:
    """Metadata of a single basis function.

    Two forms are supported:
    - Single-factor basis (the common case): acts on one dimension via
      func(x, **params).
    - Tensor-product basis (multi-dimensional): a product of single-factor
      bases, factors = [(dim, func, params), ...], evaluated as
      φ(x_1, ..., x_d) = Π_k func_k(x_{dim_k}, **params_k),
      i.e. one product basis of the expansion
      f(x, y, z) = Σ_{i,j,k} c_{ijk} X_i(x)Y_j(y)Z_k(z).
    """
    name: str
    dim: int                # 主维度：单因子基就是它作用的维度；张量积基则是第一个因子所在的维度
    params: Optional[Dict[str, Any]] = None   # 单因子基的参数（如多项式阶数、高斯核的 sigma 等）
    func: Optional[Callable] = None           # 单因子基的求值函数 func(x, **params)
    factors: Optional[List[Tuple[int, Callable, Dict[str, Any]]]] = None  # 张量积基的因子列表，每个因子是 (维度, 函数, 参数) 三元组

    def evaluate(self, point: Dict[int, float]) -> float:
        """Evaluate the basis at a data point given as a dict keyed by dimension index."""
        # 张量积形态：φ(x₁, …, x_d) = Π_k func_k(x_{dim_k}, **params_k)，
        # 即各因子在各自维度上的取值连乘（乘积基）；从 1.0 开始逐项累乘
        if self.factors:
            value = 1.0
            for dim, func, params in self.factors:
                value *= func(point[dim], **(params or {}))
            return value
        if self.func is None:
            raise ValueError(f"Basis '{self.name}' has neither func nor factors")
        # 单因子形态：只取 point 中主维度上的坐标，代入 func 求值
        return self.func(point[self.dim], **(self.params or {}))

    @property
    def dims(self) -> List[int]:
        """All dimensions involved in this basis."""
        # 张量积基涉及所有因子的维度（取每个因子三元组的第 0 项，即维度索引）
        if self.factors:
            return [f[0] for f in self.factors]
        # 单因子基只涉及主维度
        return [self.dim]


class BasisSet:
    """Flat collection of bases; assembles the design matrix Φ of shape (n_points, n_basis)."""

    def __init__(self):
        self.bases: List[BasisInfo] = []  # 扁平保存所有基（按加入顺序编号，下标即基的编号）
        self.by_dim: Dict[int, List[int]] = {}  # 维度索引 -> 作用于该维度的基在 self.bases 中的下标列表

    def add_basis(self, basis_info: BasisInfo):
        """Add a basis; tensor-product bases are registered under every dimension they involve."""
        idx = len(self.bases)
        self.bases.append(basis_info)

        # 把该基登记到它涉及的每一个维度上：单因子基登记到主维度；
        # 张量积基则登记到所有因子的维度，这样按维度查询基时不会漏掉多因子基
        for dim in basis_info.dims:
            if dim not in self.by_dim:
                self.by_dim[dim] = []
            self.by_dim[dim].append(idx)

    def get_bases_for_dim(self, dim: int) -> List[BasisInfo]:
        """Return all bases acting on the given dimension."""
        if dim not in self.by_dim:
            return []
        return [self.bases[i] for i in self.by_dim[dim]]

    def evaluate_at_point(self, point: Dict[int, float]) -> np.ndarray:
        """
        Evaluate all bases at one point.

        Returns
        -------
        np.ndarray
            Vector [φ_1(point), φ_2(point), ...] of basis values.
        """
        values = []
        # 逐个基求值：对同一个点 point 算出每个基函数的值，收集成一个向量
        for basis in self.bases:
            values.append(basis.evaluate(point))
        return np.array(values)

    def evaluate_all(self, data: MultiDimData) -> np.ndarray:
        """
        Evaluate all bases at all data points.

        Returns
        -------
        np.ndarray
            Design matrix Φ of shape (n_points, n_basis).
        """
        n_points = data.n_points
        n_basis = len(self.bases)
        # 预分配设计矩阵 Φ：行 = 数据点，列 = 基函数，形状为 (n_points, n_basis)
        Phi = np.zeros((n_points, n_basis))

        for i in range(n_points):
            point = data.get_point(i)
            # 第 i 行存放第 i 个数据点处所有基函数的值 [φ₁(x_i), φ₂(x_i), …, φ_n(x_i)]
            Phi[i, :] = self.evaluate_at_point(point)

        return Phi

    def __len__(self):
        """Number of bases in the set."""
        return len(self.bases)
