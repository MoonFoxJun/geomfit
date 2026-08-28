"""基向量容器 - 管理所有基函数"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

from .data_container import MultiDimData


@dataclass
class BasisInfo:
    """基函数的元信息。

    两种形态：
    - 单因子基（最常见）：作用在单个维度上，func(x, **params)。
    - 张量积基（多维）：由多个单因子基相乘构成，
      factors = [(dim, func, params), ...]，
      求值 φ(x_1, ..., x_d) = Π_k func_k(x_{dim_k}, **params_k)。
      即 f(x,y,z) = Σ_{i,j,k} c_{ijk} X_i(x)Y_j(y)Z_k(z) 中的单个乘积基。
    """
    name: str
    dim: int                # 主维度：单因子基的维度；张量积基取第一个因子的维度
    params: Optional[Dict[str, Any]] = None   # 单因子基的参数
    func: Optional[Callable] = None           # 单因子基的函数
    factors: Optional[List[Tuple[int, Callable, Dict[str, Any]]]] = None  # 张量积基的因子

    def evaluate(self, point: Dict[int, float]) -> float:
        """在数据点（dict 形式，键为维度索引）上求值。"""
        if self.factors:
            value = 1.0
            for dim, func, params in self.factors:
                value *= func(point[dim], **(params or {}))
            return value
        if self.func is None:
            raise ValueError(f"Basis '{self.name}' has neither func nor factors")
        return self.func(point[self.dim], **(self.params or {}))

    @property
    def dims(self) -> List[int]:
        """该基函数涉及的所有维度。"""
        if self.factors:
            return [f[0] for f in self.factors]
        return [self.dim]


class BasisSet:
    """基向量集合，支持预留、组合、自适应"""

    def __init__(self):
        self.bases: List[BasisInfo] = []  # 扁平的基函数列表
        self.by_dim: Dict[int, List[int]] = {}  # 每个维度有哪些基（索引）

    def add_basis(self, basis_info: BasisInfo):
        """添加一个基函数（张量积基会登记到其涉及的每个维度）"""
        idx = len(self.bases)
        self.bases.append(basis_info)

        for dim in basis_info.dims:
            if dim not in self.by_dim:
                self.by_dim[dim] = []
            self.by_dim[dim].append(idx)

    def get_bases_for_dim(self, dim: int) -> List[BasisInfo]:
        """获取作用在指定维度上的所有基函数"""
        if dim not in self.by_dim:
            return []
        return [self.bases[i] for i in self.by_dim[dim]]

    def evaluate_at_point(self, point: Dict[int, float]) -> np.ndarray:
        """
        计算所有基函数在某个点的值
        返回: [φ₁(point), φ₂(point), ...]
        """
        values = []
        for basis in self.bases:
            values.append(basis.evaluate(point))
        return np.array(values)

    def evaluate_all(self, data: MultiDimData) -> np.ndarray:
        """
        计算所有基函数在所有数据点的值
        返回: (n_points, n_basis) 的矩阵 Φ
        """
        n_points = data.n_points
        n_basis = len(self.bases)
        Phi = np.zeros((n_points, n_basis))

        for i in range(n_points):
            point = data.get_point(i)
            Phi[i, :] = self.evaluate_at_point(point)

        return Phi

    def __len__(self):
        return len(self.bases)
