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
    dim: int                # Primary dimension: the dimension of a single-factor basis; for tensor-product bases, the dimension of the first factor
    params: Optional[Dict[str, Any]] = None   # Parameters of a single-factor basis
    func: Optional[Callable] = None           # Function of a single-factor basis
    factors: Optional[List[Tuple[int, Callable, Dict[str, Any]]]] = None  # Factors of a tensor-product basis

    def evaluate(self, point: Dict[int, float]) -> float:
        """Evaluate the basis at a data point given as a dict keyed by dimension index."""
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
        """All dimensions involved in this basis."""
        if self.factors:
            return [f[0] for f in self.factors]
        return [self.dim]


class BasisSet:
    """Flat collection of bases; assembles the design matrix Φ of shape (n_points, n_basis)."""

    def __init__(self):
        self.bases: List[BasisInfo] = []  # Flat list of bases
        self.by_dim: Dict[int, List[int]] = {}  # Basis indices per dimension

    def add_basis(self, basis_info: BasisInfo):
        """Add a basis; tensor-product bases are registered under every dimension they involve."""
        idx = len(self.bases)
        self.bases.append(basis_info)

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
        Phi = np.zeros((n_points, n_basis))

        for i in range(n_points):
            point = data.get_point(i)
            Phi[i, :] = self.evaluate_at_point(point)

        return Phi

    def __len__(self):
        """Number of bases in the set."""
        return len(self.bases)
