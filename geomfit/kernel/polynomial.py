"""Polynomial kernel."""

import numpy as np
from typing import Dict, Any, Optional
from .base import Kernel

class PolynomialKernel(Kernel):
    """Polynomial kernel."""

    def __init__(self, degree: int = 2, coef0: float = 1.0, gamma: float = 1.0,
                 c: Optional[float] = None):
        """
        Initialize the polynomial kernel.

        Parameters
        ----------
        degree : int, default=2
            Polynomial degree.
        coef0 : float, default=1.0
            Independent term (constant term).
        gamma : float, default=1.0
            Scaling parameter.
        c : float, optional
            Alias for coef0 (independent term).
        """
        if c is not None:
            coef0 = c
        super().__init__(degree=degree, coef0=coef0, gamma=gamma)
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the polynomial kernel between two points.

        Parameters
        ----------
        x : Dict[int, float]
            Coordinates of the first point.
        y : Dict[int, float]
            Coordinates of the second point.

        Returns
        -------
        float
            k(x, y) = (γ * ⟨x, y⟩ + c)^degree.
        """
        dot_product = 0.0
        dims = set(x.keys()) & set(y.keys())  # only dimensions shared by both points

        for dim in dims:
            dot_product += x[dim] * y[dim]

        return (self.gamma * dot_product + self.coef0) ** self.degree

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        Compute the polynomial kernel matrix efficiently.

        Parameters
        ----------
        data : MultiDimData
            Data container.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_points, n_points), where
            K_ij = (γ * ⟨x_i, x_j⟩ + c)^degree.
        """
        from ..core.data_container import MultiDimData

        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")

        X = data.get_coordinate_matrix()

        # Matrix of pairwise inner products
        K = X @ X.T

        K = (self.gamma * K + self.coef0) ** self.degree

        return K
