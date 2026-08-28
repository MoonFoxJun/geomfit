"""Radial basis function (RBF) kernel."""

import numpy as np
from typing import Dict, Any, Optional
from .base import Kernel

class RBFKernel(Kernel):
    """Radial basis function (Gaussian) kernel."""

    def __init__(self, sigma: float = 1.0, length_scale: Optional[float] = None):
        """
        Initialize the RBF kernel.

        Parameters
        ----------
        sigma : float, default=1.0
            Output scale parameter.
        length_scale : float, optional
            Length scale parameter. If None (default), it is estimated
            automatically from the data as 1.5 times the mean
            nearest-neighbor distance at the first call to
            ``compute_matrix``, which keeps the kernel matrix well
            conditioned for data at any scale.
        """
        super().__init__(sigma=sigma, length_scale=length_scale)
        self.sigma = sigma
        self.length_scale = length_scale
        self._effective_length_scale = None

    def _ls(self) -> float:
        """Return the effective length scale (auto-estimated if not given)."""
        if self._effective_length_scale is not None:
            return self._effective_length_scale
        if self.length_scale is not None:
            return self.length_scale
        return 1.0

    @staticmethod
    def _estimate_length_scale(X: np.ndarray) -> float:
        """Estimate a length scale suited to the given coordinates."""
        n = X.shape[0]
        if n < 2:
            return 1.0

        # Pairwise squared Euclidean distances
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        d2 = X_squared + X_squared.T - 2 * X @ X.T
        np.fill_diagonal(d2, np.inf)

        # Mean nearest-neighbor distance
        nn = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
        return float(1.5 * nn.mean())

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the RBF kernel between two points.

        Parameters
        ----------
        x : Dict[int, float]
            Coordinates of the first point.
        y : Dict[int, float]
            Coordinates of the second point.

        Returns
        -------
        float
            k(x, y) = σ² * exp(-||x - y||² / (2 * ℓ²)).
        """
        squared_distance = 0.0
        dims = set(x.keys()) | set(y.keys())

        for dim in dims:
            x_val = x.get(dim, 0.0)
            y_val = y.get(dim, 0.0)
            squared_distance += (x_val - y_val) ** 2

        return self.sigma ** 2 * np.exp(-squared_distance / (2 * self._ls() ** 2))

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        Compute the RBF kernel matrix efficiently.

        Parameters
        ----------
        data : MultiDimData
            Data container.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_points, n_points), where
            K_ij = k(x_i, x_j).
        """
        from ..core.data_container import MultiDimData

        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")

        X = data.get_coordinate_matrix()

        # Estimate the length scale from the data if not specified
        if self._effective_length_scale is None and self.length_scale is None:
            self._effective_length_scale = self._estimate_length_scale(X)

        ls = self._ls()

        # Vectorized pairwise squared Euclidean distances
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances = X_squared + X_squared.T - 2 * X @ X.T

        K = self.sigma ** 2 * np.exp(-distances / (2 * ls ** 2))

        return K
