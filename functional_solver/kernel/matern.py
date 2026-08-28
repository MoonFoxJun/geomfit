"""Matern kernel."""

import numpy as np
from typing import Dict, Any
from scipy import special
from .base import Kernel

class MaternKernel(Kernel):
    """Matern kernel."""

    def __init__(self, nu: float = 1.5, length_scale: float = 1.0, sigma: float = 1.0):
        """
        Initialize the Matern kernel.

        Parameters
        ----------
        nu : float, default=1.5
            Smoothness parameter (typically 0.5, 1.5, 2.5, or infinity).
        length_scale : float, default=1.0
            Length scale parameter.
        sigma : float, default=1.0
            Output scale parameter.
        """
        super().__init__(nu=nu, length_scale=length_scale, sigma=sigma)
        self.nu = nu
        self.length_scale = length_scale
        self.sigma = sigma

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the Matern kernel between two points.

        Parameters
        ----------
        x : Dict[int, float]
            Coordinates of the first point.
        y : Dict[int, float]
            Coordinates of the second point.

        Returns
        -------
        float
            Matern kernel value.
        """
        distance = 0.0
        dims = set(x.keys()) | set(y.keys())

        for dim in dims:
            x_val = x.get(dim, 0.0)
            y_val = y.get(dim, 0.0)
            distance += (x_val - y_val) ** 2

        distance = np.sqrt(distance)
        scaled_distance = distance / self.length_scale

        if self.nu == 0.5:
            # Matern 1/2 (exponential kernel)
            return self.sigma ** 2 * np.exp(-scaled_distance)
        elif self.nu == 1.5:
            # Matern 3/2
            return self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distance) * np.exp(-np.sqrt(3) * scaled_distance)
        elif self.nu == 2.5:
            # Matern 5/2
            return self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distance + (5/3) * scaled_distance ** 2) * np.exp(-np.sqrt(5) * scaled_distance)
        else:
            # General Matern kernel via the modified Bessel function of the
            # second kind, K_nu
            if scaled_distance == 0:
                return self.sigma ** 2
            else:
                k_nu = special.kv(self.nu, scaled_distance)
                gamma_nu = special.gamma(self.nu)
                factor = (2 ** (1 - self.nu)) / gamma_nu
                return self.sigma ** 2 * factor * (np.sqrt(2 * self.nu) * scaled_distance) ** self.nu * k_nu

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        Compute the Matern kernel matrix efficiently.

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
        n_points = X.shape[0]

        # Vectorized pairwise distances
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances_squared = X_squared + X_squared.T - 2 * X @ X.T
        distances = np.sqrt(np.maximum(distances_squared, 0))  # clamp numerical negatives
        scaled_distances = distances / self.length_scale

        if self.nu == 0.5:
            K = self.sigma ** 2 * np.exp(-scaled_distances)
        elif self.nu == 1.5:
            K = self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distances) * np.exp(-np.sqrt(3) * scaled_distances)
        elif self.nu == 2.5:
            K = self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distances + (5/3) * scaled_distances ** 2) * np.exp(-np.sqrt(5) * scaled_distances)
        else:
            # General case -- slower but more flexible
            K = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    d = scaled_distances[i, j]
                    if d == 0:
                        K[i, j] = self.sigma ** 2
                    else:
                        k_nu = special.kv(self.nu, d)
                        gamma_nu = special.gamma(self.nu)
                        factor = (2 ** (1 - self.nu)) / gamma_nu
                        K[i, j] = self.sigma ** 2 * factor * (np.sqrt(2 * self.nu) * d) ** self.nu * k_nu
                    if i != j:
                        K[j, i] = K[i, j]

        return K
