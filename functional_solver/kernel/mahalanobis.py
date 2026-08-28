"""Mahalanobis-distance RBF kernel (absorbs inter-dimensional correlations into the distance metric).

Mathematical form
-----------------
    k(x, y) = σ² · exp(-½ (x-y)ᵀ M (x-y))

where M is a positive semidefinite metric matrix (precision matrix). When
M = Σ⁻¹ (the inverse covariance):

- Distances along strongly correlated directions are compressed and those
  along weakly correlated directions are stretched, elongating the kernel
  shape along the principal axes of the data. This is exactly equivalent to
  "PCA whitening + plain RBF" (M = V diag(1/λ) Vᵀ is the whitening transform);
- it avoids the ill-conditioning and waste of fitting narrow-band data with
  an isotropic product kernel.

Usage
-----
    kernel = MahalanobisKernel(sigma=1.0)              # metric estimated from data
    kernel = MahalanobisKernel(sigma=1.0, metric=...)  # or an explicit metric

Note: with metric=None, the metric is estimated from the training data at the
first compute_matrix call (i.e. at the solve stage); calling kernel(x, y)
directly (without a data context) falls back to a Euclidean-distance kernel.
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import Kernel


class MahalanobisKernel(Kernel):
    """Mahalanobis-distance Gaussian kernel."""

    def __init__(self, sigma: float = 1.0,
                 metric: Optional[np.ndarray] = None,
                 alpha: float = 1e-6):
        """
        Initialize the Mahalanobis kernel.

        Parameters
        ----------
        sigma : float, default=1.0
            Output scale.
        metric : np.ndarray, optional
            Metric matrix M (positive semidefinite). If None, the
            regularized precision matrix (Σ + αI)⁻¹ is estimated from
            the data at compute_matrix time.
        alpha : float, default=1e-6
            Regularization parameter for metric estimation (prevents
            blow-up along near-zero-variance directions).
        """
        super().__init__(sigma=sigma, metric=metric, alpha=alpha)
        self.sigma = sigma
        self.metric = metric
        self.alpha = alpha
        self._effective_metric = None

    # ---------- metric ----------

    def _estimate_metric(self, X: np.ndarray) -> np.ndarray:
        """Estimate the regularized precision matrix M = (Σ + αI)⁻¹ from data."""
        n, d = X.shape
        if n < 2:
            return np.eye(d)
        Xc = X - X.mean(axis=0)
        cov = (Xc.T @ Xc) / (n - 1)
        eigval, eigvec = np.linalg.eigh(cov)
        reg = np.maximum(eigval, self.alpha)          # suppress near-zero-variance directions
        return (eigvec / reg) @ eigvec.T              # V diag(1/reg) Vᵀ

    def _get_metric(self, data: Any) -> np.ndarray:
        if self.metric is not None:
            return np.asarray(self.metric, dtype=float)
        if self._effective_metric is None:
            X = data.get_coordinate_matrix()
            self._effective_metric = self._estimate_metric(X)
        return self._effective_metric

    # ---------- evaluation ----------

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the Mahalanobis kernel between two points.

        The coordinate dimensions must be 0..d-1 (rotated coordinates from
        PCA satisfy this).
        """
        M = self._effective_metric if self._effective_metric is not None \
            else (np.asarray(self.metric, dtype=float) if self.metric is not None else None)

        if M is None:
            # No data context available: fall back to the Euclidean RBF
            d2 = 0.0
            for dim in set(x.keys()) | set(y.keys()):
                d2 += (x.get(dim, 0.0) - y.get(dim, 0.0)) ** 2
            return self.sigma ** 2 * np.exp(-0.5 * d2)

        d = M.shape[0]
        diff = np.array([x.get(i, 0.0) - y.get(i, 0.0) for i in range(d)])
        d2 = float(diff @ M @ diff)
        return self.sigma ** 2 * np.exp(-0.5 * d2)

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        Compute the Mahalanobis kernel matrix efficiently.

        Uses d² = (x-y)ᵀM(x-y) = xᵀMx + yᵀMy - 2xᵀMy, vectorized.
        """
        from ..core.data_container import MultiDimData

        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")

        X = data.get_coordinate_matrix()
        M = self._get_metric(data)

        XM = X @ M
        quad = np.sum(XM * X, axis=1)                 # diag(X M Xᵀ)
        d2 = quad[:, None] + quad[None, :] - 2.0 * (XM @ X.T)
        d2 = np.maximum(d2, 0.0)                      # remove floating-point error
        return self.sigma ** 2 * np.exp(-0.5 * d2)
