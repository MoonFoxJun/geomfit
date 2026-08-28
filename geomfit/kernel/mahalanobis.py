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

    # ---------- 度量矩阵 ----------

    def _estimate_metric(self, X: np.ndarray) -> np.ndarray:
        """Estimate the regularized precision matrix M = (Σ + αI)⁻¹ from data."""
        n, d = X.shape
        if n < 2:
            return np.eye(d)
        Xc = X - X.mean(axis=0)               # 数据中心化：减去各维均值，使协方差围绕原点计算
        cov = (Xc.T @ Xc) / (n - 1)           # 样本协方差矩阵 Σ（除以 n-1 得到无偏估计）
        eigval, eigvec = np.linalg.eigh(cov)  # 对称矩阵用 eigh 做特征分解：Σ = V diag(λ) Vᵀ
        reg = np.maximum(eigval, self.alpha)  # 正则化：把近零方差方向（极小特征值）垫高到 α，防止求逆时数值爆炸
        return (eigvec / reg) @ eigvec.T      # 构造正则化精度矩阵 M = V diag(1/reg) Vᵀ = (Σ + αI)⁻¹，
                                              # 这正是 PCA 白化变换：沿主成分方向拉伸、沿噪声方向压缩

    def _get_metric(self, data: Any) -> np.ndarray:
        # 优先使用用户显式传入的度量矩阵 M（转成 float 数组并返回）
        if self.metric is not None:
            return np.asarray(self.metric, dtype=float)
        # 否则从训练数据自动估计；只估计一次并缓存，保证后续求值使用同一度量（前后一致性）
        if self._effective_metric is None:
            X = data.get_coordinate_matrix()
            self._effective_metric = self._estimate_metric(X)
        return self._effective_metric

    # ---------- 核函数求值 ----------

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the Mahalanobis kernel between two points.

        The coordinate dimensions must be 0..d-1 (rotated coordinates from
        PCA satisfy this).
        """
        M = self._effective_metric if self._effective_metric is not None \
            else (np.asarray(self.metric, dtype=float) if self.metric is not None else None)

        if M is None:
            # 没有数据上下文（直接调用 kernel(x, y) 而从未经过 compute_matrix）：
            # 无法估计度量，退化为欧氏 RBF，即马氏度量 M = I 的特例
            d2 = 0.0
            for dim in set(x.keys()) | set(y.keys()):
                d2 += (x.get(dim, 0.0) - y.get(dim, 0.0)) ** 2
            return self.sigma ** 2 * np.exp(-0.5 * d2)

        d = M.shape[0]  # 度量矩阵维数（约定坐标维度为连续的 0..d-1，PCA 旋转后的坐标满足此要求）
        diff = np.array([x.get(i, 0.0) - y.get(i, 0.0) for i in range(d)])  # 差值向量 x - y
        d2 = float(diff @ M @ diff)  # 马氏距离平方 d² = (x - y)ᵀ M (x - y)
        return self.sigma ** 2 * np.exp(-0.5 * d2)  # k(x, y) = σ² · exp(-½ d²)

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

        # 向量化计算所有点对的马氏距离平方，利用恒等式：
        # d²_ij = (x_i - x_j)ᵀM(x_i - x_j) = x_iᵀMx_i + x_jᵀMx_j - 2 x_iᵀMx_j
        XM = X @ M                             # 先算 XM = X·M，供二次项与交叉项复用
        quad = np.sum(XM * X, axis=1)          # 对角元 quad_i = x_iᵀMx_i（即 diag(X M Xᵀ)）
        d2 = quad[:, None] + quad[None, :] - 2.0 * (XM @ X.T)  # 广播成矩阵：quad_i + quad_j - 2x_iᵀMx_j
        d2 = np.maximum(d2, 0.0)               # 浮点舍入可能产生微小负值，截断到 0（保证后续指数运算有意义）
        return self.sigma ** 2 * np.exp(-0.5 * d2)  # k(x, y) = σ² · exp(-½ d²)
