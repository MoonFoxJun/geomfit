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

        # 向量化计算两两点的平方欧氏距离矩阵：
        # ‖x_i - x_j‖² = ‖x_i‖² + ‖x_j‖² - 2 x_iᵀ x_j
        # （第一项是每行模长平方构成的列向量，第二项是其转置，第三项是格拉姆矩阵 X Xᵀ）
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        d2 = X_squared + X_squared.T - 2 * X @ X.T
        np.fill_diagonal(d2, np.inf)  # 对角元是自己到自己的距离 0，置为无穷大以便取最小值时排除

        # 平均最近邻距离：对每行（每个点）取最小值得到"到最近邻的距离"，再对所有点取平均
        nn = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
        return float(1.5 * nn.mean())  # 长度尺度取 1.5 倍平均最近邻距离：保证任何数据尺度下核矩阵条件数都合理

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
        dims = set(x.keys()) | set(y.keys())  # 取两点所有维度的并集：某维度只在一个点中出现时也要计入距离

        for dim in dims:
            x_val = x.get(dim, 0.0)  # 稀疏坐标表示约定：缺失维度按 0 处理
            y_val = y.get(dim, 0.0)
            squared_distance += (x_val - y_val) ** 2  # 累加各维差的平方，得到 ‖x - y‖²

        # RBF 核公式：k(x, y) = σ² · exp(-‖x - y‖² / (2ℓ²))，σ 控制输出幅度、ℓ 控制衰减快慢
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

        # 若用户未显式给出长度尺度，则首次调用 compute_matrix 时从数据自动估计，并缓存结果供后续复用
        if self._effective_length_scale is None and self.length_scale is None:
            self._effective_length_scale = self._estimate_length_scale(X)

        ls = self._ls()

        # 向量化两两距离矩阵：‖x_i - x_j‖² = ‖x_i‖² + ‖x_j‖² - 2 x_iᵀ x_j，
        # 一次矩阵运算算出全部点对的距离（避免 Python 双层循环）
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances = X_squared + X_squared.T - 2 * X @ X.T

        # 对整个距离矩阵套用高斯公式：K_ij = σ² · exp(-‖x_i - x_j‖²/(2ℓ²))，对角元恰为 σ²
        K = self.sigma ** 2 * np.exp(-distances / (2 * ls ** 2))

        return K
