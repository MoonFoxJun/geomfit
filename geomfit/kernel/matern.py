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
        dims = set(x.keys()) | set(y.keys())  # 维度并集：缺失维度按 0 处理

        for dim in dims:
            x_val = x.get(dim, 0.0)
            y_val = y.get(dim, 0.0)
            distance += (x_val - y_val) ** 2  # 累加各维差平方得到 ‖x - y‖²

        distance = np.sqrt(distance)          # 开方得到欧氏距离 ‖x - y‖
        scaled_distance = distance / self.length_scale  # 无量纲化距离 r = ‖x - y‖/ℓ

        if self.nu == 0.5:
            # Matern 1/2：退化为指数核 k = σ²·exp(-r)（对应最粗糙、处处不可导的样本路径）
            return self.sigma ** 2 * np.exp(-scaled_distance)
        elif self.nu == 1.5:
            # Matern 3/2：k = σ²(1 + √3·r)·exp(-√3·r)，样本路径一阶可导
            return self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distance) * np.exp(-np.sqrt(3) * scaled_distance)
        elif self.nu == 2.5:
            # Matern 5/2：k = σ²(1 + √5·r + (5/3)·r²)·exp(-√5·r)，样本路径二阶可导
            return self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distance + (5/3) * scaled_distance ** 2) * np.exp(-np.sqrt(5) * scaled_distance)
        else:
            # 一般 ν 的 Matern 核通式（按本实现的实际形式）：
            # k(r) = σ² · 2^(1-ν)/Γ(ν) · (√(2ν)·r)^ν · K_ν(r)，其中 r = ‖x-y‖/ℓ
            # K_ν 是第二类修正贝塞尔函数（scipy.special.kv 计算）；
            # 数学背景：当 ν → ∞ 时通式渐近趋于高斯 RBF 核
            if scaled_distance == 0:
                return self.sigma ** 2  # r = 0 时通式含 0^ν·K_ν(0) 的奇异性，直接取极限值 k(0) = σ²
            else:
                k_nu = special.kv(self.nu, scaled_distance)  # 第二类修正贝塞尔函数 K_ν(r)
                gamma_nu = special.gamma(self.nu)            # 伽马函数 Γ(ν)，用于归一化常数
                factor = (2 ** (1 - self.nu)) / gamma_nu     # 前置系数 2^(1-ν)/Γ(ν)
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

        # 向量化两两距离：‖x_i - x_j‖² = ‖x_i‖² + ‖x_j‖² - 2 x_iᵀ x_j
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances_squared = X_squared + X_squared.T - 2 * X @ X.T
        distances = np.sqrt(np.maximum(distances_squared, 0))  # 浮点误差可能产生微小负值，先截断到 0 再开方（clamp）
        scaled_distances = distances / self.length_scale  # 无量纲化：r_ij = ‖x_i - x_j‖/ℓ

        if self.nu == 0.5:
            K = self.sigma ** 2 * np.exp(-scaled_distances)
        elif self.nu == 1.5:
            K = self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distances) * np.exp(-np.sqrt(3) * scaled_distances)
        elif self.nu == 2.5:
            K = self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distances + (5/3) * scaled_distances ** 2) * np.exp(-np.sqrt(5) * scaled_distances)
        else:
            # 一般 ν 的情况：贝塞尔函数无法整体向量化，只能逐元素计算（较慢但更灵活）
            K = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):  # 核矩阵对称，只算上三角（含对角线）
                    d = scaled_distances[i, j]
                    if d == 0:
                        K[i, j] = self.sigma ** 2  # r = 0 时取极限值 σ²，避开 0^ν·K_ν(0) 的奇异性
                    else:
                        k_nu = special.kv(self.nu, d)              # 第二类修正贝塞尔函数 K_ν(r)
                        gamma_nu = special.gamma(self.nu)          # 伽马函数 Γ(ν)（归一化常数）
                        factor = (2 ** (1 - self.nu)) / gamma_nu   # 前置系数 2^(1-ν)/Γ(ν)
                        K[i, j] = self.sigma ** 2 * factor * (np.sqrt(2 * self.nu) * d) ** self.nu * k_nu
                    if i != j:
                        K[j, i] = K[i, j]  # 对称性：镜像填充下三角

        return K
