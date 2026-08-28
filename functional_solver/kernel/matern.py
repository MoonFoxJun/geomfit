"""Matern 核。"""

import numpy as np
from typing import Dict, Any
from scipy import special
from .base import Kernel

class MaternKernel(Kernel):
    """Matern 核。"""
    
    def __init__(self, nu: float = 1.5, length_scale: float = 1.0, sigma: float = 1.0):
        """
        初始化 Matern 核。
        
        参数
        ----
        nu : float, default=1.5
            平滑度参数（通常取 0.5、1.5、2.5 或 ∞）
        length_scale : float, default=1.0
            长度尺度参数
        sigma : float, default=1.0
            输出尺度参数
        """
        super().__init__(nu=nu, length_scale=length_scale, sigma=sigma)
        self.nu = nu
        self.length_scale = length_scale
        self.sigma = sigma
    
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的 Matern 核值。
        
        参数
        ----
        x : Dict[int, float]
            第一个点的坐标
        y : Dict[int, float]
            第二个点的坐标
            
        返回
        ----
        float
            Matern 核值
        """
        # 计算欧氏距离
        distance = 0.0
        dims = set(x.keys()) | set(y.keys())
        
        for dim in dims:
            x_val = x.get(dim, 0.0)
            y_val = y.get(dim, 0.0)
            distance += (x_val - y_val) ** 2
        
        distance = np.sqrt(distance)
        scaled_distance = distance / self.length_scale
        
        if self.nu == 0.5:
            # Matern 1/2（指数核）
            return self.sigma ** 2 * np.exp(-scaled_distance)
        elif self.nu == 1.5:
            # Matern 3/2
            return self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distance) * np.exp(-np.sqrt(3) * scaled_distance)
        elif self.nu == 2.5:
            # Matern 5/2
            return self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distance + (5/3) * scaled_distance ** 2) * np.exp(-np.sqrt(5) * scaled_distance)
        else:
            # 通用 Matern 核
            if scaled_distance == 0:
                return self.sigma ** 2
            else:
                # 使用第二类修正 Bessel 函数
                k_nu = special.kv(self.nu, scaled_distance)
                gamma_nu = special.gamma(self.nu)
                factor = (2 ** (1 - self.nu)) / gamma_nu
                return self.sigma ** 2 * factor * (np.sqrt(2 * self.nu) * scaled_distance) ** self.nu * k_nu
    
    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        高效计算 Matern 核矩阵。
        
        参数
        ----
        data : MultiDimData
            数据容器
            
        返回
        ----
        np.ndarray
            核矩阵
        """
        from ..core.data_container import MultiDimData
        
        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")
        
        # 获取坐标矩阵
        X = data.get_coordinate_matrix()
        n_points = X.shape[0]
        
        # 高效计算成对距离
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances_squared = X_squared + X_squared.T - 2 * X @ X.T
        distances = np.sqrt(np.maximum(distances_squared, 0))  # 确保非负
        scaled_distances = distances / self.length_scale
        
        # 根据 ν 值计算核矩阵
        if self.nu == 0.5:
            K = self.sigma ** 2 * np.exp(-scaled_distances)
        elif self.nu == 1.5:
            K = self.sigma ** 2 * (1 + np.sqrt(3) * scaled_distances) * np.exp(-np.sqrt(3) * scaled_distances)
        elif self.nu == 2.5:
            K = self.sigma ** 2 * (1 + np.sqrt(5) * scaled_distances + (5/3) * scaled_distances ** 2) * np.exp(-np.sqrt(5) * scaled_distances)
        else:
            # 通用情形——较慢但更灵活
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
