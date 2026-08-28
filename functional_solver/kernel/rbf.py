"""径向基函数（RBF）核。"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Kernel

class RBFKernel(Kernel):
    """径向基函数（高斯）核。"""
    
    def __init__(self, sigma: float = 1.0, length_scale: Optional[float] = None):
        """
        初始化 RBF 核。
        
        参数
        ----
        sigma : float, default=1.0
            输出尺度参数
        length_scale : float, optional
            长度尺度参数。为 None（默认）时，会在首次调用 compute_matrix 时
            根据数据自动估计为平均最近邻距离的 1.5 倍，从而保证核在任意
            尺度的数据上都具有良好的条件数。
        """
        super().__init__(sigma=sigma, length_scale=length_scale)
        self.sigma = sigma
        self.length_scale = length_scale
        self._effective_length_scale = None
    
    def _ls(self) -> float:
        """返回有效长度尺度（未指定时自动估计）。"""
        if self._effective_length_scale is not None:
            return self._effective_length_scale
        if self.length_scale is not None:
            return self.length_scale
        return 1.0
    
    @staticmethod
    def _estimate_length_scale(X: np.ndarray) -> float:
        """根据坐标估计适合数据的长度尺度。"""
        n = X.shape[0]
        if n < 2:
            return 1.0
        
        # 成对欧氏距离平方
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        d2 = X_squared + X_squared.T - 2 * X @ X.T
        np.fill_diagonal(d2, np.inf)
        
        # 平均最近邻距离
        nn = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
        return float(1.5 * nn.mean())
    
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的 RBF 核值。
        
        参数
        ----
        x : Dict[int, float]
            第一个点的坐标
        y : Dict[int, float]
            第二个点的坐标
            
        返回
        ----
        float
            k(x, y) = σ² * exp(-||x - y||² / (2 * ℓ²))
        """
        # 计算欧氏距离平方
        squared_distance = 0.0
        dims = set(x.keys()) | set(y.keys())
        
        for dim in dims:
            x_val = x.get(dim, 0.0)
            y_val = y.get(dim, 0.0)
            squared_distance += (x_val - y_val) ** 2
        
        return self.sigma ** 2 * np.exp(-squared_distance / (2 * self._ls() ** 2))
    
    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        高效计算 RBF 核矩阵。
        
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
        
        # 若用户未指定长度尺度，则根据数据自动估计
        if self._effective_length_scale is None and self.length_scale is None:
            self._effective_length_scale = self._estimate_length_scale(X)
        
        ls = self._ls()
        
        # 高效计算欧氏距离平方
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances = X_squared + X_squared.T - 2 * X @ X.T
        
        # 计算核矩阵
        K = self.sigma ** 2 * np.exp(-distances / (2 * ls ** 2))
        
        return K
