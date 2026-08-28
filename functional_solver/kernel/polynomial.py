"""多项式核。"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Kernel

class PolynomialKernel(Kernel):
    """多项式核。"""
    
    def __init__(self, degree: int = 2, coef0: float = 1.0, gamma: float = 1.0,
                 c: Optional[float] = None):
        """
        初始化多项式核。
        
        参数
        ----
        degree : int, default=2
            多项式次数
        coef0 : float, default=1.0
            独立项（常数项）
        gamma : float, default=1.0
            缩放参数
        c : float, optional
            coef0（独立项）的别名
        """
        if c is not None:
            coef0 = c
        super().__init__(degree=degree, coef0=coef0, gamma=gamma)
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
    
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的多项式核值。
        
        参数
        ----
        x : Dict[int, float]
            第一个点的坐标
        y : Dict[int, float]
            第二个点的坐标
            
        返回
        ----
        float
            k(x, y) = (γ * ⟨x, y⟩ + c)ᵈ
        """
        # 计算内积（点积）
        dot_product = 0.0
        dims = set(x.keys()) & set(y.keys())  # 仅取两个点共有的维度
        
        for dim in dims:
            dot_product += x[dim] * y[dim]
        
        return (self.gamma * dot_product + self.coef0) ** self.degree
    
    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        高效计算多项式核矩阵。
        
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
        
        # 高效计算内积
        K = X @ X.T
        
        # 应用多项式核公式
        K = (self.gamma * K + self.coef0) ** self.degree
        
        return K
