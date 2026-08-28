"""加权内积的权重函数。"""

import numpy as np
from typing import Callable, Dict, Any, Union

# numpy>=2.0 已将 np.trapz 改名为 np.trapezoid；此处保持与 numpy 1.x 兼容
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

class WeightFunction:
    """加权内积的权重函数。"""
    
    def __init__(self, weight_func: Callable):
        """
        初始化权重函数。
        
        参数
        ----
        weight_func : Callable
            返回某一点或索引处权重的函数
            签名：weight_func(x) -> float 或 np.ndarray
        """
        self.weight_func = weight_func
    
    def __call__(self, x: Union[float, np.ndarray, Dict[int, float]]) -> Union[float, np.ndarray]:
        """
        求值权重函数。
        
        参数
        ----
        x : Union[float, np.ndarray, Dict[int, float]]
            输入的点（或点的集合）
            
        返回
        ----
        Union[float, np.ndarray]
            权重值（或权重值数组）
        """
        return self.weight_func(x)
    
    @staticmethod
    def constant(c: float = 1.0) -> 'WeightFunction':
        """常量权重函数。"""
        def weight_func(x):
            if isinstance(x, dict):
                # 对于字典形式的点，返回常量
                return c
            elif isinstance(x, np.ndarray):
                # 对于数组，返回常量组成的数组
                return np.full_like(x, c, dtype=float)
            else:
                # 对于标量
                return c
        
        return WeightFunction(weight_func)
    
    @staticmethod
    def exponential(alpha: float = 1.0) -> 'WeightFunction':
        """指数权重函数 w(x) = exp(-α|x|)。"""
        def weight_func(x):
            if isinstance(x, dict):
                # 对于字典形式的点，计算范数
                norm = np.sqrt(sum(val ** 2 for val in x.values()))
                return np.exp(-alpha * norm)
            elif isinstance(x, np.ndarray):
                return np.exp(-alpha * np.abs(x))
            else:
                return np.exp(-alpha * np.abs(x))
        
        return WeightFunction(weight_func)
    
    @staticmethod
    def gaussian(sigma: float = 1.0, center: float = 0.0) -> 'WeightFunction':
        """高斯权重函数 w(x) = exp(-(x - center)²/(2σ²))。"""
        def weight_func(x):
            if isinstance(x, dict):
                # 对于字典形式的点，使用第一个维度
                if x:
                    first_key = list(x.keys())[0]
                    x_val = x[first_key]
                else:
                    x_val = 0.0
                return np.exp(-(x_val - center) ** 2 / (2 * sigma ** 2))
            elif isinstance(x, np.ndarray):
                return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
            else:
                return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        
        return WeightFunction(weight_func)
    
    @staticmethod
    def polynomial(degree: int = 2, coef: float = 1.0) -> 'WeightFunction':
        """多项式权重函数 w(x) = 1/(1 + coef * |x|^degree)。"""
        def weight_func(x):
            if isinstance(x, dict):
                # 对于字典形式的点，计算范数
                norm = np.sqrt(sum(val ** 2 for val in x.values()))
                return 1.0 / (1.0 + coef * (norm ** degree))
            elif isinstance(x, np.ndarray):
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))
            else:
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))
        
        return WeightFunction(weight_func)
    
    @staticmethod
    def chebyshev_weight(kind: str = "first") -> 'WeightFunction':
        """用于正交多项式的 Chebyshev 权重函数。"""
        if kind == "first":
            # w(x) = 1/√(1 - x²)，其中 x ∈ (-1, 1)
            def weight_func(x):
                if isinstance(x, dict):
                    # 使用第一个维度
                    if x:
                        first_key = list(x.keys())[0]
                        x_val = x[first_key]
                    else:
                        x_val = 0.0
                    if -1 < x_val < 1:
                        return 1.0 / np.sqrt(1 - x_val ** 2)
                    else:
                        return 0.0
                elif isinstance(x, np.ndarray):
                    mask = (x > -1) & (x < 1)
                    result = np.zeros_like(x)
                    result[mask] = 1.0 / np.sqrt(1 - x[mask] ** 2)
                    return result
                else:
                    if -1 < x < 1:
                        return 1.0 / np.sqrt(1 - x ** 2)
                    else:
                        return 0.0
        elif kind == "second":
            # w(x) = √(1 - x²)，其中 x ∈ (-1, 1)
            def weight_func(x):
                if isinstance(x, dict):
                    if x:
                        first_key = list(x.keys())[0]
                        x_val = x[first_key]
                    else:
                        x_val = 0.0
                    if -1 < x_val < 1:
                        return np.sqrt(1 - x_val ** 2)
                    else:
                        return 0.0
                elif isinstance(x, np.ndarray):
                    mask = (x > -1) & (x < 1)
                    result = np.zeros_like(x)
                    result[mask] = np.sqrt(1 - x[mask] ** 2)
                    return result
                else:
                    if -1 < x < 1:
                        return np.sqrt(1 - x ** 2)
                    else:
                        return 0.0
        else:
            raise ValueError(f"Unknown Chebyshev kind: {kind}")
        
        return WeightFunction(weight_func)
    
    @staticmethod
    def custom(func: Callable) -> 'WeightFunction':
        """自定义权重函数。"""
        return WeightFunction(func)
    
    def integrate(self, a: float, b: float, n_points: int = 1000) -> float:
        """
        在区间上对权重函数求积分。
        
        参数
        ----
        a : float
            下界
        b : float
            上界
        n_points : int, 默认=1000
            数值积分的采样点数
            
        返回
        ----
        float
            近似积分 ∫[a,b] w(x) dx
        """
        x = np.linspace(a, b, n_points)
        w = self(x)
        return _trapz(w, x)
