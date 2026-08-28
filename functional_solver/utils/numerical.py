"""函数求解器的数值工具。"""

import numpy as np
from typing import Callable, Dict, Any, Tuple, List
from ..core.data_container import MultiDimData

class NumericalUtils:
    """用于积分、微分等的数值工具。"""
    
    @staticmethod
    def integrate_1d(f: Callable, a: float, b: float, n_points: int = 1000) -> float:
        """
        使用辛普森法则对一维函数进行积分。
        
        参数
        ----------
        f : Callable
            被积函数
        a : float
            积分下限
        b : float
            积分上限
        n_points : int, 默认=1000
            积分使用的点数
            
        返回
        -------
        float
            积分的近似值
        """
        if n_points % 2 == 0:
            n_points += 1  # 辛普森法则要求奇数个点
        
        x = np.linspace(a, b, n_points)
        y = f(x)
        
        h = (b - a) / (n_points - 1)
        
        # 辛普森法则
        integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
        return integral
    
    @staticmethod
    def integrate_nd(f: Callable, bounds: List[Tuple[float, float]], 
                    n_points_per_dim: int = 50) -> float:
        """
        使用蒙特卡洛积分对 n 维函数进行积分。
        
        参数
        ----------
        f : Callable
            被积函数
        bounds : List[Tuple[float, float]]
            每个维度的（下界，上界）列表
        n_points_per_dim : int, 默认=50
            每个维度的点数
            
        返回
        -------
        float
            积分的近似值
        """
        dim = len(bounds)
        n_samples = n_points_per_dim ** dim
        
        if n_samples > 1e6:  # 限制总样本数
            n_samples = 1000000
        
        # 生成随机样本
        samples = []
        for i in range(dim):
            a, b = bounds[i]
            samples.append(np.random.uniform(a, b, n_samples))
        
        # 评估函数
        points = np.column_stack(samples)
        values = np.apply_along_axis(f, 1, points)
        
        # 计算体积
        volume = 1.0
        for a, b in bounds:
            volume *= (b - a)
        
        # 蒙特卡洛积分
        integral = volume * np.mean(values)
        return integral
    
    @staticmethod
    def differentiate(f: Callable, x: float, h: float = 1e-5, method: str = "central") -> float:
        """
        计算函数的数值导数。
        
        参数
        ----------
        f : Callable
            要求导的函数
        x : float
            计算导数的位置
        h : float, 默认=1e-5
            步长
        method : str, 默认="central"
            求导方法："forward"（前向）、"backward"（后向）或 "central"（中心）
            
        返回
        -------
        float
            导数的近似值
        """
        if method == "forward":
            return (f(x + h) - f(x)) / h
        elif method == "backward":
            return (f(x) - f(x - h)) / h
        elif method == "central":
            return (f(x + h) - f(x - h)) / (2 * h)
        else:
            raise ValueError(f"Unknown differentiation method: {method}")
    
    @staticmethod
    def gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        计算多元函数的数值梯度。
        
        参数
        ----------
        f : Callable
            要求导的函数
        x : np.ndarray
            计算梯度的位置
        h : float, 默认=1e-5
            步长
            
        返回
        -------
        np.ndarray
            梯度向量
        """
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        return grad
    
    @staticmethod
    def hessian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        计算多元函数的数值 Hessian 矩阵。
        
        参数
        ----------
        f : Callable
            要求导的函数
        x : np.ndarray
            计算 Hessian 矩阵的位置
        h : float, 默认=1e-5
            步长
            
        返回
        -------
        np.ndarray
            Hessian 矩阵
        """
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # 对角线元素：二阶导数
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    hess[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
                else:
                    # 非对角线元素：混合偏导数
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[i] += h
                    x_pp[j] += h
                    
                    x_pm[i] += h
                    x_pm[j] -= h
                    
                    x_mp[i] -= h
                    x_mp[j] += h
                    
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
        
        return hess
    
    @staticmethod
    def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, 
                      method: str = "linear") -> np.ndarray:
        """
        对一维数据进行插值。
        
        参数
        ----------
        x : np.ndarray
            原始 x 值
        y : np.ndarray
            原始 y 值
        x_new : np.ndarray
            用于插值的新 x 值
        method : str, 默认="linear"
            插值方法："linear"（线性）、"cubic"（三次）或 "nearest"（最近邻）
            
        返回
        -------
        np.ndarray
            插值得到的 y 值
        """
        from scipy import interpolate
        
        if method == "linear":
            f = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
        elif method == "cubic":
            f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
        elif method == "nearest":
            f = interpolate.interp1d(x, y, kind='nearest', fill_value="extrapolate")
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return f(x_new)
    
    @staticmethod
    def interpolate_nd(points: np.ndarray, values: np.ndarray, 
                      new_points: np.ndarray, method: str = "linear") -> np.ndarray:
        """
        对 n 维数据进行插值。
        
        参数
        ----------
        points : np.ndarray
            原始点 (n_samples, n_dims)
        values : np.ndarray
            原始值 (n_samples,)
        new_points : np.ndarray
            用于插值的新点 (n_new, n_dims)
        method : str, 默认="linear"
            插值方法："linear"（线性）或 "nearest"（最近邻）
            
        返回
        -------
        np.ndarray
            插值结果
        """
        from scipy.interpolate import griddata
        
        return griddata(points, values, new_points, method=method)
    
    @staticmethod
    def compute_quadrature_points(n: int, method: str = "gauss-legendre") -> Tuple[np.ndarray, np.ndarray]:
        """
        计算求积点和权重。
        
        参数
        ----------
        n : int
            求积点的数量
        method : str, 默认="gauss-legendre"
            求积方法："gauss-legendre"（高斯-勒让德）、"gauss-chebyshev"（高斯-切比雪夫）或 "trapezoidal"（梯形）
            
        返回
        -------
        Tuple[np.ndarray, np.ndarray]
            求积点和权重
        """
        if method == "gauss-legendre":
            from numpy.polynomial.legendre import leggauss
            points, weights = leggauss(n)
            # 将区间从 [-1, 1] 变换到 [0, 1]
            points = 0.5 * (points + 1)
            weights = 0.5 * weights
            return points, weights
        
        elif method == "gauss-chebyshev":
            # 第一类切比雪夫节点
            k = np.arange(1, n + 1)
            points = np.cos((2 * k - 1) * np.pi / (2 * n))
            weights = np.pi / n * np.ones(n)
            # 将区间从 [-1, 1] 变换到 [0, 1]
            points = 0.5 * (points + 1)
            weights = 0.5 * weights
            return points, weights
        
        elif method == "trapezoidal":
            points = np.linspace(0, 1, n)
            weights = np.ones(n) / (n - 1)
            weights[0] = weights[-1] = 0.5 / (n - 1)
            return points, weights
        
        else:
            raise ValueError(f"Unknown quadrature method: {method}")
    
    @staticmethod
    def compute_fourier_coefficients(f: Callable, L: float = 2*np.pi, 
                                    n_terms: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算周期函数的傅里叶系数。
        
        参数
        ----------
        f : Callable
            周期函数
        L : float, 默认=2*pi
            周期
        n_terms : int, 默认=10
            傅里叶项数
            
        返回
        -------
        Tuple[np.ndarray, np.ndarray]
            a_n（余弦）和 b_n（正弦）系数
        """
        a = np.zeros(n_terms + 1)
        b = np.zeros(n_terms)
        
        # a0
        def integrand_a0(x):
            return f(x)
        
        a[0] = (1/L) * NumericalUtils.integrate_1d(integrand_a0, -L/2, L/2)
        
        # 其余系数
        for n in range(1, n_terms + 1):
            def integrand_an(x):
                return f(x) * np.cos(2 * np.pi * n * x / L)
            
            def integrand_bn(x):
                return f(x) * np.sin(2 * np.pi * n * x / L)
            
            a[n] = (2/L) * NumericalUtils.integrate_1d(integrand_an, -L/2, L/2)
            
            if n < n_terms:
                b[n-1] = (2/L) * NumericalUtils.integrate_1d(integrand_bn, -L/2, L/2)
        
        return a, b
