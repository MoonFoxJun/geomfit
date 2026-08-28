"""Numerical utilities for the functional solver."""

import numpy as np
from typing import Callable, Dict, Any, Tuple, List
from ..core.data_container import MultiDimData

class NumericalUtils:
    """Numerical utilities for integration, differentiation, interpolation, and quadrature."""
    
    @staticmethod
    def integrate_1d(f: Callable, a: float, b: float, n_points: int = 1000) -> float:
        """
        Integrate a 1-D function using Simpson's rule.
        
        Parameters
        ----------
        f : Callable
            Integrand.
        a : float
            Lower integration bound.
        b : float
            Upper integration bound.
        n_points : int, default=1000
            Number of integration points.
            
        Returns
        -------
        float
            Approximation of the integral.
        """
        if n_points % 2 == 0:
            n_points += 1  # 辛普森法则要求节点数为奇数（偶数个区间）
        
        x = np.linspace(a, b, n_points)
        y = f(x)
        
        h = (b - a) / (n_points - 1)
        
        # 辛普森法则：(h/3)(y_0 + y_n + 4·Σ(奇数下标) + 2·Σ(偶数下标))
        # 即用分段抛物线逼近被积函数，奇数节点权 4、偶数节点权 2、
        # 两端权 1；比梯形法则精度高两阶（O(h⁴)）。
        integral = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
        return integral
    
    @staticmethod
    def integrate_nd(f: Callable, bounds: List[Tuple[float, float]], 
                    n_points_per_dim: int = 50) -> float:
        """
        Integrate an n-dimensional function by Monte Carlo sampling.
        
        Parameters
        ----------
        f : Callable
            Integrand.
        bounds : List[Tuple[float, float]]
            (lower, upper) bound for each dimension.
        n_points_per_dim : int, default=50
            Number of samples per dimension.
            
        Returns
        -------
        float
            Approximation of the integral.
        """
        dim = len(bounds)
        n_samples = n_points_per_dim ** dim
        
        if n_samples > 1e6:  # 样本数上限：防止高维时样本数爆炸（d 维下呈指数增长）
            n_samples = 1000000
        
        # 在每个维度上独立均匀采样，得到 d 组随机样本
        samples = []
        for i in range(dim):
            a, b = bounds[i]
            samples.append(np.random.uniform(a, b, n_samples))
        
        # 按列拼接成 (n_samples, dim) 的采样点矩阵，逐行调用被积函数
        points = np.column_stack(samples)
        values = np.apply_along_axis(f, 1, points)
        
        # 积分区域的体积 = 各维度区间长度的乘积
        volume = 1.0
        for a, b in bounds:
            volume *= (b - a)
        
        # 蒙特卡洛估计：积分 ≈ 体积 × 采样点函数值的均值。
        # 原理是 ∫_Ω f dV ≈ |Ω| · E[f]，误差按 1/√N 收敛，
        # 收敛速度与维度无关（这正是高维积分选 MC 而非网格法的原因）。
        integral = volume * np.mean(values)
        return integral
    
    @staticmethod
    def differentiate(f: Callable, x: float, h: float = 1e-5, method: str = "central") -> float:
        """
        Compute the numerical derivative of a function.
        
        Parameters
        ----------
        f : Callable
            Function to differentiate.
        x : float
            Point at which the derivative is evaluated.
        h : float, default=1e-5
            Step size.
        method : str, default="central"
            Differentiation scheme: "forward", "backward", or "central".
            
        Returns
        -------
        float
            Approximation of the derivative.
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
        Compute the numerical gradient of a multivariate function by central differences.
        
        Parameters
        ----------
        f : Callable
            Function to differentiate.
        x : np.ndarray
            Point at which the gradient is evaluated.
        h : float, default=1e-5
            Step size.
            
        Returns
        -------
        np.ndarray
            Gradient vector.
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
        Compute the numerical Hessian matrix of a multivariate function.
        
        Parameters
        ----------
        f : Callable
            Function to differentiate.
        x : np.ndarray
            Point at which the Hessian is evaluated.
        h : float, default=1e-5
            Step size.
            
        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # 对角元：二阶导数（对第 i 个变量求两次导），
                    # 用中心差分 (f(x+h) − 2f(x) + f(x−h))/h²
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    hess[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
                else:
                    # 非对角元：混合偏导数 ∂²f/∂xᵢ∂xⱼ，
                    # 用四个角点组合的中心差分公式
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
                    
                    # 公式：∂²f/∂xᵢ∂xⱼ ≈ (f(+,+) − f(+,−) − f(−,+) + f(−,−)) / (4h²)
                    hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
        
        return hess
    
    @staticmethod
    def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, 
                      method: str = "linear") -> np.ndarray:
        """
        Interpolate 1-D data.
        
        Parameters
        ----------
        x : np.ndarray
            Original x values.
        y : np.ndarray
            Original y values.
        x_new : np.ndarray
            New x values at which to interpolate.
        method : str, default="linear"
            Interpolation method: "linear", "cubic", or "nearest".
            
        Returns
        -------
        np.ndarray
            Interpolated y values.
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
        Interpolate n-dimensional data.
        
        Parameters
        ----------
        points : np.ndarray
            Original points, shape (n_samples, n_dims).
        values : np.ndarray
            Original values, shape (n_samples,).
        new_points : np.ndarray
            Points at which to interpolate, shape (n_new, n_dims).
        method : str, default="linear"
            Interpolation method: "linear" or "nearest".
            
        Returns
        -------
        np.ndarray
            Interpolated values.
        """
        from scipy.interpolate import griddata
        
        return griddata(points, values, new_points, method=method)
    
    @staticmethod
    def compute_quadrature_points(n: int, method: str = "gauss-legendre") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quadrature nodes and weights on [0, 1].
        
        Parameters
        ----------
        n : int
            Number of quadrature nodes.
        method : str, default="gauss-legendre"
            Quadrature rule: "gauss-legendre", "gauss-chebyshev", or "trapezoidal".
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Quadrature nodes and weights.
        """
        if method == "gauss-legendre":
            from numpy.polynomial.legendre import leggauss
            points, weights = leggauss(n)
            # 高斯-勒让德节点定义在 [-1, 1] 上，通过仿射变换
            # x' = (x+1)/2 映射到 [0, 1]（权重相应减半，
            # 因为 dx' = dx/2）。
            points = 0.5 * (points + 1)
            weights = 0.5 * weights
            return points, weights
        
        elif method == "gauss-chebyshev":
            # 第一类切比雪夫节点：x_k = cos((2k−1)π/(2n))，k=1..n，
            # 它们是切比雪夫多项式 T_n 的根，位于 [-1, 1] 且两端更密集
            k = np.arange(1, n + 1)
            points = np.cos((2 * k - 1) * np.pi / (2 * n))
            weights = np.pi / n * np.ones(n)
            # 同样映射到 [0, 1]
            points = 0.5 * (points + 1)
            weights = 0.5 * weights
            return points, weights
        
        elif method == "trapezoidal":
            # 梯形法则：等距节点，两端权重为中间的一半
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
        Compute the Fourier coefficients of a periodic function.
        
        Parameters
        ----------
        f : Callable
            Periodic function.
        L : float, default=2*pi
            Period.
        n_terms : int, default=10
            Number of Fourier terms.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            a_n (cosine) and b_n (sine) coefficients.
        """
        a = np.zeros(n_terms + 1)
        b = np.zeros(n_terms)
        
        # 常数项 a₀：f 在一个周期内的平均值（傅里叶级数的直流分量）
        def integrand_a0(x):
            return f(x)
        
        a[0] = (1/L) * NumericalUtils.integrate_1d(integrand_a0, -L/2, L/2)
        
        # 对每个谐波阶数 n 计算余弦系数 a_n 与正弦系数 b_n：
        #   a_n = (2/L) ∫ f(x)cos(2πnx/L) dx
        #   b_n = (2/L) ∫ f(x)sin(2πnx/L) dx
        # 积分核里的 2πn/L 保证谐波频率为 n 倍的基频。
        for n in range(1, n_terms + 1):
            def integrand_an(x):
                return f(x) * np.cos(2 * np.pi * n * x / L)
            
            def integrand_bn(x):
                return f(x) * np.sin(2 * np.pi * n * x / L)
            
            a[n] = (2/L) * NumericalUtils.integrate_1d(integrand_an, -L/2, L/2)
            
            # b 数组只有 n_terms 个元素（没有 b_0），所以最后一个
            # 正弦系数不计算（n == n_terms 时跳过）
            if n < n_terms:
                b[n-1] = (2/L) * NumericalUtils.integrate_1d(integrand_bn, -L/2, L/2)
        
        return a, b
