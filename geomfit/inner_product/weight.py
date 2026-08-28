"""Weight functions for weighted inner products."""

import numpy as np
from typing import Callable, Dict, Any, Union

# numpy 2.0 起 np.trapz 被改名为 np.trapezoid；这里做兼容处理以同时支持 numpy 1.x 与 2.x
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

class WeightFunction:
    """Weight function for a weighted inner product."""

    def __init__(self, weight_func: Callable):
        """
        Initialize the weight function.

        Parameters
        ----------
        weight_func : Callable
            Function returning the weight at a point or index.
            Signature: weight_func(x) -> float or np.ndarray.
        """
        self.weight_func = weight_func

    def __call__(self, x: Union[float, np.ndarray, Dict[int, float]]) -> Union[float, np.ndarray]:
        """
        Evaluate the weight function.

        Parameters
        ----------
        x : Union[float, np.ndarray, Dict[int, float]]
            Input point (or collection of points).

        Returns
        -------
        Union[float, np.ndarray]
            Weight value (or array of weight values).
        """
        return self.weight_func(x)

    @staticmethod
    def constant(c: float = 1.0) -> 'WeightFunction':
        """Constant weight function."""
        def weight_func(x):
            # 常数权重 w(x) ≡ c：无论输入是字典点、数组还是标量，都返回常数 c
            if isinstance(x, dict):
                return c
            elif isinstance(x, np.ndarray):
                return np.full_like(x, c, dtype=float)  # 数组输入：返回与 x 同形状的全 c 数组
            else:
                return c

        return WeightFunction(weight_func)

    @staticmethod
    def exponential(alpha: float = 1.0) -> 'WeightFunction':
        """Exponential weight function w(x) = exp(-α|x|)."""
        def weight_func(x):
            # 指数权重：w(x) = exp(-α|x|)；对字典点 x 取欧氏范数 ‖x‖ 作为自变量
            if isinstance(x, dict):
                norm = np.sqrt(sum(val ** 2 for val in x.values()))  # 多维字典点：‖x‖ = √(Σ_d x_d²)
                return np.exp(-alpha * norm)
            elif isinstance(x, np.ndarray):
                return np.exp(-alpha * np.abs(x))  # 数组：逐元素取 |x| 再指数衰减
            else:
                return np.exp(-alpha * np.abs(x))  # 标量：直接代入

        return WeightFunction(weight_func)

    @staticmethod
    def gaussian(sigma: float = 1.0, center: float = 0.0) -> 'WeightFunction':
        """Gaussian weight function w(x) = exp(-(x - center)²/(2σ²))."""
        def weight_func(x):
            # 高斯权重：w(x) = exp(-(x - center)²/(2σ²))，以 center 为中心、σ 控制衰减宽度
            if isinstance(x, dict):
                # 字典点只取第一个维度的坐标作为 x（一维高斯权重；多维时按主维度处理）
                if x:
                    first_key = list(x.keys())[0]
                    x_val = x[first_key]
                else:
                    x_val = 0.0  # 空字典：视作坐标 0
                return np.exp(-(x_val - center) ** 2 / (2 * sigma ** 2))
            elif isinstance(x, np.ndarray):
                return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))  # 数组：逐元素求值
            else:
                return np.exp(-(x - center) ** 2 / (2 * sigma ** 2))  # 标量：直接代入

        return WeightFunction(weight_func)

    @staticmethod
    def polynomial(degree: int = 2, coef: float = 1.0) -> 'WeightFunction':
        """Polynomial weight function w(x) = 1/(1 + coef * |x|^degree)."""
        def weight_func(x):
            # 多项式倒数权重：w(x) = 1/(1 + coef·|x|^degree)，距离越远衰减越慢（重尾分布）
            if isinstance(x, dict):
                norm = np.sqrt(sum(val ** 2 for val in x.values()))  # 字典点取欧氏范数 ‖x‖
                return 1.0 / (1.0 + coef * (norm ** degree))
            elif isinstance(x, np.ndarray):
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))  # 数组：逐元素取 |x|^degree
            else:
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))  # 标量：直接代入

        return WeightFunction(weight_func)

    @staticmethod
    def chebyshev_weight(kind: str = "first") -> 'WeightFunction':
        """Chebyshev weight function for orthogonal polynomials."""
        if kind == "first":
            # 第一类 Chebyshev 权：w(x) = 1/√(1 - x²)，仅定义于 x ∈ (-1, 1)
            # （在端点 ±1 处奇异，故区间外权重一律取 0）
            def weight_func(x):
                if isinstance(x, dict):
                    # 字典点只取第一个维度的坐标作为 x
                    if x:
                        first_key = list(x.keys())[0]
                        x_val = x[first_key]
                    else:
                        x_val = 0.0
                    if -1 < x_val < 1:
                        return 1.0 / np.sqrt(1 - x_val ** 2)  # 区间内按公式求值
                    else:
                        return 0.0  # 区间外（含端点）：权重为 0
                elif isinstance(x, np.ndarray):
                    mask = (x > -1) & (x < 1)  # 布尔掩码：只保留 (-1, 1) 内的元素
                    result = np.zeros_like(x)
                    result[mask] = 1.0 / np.sqrt(1 - x[mask] ** 2)  # 区间内逐元素求值，区间外保持 0
                    return result
                else:
                    if -1 < x < 1:
                        return 1.0 / np.sqrt(1 - x ** 2)
                    else:
                        return 0.0
        elif kind == "second":
            # 第二类 Chebyshev 权：w(x) = √(1 - x²)，仅定义于 x ∈ (-1, 1)
            # （与第一类相反：在端点处权重趋于 0，区间外一律取 0）
            def weight_func(x):
                if isinstance(x, dict):
                    # 字典点只取第一个维度的坐标作为 x
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
                    mask = (x > -1) & (x < 1)  # 布尔掩码：只保留 (-1, 1) 内的元素
                    result = np.zeros_like(x)
                    result[mask] = np.sqrt(1 - x[mask] ** 2)  # 区间内逐元素求值，区间外保持 0
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
        """Custom weight function."""
        return WeightFunction(func)

    def integrate(self, a: float, b: float, n_points: int = 1000) -> float:
        """
        Integrate the weight function over an interval.

        Parameters
        ----------
        a : float
            Lower bound.
        b : float
            Upper bound.
        n_points : int, default=1000
            Number of sampling points for the numerical integration.

        Returns
        -------
        float
            Approximate integral ∫[a,b] w(x) dx.
        """
        x = np.linspace(a, b, n_points)  # 在 [a, b] 上均匀采样 n_points 个点
        w = self(x)                      # 在这些采样点上求权重值
        return _trapz(w, x)              # 用复合梯形法则数值积分（_trapz 为 np.trapezoid/np.trapz 的兼容别名）
