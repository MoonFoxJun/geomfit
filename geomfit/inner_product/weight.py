"""Weight functions for weighted inner products."""

import numpy as np
from typing import Callable, Dict, Any, Union

# numpy>=2.0 renamed np.trapz to np.trapezoid; keep compatibility with numpy 1.x
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
            if isinstance(x, dict):
                return c
            elif isinstance(x, np.ndarray):
                return np.full_like(x, c, dtype=float)
            else:
                return c

        return WeightFunction(weight_func)

    @staticmethod
    def exponential(alpha: float = 1.0) -> 'WeightFunction':
        """Exponential weight function w(x) = exp(-α|x|)."""
        def weight_func(x):
            if isinstance(x, dict):
                norm = np.sqrt(sum(val ** 2 for val in x.values()))
                return np.exp(-alpha * norm)
            elif isinstance(x, np.ndarray):
                return np.exp(-alpha * np.abs(x))
            else:
                return np.exp(-alpha * np.abs(x))

        return WeightFunction(weight_func)

    @staticmethod
    def gaussian(sigma: float = 1.0, center: float = 0.0) -> 'WeightFunction':
        """Gaussian weight function w(x) = exp(-(x - center)²/(2σ²))."""
        def weight_func(x):
            if isinstance(x, dict):
                # use the first dimension for dict points
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
        """Polynomial weight function w(x) = 1/(1 + coef * |x|^degree)."""
        def weight_func(x):
            if isinstance(x, dict):
                norm = np.sqrt(sum(val ** 2 for val in x.values()))
                return 1.0 / (1.0 + coef * (norm ** degree))
            elif isinstance(x, np.ndarray):
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))
            else:
                return 1.0 / (1.0 + coef * (np.abs(x) ** degree))

        return WeightFunction(weight_func)

    @staticmethod
    def chebyshev_weight(kind: str = "first") -> 'WeightFunction':
        """Chebyshev weight function for orthogonal polynomials."""
        if kind == "first":
            # w(x) = 1/√(1 - x²), for x ∈ (-1, 1)
            def weight_func(x):
                if isinstance(x, dict):
                    # use the first dimension for dict points
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
            # w(x) = √(1 - x²), for x ∈ (-1, 1)
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
        x = np.linspace(a, b, n_points)
        w = self(x)
        return _trapz(w, x)
