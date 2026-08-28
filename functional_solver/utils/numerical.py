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
            n_points += 1  # Simpson's rule requires an odd number of points
        
        x = np.linspace(a, b, n_points)
        y = f(x)
        
        h = (b - a) / (n_points - 1)
        
        # Simpson's rule: (h/3)(y_0 + y_n + 4*sum(odd) + 2*sum(even))
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
        
        if n_samples > 1e6:  # Cap the total number of samples
            n_samples = 1000000
        
        # Draw uniform random samples per dimension
        samples = []
        for i in range(dim):
            a, b = bounds[i]
            samples.append(np.random.uniform(a, b, n_samples))
        
        # Evaluate the integrand
        points = np.column_stack(samples)
        values = np.apply_along_axis(f, 1, points)
        
        # Volume of the integration domain
        volume = 1.0
        for a, b in bounds:
            volume *= (b - a)
        
        # Monte Carlo estimate: volume * mean(f)
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
                    # Diagonal entries: second derivative
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    hess[i, i] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
                else:
                    # Off-diagonal entries: mixed partial derivative
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
            # Map the nodes from [-1, 1] to [0, 1]
            points = 0.5 * (points + 1)
            weights = 0.5 * weights
            return points, weights
        
        elif method == "gauss-chebyshev":
            # Chebyshev nodes of the first kind
            k = np.arange(1, n + 1)
            points = np.cos((2 * k - 1) * np.pi / (2 * n))
            weights = np.pi / n * np.ones(n)
            # Map the nodes from [-1, 1] to [0, 1]
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
        
        # Constant term a0
        def integrand_a0(x):
            return f(x)
        
        a[0] = (1/L) * NumericalUtils.integrate_1d(integrand_a0, -L/2, L/2)
        
        for n in range(1, n_terms + 1):
            def integrand_an(x):
                return f(x) * np.cos(2 * np.pi * n * x / L)
            
            def integrand_bn(x):
                return f(x) * np.sin(2 * np.pi * n * x / L)
            
            a[n] = (2/L) * NumericalUtils.integrate_1d(integrand_an, -L/2, L/2)
            
            if n < n_terms:
                b[n-1] = (2/L) * NumericalUtils.integrate_1d(integrand_bn, -L/2, L/2)
        
        return a, b
