"""Custom basis function utilities."""

from typing import Callable, Dict, Any, List
from ..core.basis_container import BasisInfo

class CustomBasis:
    """Utilities for creating custom basis functions."""
    
    @staticmethod
    def from_function(dim: int, name: str, func: Callable, params: Dict[str, Any] = None) -> BasisInfo:
        """Create a basis function from a custom function."""
        return BasisInfo(
            name=name,
            dim=dim,
            params=params or {},
            func=func
        )
    
    @staticmethod
    def piecewise_linear(dim: int, breakpoints: List[float], values: List[float]) -> List[BasisInfo]:
        """Create piecewise linear basis functions (hat functions)."""
        if len(breakpoints) != len(values):
            raise ValueError("breakpoints and values must have the same length")
        
        bases = []
        n = len(breakpoints)
        
        for i in range(n):
            def make_hat_func(i):
                def hat_func(x):
                    if i == 0:
                        if breakpoints[0] <= x <= breakpoints[1]:
                            return (breakpoints[1] - x) / (breakpoints[1] - breakpoints[0])
                    elif i == n - 1:
                        if breakpoints[n-2] <= x <= breakpoints[n-1]:
                            return (x - breakpoints[n-2]) / (breakpoints[n-1] - breakpoints[n-2])
                    else:
                        if breakpoints[i-1] <= x <= breakpoints[i]:
                            return (x - breakpoints[i-1]) / (breakpoints[i] - breakpoints[i-1])
                        elif breakpoints[i] <= x <= breakpoints[i+1]:
                            return (breakpoints[i+1] - x) / (breakpoints[i+1] - breakpoints[i])
                    return 0.0
                return hat_func
            
            bases.append(BasisInfo(
                name=f"PiecewiseLinear_breakpoint{i}",
                dim=dim,
                params={"breakpoint": breakpoints[i], "value": values[i]},
                func=make_hat_func(i)
            ))
        
        return bases
    
    @staticmethod
    def spline_basis(dim: int, knots: List[float], degree: int = 3) -> List[BasisInfo]:
        """Create B-spline basis functions."""
        from scipy.interpolate import BSpline
        import warnings
        
        # Ensure knots are sorted
        knots = sorted(knots)
        
        bases = []
        n_basis = len(knots) - degree - 1
        
        for i in range(n_basis):
            # Create B-spline basis function
            coeffs = [0] * n_basis
            coeffs[i] = 1.0
            
            def make_spline_func(i, coeffs, knots, degree):
                def spline_func(x):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            return BSpline(knots, coeffs, degree)(x)
                        except:
                            return 0.0
                return spline_func
            
            bases.append(BasisInfo(
                name=f"BSpline_degree{degree}_basis{i}",
                dim=dim,
                params={"knots": knots, "degree": degree, "basis_index": i},
                func=make_spline_func(i, coeffs, knots, degree)
            ))
        
        return bases
    
    @staticmethod
    def composite_basis(dim: int, basis_list: List[BasisInfo]) -> List[BasisInfo]:
        """Combine multiple basis functions into a single list."""
        # 要求每个基都涉及指定维度（张量积基按其因子维度判断）
        for basis in basis_list:
            if dim not in basis.dims:
                raise ValueError(f"Basis dimension mismatch: expected {dim}, got {basis.dims}")
        
        return basis_list
