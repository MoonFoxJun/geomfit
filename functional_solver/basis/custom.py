"""Utilities for constructing custom basis functions."""

from typing import Callable, Dict, Any, List
from ..core.basis_container import BasisInfo

class CustomBasis:
    """Utilities for creating custom basis functions."""
    
    @staticmethod
    def from_function(dim: int, name: str, func: Callable, params: Dict[str, Any] = None) -> BasisInfo:
        """Create a single-factor basis from an arbitrary callable func(x, **params).

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        name : str
            Name of the basis.
        func : Callable
            Evaluation function, called as func(x, **params).
        params : Dict[str, Any], optional
            Parameters passed to `func`.

        Returns
        -------
        BasisInfo
            Single-factor custom basis.
        """
        return BasisInfo(
            name=name,
            dim=dim,
            params=params or {},
            func=func
        )
    
    @staticmethod
    def piecewise_linear(dim: int, breakpoints: List[float], values: List[float]) -> List[BasisInfo]:
        """Create piecewise-linear (hat) basis functions on the given breakpoints.

        The i-th hat function is 1 at breakpoints[i] and decays linearly to 0 at
        the neighbouring breakpoints; the boundary hats are zero outside the
        breakpoint range. `values` is stored per basis but is not used by the
        evaluation.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        breakpoints : List[float]
            Breakpoints defining the hat functions (assumed in ascending order).
        values : List[float]
            Value stored per basis, one entry per breakpoint.

        Returns
        -------
        List[BasisInfo]
            One hat-function basis per breakpoint.
        """
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
        """Create B-spline basis functions of the given degree from a knot vector.

        The i-th basis function is the B-spline with a single unit coefficient
        at index i. Evaluation errors (e.g. outside the spline domain) are
        suppressed and yield 0.0.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        knots : List[float]
            Knot vector (sorted internally).
        degree : int, default=3
            Spline degree.

        Returns
        -------
        List[BasisInfo]
            One B-spline basis per basis function, len(knots) - degree - 1 in total.
        """
        from scipy.interpolate import BSpline
        import warnings
        
        # Sort the knot vector
        knots = sorted(knots)
        
        bases = []
        n_basis = len(knots) - degree - 1
        
        for i in range(n_basis):
            # Basis function with a single unit coefficient at index i
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
        """Validate that every basis in `basis_list` involves `dim` and return the list unchanged.

        Parameters
        ----------
        dim : int
            Dimension that every basis must involve.
        basis_list : List[BasisInfo]
            Bases to validate (tensor-product bases are checked against their
            factor dimensions).

        Returns
        -------
        List[BasisInfo]
            The validated basis list.
        """
        # Every basis must involve the given dimension; tensor-product bases are
        # judged by their factor dimensions
        for basis in basis_list:
            if dim not in basis.dims:
                raise ValueError(f"Basis dimension mismatch: expected {dim}, got {basis.dims}")
        
        return basis_list
