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
            # params 缺省（None）时统一转成空字典，保证求值时 **(params or {}) 解包安全
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
        
        # 对每个断点 breakpoints[i] 构造一个"帽子函数"（hat/tent 函数）：
        # 在第 i 个断点处取峰值 1，向两侧线性衰减到相邻断点处为 0；
        # 全部帽子函数构成分片线性插值的标准基（一维有限元基）
        for i in range(n):
            # 用 make_hat_func(i) 工厂函数包一层：把循环变量 i 作为参数传入，
            # 闭包才能捕获到正确的 i（直接定义闭包会共享循环变量，最后全是同一个 i）
            def make_hat_func(i):
                def hat_func(x):
                    if i == 0:
                        # 最左端帽子：只在 [b0, b1] 上非零，从 b0 处的 1 线性降到 b1 处的 0
                        if breakpoints[0] <= x <= breakpoints[1]:
                            return (breakpoints[1] - x) / (breakpoints[1] - breakpoints[0])
                    elif i == n - 1:
                        # 最右端帽子：只在 [b_{n-2}, b_{n-1}] 上非零，从 b_{n-2} 处的 0
                        # 线性升到 b_{n-1} 处的 1
                        if breakpoints[n-2] <= x <= breakpoints[n-1]:
                            return (x - breakpoints[n-2]) / (breakpoints[n-1] - breakpoints[n-2])
                    else:
                        # 内部帽子：三角形，左支在 [b_{i-1}, b_i] 上从 0 升到 1，
                        # 右支在 [b_i, b_{i+1}] 上从 1 降到 0
                        if breakpoints[i-1] <= x <= breakpoints[i]:
                            return (x - breakpoints[i-1]) / (breakpoints[i] - breakpoints[i-1])
                        elif breakpoints[i] <= x <= breakpoints[i+1]:
                            return (breakpoints[i+1] - x) / (breakpoints[i+1] - breakpoints[i])
                    # 落在支撑区间之外时，帽子函数取 0（边界帽子在断点范围外也是 0）
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
        
        # 先把节点向量排序：scipy 的 BSpline 要求节点序列非递减
        knots = sorted(knots)
        
        bases = []
        # B 样条基函数的个数 = 节点数 − 阶数 − 1（degree 即多项式阶数）
        n_basis = len(knots) - degree - 1
        
        for i in range(n_basis):
            # 第 i 个基函数：系数向量只在第 i 位取 1、其余为 0。因为 B 样条曲线是
            # "控制系数 × 基函数"的线性组合 Σ cᵢ Nᵢ(x)，单独置一个系数为 1 就得到单个基
            coeffs = [0] * n_basis
            coeffs[i] = 1.0
            
            # 用默认参数把 i/coeffs/knots/degree 冻结进闭包，避免循环变量共享
            def make_spline_func(i, coeffs, knots, degree):
                def spline_func(x):
                    # 屏蔽 scipy 的数值警告（如求值点落在节点区间外），并把一切求值异常
                    # 统一吞掉返回 0.0，保证任何输入都不会让整个拟合流程崩溃
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
        # 校验每个基都确实作用于目标维度 dim；对张量积基用它的 dims 属性
        # （所有因子的维度集合）来判断，而不是只看主维度，避免漏检
        for basis in basis_list:
            if dim not in basis.dims:
                raise ValueError(f"Basis dimension mismatch: expected {dim}, got {basis.dims}")
        
        return basis_list
