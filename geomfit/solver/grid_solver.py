"""Separable (tensor-product) solver for full-grid data.

Mathematical background
-----------------------
For a tensor-product basis ``phi_p = X_i(x) Y_j(y) Z_k(z)`` the joint normal
equations are

    G c = b,     G in R^{P x P},  P = I * J * K,

but on a full Cartesian grid with a separable inner product every entry
factorises: ``<X_i Y_j Z_k, X_i' Y_j' Z_k'> = Gx[i,i'] Gy[j,j'] Gz[k,k']``.
The system therefore becomes a *separable sandwich* over the data tensor ``Z``:

    B = Z x_1 (Phi_1^T W_1) x_2 ... x_D (Phi_D^T W_D)      (readings)
    C = B x_1 G_1^{-1}      x_2 ... x_D G_D^{-1}           (coefficients)

with ``x_d`` denoting contraction along mode d, ``W_d`` the 1-D quadrature
weights (trapezoid weights for continuous inner products, ones for discrete
ones) and ``G_d = Phi_d^T W_d Phi_d`` the per-dimension Gram matrices. The
O(P^3) joint solve is replaced by D independent solves of size I_d x I_d.

Fallback
--------
The separable path is used only when all of the following hold:

- the points form a full Cartesian grid (:func:`geomfit.core.grid.detect_grid`);
- the basis set has a Cartesian tensor structure (every basis function is a
  product of exactly one factor per dimension, and all combinations appear);
- the inner product is separable, i.e. no weight function is given;
- the requested regularization is Tikhonov (``alpha``) or none.

Otherwise the solver transparently delegates to the general joint
:class:`~geomfit.solver.gram_solver.GramSolver` and records the reason in
``debug_info["fallback_reason"]``.

Numerical note
--------------
The separable path reports the per-dimension condition numbers through
:meth:`GridSolver.get_condition_numbers`. They diagnose the *basis*, not the
algorithm: ten monomial terms on [0, 1] already give cond ~ 1e13, and in that
regime both the joint and the separable solves are numerically unreliable --
use an orthogonal basis (Fourier on [0, 1], or a domain-adapted family) or
regularization. With an orthogonal basis the separable solve reproduces the
joint solve to machine precision (measured: P = 225 on a 24x24 grid agrees to
~1e-13) while being roughly two orders of magnitude faster.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..core.grid import detect_grid
from ..inner_product.base import InnerProduct, trapezoid_weights
from .gram_solver import GramSolver


class GridSolver:
    """Separable solver for tensor-product bases on Cartesian grids."""

    def __init__(self, basis_set: BasisSet, inner_product: InnerProduct):
        """
        Parameters
        ----------
        basis_set : BasisSet
            Basis functions; a Cartesian tensor-product structure is required
            for the separable path (otherwise the joint solver is used).
        inner_product : InnerProduct
            Inner-product definition. A weighted inner product disables the
            separable path because its quadrature does not factorise.
        """
        self.basis_set = basis_set
        self.inner_product = inner_product
        self.data: Optional[MultiDimData] = None
        self.target: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None      # 与 basis_set.bases 顺序对齐
        self.coefficient_tensor: Optional[np.ndarray] = None  # 多维系数张量 C（仅分离路径）
        self.debug_info: Dict[str, Any] = {}
        self._regularization: Optional[Dict] = None
        # 兜底：通用联合求解器，分离条件不满足时直接委托给它
        self._fallback = GramSolver(basis_set, inner_product)

    # ------------------------------------------------------------------ data

    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the training data."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"

    # ------------------------------------------------------- basis structure

    def _factor_structure(self):
        """Factor the basis set into per-dimension 1-D factor lists.

        Returns ``(dims, keys, factor_data, shape, multi)`` where ``multi[p]`` is
        the multi-index of basis ``p`` in the per-dimension factor lists, or
        ``None`` when the basis set is not a Cartesian tensor-product structure.
        """
        dims = list(self.data.dims)

        # 把每个基函数统一拆成"因子列表"：张量积基用其 factors，单因子基包装成长度 1 的列表
        per_basis: List[List[Tuple]] = []
        for basis in self.basis_set.bases:
            if basis.factors:
                per_basis.append(list(basis.factors))
            else:
                per_basis.append([(basis.dim, basis.func, basis.params)])

        # 逐维收集出现过的因子；因子身份 = (函数对象 id, 参数字典)
        keys: Dict[int, List[Tuple]] = {d: [] for d in dims}
        factor_data: Dict[int, Dict[Tuple, Tuple]] = {d: {} for d in dims}
        for facs in per_basis:
            for dim, func, params in facs:
                if dim not in keys:
                    return None
                key = (id(func), repr(sorted((params or {}).items())))
                if key not in factor_data[dim]:
                    keys[dim].append(key)
                    factor_data[dim][key] = (func, params)

        # 每个基函数必须"每维恰好一个因子"
        multi: List[Tuple[int, ...]] = []
        for facs in per_basis:
            seen: Dict[int, int] = {}
            for dim, func, params in facs:
                key = (id(func), repr(sorted((params or {}).items())))
                if dim in seen:
                    return None  # 同一维度出现两个因子 → 不是纯张量积
                seen[dim] = keys[dim].index(key)
            if set(seen.keys()) != set(dims):
                return None  # 缺少某个维度的因子（例如直和/加法模型的基）
            multi.append(tuple(seen[d] for d in dims))

        shape = tuple(len(keys[d]) for d in dims)
        if len(set(multi)) != len(multi):
            return None
        if len(multi) != int(np.prod(shape)):
            return None  # 组合不全 → 不是完整笛卡尔积

        return dims, keys, factor_data, shape, multi

    # -------------------------------------------------------- factor matrices

    def _axis_weights(self, axis: np.ndarray) -> np.ndarray:
        """1-D quadrature weights for one axis (separable volume element)."""
        if self.inner_product.is_continuous:
            return trapezoid_weights(axis)  # 连续内积：梯形权重（体积元的 1D 因子）
        return np.ones(len(axis))           # 离散内积：无权重的经验测度

    def _factor_matrices(self, dims, keys, factor_data, axes):
        """Evaluate the per-dimension factor matrices Phi_d (n_d x I_d)."""
        phis = []
        weights = []
        for dim, axis in zip(dims, axes):
            cols = []
            for key in keys[dim]:
                func, params = factor_data[dim][key]
                try:
                    vals = np.asarray(func(axis, **(params or {})), dtype=float)
                    if vals.shape != np.shape(axis):
                        raise ValueError
                except Exception:
                    # 非向量化因子（如 Haar 小波的区间判断）：逐点求值
                    vals = np.array([
                        float(np.asarray(func(float(a), **(params or {}))).ravel()[0])
                        for a in axis
                    ])
                cols.append(vals)
            phis.append(np.column_stack(cols) if cols else np.zeros((len(axis), 0)))
            weights.append(self._axis_weights(np.asarray(axis, dtype=float)))
        return phis, weights

    @staticmethod
    def _contract(matrix: np.ndarray, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Contract ``matrix`` (I x n_mode) with axis ``mode`` of ``tensor``."""
        out = np.tensordot(matrix, tensor, axes=(1, mode))
        return np.moveaxis(out, 0, mode)  # 把新产生的基指标放回它原来的位置

    # ----------------------------------------------------------------- solve

    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve for the coefficients (separable path when applicable)."""
        if self.data is None or self.target is None:
            raise ValueError("Load data first")

        self._regularization = regularization

        # 条件一：权重函数不可分离
        if self.inner_product.weight_func is not None:
            return self._solve_fallback("weighted inner product is not separable")

        # 条件二：正则化方式（分离路径只实现逐维 Tikhonov；其它方式交给联合求解以保证语义一致）
        method = (regularization or {}).get("method", "tikhonov")
        if regularization and method not in ("tikhonov",):
            return self._solve_fallback(f"regularization method '{method}' is not separable")

        # 条件三：数据必须是完整笛卡尔网格
        grid = detect_grid(self.data)
        if grid is None:
            return self._solve_fallback("data points do not form a full Cartesian grid")

        # 条件四：基函数必须具备笛卡尔张量积结构
        struct = self._factor_structure()
        if struct is None:
            return self._solve_fallback("basis set is not a Cartesian tensor-product structure")

        dims, keys, factor_data, factor_shape, multi = struct
        axes = grid["axes"]
        phis, weights = self._factor_matrices(dims, keys, factor_data, axes)

        alpha = float((regularization or {}).get("alpha", 0.0))

        # 数据张量 Z：把数据点按多重指标散布进网格张量（允许数据点任意排列）
        Z = np.empty(grid["shape"], dtype=float)
        Z.flat[grid["flat_index"]] = self.target

        # 第一步：读数 B = Z ×_d (Φ_dᵀ W_d)
        B: np.ndarray = Z
        for d in range(len(dims)):
            M = (phis[d] * weights[d][:, None]).T  # (I_d, n_d)
            B = self._contract(M, B, d)

        # 第二步：系数 C = B ×_d (G_d + αI)^{-1}，G_d = Φ_dᵀ W_d Φ_d
        C: np.ndarray = B
        condition_numbers = []
        for d in range(len(dims)):
            G = phis[d].T @ (phis[d] * weights[d][:, None])
            condition_numbers.append(float(np.linalg.cond(G)))
            if alpha > 0.0:
                G = G + alpha * np.eye(G.shape[0])  # 逐维 Tikhonov：分离路径下的正则化写法
            C = self._contract(np.linalg.pinv(G), C, d)

        # 把系数张量摊平回 basis_set.bases 的顺序，保证 predict 与全库语义一致
        c_flat = np.empty(len(self.basis_set.bases), dtype=float)
        for p, midx in enumerate(multi):
            c_flat[p] = C[midx]

        self.coefficient_tensor = C
        self.coefficients = c_flat
        self.debug_info = {
            "method": "separable",
            "grid_shape": grid["shape"],
            "factor_shape": factor_shape,
            "condition_numbers": condition_numbers,
            "alpha": alpha,
        }
        return c_flat

    def _solve_fallback(self, reason: str) -> np.ndarray:
        """Delegate to the general joint Gram solver."""
        self._fallback.load_data(self.data, self.target)
        coeff = self._fallback.solve(self._regularization)
        self.coefficients = coeff
        self.coefficient_tensor = None
        self.debug_info = {
            "method": "joint",
            "fallback_reason": reason,
            "condition_number": float(self._fallback.get_condition_number()),
            "alpha": float((self._regularization or {}).get("alpha", 0.0)),
        }
        return coeff

    # ------------------------------------------------------------- prediction

    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict target values at new points."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients

    # ------------------------------------------------------------ diagnostics

    def get_condition_numbers(self) -> List[float]:
        """Condition numbers of the separable per-dimension systems (or the joint one)."""
        if not self.debug_info:
            raise ValueError("Must call solve() first")
        if self.debug_info["method"] == "separable":
            return list(self.debug_info["condition_numbers"])
        return [self.debug_info["condition_number"]]

    def is_separable(self) -> bool:
        """Whether the last solve used the separable sandwich path."""
        return bool(self.debug_info.get("method") == "separable")
