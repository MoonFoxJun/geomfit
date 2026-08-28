"""Polynomial basis functions over a single dimension: monomial, Legendre, and Chebyshev families."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class PolynomialBasis:
    """Polynomial basis functions over a single dimension."""
    
    @staticmethod
    def create_basis(dim: int, max_order: int) -> List[BasisInfo]:
        """Create monomial basis functions φ(x) = x^n for n = 0, ..., max_order.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_order : int
            Highest polynomial order (inclusive).

        Returns
        -------
        List[BasisInfo]
            One monomial basis per order from 0 to max_order.
        """
        bases = []
        # 生成 0 到 max_order 的单项式基 φ(x) = x^n；注意 lambda 用 order=order
        # 作为默认参数把循环变量"冻结"下来，否则所有闭包共享同一个 order 变量，
        # 循环结束后会全部指向最后一个值（经典 Python 闭包陷阱）
        for order in range(max_order + 1):
            bases.append(BasisInfo(
                name=f"Polynomial_order{order}",
                dim=dim,
                params={"order": order},
                func=lambda x, order=order: x ** order
            ))
        return bases
    
    @staticmethod
    def legendre_basis(dim: int, max_order: int) -> List[BasisInfo]:
        """Create Legendre polynomial basis functions P_n(x) for n = 0, ..., max_order.

        The polynomials are evaluated with numpy's Legendre routine (three-term
        recurrence), which is numerically more stable than monomial powers.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_order : int
            Highest polynomial order (inclusive).

        Returns
        -------
        List[BasisInfo]
            One Legendre basis per order from 0 to max_order.
        """
        from numpy.polynomial.legendre import legval
        bases = []
        for order in range(max_order + 1):
            # 构造系数向量：只有第 order 个分量为 1、其余为 0，
            # 代入 legval 后得到恰为 P_order(x)（Legendre 多项式系）
            coeffs = [0] * (order + 1)
            coeffs[order] = 1
            bases.append(BasisInfo(
                name=f"Legendre_order{order}",
                dim=dim,
                params={"order": order, "coeffs": coeffs},
                # 同样用默认参数冻结 coeffs，避免闭包共享同一个列表
                func=lambda x, coeffs=coeffs: legval(x, coeffs)
            ))
        return bases
    
    @staticmethod
    def chebyshev_basis(dim: int, max_order: int, kind: str = "first") -> List[BasisInfo]:
        """Create Chebyshev polynomial basis functions up to the given order.

        `kind` selects the family naming: "first" produces T_n(x), "second"
        produces U_n(x); both are evaluated with numpy's chebval.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_order : int
            Highest polynomial order (inclusive).
        kind : str, default="first"
            "first" (Chebyshev T_n) or "second" (Chebyshev U_n).

        Returns
        -------
        List[BasisInfo]
            One Chebyshev basis per order from 0 to max_order.
        """
        from numpy.polynomial.chebyshev import chebval
        bases = []
        for order in range(max_order + 1):
            # 系数向量第 order 个分量为 1、其余为 0 → 求值结果恰为单个 Chebyshev 多项式
            coeffs = [0] * (order + 1)
            coeffs[order] = 1
            if kind == "first":
                # 第一类 Chebyshev 多项式 T_n(x) = cos(n·arccos x)，在 [-1,1] 上关于
                # 权重 1/√(1−x²) 正交，且是极小极大意义下的最优多项式逼近（龙格现象更弱）
                bases.append(BasisInfo(
                    name=f"ChebyshevT_order{order}",
                    dim=dim,
                    params={"order": order, "coeffs": coeffs, "kind": "first"},
                    func=lambda x, coeffs=coeffs: chebval(x, coeffs)
                ))
            else:  # second kind
                # 第二类 Chebyshev 多项式 U_n(x) = sin((n+1)θ)/sin θ（x = cos θ），
                # 关于权重 √(1−x²) 正交，边界处的行为与 T_n 不同
                bases.append(BasisInfo(
                    name=f"ChebyshevU_order{order}",
                    dim=dim,
                    params={"order": order, "coeffs": coeffs, "kind": "second"},
                    func=lambda x, coeffs=coeffs: chebval(x, coeffs)
                ))
        return bases
