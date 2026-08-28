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
            coeffs = [0] * (order + 1)
            coeffs[order] = 1
            bases.append(BasisInfo(
                name=f"Legendre_order{order}",
                dim=dim,
                params={"order": order, "coeffs": coeffs},
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
            coeffs = [0] * (order + 1)
            coeffs[order] = 1
            if kind == "first":
                bases.append(BasisInfo(
                    name=f"ChebyshevT_order{order}",
                    dim=dim,
                    params={"order": order, "coeffs": coeffs, "kind": "first"},
                    func=lambda x, coeffs=coeffs: chebval(x, coeffs)
                ))
            else:  # second kind
                bases.append(BasisInfo(
                    name=f"ChebyshevU_order{order}",
                    dim=dim,
                    params={"order": order, "coeffs": coeffs, "kind": "second"},
                    func=lambda x, coeffs=coeffs: chebval(x, coeffs)
                ))
        return bases
