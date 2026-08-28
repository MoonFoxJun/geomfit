"""Polynomial basis functions."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class PolynomialBasis:
    """Polynomial basis functions."""
    
    @staticmethod
    def create_basis(dim: int, max_order: int) -> List[BasisInfo]:
        """Create polynomial basis functions up to specified order."""
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
        """Create Legendre polynomial basis functions."""
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
        """Create Chebyshev polynomial basis functions."""
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
