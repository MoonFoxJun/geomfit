"""Direct-sum (additive) basis construction — retained as an independent implementation.

Mathematical structure
----------------------
Functions in the direct-sum space V = V₀ ⊕ V₁ ⊕ … take the additive form

    f(x₁, …, x_d) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d)

i.e. the per-dimension basis contributions are summed directly, with no
cross terms such as x·y. This is the counterpart of the tensor-product
basis (f = Σ cᵢⱼ Xᵢ(x)Yⱼ(y)).

Known limitations (the algorithm is retained; be aware of these)
----------------------------------------------------------------
1. Limited expressiveness: interactions or products between dimensions
   (e.g. x·y, sin(2πx)cos(2πy)) cannot be represented, so completeness is
   poor and information loss is high for general multi-dimensional
   function approximation.
2. Constant overlap: if every dimension provides a constant basis
   (x⁰, y⁰, …), the constant component is duplicated in the direct sum,
   making the Gram matrix singular. This construction keeps only the first
   constant basis by default (deduplication).
3. The direct sum is an appropriate model when the per-dimension
   contributions are genuinely independent (and is also more interpretable).

Relation to the tensor product
------------------------------
The direct-sum space is a low-dimensional subspace of the tensor-product
space: restricting the tensor-product basis to the factors in which all
but one dimension are constant recovers the direct sum. Hence
tensor-product basis ⊇ direct-sum basis, and the former is strictly more
expressive.

Rationale / use cases
---------------------
- High-dimensional data: the number of tensor-product bases grows
  exponentially with dimension (curse of dimensionality), whereas the
  direct sum grows only linearly — the only feasible explicit-basis route
  in high dimensions.
- Problems whose per-dimension contributions are independent, for which the
  direct sum is the true structure and no interaction terms are needed.

Usage
-----
    from geomfit.basis.additive import AdditiveBasis

    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
    basis_set = AdditiveBasis.build({0: x_bases, 1: y_bases})
    # Equivalent to concatenating the per-dimension bases into one BasisSet,
    # except that duplicate constant bases are removed.
"""

from typing import Dict, List
import numpy as np

from ..core.basis_container import BasisInfo, BasisSet


def _is_constant(basis: BasisInfo, rtol: float = 1e-12) -> bool:
    """Return True if a single-dimensional basis is a constant function.

    The basis is evaluated at three probe points; if the relative variation
    is negligible it is treated as constant. Tensor-product (multi-dimensional)
    bases cannot be probed on a single dimension and return False.

    Parameters
    ----------
    basis : BasisInfo
        Basis to test.
    rtol : float, default=1e-12
        Relative tolerance for the variation test.

    Returns
    -------
    bool
        True if the basis is constant, False otherwise.
    """
    dim = basis.dims[0]
    try:
        vals = np.asarray(
            [basis.evaluate({dim: p}) for p in (0.0, 1.0, 0.5)],
            dtype=float,
        )
    except Exception:
        return False
    scale = float(np.max(np.abs(vals)))
    if scale == 0.0:
        return True  # a basis that is zero everywhere is also treated as constant
    return bool(np.max(np.abs(vals - vals[0])) <= rtol * scale)


class AdditiveBasis:
    """Constructor of direct-sum (additive-model) basis sets.

    Concatenates the per-dimension bases into a single BasisSet; the resulting
    model is f(x₁, …, x_d) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d).
    """

    @staticmethod
    def build(dim_bases: Dict[int, List[BasisInfo]],
              deduplicate_constants: bool = True) -> BasisSet:
        """
        Build the direct-sum basis set.

        Parameters
        ----------
        dim_bases : Dict[int, List[BasisInfo]]
            Dimension index -> list of bases for that dimension.
        deduplicate_constants : bool, default=True
            Keep only the first constant basis (the constant component shared
            across dimensions), avoiding repeated constant columns that make
            the Gram matrix singular.

        Returns
        -------
        BasisSet
            The direct-sum basis set (per-dimension bases concatenated;
            tensor-product bases may also be passed as elements).
        """
        basis_set = BasisSet()
        seen_constant = False

        for dim in sorted(dim_bases.keys()):
            for basis in dim_bases[dim]:
                if deduplicate_constants and _is_constant(basis):
                    if not seen_constant:
                        seen_constant = True
                        basis_set.add_basis(basis)
                    # a constant basis was already added; skip subsequent duplicates
                else:
                    basis_set.add_basis(basis)

        return basis_set
