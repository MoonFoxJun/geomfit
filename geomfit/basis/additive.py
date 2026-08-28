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
        # 在三个探针点 (0.0, 1.0, 0.5) 上求值：若函数是常数，这三处取值应几乎相同。
        # 用"相对变化量"判断比绝对阈值更稳健（不受函数本身量级影响）
        vals = np.asarray(
            [basis.evaluate({dim: p}) for p in (0.0, 1.0, 0.5)],
            dtype=float,
        )
    except Exception:
        # 求值失败（例如基在探针点处无定义）时保守地判为"不是常数"
        return False
    scale = float(np.max(np.abs(vals)))
    if scale == 0.0:
        return True  # 处处为 0 的基同样视为常数（它也会产生重复的常数列）
    # 相对变化判据：各点取值与第一点的最大偏差 ≤ rtol × |最大值|，即认为是常数函数
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

        # 直和（additive）模型：f(x₁, …, x_d) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d)，
        # 各维基函数直接相加、没有 x·y 之类的交叉项。它对应张量积空间中的一个
        # 低维子空间：在张量积展开里只保留"其余维度取常数基"的项即退化为此模型，
        # 因此 直和 ⊂ 张量积，表达能力更弱，但基数量随维度线性增长（无维数灾难）
        for dim in sorted(dim_bases.keys()):
            for basis in dim_bases[dim]:
                if deduplicate_constants and _is_constant(basis):
                    # 常数基去重：若每个维度都提供常数基（x⁰, y⁰, …），直和中会出现
                    # 多列全 1 的重复列，Gram 矩阵 ΦᵀΦ 必然奇异、法方程无法求解，
                    # 所以只保留第一个常数基（它已代表所有维度的公共常数分量）
                    if not seen_constant:
                        seen_constant = True
                        basis_set.add_basis(basis)
                    # 已经加入过常数基了，跳过后续维度的重复常数基
                else:
                    basis_set.add_basis(basis)

        return basis_set
