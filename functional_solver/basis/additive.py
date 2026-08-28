"""直和（加法模型）基构造 —— 独立保留的实现。

数学结构
--------
直和空间 V = V₀ ⊕ V₁ ⊕ … 中的函数具有“加法”形式：

    f(x₁, …, x_d) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d)

即每个维度上的基函数贡献直接相加，不产生任何交叉项（如 x·y）。
这是与张量积基（f = Σ cᵢⱼ Xᵢ(x)Yⱼ(y)）相对的另一条路线。

已知的数学问题（本模块仍保留该算法，但请知悉其局限）
----------------------------------------------------
1. 表达力受限：无法表示维度间的交互/乘积结构（如 x·y、sin(2πx)cos(2πy)），
   完备性差、信息损失率高，对一般的多维函数逼近结果不佳。
2. 常数重叠：若每个维度都提供常数基（x⁰、y⁰ …），直和中会重复出现常数
   分量，使 Gram 矩阵奇异。本构造默认只保留第一个常数基（去重）。
3. 当各维度贡献确实独立时，直和是合适的模型（可解释性也更好）。

与张量积的关系
-------------
直和空间是张量积空间的一个低维子空间：取张量积基中“其他维度恒为常数因子”
的那一部分即退化为直和。因此张量积基 ⊇ 直和基，表达能力更强。

保留理由 / 适用场景
-------------------
- 高维数据：张量积基数量随维数指数增长（维数灾难），直和只按线性增长，
  是高维下唯一可行的显式基路线。
- 各维度独立贡献的领域问题（此时直和即是真实结构，无需交互项）。

用法
----
    from functional_solver.basis.additive import AdditiveBasis

    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
    basis_set = AdditiveBasis.build({0: x_bases, 1: y_bases})
    # 等价于把各维基函数直接拼接进同一个 BasisSet（但会去重重复的常数基）
"""

from typing import Dict, List
import numpy as np

from ..core.basis_container import BasisInfo, BasisSet


def _is_constant(basis: BasisInfo, rtol: float = 1e-12) -> bool:
    """判断单维度基函数是否为常数函数（值不随坐标变化）。

    在三个探针点上求值，若相对变化可忽略则视为常数。
    张量积基（多维度）无法用单维度探针判断，返回 False。
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
        return True  # 处处为零的基也算“常数型”
    return bool(np.max(np.abs(vals - vals[0])) <= rtol * scale)


class AdditiveBasis:
    """直和（加法模型）基集合构造器。

    把每个维度的基函数拼接成同一个 BasisSet，函数形式为
        f(x₁, …, x_d) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d)
    """

    @staticmethod
    def build(dim_bases: Dict[int, List[BasisInfo]],
              deduplicate_constants: bool = True) -> BasisSet:
        """
        构造直和基集合。

        参数
        ----
        dim_bases : Dict[int, List[BasisInfo]]
            维度索引 -> 该维度的基函数列表
        deduplicate_constants : bool, 默认=True
            是否只保留第一个常数基（跨维度共享的常数分量），
            避免重复的常数列导致 Gram 矩阵奇异

        返回
        ----
        BasisSet
            直和基集合（各维基函数拼接；张量积基也可作为其中的元素传入）
        """
        basis_set = BasisSet()
        seen_constant = False

        for dim in sorted(dim_bases.keys()):
            for basis in dim_bases[dim]:
                if deduplicate_constants and _is_constant(basis):
                    if not seen_constant:
                        seen_constant = True
                        basis_set.add_basis(basis)
                    # 已见过常数基：跳过后续的重复常数
                else:
                    basis_set.add_basis(basis)

        return basis_set
