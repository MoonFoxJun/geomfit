"""Tests for tensor-product bases and multi-dimensional fitting."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver


def test_tensor_product_evaluation():
    """The value of a tensor-product basis equals the product of its factor values in each dimension."""
    x_bases = [BasisFactory.polynomial(dim=0, order=1)]   # 因子基：X₁(x) = x（维度 0 上的一次多项式）
    y_bases = [BasisFactory.polynomial(dim=1, order=2)]   # 因子基：Y₂(y) = y²（维度 1 上的二次多项式）
    products = BasisFactory.tensor_product({0: x_bases, 1: y_bases})
    assert len(products) == 1

    # 张量积基函数 φ(x, y) = X₁(x)·Y₂(y) = x·y²
    phi = products[0]
    point = {0: 3.0, 1: 4.0}
    # 验证张量积基的值 = 各因子乘积：φ(3, 4) = X₁(3)·Y₂(4) = 3·4² = 48
    assert phi.evaluate(point) == 3.0 * 16.0 == 48.0

    # 张量积基应登记所有涉及的维度（0 和 1），供按维度组织的功能使用
    assert set(phi.dims) == {0, 1}


def test_tensor_product_count():
    """The number of tensor-product bases equals the product of per-dimension basis counts."""
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(2)]
    z_bases = [BasisFactory.polynomial(dim=2, order=o) for o in range(2)]
    # 张量积的基函数总数 = 各维度基函数个数之积 = 3×2×2 = 12
    products = BasisFactory.tensor_product({0: x_bases, 1: y_bases, 2: z_bases})
    assert len(products) == 3 * 2 * 2 == 12


def test_tensor_product_fits_interaction():
    """A tensor-product basis fits the interaction function f(x,y)=1+2x-y+3xy exactly, while an additive model cannot."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z_true = 1.0 + 2.0 * x - y + 3.0 * x * y   # 目标函数含交叉项 x·y（交互项），张量积能表达而加法模型表达不了
    z = z_true + 0.01 * rng.standard_normal(n)
    data = MultiDimData({0: x, 1: y})

    # 张量积基：两个维度各取 0~2 阶，共 3×3 = 9 个基函数（x⁰..² ⊗ y⁰..²）
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
    basis_set = BasisSet()
    for b in BasisFactory.tensor_product({0: x_bases, 1: y_bases}):
        basis_set.add_basis(b)

    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    solver.solve()
    # 9 个张量积基恰好覆盖 1, x, y, x², xy, y² 等全部单项式，可精确拟合该交互函数
    mse_tensor = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse_tensor < 1e-3  # 拟合误差应接近噪声水平（0.01² 量级），故要求 MSE < 1e-3

    # 加法模型（直和）：只有 1, x, x², y, y² 这 5 个基（其中常数基 1 重复出现）
    add_set = BasisSet()
    for o in range(3):
        add_set.add_basis(BasisFactory.polynomial(dim=0, order=o))
        add_set.add_basis(BasisFactory.polynomial(dim=1, order=o))
    solver_add = FunctionalSolver()
    solver_add.set_basis(add_set)
    solver_add.set_inner_product(InnerProduct(is_continuous=False))
    solver_add.load_data(data, z)
    solver_add.solve()
    mse_add = np.mean((solver_add.predict(data) - z_true) ** 2)
    # 加法模型在结构上无法表示 x·y 交叉项，因此拟合误差应显著大于张量积（至少 10 倍）
    assert mse_add > 10 * mse_tensor


def test_continuous_multidim_gram():
    """Multi-dimensional continuous inner product (tensor-product grid): ∫∫1 = 1, ∫∫x²y² = 1/9, Gram matrix symmetric."""
    n = 81
    t = np.linspace(0, 1, n)
    # 在 [0,1]² 上构造 81×81 的张量积网格，摊平后作为二维采样点（用于数值积分）
    xv, yv = np.meshgrid(t, t)
    data = MultiDimData({0: xv.ravel(), 1: yv.ravel()})
    inner = InnerProduct(is_continuous=True)

    # 常数函数 f ≡ 1 在所有采样点上的取值
    ones = np.ones(data.n_points)
    # 连续内积 ⟨1, 1⟩ = ∫∫₁ 1 dx dy = 1（即单位正方形的面积）
    assert abs(inner(ones, ones, data) - 1.0) < 1e-3

    # 被积函数 f(x, y) = x²y²
    xy2 = xv.ravel() ** 2 * yv.ravel() ** 2
    # ⟨x²y², 1⟩ = ∫₀¹x²dx · ∫₀¹y²dy = (1/3)·(1/3) = 1/9（二维积分可分离变量）
    assert abs(inner(xy2, ones, data) - 1.0 / 9.0) < 1e-3

    # 设计矩阵：第 1 列为常数 1，第 2 列为 x 的取值
    Phi = np.column_stack([ones, xv.ravel()])
    # 连续内积下计算 Gram 矩阵，验证其对称性
    G = inner.compute_gram_matrix(Phi, data)
    assert np.allclose(G, G.T)


if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
    test_tensor_product_evaluation()
    print("✓ test_tensor_product_evaluation passed")
    test_tensor_product_count()
    print("✓ test_tensor_product_count passed")
    test_tensor_product_fits_interaction()
    print("✓ test_tensor_product_fits_interaction passed")
    test_continuous_multidim_gram()
    print("✓ test_continuous_multidim_gram passed")
    print("\nAll tensor-product tests passed!")
