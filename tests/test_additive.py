"""Tests for the direct-sum (additive model) basis construction (legacy module)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.basis.factory import BasisFactory
from geomfit.basis.additive import AdditiveBasis
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver


def test_additive_constant_dedup():
    """The direct-sum construction should deduplicate constant bases repeated across dimensions."""
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]

    # 不去重：3 + 3 = 6 个基（其中包含两个相同的常数基 1，分别来自 x 维和 y 维）
    raw = AdditiveBasis.build({0: x_bases, 1: y_bases}, deduplicate_constants=False)
    assert len(raw) == 6

    # 去重后：两个常数基合并为一个，共 6 - 1 = 5 个基
    bs = AdditiveBasis.build({0: x_bases, 1: y_bases})
    assert len(bs) == 5

    # 设计矩阵中应恰好只有一列全为 1（常数基只保留一个）
    data = MultiDimData({0: np.linspace(0, 1, 10), 1: np.linspace(0, 1, 10)})
    Phi = bs.evaluate_all(data)
    # 逐列判断是否所有元素都 ≈ 1（容差 1e-12），统计这样的"全 1 列"的数量
    ones_cols = int(np.sum(np.all(np.abs(Phi - 1.0) < 1e-12, axis=0)))
    assert ones_cols == 1


def test_additive_fits_additive_function():
    """The direct sum can fit the purely additive function f = 1 + 2x - y exactly."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 40)
    y = rng.uniform(0, 1, 40)
    z_true = 1 + 2.0 * x - y
    z = z_true + 0.01 * rng.standard_normal(40)
    data = MultiDimData({0: x, 1: y})

    # 直和构造：每个维度取 0、1 阶多项式，去重后为 {1, x, y}
    bs = AdditiveBasis.build({
        0: [BasisFactory.polynomial(dim=0, order=o) for o in range(2)],
        1: [BasisFactory.polynomial(dim=1, order=o) for o in range(2)],
    })
    solver = FunctionalSolver()
    solver.set_basis(bs)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    solver.solve()

    # 目标函数 1 + 2x - y 是纯加法形式（无交叉项），直和基可以精确表示，拟合误差应≈噪声水平（0.01² 量级）
    mse = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse < 1e-3


if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
    test_additive_constant_dedup()
    print("✓ test_additive_constant_dedup passed")
    test_additive_fits_additive_function()
    print("✓ test_additive_fits_additive_function passed")
    print("\nAll direct-sum basis tests passed!")
