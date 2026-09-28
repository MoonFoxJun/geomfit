"""Tests for the separable (grid) solver."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.core.grid import detect_grid
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.inner_product.weight import WeightFunction
from geomfit.solver.gram_solver import GramSolver
from geomfit.solver.grid_solver import GridSolver


def _tensor_basis(orders_per_dim):
    """Build a polynomial tensor-product basis: orders_per_dim[d] orders on dim d."""
    per_dim = {}
    for dim, n_orders in enumerate(orders_per_dim):
        per_dim[dim] = [BasisFactory.polynomial(dim=dim, order=o) for o in range(n_orders)]
    basis_set = BasisSet()
    for b in BasisFactory.tensor_product(per_dim):
        basis_set.add_basis(b)
    return basis_set


def _grid_data(axes_per_dim):
    """Return MultiDimData on the full Cartesian grid (C-order flattened)."""
    meshes = np.meshgrid(*axes_per_dim, indexing='ij')
    return MultiDimData({d: meshes[d].ravel() for d in range(len(axes_per_dim))}), meshes


def test_detect_grid_on_grid_and_scattered():
    """detect_grid 能区分完整网格与散点"""
    axes = [np.linspace(0, 1, 7), np.linspace(0, 1, 5)]
    data, _ = _grid_data(axes)
    info = detect_grid(data)
    assert info is not None
    assert info["shape"] == (7, 5)
    assert len(info["flat_index"]) == 35

    # 散点数据：不是完整笛卡尔积，应返回 None
    rng = np.random.default_rng(0)
    scattered = MultiDimData({0: rng.uniform(0, 1, 30), 1: rng.uniform(0, 1, 30)})
    assert detect_grid(scattered) is None

    # 网格但有点重复（点数不等于轴长乘积）也应判定为非网格
    dup = MultiDimData({0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 1.0, 1.0])})
    assert detect_grid(dup) is None


def test_separable_matches_joint_2d():
    """二维网格上，三明治解与联合 Gram 解给出相同系数与预测"""
    orders = [3, 3]  # 每维 0..2 阶 → P = 9
    basis_set = _tensor_basis(orders)
    axes = [np.linspace(0, 1, 11), np.linspace(0, 1, 11)]
    data, meshes = _grid_data(axes)

    X, Y = meshes
    z_true = 1.0 + 2.0 * X - Y + 3.0 * X * Y + X ** 2 * Y
    target = z_true.ravel()

    inner = InnerProduct(is_continuous=False)
    joint = GramSolver(basis_set, inner)
    joint.load_data(data, target)
    c_joint = joint.solve()

    sep = GridSolver(basis_set, inner)
    sep.load_data(data, target)
    c_sep = sep.solve()

    assert sep.is_separable()
    assert len(sep.get_condition_numbers()) == 2      # 每一维一个条件数
    assert np.allclose(c_sep, c_joint, atol=1e-8, rtol=1e-6)

    # 预测一致，并且能还原真函数
    grid_new = MultiDimData({0: np.array([0.2, 0.7]), 1: np.array([0.3, 0.9])})
    pred_sep = sep.predict(grid_new)
    pred_joint = joint.predict(grid_new)
    assert np.allclose(pred_sep, pred_joint, atol=1e-8)
    z_expected = 1.0 + 2.0 * 0.2 - 0.3 + 3.0 * 0.2 * 0.3 + 0.2 ** 2 * 0.3
    assert abs(pred_sep[0] - z_expected) < 1e-8


def test_separable_matches_joint_3d():
    """三维网格上同样一致（P = 27 vs 三个 3×3 小求解）"""
    orders = [3, 3, 3]
    basis_set = _tensor_basis(orders)
    axes = [np.linspace(0, 1, 6)] * 3
    data, meshes = _grid_data(axes)
    X, Y, Z = meshes
    target = (1.0 + X + 2.0 * Y - Z + X * Z).ravel()

    inner = InnerProduct(is_continuous=False)
    joint = GramSolver(basis_set, inner)
    joint.load_data(data, target)
    c_joint = joint.solve()

    sep = GridSolver(basis_set, inner)
    sep.load_data(data, target)
    c_sep = sep.solve()

    assert sep.is_separable()
    assert len(sep.get_condition_numbers()) == 3
    assert np.allclose(c_sep, c_joint, atol=1e-8, rtol=1e-6)


def test_separable_with_continuous_inner_product():
    """连续内积（梯形体积元）下，三明治的逐维权重与联合路径一致"""
    orders = [3, 3]
    basis_set = _tensor_basis(orders)
    axes = [np.linspace(0, 1, 9), np.linspace(0, 1, 9)]
    data, meshes = _grid_data(axes)
    X, Y = meshes
    target = (X ** 2 + Y ** 2 + X * Y).ravel()

    inner = InnerProduct(is_continuous=True)
    joint = GramSolver(basis_set, inner)
    joint.load_data(data, target)
    c_joint = joint.solve()

    sep = GridSolver(basis_set, inner)
    sep.load_data(data, target)
    c_sep = sep.solve()

    assert sep.is_separable()
    assert np.allclose(c_sep, c_joint, atol=1e-8, rtol=1e-6)


def test_fallback_on_scattered_data():
    """散点数据自动回落到联合求解，结果与 GramSolver 完全一致"""
    rng = np.random.default_rng(42)
    basis_set = _tensor_basis([3, 3])
    data = MultiDimData({0: rng.uniform(0, 1, 40), 1: rng.uniform(0, 1, 40)})
    target = rng.standard_normal(40)

    inner = InnerProduct(is_continuous=False)
    joint = GramSolver(basis_set, inner)
    joint.load_data(data, target)
    c_joint = joint.solve()

    sep = GridSolver(basis_set, inner)
    sep.load_data(data, target)
    c_sep = sep.solve()

    assert not sep.is_separable()
    assert "Cartesian grid" in sep.debug_info["fallback_reason"]
    assert np.allclose(c_sep, c_joint, atol=1e-10)
    assert len(sep.get_condition_numbers()) == 1  # 回落路径只有一个联合条件数


def test_fallback_on_additive_basis():
    """直和（加法）基不是笛卡尔张量积结构 → 自动回落"""
    rng = np.random.default_rng(7)
    basis_set = BasisSet()
    for o in range(3):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=o))
        basis_set.add_basis(BasisFactory.polynomial(dim=1, order=o))
    axes = [np.linspace(0, 1, 8), np.linspace(0, 1, 8)]
    data, meshes = _grid_data(axes)
    target = (meshes[0] + meshes[1]).ravel()

    sep = GridSolver(basis_set, InnerProduct(is_continuous=False))
    sep.load_data(data, target)
    sep.solve()
    assert not sep.is_separable()
    assert "tensor-product structure" in sep.debug_info["fallback_reason"]


def test_fallback_on_weighted_inner_product():
    """带权重函数的内积不可分离 → 自动回落"""
    basis_set = _tensor_basis([3, 3])
    axes = [np.linspace(0.1, 1.0, 8), np.linspace(0.1, 1.0, 8)]
    data, meshes = _grid_data(axes)
    target = (meshes[0] * meshes[1]).ravel()

    # 多维权重函数会被以"点字典"的形式逐点调用，因此这里用库自带的 WeightFunction
    # （其 polynomial 分支支持字典输入），而不是裸的 lambda
    inner = InnerProduct(weight_func=WeightFunction.polynomial(degree=1), is_continuous=False)
    sep = GridSolver(basis_set, inner)
    sep.load_data(data, target)
    sep.solve()
    assert not sep.is_separable()
    assert "weighted inner product" in sep.debug_info["fallback_reason"]


def test_separable_with_tikhonov_alpha():
    """分离路径支持逐维 Tikhonov：α 越大系数范数越小"""
    basis_set = _tensor_basis([4, 4])
    axes = [np.linspace(0, 1, 9), np.linspace(0, 1, 9)]
    data, meshes = _grid_data(axes)
    target = (np.sin(2 * np.pi * meshes[0]) * np.cos(2 * np.pi * meshes[1])).ravel()

    sep = GridSolver(basis_set, InnerProduct(is_continuous=False))
    sep.load_data(data, target)
    c_small = sep.solve(regularization={"method": "tikhonov", "alpha": 1e-8})
    norm_small = np.linalg.norm(c_small)
    c_large = sep.solve(regularization={"method": "tikhonov", "alpha": 1.0})
    norm_large = np.linalg.norm(c_large)

    assert sep.is_separable()
    assert norm_large < norm_small


def test_separable_requires_full_grid_point_ordering():
    """打乱数据点顺序后，三明治结果不变（多重指标散射保证顺序无关）"""
    basis_set = _tensor_basis([3, 3])
    axes = [np.linspace(0, 1, 8), np.linspace(0, 1, 8)]
    data, meshes = _grid_data(axes)
    target = (meshes[0] ** 2 * meshes[1]).ravel()

    sep = GridSolver(basis_set, InnerProduct(is_continuous=False))
    sep.load_data(data, target)
    c_ordered = sep.solve()

    rng = np.random.default_rng(3)
    perm = rng.permutation(len(target))
    shuffled = MultiDimData({0: data.get_dim(0)[perm], 1: data.get_dim(1)[perm]})
    sep2 = GridSolver(basis_set, InnerProduct(is_continuous=False))
    sep2.load_data(shuffled, target[perm])
    c_shuffled = sep2.solve()

    assert sep2.is_separable()
    assert np.allclose(c_ordered, c_shuffled, atol=1e-10)


if __name__ == "__main__":
    test_detect_grid_on_grid_and_scattered()
    print("✓ test_detect_grid_on_grid_and_scattered passed")
    test_separable_matches_joint_2d()
    print("✓ test_separable_matches_joint_2d passed")
    test_separable_matches_joint_3d()
    print("✓ test_separable_matches_joint_3d passed")
    test_separable_with_continuous_inner_product()
    print("✓ test_separable_with_continuous_inner_product passed")
    test_fallback_on_scattered_data()
    print("✓ test_fallback_on_scattered_data passed")
    test_fallback_on_additive_basis()
    print("✓ test_fallback_on_additive_basis passed")
    test_fallback_on_weighted_inner_product()
    print("✓ test_fallback_on_weighted_inner_product passed")
    test_separable_with_tikhonov_alpha()
    print("✓ test_separable_with_tikhonov_alpha passed")
    test_separable_requires_full_grid_point_ordering()
    print("✓ test_separable_requires_full_grid_point_ordering passed")
    print("\n所有网格求解器测试通过！")
