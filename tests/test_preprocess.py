"""测试 PCA 坐标旋转预处理"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory
from functional_solver.inner_product.base import InnerProduct
from functional_solver.utils.preprocess import pca_rotate, pca_transform, pca_rotate_back


def test_pca_rotate_decorrelates():
    """PCA 旋转后主成分彼此不相关"""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = x + rng.normal(0, 0.02, 200)
    data = MultiDimData({0: x, 1: y})

    rotated, info = pca_rotate(data)
    assert rotated.n_dims == 2

    R = np.corrcoef(rotated.get_dim(0), rotated.get_dim(1))
    assert abs(R[0, 1]) < 1e-8

    assert np.allclose(np.sum(info["explained_variance_ratio"]), 1.0)
    # 主轴方差占比应远大于次轴（数据强相关）
    assert info["explained_variance_ratio"][0] > 0.99


def test_pca_rotate_truncate_and_values():
    """n_components 降维 + 目标值跟随数据点"""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.01, 50)
    data = MultiDimData({0: x, 1: y})
    data.values = np.ones(50)

    rotated, _ = pca_rotate(data, n_components=1)
    assert rotated.n_dims == 1
    assert rotated.values is not None
    assert np.allclose(rotated.values, 1.0)


def test_pca_rotate_back_roundtrip():
    """旋转再还原应回到原始坐标"""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.05, 30)
    data = MultiDimData({0: x, 1: y})
    rotated, info = pca_rotate(data)

    for i in range(0, len(x), 7):
        p = rotated.get_point(i)
        back = pca_rotate_back(p, info)
        assert np.allclose(np.array([back[0], back[1]]),
                           np.array([x[i], y[i]]), atol=1e-10)


def test_pca_transform_consistent():
    """新数据用训练时拟合的 PCA 信息变换（同一坐标系）"""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.02, 100)
    train = MultiDimData({0: x, 1: y})
    _, info = pca_rotate(train, n_components=1)

    x2 = rng.uniform(0, 1, 20)
    y2 = x2 + rng.normal(0, 0.02, 20)
    test = MultiDimData({0: x2, 1: y2})
    t2 = pca_transform(test, info)

    assert t2.n_dims == 1
    u2 = (x2 + y2) / np.sqrt(2.0)  # 近似主轴
    assert np.corrcoef(t2.get_dim(0), u2)[0, 1] > 0.999


def test_pca_improves_tensor_conditioning():
    """强相关窄带上，PCA 解耦后张量积的 Gram 条件数远小于原始坐标"""
    rng = np.random.default_rng(5)
    n = 60
    x = rng.uniform(0.1, 0.9, n)
    y = x + rng.normal(0, 0.02, n)
    z = np.sin(np.pi * (x + y))
    data = MultiDimData({0: x, 1: y})

    # 原始坐标张量积傅里叶（频率 0..2, L=1）
    xb = []
    for f in range(3):
        xb.extend(BasisFactory.fourier(dim=0, freq=f, L=1.0))
    yb = []
    for f in range(3):
        yb.extend(BasisFactory.fourier(dim=1, freq=f, L=1.0))
    raw = BasisSet()
    for b in BasisFactory.tensor_product({0: xb, 1: yb}):
        raw.add_basis(b)
    G_raw = InnerProduct(is_continuous=False).compute_gram_matrix(
        raw.evaluate_all(data), data)
    cond_raw = np.linalg.cond(G_raw)

    # PCA 解耦后（丢弃近零方差方向）的一维傅里叶基
    rot, _ = pca_rotate(data, n_components=1)
    pca = BasisSet()
    for f in range(3):
        for b in BasisFactory.fourier(dim=0, freq=f, L=np.sqrt(2.0)):
            pca.add_basis(b)
    G_pca = InnerProduct(is_continuous=False).compute_gram_matrix(
        pca.evaluate_all(rot), rot)
    cond_pca = np.linalg.cond(G_pca)

    assert cond_pca < 1e3          # PCA 后良态
    assert cond_raw > 1e6 * cond_pca   # 原始坐标病态（近共线）


if __name__ == "__main__":
    test_pca_rotate_decorrelates()
    print("✓ test_pca_rotate_decorrelates passed")
    test_pca_rotate_truncate_and_values()
    print("✓ test_pca_rotate_truncate_and_values passed")
    test_pca_rotate_back_roundtrip()
    print("✓ test_pca_rotate_back_roundtrip passed")
    test_pca_transform_consistent()
    print("✓ test_pca_transform_consistent passed")
    test_pca_improves_tensor_conditioning()
    print("✓ test_pca_improves_tensor_conditioning passed")
    print("\n所有 PCA 预处理测试通过！")
