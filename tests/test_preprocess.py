"""Tests for the PCA coordinate rotation preprocessing."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.utils.preprocess import pca_rotate, pca_transform, pca_rotate_back


def test_pca_rotate_decorrelates():
    """Principal components are mutually uncorrelated after PCA rotation."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    # 构造强相关的二维数据：y ≈ x + 小噪声，两个维度几乎完全共线
    y = x + rng.normal(0, 0.02, 200)
    data = MultiDimData({0: x, 1: y})

    # 对数据做 PCA 旋转；info 中保存特征向量、方差贡献率等参数
    rotated, info = pca_rotate(data)
    assert rotated.n_dims == 2

    # 计算旋转后两个主成分之间的相关系数
    R = np.corrcoef(rotated.get_dim(0), rotated.get_dim(1))
    assert abs(R[0, 1]) < 1e-8  # 经 PCA 解耦后，主成分应互不相关（相关系数≈0）

    # 所有主成分的方差贡献率之和应为 1（说明没有截断、保留了全部信息）
    assert np.allclose(np.sum(info["explained_variance_ratio"]), 1.0)
    # 数据高度相关时，第一个主成分应吸收绝大部分方差（占比 > 99%）
    assert info["explained_variance_ratio"][0] > 0.99


def test_pca_rotate_truncate_and_values():
    """Dimensionality reduction via n_components keeps target values aligned with points."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.01, 50)
    data = MultiDimData({0: x, 1: y})
    # 给每个采样点设置目标值 values = 1，用于验证降维后目标值仍与点一一对应
    data.values = np.ones(50)

    # 只保留第一个主成分（n_components=1），把维度从 2 压缩到 1
    rotated, _ = pca_rotate(data, n_components=1)
    assert rotated.n_dims == 1
    # 降维不应破坏目标值与点的对应关系：values 应原样保留
    assert rotated.values is not None
    assert np.allclose(rotated.values, 1.0)


def test_pca_rotate_back_roundtrip():
    """Rotating and rotating back should recover the original coordinates."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.05, 30)
    data = MultiDimData({0: x, 1: y})
    # 先做完整的 PCA 旋转，info 保存旋转矩阵（特征向量）等参数
    rotated, info = pca_rotate(data)

    # 每隔 7 个点抽查一个（索引 0, 7, 14, ...），无需全部验证
    for i in range(0, len(x), 7):
        p = rotated.get_point(i)  # 取出旋转后的坐标
        back = pca_rotate_back(p, info)  # 用保存的 PCA 参数逆旋转回原坐标系
        # 往返变换应还原出原始坐标（容差 1e-10）
        assert np.allclose(np.array([back[0], back[1]]),
                           np.array([x[i], y[i]]), atol=1e-10)


def test_pca_transform_consistent():
    """New data is transformed with PCA information fitted at training time (shared coordinate system)."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.02, 100)
    train = MultiDimData({0: x, 1: y})
    # 在训练数据上拟合 PCA（只保留 1 个主成分），得到参数 info
    _, info = pca_rotate(train, n_components=1)

    x2 = rng.uniform(0, 1, 20)
    y2 = x2 + rng.normal(0, 0.02, 20)
    test = MultiDimData({0: x2, 1: y2})
    # 用训练阶段学到的 PCA 参数变换新数据，保证新数据与训练数据共享同一坐标系
    t2 = pca_transform(test, info)

    assert t2.n_dims == 1
    # 理论近似：当 y ≈ x 时，第一主成分方向约为 (1,1)/√2，其投影值 ≈ (x+y)/√2
    u2 = (x2 + y2) / np.sqrt(2.0)  # 第一主成分方向的近似投影值
    # 新数据的 PCA 投影应与理论第一主成分高度相关（> 0.999）
    assert np.corrcoef(t2.get_dim(0), u2)[0, 1] > 0.999


def test_pca_improves_tensor_conditioning():
    """On a strongly correlated band, PCA-decoupled tensor-product Gram conditioning is far better than in original coordinates."""
    rng = np.random.default_rng(5)
    n = 60
    x = rng.uniform(0.1, 0.9, n)
    y = x + rng.normal(0, 0.02, n)
    z = np.sin(np.pi * (x + y))
    # 强相关带状数据：y ≈ x + 小噪声，两维近共线
    data = MultiDimData({0: x, 1: y})

    # 在原坐标系下构造傅里叶张量积基：每个维度取频率 0~2、周期 L=1
    xb = []
    for f in range(3):
        xb.extend(BasisFactory.fourier(dim=0, freq=f, L=1.0))
    yb = []
    for f in range(3):
        yb.extend(BasisFactory.fourier(dim=1, freq=f, L=1.0))
    raw = BasisSet()
    for b in BasisFactory.tensor_product({0: xb, 1: yb}):
        raw.add_basis(b)
    # 在原坐标系下计算 Gram 矩阵及其条件数
    G_raw = InnerProduct(is_continuous=False).compute_gram_matrix(
        raw.evaluate_all(data), data)
    cond_raw = np.linalg.cond(G_raw)

    # 解耦后只保留一个主成分（丢掉近零方差方向），在旋转后的坐标上构造一维傅里叶基
    rot, _ = pca_rotate(data, n_components=1)
    pca = BasisSet()
    for f in range(3):
        # 旋转后主成分坐标 u = (x+y)/√2 的取值范围变为 [0, √2]，故周期参数取 L = √2
        for b in BasisFactory.fourier(dim=0, freq=f, L=np.sqrt(2.0)):
            pca.add_basis(b)
    G_pca = InnerProduct(is_continuous=False).compute_gram_matrix(
        pca.evaluate_all(rot), rot)
    cond_pca = np.linalg.cond(G_pca)

    assert cond_pca < 1e3          # 解耦后 Gram 矩阵良态（条件数 < 1e3）
    assert cond_raw > 1e6 * cond_pca   # 原坐标系下两维近共线，条件数应比 PCA 后大 1e6 倍以上（病态）


if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
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
    print("\nAll PCA preprocessing tests passed!")
