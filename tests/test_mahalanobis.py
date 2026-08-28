"""Tests for the Mahalanobis distance kernel."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.solver.functional_solver import FunctionalSolver
from geomfit.kernel.mahalanobis import MahalanobisKernel
from geomfit.kernel.rbf import RBFKernel


def test_mahalanobis_equals_rbf_for_isotropic_metric():
    """With metric = I, the Mahalanobis kernel reduces to a plain RBF with length scale 1."""
    # 度量矩阵取单位阵 I：马氏距离退化为普通的欧氏距离
    k_m = MahalanobisKernel(sigma=2.0, metric=np.eye(1))
    # 对照用的普通 RBF 核：长度尺度 length_scale = 1
    k_r = RBFKernel(sigma=2.0, length_scale=1.0)
    # 当度量矩阵 metric = I 时，马氏核应等价于 RBF 核（1 维情形）
    assert np.allclose(k_m({0: 0.3}, {0: 0.7}), k_r({0: 0.3}, {0: 0.7}))

    # 二维情形同理：单位度量下马氏核应完全等价于 RBF 核
    k_m2 = MahalanobisKernel(sigma=1.5, metric=np.eye(2))
    k_r2 = RBFKernel(sigma=1.5, length_scale=1.0)
    assert np.allclose(
        k_m2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}),
        k_r2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}))


def test_mahalanobis_matrix_symmetry_psd():
    """The kernel matrix is symmetric and numerically positive semi-definite."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 30)
    # 构造强相关的二维数据：y ≈ x + 小噪声
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    # 不显式指定度量矩阵，之后会从数据自动估计
    kern = MahalanobisKernel(sigma=1.0)
    # 计算核矩阵 K：K[i,j] = k(x_i, x_j)
    K = kern.compute_matrix(data)

    # 核函数对称，核矩阵应满足 K = Kᵀ
    assert np.allclose(K, K.T)
    # 核矩阵应数值上半正定：最小特征值 > -1e-8（该容差内的微小负值可视为数值误差）
    assert np.all(np.linalg.eigvalsh(K) > -1e-8)


def test_mahalanobis_fits_correlated_band():
    """The Mahalanobis kernel with ridge regularization interpolates accurately on a strongly correlated band (error ≈ noise level)."""
    rng = np.random.default_rng(4)
    n = 60
    x = rng.uniform(0.1, 0.9, n)
    y = x + rng.normal(0, 0.02, n)
    z_true = np.sin(np.pi * (x + y))
    z = z_true + 0.02 * rng.standard_normal(n)
    # 强相关带状数据（y ≈ x + 0.02 噪声），目标值 z = sin(π(x+y)) + 小噪声
    data = MultiDimData({0: x, 1: y})

    solver = FunctionalSolver()
    solver.set_kernel(MahalanobisKernel(sigma=1.0))
    solver.load_data(data, z)
    # 用岭正则化（α = 1e-6）求解：核矩阵接近奇异时保证数值稳定
    solver.solve(regularization={"alpha": 1e-6})
    # 核岭回归插值误差应≈噪声水平（0.02² 量级）
    mse = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse < 1e-2


def test_mahalanobis_auto_metric_from_data():
    """With metric=None, the first compute_matrix call estimates the metric automatically from the data."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    kern = MahalanobisKernel(sigma=1.0)
    # 初始状态下度量尚未估计，_effective_metric 应为 None
    assert kern._effective_metric is None
    # 第一次调用 compute_matrix 会触发从数据自动估计度量矩阵
    K = kern.compute_matrix(data)
    # 估计完成后，内部保存了度量矩阵
    assert kern._effective_metric is not None
    assert K.shape == (30, 30)
    # 自动估计的度量应反映数据相关性：沿次轴（方差小的方向）的距离被赋予更大权重
    M = kern._effective_metric
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)   # 次轴方向：y - x（该方向上方差最小，接近噪声尺度）
    assert v @ M @ v > M[0, 0]                  # 沿次轴的马氏距离被放大：vᵀMv > M[0,0]，即度量在次轴方向权重更大


if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
    test_mahalanobis_equals_rbf_for_isotropic_metric()
    print("✓ test_mahalanobis_equals_rbf_for_isotropic_metric passed")
    test_mahalanobis_matrix_symmetry_psd()
    print("✓ test_mahalanobis_matrix_symmetry_psd passed")
    test_mahalanobis_fits_correlated_band()
    print("✓ test_mahalanobis_fits_correlated_band passed")
    test_mahalanobis_auto_metric_from_data()
    print("✓ test_mahalanobis_auto_metric_from_data passed")
    print("\nAll Mahalanobis kernel tests passed!")
