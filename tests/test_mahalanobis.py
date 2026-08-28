"""测试马氏距离核"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.solver.functional_solver import FunctionalSolver
from functional_solver.kernel.mahalanobis import MahalanobisKernel
from functional_solver.kernel.rbf import RBFKernel


def test_mahalanobis_equals_rbf_for_isotropic_metric():
    """metric = I 时马氏核退化为长度尺度 1 的普通 RBF"""
    k_m = MahalanobisKernel(sigma=2.0, metric=np.eye(1))
    k_r = RBFKernel(sigma=2.0, length_scale=1.0)
    assert np.allclose(k_m({0: 0.3}, {0: 0.7}), k_r({0: 0.3}, {0: 0.7}))

    k_m2 = MahalanobisKernel(sigma=1.5, metric=np.eye(2))
    k_r2 = RBFKernel(sigma=1.5, length_scale=1.0)
    assert np.allclose(
        k_m2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}),
        k_r2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}))


def test_mahalanobis_matrix_symmetry_psd():
    """核矩阵对称且（数值上）半正定"""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    kern = MahalanobisKernel(sigma=1.0)
    K = kern.compute_matrix(data)

    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) > -1e-8)


def test_mahalanobis_fits_correlated_band():
    """马氏核 + 岭正则能在强相关窄带上精确插值（误差≈噪声水平）"""
    rng = np.random.default_rng(4)
    n = 60
    x = rng.uniform(0.1, 0.9, n)
    y = x + rng.normal(0, 0.02, n)
    z_true = np.sin(np.pi * (x + y))
    z = z_true + 0.02 * rng.standard_normal(n)
    data = MultiDimData({0: x, 1: y})

    solver = FunctionalSolver()
    solver.set_kernel(MahalanobisKernel(sigma=1.0))
    solver.load_data(data, z)
    solver.solve(regularization={"alpha": 1e-6})
    mse = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse < 1e-2


def test_mahalanobis_auto_metric_from_data():
    """metric=None 时首次 compute_matrix 自动从数据估计度量"""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    kern = MahalanobisKernel(sigma=1.0)
    assert kern._effective_metric is None
    K = kern.compute_matrix(data)
    assert kern._effective_metric is not None
    assert K.shape == (30, 30)
    # 自动估计的度量应反映数据相关性（沿次轴方向权重更大）
    M = kern._effective_metric
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)   # 次轴方向（y-x）
    assert v @ M @ v > M[0, 0]                  # 次轴方向距离权重更大


if __name__ == "__main__":
    test_mahalanobis_equals_rbf_for_isotropic_metric()
    print("✓ test_mahalanobis_equals_rbf_for_isotropic_metric passed")
    test_mahalanobis_matrix_symmetry_psd()
    print("✓ test_mahalanobis_matrix_symmetry_psd passed")
    test_mahalanobis_fits_correlated_band()
    print("✓ test_mahalanobis_fits_correlated_band passed")
    test_mahalanobis_auto_metric_from_data()
    print("✓ test_mahalanobis_auto_metric_from_data passed")
    print("\n所有马氏核测试通过！")
