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
    k_m = MahalanobisKernel(sigma=2.0, metric=np.eye(1))
    k_r = RBFKernel(sigma=2.0, length_scale=1.0)
    assert np.allclose(k_m({0: 0.3}, {0: 0.7}), k_r({0: 0.3}, {0: 0.7}))

    k_m2 = MahalanobisKernel(sigma=1.5, metric=np.eye(2))
    k_r2 = RBFKernel(sigma=1.5, length_scale=1.0)
    assert np.allclose(
        k_m2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}),
        k_r2({0: 0.3, 1: 0.6}, {0: 0.7, 1: 0.2}))


def test_mahalanobis_matrix_symmetry_psd():
    """The kernel matrix is symmetric and numerically positive semi-definite."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    kern = MahalanobisKernel(sigma=1.0)
    K = kern.compute_matrix(data)

    assert np.allclose(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) > -1e-8)


def test_mahalanobis_fits_correlated_band():
    """The Mahalanobis kernel with ridge regularization interpolates accurately on a strongly correlated band (error ≈ noise level)."""
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
    """With metric=None, the first compute_matrix call estimates the metric automatically from the data."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.02, 30)
    data = MultiDimData({0: x, 1: y})

    kern = MahalanobisKernel(sigma=1.0)
    assert kern._effective_metric is None
    K = kern.compute_matrix(data)
    assert kern._effective_metric is not None
    assert K.shape == (30, 30)
    # The auto-estimated metric should reflect data correlation (larger weight along the minor axis)
    M = kern._effective_metric
    v = np.array([1.0, -1.0]) / np.sqrt(2.0)   # minor-axis direction (y-x)
    assert v @ M @ v > M[0, 0]                  # distances along the minor axis are weighted more heavily


if __name__ == "__main__":
    test_mahalanobis_equals_rbf_for_isotropic_metric()
    print("✓ test_mahalanobis_equals_rbf_for_isotropic_metric passed")
    test_mahalanobis_matrix_symmetry_psd()
    print("✓ test_mahalanobis_matrix_symmetry_psd passed")
    test_mahalanobis_fits_correlated_band()
    print("✓ test_mahalanobis_fits_correlated_band passed")
    test_mahalanobis_auto_metric_from_data()
    print("✓ test_mahalanobis_auto_metric_from_data passed")
    print("\nAll Mahalanobis kernel tests passed!")
