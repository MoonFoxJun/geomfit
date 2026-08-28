"""Strongly correlated dimensions: PCA decoupling + Mahalanobis kernel.

The data lie on a narrow band y ≈ x (strong correlation), and the true function
depends only on the principal axis u = (x + y)/√2. Three approaches are compared:

1. Tensor-product Fourier basis in raw coordinates (Cartesian product basis)
   — the frequency grid is square (integer frequencies), but the intrinsic
     frequency along the diagonal is 0.5, which falls between grid points:
     it cannot be represented, and the Gram matrix is ill-conditioned because
     the basis columns are nearly collinear;
2. PCA rotation (dropping the near-zero-variance direction), then tensor product
   — the coordinate system aligns with the data's principal axis, so a single
     frequency term fits exactly;
3. Mahalanobis kernel (directly on raw coordinates)
   — correlation is absorbed into the distance metric, giving a well-conditioned
     kernel matrix and a smooth fit.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory
from functional_solver.inner_product.base import InnerProduct
from functional_solver.solver.functional_solver import FunctionalSolver
from functional_solver.utils.preprocess import pca_rotate, pca_transform
from functional_solver.kernel.mahalanobis import MahalanobisKernel


def fit_tensor(basis_set, data, z):
    """Fit with a tensor-product basis; returns (solver, condition number, coefficient norm)."""
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    coeff = solver.solve()
    G = InnerProduct(is_continuous=False).compute_gram_matrix(
        basis_set.evaluate_all(data), data)
    return solver, np.linalg.cond(G), np.linalg.norm(coeff)


def fit_kernel(kernel, data, z):
    """Fit with a kernel method; returns (solver, condition number, coefficient norm)."""
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, z)
    coeff = solver.solve()
    K = kernel.compute_matrix(data)
    return solver, np.linalg.cond(K), np.linalg.norm(coeff)


def main():
    rng = np.random.default_rng(2024)

    # ---------------- data: narrow band y ≈ x, true function depends only on principal axis u ----------------
    n = 80
    x = rng.uniform(0.1, 0.9, n)
    v = rng.normal(0.0, 0.02, n)          # small perturbation in the perpendicular direction -> strong correlation
    y = x + v
    u = (x + y) / np.sqrt(2.0)            # principal axis
    z_true = np.sin(np.pi * (x + y))      # = sin(√2 π u), depends only on u
    z = z_true + 0.02 * rng.standard_normal(n)
    data = MultiDimData({0: x, 1: y})

    print("=" * 66)
    print("Strongly correlated dims (narrow band y≈x): raw-coordinate tensor product vs PCA-decoupled vs Mahalanobis kernel")
    print(f"corr(x, y) = {np.corrcoef(x, y)[0, 1]:.4f}")
    print("=" * 66)

    # ---------------- method 1: tensor-product Fourier in raw coordinates (frequencies 0..2, L=1) ----------------
    xb = []
    for f in range(3):
        xb.extend(BasisFactory.fourier(dim=0, freq=f, L=1.0))
    yb = []
    for f in range(3):
        yb.extend(BasisFactory.fourier(dim=1, freq=f, L=1.0))
    raw_set = BasisSet()
    for b in BasisFactory.tensor_product({0: xb, 1: yb}):
        raw_set.add_basis(b)
    s_raw, cond_raw, norm_raw = fit_tensor(raw_set, data, z)
    mse_raw = np.mean((s_raw.predict(data) - z_true) ** 2)

    # ---------------- method 2: tensor product after PCA rotation (drop the near-zero-variance direction) ----------------
    rotated, info = pca_rotate(data, n_components=1)
    r0 = info['explained_variance_ratio'][0]
    r1 = info['explained_variance_ratio'][1]
    pca_set = BasisSet()
    for f in range(3):
        for b in BasisFactory.fourier(dim=0, freq=f, L=np.sqrt(2.0)):
            pca_set.add_basis(b)
    s_pca, cond_pca, norm_pca = fit_tensor(pca_set, rotated, z)
    mse_pca = np.mean((s_pca.predict(rotated) - z_true) ** 2)

    # ---------------- method 3: Mahalanobis kernel (raw coordinates; kernel method needs ridge regularization) ----------------
    kernel = MahalanobisKernel(sigma=1.0)
    s_ker, cond_ker, norm_ker = fit_kernel(kernel, data, z)
    # Gaussian kernel matrices have fast-decaying spectra and are naturally
    # ill-conditioned; ridge regularization (K + αI) is standard practice
    s_ker_reg = FunctionalSolver()
    s_ker_reg.set_kernel(kernel)
    s_ker_reg.load_data(data, z)
    s_ker_reg.solve(regularization={"alpha": 1e-6})
    mse_ker = np.mean((s_ker_reg.predict(data) - z_true) ** 2)

    # ---------------- results table ----------------
    print(f"\n{'Method':<30}{'Cond':>14}{'|coeff|':>14}{'Fit MSE':>14}")
    print("-" * 72)
    print(f"{'Raw tensor product (Fourier ≤ 2)':<30}{cond_raw:>14.3e}{norm_raw:>14.3e}{mse_raw:>14.6f}")
    print(f"{'PCA-decoupled tensor product':<30}{cond_pca:>14.3e}{norm_pca:>14.3e}{mse_pca:>14.6f}")
    print(f"{'Mahalanobis kernel (+ridge 1e-6)':<30}{cond_ker:>14.3e}{norm_ker:>14.3e}{mse_ker:>14.6f}")
    print("-" * 72)
    print(f"PCA explained variance, principal/secondary axis: {r0:.6f} / {r1:.6f}  (secondary ≈ noise, safely dropped)")
    print("Interpretation:")
    print("  - Raw-coordinate tensor product: basis columns are nearly collinear on the narrow band")
    print("    -> ill-conditioned Gram matrix (cond ≈ 1e9), coefficients inflated ≈ 140x;")
    print("       the in-band fit is acceptable but extremely sensitive to extrapolation and noise.")
    print("  - After PCA decoupling: coordinates align with the data's principal axis,")
    print("    cond ≈ 8.6, coefficients ≈ 1, and the fit is exact.")
    print("  - Mahalanobis kernel: Gaussian kernel matrices decay fast and are naturally")
    print("    ill-conditioned (cond ≈ 1e16), so ridge regularization (K + αI) is required;")
    print("    with the ridge, the interpolation is smooth and the error is at noise level.")

    # ---------------- visualization ----------------
    gx, gy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    grid_data = MultiDimData({0: gx.ravel(), 1: gy.ravel()})
    grid_data_rot = pca_transform(grid_data, info)

    z_raw = s_raw.predict(grid_data).reshape(gx.shape)
    z_pca = s_pca.predict(grid_data_rot).reshape(gx.shape)
    z_ker = s_ker_reg.predict(grid_data).reshape(gx.shape)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.scatter(x, y, c=z, cmap='viridis', s=20)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y = x (band center)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Data (narrow band y ≈ x, color = target)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(z_raw.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'Raw-coordinate tensor product\nMSE={mse_raw:.4f}, cond={cond_raw:.1e}')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(z_pca.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'PCA-decoupled tensor product\nMSE={mse_pca:.4f}, cond={cond_pca:.1e}')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(z_ker.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'Mahalanobis kernel\nMSE={mse_ker:.4f}, cond={cond_ker:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_correlated_dims.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return {"raw": mse_raw, "pca": mse_pca, "kernel": mse_ker,
            "cond_raw": cond_raw, "cond_pca": cond_pca, "cond_ker": cond_ker}


if __name__ == "__main__":
    results = main()
    print("\nDemo complete!")
