"""Attention as a kernel machine: a minimal demonstration.

Three points (see docs/kernel_convolution_attention.md for details):

1. [Exact identity] The self-attention similarity matrix QKᵀ (with W_q = W_k = I)
   is exactly the kernel matrix computed by PolynomialKernel(1).compute_matrix():
       max|QKᵀ − K| = 0.0
   Attention = softmax(kernel matrix) × value vectors  ->  kernel smoothing.

2. [One machine, two estimators]
   - KernelSolver: solves (K + αI)α = y (kernel ridge regression; can interpolate
     or overfit, and decays by the kernel outside the data range);
   - Fixed attention: row-wise softmax normalization (a convex combination;
     the output is bounded and does not extrapolate).
   Both compute "prediction = Σ w_j(x*) y_j"; they differ only in how the
   weights w are obtained.

3. [Where "content-adaptivity" comes from] When W_q and W_k are learnable, the
   kernel is no longer fixed — the weights are determined by all inputs jointly.
   This is what makes attention more flexible than a fixed convolution kernel.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from functional_solver.core.data_container import MultiDimData
from functional_solver.kernel.polynomial import PolynomialKernel
from functional_solver.kernel.rbf import RBFKernel
from functional_solver.solver.functional_solver import FunctionalSolver


def main():
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.05 * rng.standard_normal(n)
    data = MultiDimData({0: x})

    # ---------- 1. exact identity: QKᵀ = kernel matrix ----------
    K = PolynomialKernel(degree=1, coef0=0.0, gamma=1.0).compute_matrix(data)
    QK = x[:, None] @ x[None, :]              # here W_q = W_k = I
    print("=" * 64)
    print("1) Exact identity: attention similarity matrix == library kernel matrix")
    print(f"   max|QKᵀ - PolynomialKernel(1).compute_matrix()| = "
          f"{np.max(np.abs(QK - K)):.2e}")
    print("   → attention softmax(QKᵀ/√d)·V is kernel smoothing softmax(kernel matrix)·V.")

    # ---------- 2. one machine, two estimators ----------
    ell, alpha = 0.12, 0.05
    x_grid = np.linspace(-0.25, 1.25, 400)

    # 2a) KernelSolver: kernel ridge regression (K+αI)α = y
    s = FunctionalSolver()
    s.set_kernel(RBFKernel(sigma=1.0, length_scale=ell))
    s.load_data(data, y)
    s.solve(regularization={"alpha": alpha})
    f_ridge = s.predict(MultiDimData({0: x_grid}))

    # 2b) Fixed attention: softmax-normalized kernel smoothing (Nadaraya-Watson)
    d2 = (x_grid[:, None] - x[None, :]) ** 2
    A = np.exp(-d2 / (2 * ell ** 2))
    A = A / A.sum(axis=1, keepdims=True)      # row-wise softmax
    f_attn = A @ y

    in_range = (x_grid >= 0) & (x_grid <= 1)
    print("-" * 64)
    print("2) One kernel machine, two estimators (RBF kernel, ℓ=0.12)")
    print(f"   Max in-range difference = {np.max(np.abs(f_ridge[in_range] - f_attn[in_range])):.3f}")
    print(f"   Out-of-range behavior differs: kernel ridge at x→1.25 → {f_ridge[-1]:+.3f},")
    print(f"                  softmax attention → {f_attn[-1]:+.3f} (bounded, no extrapolation)")
    print("   Ridge regression solves a linear system and can overfit; softmax is a convex")
    print("   combination and is bounded — that is the difference between the two 'knobs',")
    print("   α and softmax normalization.")

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=20, alpha=0.6, label='Data')
    ax.plot(x_grid, np.sin(2 * np.pi * x_grid), 'g-', lw=1.5, alpha=0.6, label='True function')
    ax.plot(x_grid, f_ridge, 'b-', lw=2, label='KernelSolver ridge regression (K+αI)α=y')
    ax.plot(x_grid, f_attn, 'r--', lw=2, label='Fixed attention = softmax kernel smoothing')
    ax.axvspan(-0.25, 0, color='gray', alpha=0.15)
    ax.axvspan(1, 1.25, color='gray', alpha=0.15)
    ax.text(1.03, 0.9, 'Out-of-range region\n(estimator differences)', fontsize=8, color='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('One kernel machine: solve (ridge) vs softmax normalization (attention)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_attention_as_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
    print("Demo complete!")
