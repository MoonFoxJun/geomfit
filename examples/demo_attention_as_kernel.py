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
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # 把标准输出重设为 UTF-8 编码,确保数学符号(如 α、ᵀ)能在任何控制台正常打印(例如 Windows 的 GBK 代码页)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from geomfit.core.data_container import MultiDimData
from geomfit.kernel.polynomial import PolynomialKernel
from geomfit.kernel.rbf import RBFKernel
from geomfit.solver.functional_solver import FunctionalSolver


def main():
    rng = np.random.default_rng(42)     # 固定随机种子,保证结果可复现
    n = 60
    x = rng.uniform(0, 1, n)
    y_true = np.sin(2 * np.pi * x)      # 真实函数:正弦波
    y = y_true + 0.05 * rng.standard_normal(n)
    data = MultiDimData({0: x})

    # ---------- 1. 精确恒等式:注意力相似度矩阵 QKᵀ = 核矩阵 ----------
    # 一阶多项式核 k(x,x') = γ·xx' + coef0 = xx'(取 γ=1、coef0=0),与注意力相似度矩阵的形式完全一致
    K = PolynomialKernel(degree=1, coef0=0.0, gamma=1.0).compute_matrix(data)
    QK = x[:, None] @ x[None, :]              # 注意力相似度矩阵 QKᵀ = XXᵀ(这里取 W_q = W_k = I,即不学习的单位映射)
    print("=" * 64)
    print("1) Exact identity: attention similarity matrix == library kernel matrix")
    print(f"   max|QKᵀ - PolynomialKernel(1).compute_matrix()| = "
          f"{np.max(np.abs(QK - K)):.2e}")
    print("   → attention softmax(QKᵀ/√d)·V is kernel smoothing softmax(kernel matrix)·V.")

    # ---------- 2. 同一个核机器,两种估计器 ----------
    ell, alpha = 0.12, 0.05      # 核宽度 ℓ 与岭正则化强度 α
    x_grid = np.linspace(-0.25, 1.25, 400)   # 预测网格超出数据范围 [0,1],两侧各外延 0.25,用于考察带外行为

    # 2a) KernelSolver:核岭回归,求解线性方程组 (K + αI)α = y
    s = FunctionalSolver()
    s.set_kernel(RBFKernel(sigma=1.0, length_scale=ell))   # 用 RBF 核装配求解器,ℓ 控制核的平滑宽度
    s.load_data(data, y)
    s.solve(regularization={"alpha": alpha})
    f_ridge = s.predict(MultiDimData({0: x_grid}))          # 岭回归预测:线性方程组的解,可以插值甚至过拟合,带外按核衰减

    # 2b) 固定注意力:softmax 归一化的核平滑(即 Nadaraya-Watson 估计器)
    d2 = (x_grid[:, None] - x[None, :]) ** 2                # 网格点与全部数据点的距离平方,广播成 (400, 60) 矩阵
    A = np.exp(-d2 / (2 * ell ** 2))                        # RBF 权重:exp(−‖x−x'‖²/(2ℓ²)),距离越近权重越大
    A = A / A.sum(axis=1, keepdims=True)      # 逐行 softmax 归一化:每行权重和为 1,输出是数据的凸组合(有界)
    f_attn = A @ y                                          # 注意力输出 = 加权平均 Σⱼ wⱼ yⱼ,不会外推(带外预测趋近常数)

    in_range = (x_grid >= 0) & (x_grid <= 1)                # 标记落在数据范围 [0,1] 内的网格点,用于分别统计带内/带外差异
    print("-" * 64)
    print("2) One kernel machine, two estimators (RBF kernel, ℓ=0.12)")
    print(f"   Max in-range difference = {np.max(np.abs(f_ridge[in_range] - f_attn[in_range])):.3f}")
    print(f"   Out-of-range behavior differs: kernel ridge at x→1.25 → {f_ridge[-1]:+.3f},")
    print(f"                  softmax attention → {f_attn[-1]:+.3f} (bounded, no extrapolation)")
    print("   Ridge regression solves a linear system and can overfit; softmax is a convex")
    print("   combination and is bounded — that is the difference between the two 'knobs',")
    print("   α and softmax normalization.")

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(10, 6))                 # 绘制两种估计器与真函数、数据的对比
    ax.scatter(x, y, s=20, alpha=0.6, label='Data')
    ax.plot(x_grid, np.sin(2 * np.pi * x_grid), 'g-', lw=1.5, alpha=0.6, label='True function')
    ax.plot(x_grid, f_ridge, 'b-', lw=2, label='KernelSolver ridge regression (K+αI)α=y')
    ax.plot(x_grid, f_attn, 'r--', lw=2, label='Fixed attention = softmax kernel smoothing')
    ax.axvspan(-0.25, 0, color='gray', alpha=0.15)          # 灰色阴影标出带外区域,两种估计器在此处的行为差异是本节重点
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
