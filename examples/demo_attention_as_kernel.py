"""注意力 = 核机器 的最小演示。

三个要点（详见 docs/kernel_convolution_attention.md）：

1. 【精确恒等式】自注意力的相似度矩阵 QKᵀ（取 W_q = W_k = I）就是本库
   PolynomialKernel(1).compute_matrix() 算出来的核矩阵：
       max|QKᵀ − K| = 0.0
   注意力 = softmax(核矩阵) × 值向量  →  核平滑。

2. 【同一台机器，两种估计器】
   - KernelSolver：解方程 (K+αI)α = y（核岭回归，可插值/过拟合、带外按核衰减）；
   - 固定注意力：逐行 softmax 归一化（凸组合，输出必有界、不外推）。
   两者都是"预测 = Σ w_j(x*) y_j"，只是权重 w 的算法不同。

3. 【注意力的"内容自适应"来自哪里】当 W_q、W_k 可学习时，核不再固定——
   权重由全部输入共同决定，这就是它比固定卷积核"自由"的地方。
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

    # ---------- 1. 精确恒等式：QKᵀ = 核矩阵 ----------
    K = PolynomialKernel(degree=1, coef0=0.0, gamma=1.0).compute_matrix(data)
    QK = x[:, None] @ x[None, :]              # 取 W_q = W_k = I
    print("=" * 64)
    print("1) 精确恒等式：注意力相似度矩阵 == 库的核矩阵")
    print(f"   max|QKᵀ - PolynomialKernel(1).compute_matrix()| = "
          f"{np.max(np.abs(QK - K)):.2e}")
    print("   → 注意力 softmax(QKᵀ/√d)·V 就是  softmax(核矩阵)·V 的核平滑。")

    # ---------- 2. 同一台机器，两种估计器 ----------
    ell, alpha = 0.12, 0.05
    x_grid = np.linspace(-0.25, 1.25, 400)

    # 2a) KernelSolver：核岭回归 (K+αI)α = y
    s = FunctionalSolver()
    s.set_kernel(RBFKernel(sigma=1.0, length_scale=ell))
    s.load_data(data, y)
    s.solve(regularization={"alpha": alpha})
    f_ridge = s.predict(MultiDimData({0: x_grid}))

    # 2b) 固定注意力：softmax 归一化的核平滑（Nadaraya-Watson）
    d2 = (x_grid[:, None] - x[None, :]) ** 2
    A = np.exp(-d2 / (2 * ell ** 2))
    A = A / A.sum(axis=1, keepdims=True)      # 逐行 softmax
    f_attn = A @ y

    in_range = (x_grid >= 0) & (x_grid <= 1)
    print("-" * 64)
    print("2) 同一台核机器，两种估计器（RBF 核, ℓ=0.12）")
    print(f"   带内最大差异 = {np.max(np.abs(f_ridge[in_range] - f_attn[in_range])):.3f}")
    print(f"   带外行为不同：核岭回归在 x→1.25 处 → {f_ridge[-1]:+.3f}，")
    print(f"                  softmax 注意力 → {f_attn[-1]:+.3f}（有界，不外推）")
    print("   岭回归解方程、可过拟合；softmax 是凸组合、必有界——")
    print("   这正是 α 与 softmax 归一化这两个'旋钮'的差别。")

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=20, alpha=0.6, label='数据')
    ax.plot(x_grid, np.sin(2 * np.pi * x_grid), 'g-', lw=1.5, alpha=0.6, label='真实函数')
    ax.plot(x_grid, f_ridge, 'b-', lw=2, label='KernelSolver 核岭回归 (K+αI)α=y')
    ax.plot(x_grid, f_attn, 'r--', lw=2, label='固定注意力 = softmax 核平滑')
    ax.axvspan(-0.25, 0, color='gray', alpha=0.15)
    ax.axvspan(1, 1.25, color='gray', alpha=0.15)
    ax.text(1.03, 0.9, '带外区域\n(展示估计器差异)', fontsize=8, color='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('同一台核机器：解方程(岭) vs softmax 归一化(注意力)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_attention_as_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
    print("示例完成！")
