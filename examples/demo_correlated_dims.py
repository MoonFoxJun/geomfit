"""强相关维度示例：PCA 解耦 + 马氏距离核。

演示内容：数据躺在窄带 y ≈ x（强相关）上，真函数只依赖主轴
u = (x+y)/√2。对比三种做法：

1. 原始坐标上的张量积傅里叶基（笛卡尔直积基）
   —— 频率网格是方形的（整数频率），而对角方向的固有频率是 0.5，
      落在网格缝隙里：既表示不了，Gram 又因基列近共线而病态；
2. PCA 旋转（丢弃近零方差方向）后再做张量积
   —— 坐标系对齐数据主轴，一个频率项就精确拟合；
3. 马氏距离核（直接在原始坐标上）
   —— 把相关性吸收进距离度量，核矩阵良态、拟合平滑。
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
    """用张量积基拟合，返回 (解算器, 条件数, 系数范数)。"""
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    coeff = solver.solve()
    G = InnerProduct(is_continuous=False).compute_gram_matrix(
        basis_set.evaluate_all(data), data)
    return solver, np.linalg.cond(G), np.linalg.norm(coeff)


def fit_kernel(kernel, data, z):
    """用核方法拟合，返回 (解算器, 条件数, 系数范数)。"""
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, z)
    coeff = solver.solve()
    K = kernel.compute_matrix(data)
    return solver, np.linalg.cond(K), np.linalg.norm(coeff)


def main():
    rng = np.random.default_rng(2024)

    # ---------------- 数据：窄带 y ≈ x，真函数只依赖主轴 u ----------------
    n = 80
    x = rng.uniform(0.1, 0.9, n)
    v = rng.normal(0.0, 0.02, n)          # 垂直方向抖动很小 → 强相关
    y = x + v
    u = (x + y) / np.sqrt(2.0)            # 主轴
    z_true = np.sin(np.pi * (x + y))      # = sin(√2 π u)，只依赖 u
    z = z_true + 0.02 * rng.standard_normal(n)
    data = MultiDimData({0: x, 1: y})

    print("=" * 66)
    print("强相关维度（窄带 y≈x）: 原始坐标张量积 vs PCA 解耦 vs 马氏核")
    print(f"corr(x, y) = {np.corrcoef(x, y)[0, 1]:.4f}")
    print("=" * 66)

    # ---------------- 方法 1：原始坐标张量积傅里叶（频率 0..2, L=1） ----------------
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

    # ---------------- 方法 2：PCA 旋转（丢弃近零方差方向）后张量积 ----------------
    rotated, info = pca_rotate(data, n_components=1)
    r0 = info['explained_variance_ratio'][0]
    r1 = info['explained_variance_ratio'][1]
    pca_set = BasisSet()
    for f in range(3):
        for b in BasisFactory.fourier(dim=0, freq=f, L=np.sqrt(2.0)):
            pca_set.add_basis(b)
    s_pca, cond_pca, norm_pca = fit_tensor(pca_set, rotated, z)
    mse_pca = np.mean((s_pca.predict(rotated) - z_true) ** 2)

    # ---------------- 方法 3：马氏距离核（原始坐标，核方法需岭正则） ----------------
    kernel = MahalanobisKernel(sigma=1.0)
    s_ker, cond_ker, norm_ker = fit_kernel(kernel, data, z)
    # 高斯核矩阵谱衰减快、天然病态——配岭正则化（核方法的常规操作）
    s_ker_reg = FunctionalSolver()
    s_ker_reg.set_kernel(kernel)
    s_ker_reg.load_data(data, z)
    s_ker_reg.solve(regularization={"alpha": 1e-6})
    mse_ker = np.mean((s_ker_reg.predict(data) - z_true) ** 2)

    # ---------------- 结果表 ----------------
    print(f"\n{'方法':<28}{'条件数':>12}{'|系数|':>12}{'拟合MSE':>14}")
    print("-" * 66)
    print(f"{'原始坐标张量积(傅里叶≤2)':<22}{cond_raw:>14.3e}{norm_raw:>12.3e}{mse_raw:>14.6f}")
    print(f"{'PCA解耦后张量积':<26}{cond_pca:>12.3e}{norm_pca:>12.3e}{mse_pca:>14.6f}")
    print(f"{'马氏距离核(+岭1e-6)':<24}{cond_ker:>14.3e}{norm_ker:>12.3e}{mse_ker:>14.6f}")
    print("-" * 66)
    print(f"PCA 主轴/次轴方差占比: {r0:.6f} / {r1:.6f}  (次轴≈噪声，可丢弃)")
    print("解读:")
    print("  - 原始坐标张量积：窄带上基列近共线 → Gram 病态(条件数≈1e9)、")
    print("    系数被放大≈140倍；带内勉强拟合，但对外推/噪声极敏感。")
    print("  - PCA 解耦后：坐标对齐数据主轴，条件数≈8.6、系数≈1，精确拟合。")
    print("  - 马氏核：高斯核矩阵谱衰减快、天然病态(条件数≈1e16)，")
    print("    必须配岭正则(K+αI)；加岭后平滑插值，误差≈噪声水平。")

    # ---------------- 可视化 ----------------
    gx, gy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    grid_data = MultiDimData({0: gx.ravel(), 1: gy.ravel()})
    grid_data_rot = pca_transform(grid_data, info)

    z_raw = s_raw.predict(grid_data).reshape(gx.shape)
    z_pca = s_pca.predict(grid_data_rot).reshape(gx.shape)
    z_ker = s_ker_reg.predict(grid_data).reshape(gx.shape)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.scatter(x, y, c=z, cmap='viridis', s=20)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x（带中心）')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('数据（窄带 y≈x，颜色=目标值）')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(z_raw.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'原始坐标张量积\nMSE={mse_raw:.4f}, cond={cond_raw:.1e}')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(z_pca.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'PCA 解耦后张量积\nMSE={mse_pca:.4f}, cond={cond_pca:.1e}')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(z_ker.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu', vmin=-1, vmax=1)
    ax.scatter(x, y, s=8, c='k', alpha=0.4)
    ax.set_title(f'马氏距离核\nMSE={mse_ker:.4f}, cond={cond_ker:.1e}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_correlated_dims.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return {"raw": mse_raw, "pca": mse_pca, "kernel": mse_ker,
            "cond_raw": cond_raw, "cond_pca": cond_pca, "cond_ker": cond_ker}


if __name__ == "__main__":
    results = main()
    print("\n示例完成！")
