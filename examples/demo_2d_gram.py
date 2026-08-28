"""二维Gram矩阵示例"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory
from functional_solver.basis.additive import AdditiveBasis
from functional_solver.inner_product.base import InnerProduct
from functional_solver.solver.functional_solver import FunctionalSolver

def demo_2d_polynomial_basis():
    """二维多项式基函数示例"""
    print("=" * 60)
    print("二维多项式基函数示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 创建基函数集合：二维张量积基 f(x,y) = Σᵢⱼ cᵢⱼ Xᵢ(x)Yⱼ(y)
    print("构建二维张量积多项式基 f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y):")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]  # x 的 0-2 阶
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]  # y 的 0-2 阶
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})   # 3×3 = 9 个乘积基
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 创建二维测试数据
    n_points = 20
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真实函数：带交互项的多项式（张量积基能精确表示，加法模型不能）
    z_true = 1.0 + 2.0*x - y + 3.0*x*y + x**2 * y
    z = z_true + 0.05 * np.random.randn(n_points)  # 添加少量噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 创建内积（离散内积）
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数数量: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((z_pred - z)**2)
    print(f"张量积基 MSE: {mse:.6f}")

    # 对照：直和（加法模型）——保留的 AdditiveBasis 实现，结构上无法表示 X(x)Y(y) 交互项
    additive_set = AdditiveBasis.build({
        0: [BasisFactory.polynomial(dim=0, order=o) for o in range(3)],
        1: [BasisFactory.polynomial(dim=1, order=o) for o in range(3)],
    })
    solver_add = FunctionalSolver()
    solver_add.set_basis(additive_set)
    solver_add.set_inner_product(InnerProduct(is_continuous=False))
    solver_add.load_data(data, z)
    solver_add.solve()
    z_pred_add = solver_add.predict(data)
    mse_add = np.mean((z_pred_add - z)**2)
    print(f"加法模型(直和) MSE: {mse_add:.6f}  <- 无法表示交互项")
    
    # 创建网格用于可视化
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    
    # 将网格点转换为数据格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上预测
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 计算真实函数在网格上的值
    z_grid_true = 1.0 + 2.0*x_grid - y_grid + 3.0*x_grid*y_grid + x_grid**2 * y_grid
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8, label='数据点')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('原始数据点')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真实函数表面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('真实函数: sin(2πx)cos(2πy)')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合表面
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('多项式基函数拟合')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差表面
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('误差')
    ax4.set_title('拟合误差')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('基函数索引')
    ax5.set_ylabel('系数值')
    ax5.set_title('基函数系数')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 残差散点图
    ax6 = fig.add_subplot(236)
    residuals = z - z_pred
    scatter6 = ax6.scatter(z_pred, residuals, c=residuals, cmap='RdBu', alpha=0.7, s=50)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('预测值')
    ax6.set_ylabel('残差')
    ax6.set_title('残差图')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_polynomial_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_2d_fourier_basis():
    """二维傅里叶基函数示例"""
    print("\n" + "=" * 60)
    print("二维傅里叶基函数示例")
    print("=" * 60)
    
    # 创建基函数集合：二维张量积傅里叶基（频率 0-1，共 3×3 = 9 个乘积基）
    print("构建二维张量积傅里叶基 f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y):")
    x_bases = []
    for freq in range(2):
        x_bases.extend(BasisFactory.fourier(dim=0, freq=freq, L=1.0))
    y_bases = []
    for freq in range(2):
        y_bases.extend(BasisFactory.fourier(dim=1, freq=freq, L=1.0))
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 创建二维测试数据（周期函数）
    n_points = 25
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真实函数：f(x,y) = sin(2πx)cos(2πy)，正是 X⊗Y 形式，张量积基可精确表示
    z_true = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    z = z_true + 0.05 * np.random.randn(n_points)  # 添加少量噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数数量: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((z_pred - z)**2)
    print(f"张量积基 MSE: {mse:.6f}")

    # 对照：直和（加法模型）——保留的 AdditiveBasis 实现，结构上无法表示 X(x)Y(y) 交互项
    additive_set = AdditiveBasis.build({
        0: [BasisFactory.polynomial(dim=0, order=o) for o in range(3)],
        1: [BasisFactory.polynomial(dim=1, order=o) for o in range(3)],
    })
    solver_add = FunctionalSolver()
    solver_add.set_basis(additive_set)
    solver_add.set_inner_product(InnerProduct(is_continuous=False))
    solver_add.load_data(data, z)
    solver_add.solve()
    z_pred_add = solver_add.predict(data)
    mse_add = np.mean((z_pred_add - z)**2)
    print(f"加法模型(直和) MSE: {mse_add:.6f}  <- 无法表示交互项")
    
    # 创建网格用于可视化
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # 将网格点转换为数据格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上预测
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 计算真实函数在网格上的值
    z_grid_true = np.sin(2 * np.pi * x_grid) * np.cos(2 * np.pi * y_grid)
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('原始数据点')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真实函数表面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('真实函数: sin(2πx)cos(2πy)')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合表面
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('傅里叶基函数拟合')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差表面
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('误差')
    ax4.set_title('拟合误差')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('基函数索引')
    ax5.set_ylabel('系数值')
    ax5.set_title('傅里叶基函数系数')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 系数幅度
    ax6 = fig.add_subplot(236)
    indices = np.arange(len(coefficients))
    ax6.bar(indices, np.abs(coefficients), alpha=0.7, color='steelblue')
    ax6.set_xlabel('张量积基索引')
    ax6.set_ylabel('|系数|')
    ax6.set_title('张量积傅里叶基系数')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_fourier_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_2d_mixed_basis():
    """二维混合基函数示例"""
    print("\n" + "=" * 60)
    print("二维混合基函数示例")
    print("=" * 60)
    
    # 创建基函数集合：混合张量积（x 方向多项式 + y 方向傅里叶 + 各维高斯）
    # 二维高斯 = 一维高斯(x) ⊗ 一维高斯(y)，正是张量积结构
    print("构建二维混合张量积基:")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    x_bases.append(BasisFactory.gaussian_rbf(dim=0, center=0.5, sigma=0.1))
    y_bases = []
    for freq in range(2):  # y的0-1频率傅里叶基
        y_bases.extend(BasisFactory.fourier(dim=1, freq=freq, L=1.0))
    y_bases.append(BasisFactory.gaussian_rbf(dim=1, center=0.5, sigma=0.1))
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})  # 4×4 = 16 个乘积基
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 创建二维测试数据
    n_points = 30
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真实函数：x²·sin(2πy)（= x² ⊗ sin） + 二维高斯（= gauss ⊗ gauss） + 0.3x（= x ⊗ 1）
    z_true = x**2 * np.sin(2 * np.pi * y) + np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) + 0.3*x
    z = z_true + 0.05 * np.random.randn(n_points)  # 添加少量噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数数量: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((z_pred - z)**2)
    print(f"张量积基 MSE: {mse:.6f}")

    # 对照：直和（加法模型）——保留的 AdditiveBasis 实现，结构上无法表示 X(x)Y(y) 交互项
    additive_set = AdditiveBasis.build({
        0: [BasisFactory.polynomial(dim=0, order=o) for o in range(3)],
        1: [BasisFactory.polynomial(dim=1, order=o) for o in range(3)],
    })
    solver_add = FunctionalSolver()
    solver_add.set_basis(additive_set)
    solver_add.set_inner_product(InnerProduct(is_continuous=False))
    solver_add.load_data(data, z)
    solver_add.solve()
    z_pred_add = solver_add.predict(data)
    mse_add = np.mean((z_pred_add - z)**2)
    print(f"加法模型(直和) MSE: {mse_add:.6f}  <- 无法表示交互项")
    
    # 创建网格用于可视化
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # 将网格点转换为数据格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上预测
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 计算真实函数在网格上的值
    z_grid_true = x_grid**2 * np.sin(2 * np.pi * y_grid) + np.exp(-((x_grid-0.5)**2 + (y_grid-0.5)**2) / 0.02) + 0.3*x_grid
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('原始数据点')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真实函数表面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('真实函数')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合表面
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('混合基函数拟合')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差表面
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('误差')
    ax4.set_title('拟合误差')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('基函数索引')
    ax5.set_ylabel('系数值')
    ax5.set_title('混合基函数系数')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 等高线图比较
    ax6 = fig.add_subplot(236)
    contour_true = ax6.contourf(x_grid, y_grid, z_grid_true, levels=20, cmap='viridis')
    ax6.contour(x_grid, y_grid, z_grid_pred, levels=20, colors='red', alpha=0.5, linewidths=0.5)
    ax6.scatter(x, y, c='white', s=20, alpha=0.7, edgecolors='black')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('真实函数（填充） vs 拟合（等高线）')
    plt.colorbar(contour_true, ax=ax6, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_mixed_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_gram_matrix_analysis():
    """Gram矩阵分析示例"""
    print("\n" + "=" * 60)
    print("Gram矩阵分析示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 添加多项式基函数
    print("创建基函数集合...")
    for order in range(5):  # 0-4阶多项式
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
    
    # 创建测试数据
    x = np.linspace(0, 1, 50)
    data = MultiDimData({0: x})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    print(f"基函数矩阵形状: {Phi.shape}")
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    print(f"Gram矩阵形状: {G.shape}")
    
    # 分析Gram矩阵
    eigenvalues = np.linalg.eigvalsh(G)  # 计算特征值（对称矩阵）
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # 降序排列
    
    condition_number = np.linalg.cond(G)
    rank = np.linalg.matrix_rank(G)
    
    print(f"\nGram矩阵分析:")
    print(f"  条件数: {condition_number:.2e}")
    print(f"  数值秩: {rank}")
    print(f"  特征值范围: {eigenvalues_sorted[0]:.2e} 到 {eigenvalues_sorted[-1]:.2e}")
    print(f"  特征值比率: {eigenvalues_sorted[0]/eigenvalues_sorted[-1]:.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Gram矩阵热图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(G, cmap='viridis', aspect='auto')
    ax1.set_xlabel('基函数索引')
    ax1.set_ylabel('基函数索引')
    ax1.set_title('Gram矩阵')
    plt.colorbar(im1, ax=ax1, shrink=0.7, aspect=10)
    
    # 2. 特征值谱
    ax2 = axes[0, 1]
    indices = np.arange(1, len(eigenvalues_sorted) + 1)
    ax2.semilogy(indices, eigenvalues_sorted, 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='阈值: 1e-10')
    ax2.set_xlabel('特征值索引')
    ax2.set_ylabel('特征值（对数尺度）')
    ax2.set_title('Gram矩阵特征值谱')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 基函数矩阵
    ax3 = axes[0, 2]
    im3 = ax3.imshow(Phi, cmap='plasma', aspect='auto')
    ax3.set_xlabel('基函数索引')
    ax3.set_ylabel('数据点索引')
    ax3.set_title('基函数矩阵 Φ')
    plt.colorbar(im3, ax=ax3, shrink=0.7, aspect=10)
    
    # 4. 基函数
    ax4 = axes[1, 0]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax4.plot(x_plot, y_basis, label=f'阶数 {i}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('φ(x)')
    ax4.set_title('多项式基函数')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 奇异值分解
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    ax5 = axes[1, 1]
    indices_svd = np.arange(1, len(s) + 1)
    ax5.semilogy(indices_svd, s, 'go-', linewidth=2, markersize=6)
    ax5.set_xlabel('奇异值索引')
    ax5.set_ylabel('奇异值（对数尺度）')
    ax5.set_title('基函数矩阵奇异值')
    ax5.grid(True, alpha=0.3)
    
    # 6. 条件数随基函数数量的变化
    ax6 = axes[1, 2]
    cond_numbers = []
    n_basis_range = range(2, 8)
    
    for n_basis in n_basis_range:
        # 使用前n_basis个基函数
        Phi_partial = Phi[:, :n_basis]
        G_partial = inner_product.compute_gram_matrix(Phi_partial, data)
        cond = np.linalg.cond(G_partial)
        cond_numbers.append(cond)
    
    ax6.plot(list(n_basis_range), cond_numbers, 'ro-', linewidth=2, markersize=8)
    ax6.set_xlabel('基函数数量')
    ax6.set_ylabel('条件数（对数尺度）')
    ax6.set_title('条件数 vs 基函数数量')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_gram_matrix_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return G, eigenvalues_sorted, condition_number

if __name__ == "__main__":
    print("二维Gram矩阵示例")
    print("=" * 60)
    
    # 运行所有示例
    results = {}
    
    # 1. 二维多项式基函数
    coeff_2d_poly, mse_2d_poly = demo_2d_polynomial_basis()
    results['二维多项式'] = mse_2d_poly
    
    # 2. 二维傅里叶基函数
    coeff_2d_fourier, mse_2d_fourier = demo_2d_fourier_basis()
    results['二维傅里叶'] = mse_2d_fourier
    
    # 3. 二维混合基函数
    coeff_2d_mixed, mse_2d_mixed = demo_2d_mixed_basis()
    results['二维混合'] = mse_2d_mixed
    
    # 4. Gram矩阵分析
    G, eigenvalues, cond_number = demo_gram_matrix_analysis()
    results['条件数'] = cond_number
    
    # 打印结果比较
    print("\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)
    for method, value in results.items():
        if method == '条件数':
            print(f"{method:10}: {value:.2e}")
        else:
            print(f"{method:10} 基函数: MSE = {value:.6f}")
    
    print("\n示例完成！")
