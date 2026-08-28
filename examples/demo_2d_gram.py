"""2D Gram matrix demo"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # 把标准输出重设为 UTF-8 编码,确保数学符号(如 φ、⊗、Σ)能在任何控制台正常打印(例如 Windows 的 GBK 代码页)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.basis.additive import AdditiveBasis
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver

def demo_2d_polynomial_basis():
    """2D polynomial basis demo"""
    print("=" * 60)
    print("2D polynomial basis demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器
    basis_set = BasisSet()
    
    # 二维张量积基:f(x,y) = Σᵢⱼ cᵢⱼ Xᵢ(x)Yⱼ(y),把两个一维基函数按"乘积"组合成二维基
    print("Building 2D tensor-product polynomial basis f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y):")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]  # x 方向:0 到 2 阶多项式基函数
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]  # y 方向:0 到 2 阶多项式基函数
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})   # 张量积:3×3 = 9 个乘积基函数
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 生成二维测试数据:在 [0,1]² 内随机取 20 个点
    n_points = 20
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真实函数:含交互项的多项式 z = 1 + 2x − y + 3xy + x²y。
    # 交互项(如 xy、x²y)恰好能被张量积基精确表示,而加法(直和)模型无法表示。
    z_true = 1.0 + 2.0*x - y + 3.0*x*y + x**2 * y
    z = z_true + 0.05 * np.random.randn(n_points)  # 叠加少量高斯噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 构造离散内积(基于数据点求和)
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配基函数、内积与数据
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # 对照实验:直和(加法)模型 —— AdditiveBasis.build 生成形如 f(x)+g(y) 的基,
    # 结构上无法表示 X(x)Y(y) 这类交互项,因此对含交互项的真函数拟合误差会更大
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
    print(f"Additive (direct-sum) MSE: {mse_add:.6f}  <- cannot represent interactions")
    
    # 构造 30×30 的规则网格,用于绘制曲面
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    
    # 把网格坐标展平并拼成 (900, 2) 的数组,再转成 MultiDimData 格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上做预测,再还原成网格形状便于绘图
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 在网格上计算真函数的精确值,用于对比拟合误差
    z_grid_true = 1.0 + 2.0*x_grid - y_grid + 3.0*x_grid*y_grid + x_grid**2 * y_grid
    
    # 可视化:3×2 布局,前四个为 3D 曲面,后两个为 2D 图
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点:散点颜色表示目标值 z
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8, label='Data points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真函数曲面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function: 1 + 2x − y + 3xy + x²y')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合曲面:张量积基应能近乎完美地还原真函数
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Polynomial basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差曲面:拟合值减去真值,颜色越接近 0 说明拟合越好
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数柱状图:注意交互项基(如 xy、x²y)对应的系数非零,这正是张量积基的优势
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('Basis index')
    ax5.set_ylabel('Coefficient')
    ax5.set_title('Basis coefficients')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 残差散点:横轴为预测值、纵轴为残差,应无明显趋势
    ax6 = fig.add_subplot(236)
    residuals = z - z_pred
    scatter6 = ax6.scatter(z_pred, residuals, c=residuals, cmap='RdBu', alpha=0.7, s=50)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Residual')
    ax6.set_title('Residuals')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_polynomial_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_2d_fourier_basis():
    """2D Fourier basis demo"""
    print("\n" + "=" * 60)
    print("2D Fourier basis demo")
    print("=" * 60)
    
    # 二维张量积傅里叶基:每个维度取频率 0..1(常数 + 一阶正弦/余弦),张量积后共 3×3 = 9 个乘积基
    print("Building 2D tensor-product Fourier basis f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y):")
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
    
    # 生成周期性二维测试数据:在 [0,1]² 内随机取 25 个点
    n_points = 25
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真实函数 f(x,y) = sin(2πx)·cos(2πy) 本身就是一个乘积 X(x)Y(y),
    # 因此张量积基可以精确表示它(只需对应的一个乘积基非零)
    z_true = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    z = z_true + 0.05 * np.random.randn(n_points)  # 叠加少量高斯噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 构造离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # 对照实验:直和(加法)模型 —— AdditiveBasis.build 生成形如 f(x)+g(y) 的基,
    # 结构上无法表示 X(x)Y(y) 这类交互项,因此对乘积型真函数拟合误差会更大
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
    print(f"Additive (direct-sum) MSE: {mse_add:.6f}  <- cannot represent interactions")
    
    # 构造 40×40 的规则网格,用于绘制曲面
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # 把网格坐标展平并拼成 (1600, 2) 的数组,再转成 MultiDimData 格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上做预测,再还原成网格形状便于绘图
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 在网格上计算真函数值
    z_grid_true = np.sin(2 * np.pi * x_grid) * np.cos(2 * np.pi * y_grid)
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真函数曲面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function: sin(2πx)cos(2πy)')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合曲面
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Fourier basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差曲面
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数柱状图:每个系数对应一个 sin/cos 乘积基
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('Basis index')
    ax5.set_ylabel('Coefficient')
    ax5.set_title('Fourier basis coefficients')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 系数幅度(|cᵢⱼ|)柱状图:展示各频率分量的能量大小
    ax6 = fig.add_subplot(236)
    indices = np.arange(len(coefficients))
    ax6.bar(indices, np.abs(coefficients), alpha=0.7, color='steelblue')
    ax6.set_xlabel('Tensor-product basis index')
    ax6.set_ylabel('|Coefficient|')
    ax6.set_title('Tensor-product Fourier coefficients')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_fourier_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_2d_mixed_basis():
    """2D mixed basis demo"""
    print("\n" + "=" * 60)
    print("2D mixed basis demo")
    print("=" * 60)
    
    # 混合张量积:x 方向用多项式 + 高斯 RBF,y 方向用傅里叶 + 高斯 RBF。
    # 注意:二维高斯 = 一维高斯(x) ⊗ 一维高斯(y),本身就具有张量积结构。
    print("Building 2D mixed tensor-product basis:")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    x_bases.append(BasisFactory.gaussian_rbf(dim=0, center=0.5, sigma=0.1))
    y_bases = []
    for freq in range(2):  # y 方向:频率 0..1 的傅里叶基
        y_bases.extend(BasisFactory.fourier(dim=1, freq=freq, L=1.0))
    y_bases.append(BasisFactory.gaussian_rbf(dim=1, center=0.5, sigma=0.1))
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})  # 张量积:4×4 = 16 个乘积基
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 生成二维测试数据:在 [0,1]² 内随机取 30 个点
    n_points = 30
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # 真函数 = x²·sin(2πy)(即 x² ⊗ sin)+ 二维高斯(即 gauss ⊗ gauss)+ 0.3x(即 x ⊗ 1),每一项都是乘积形式
    z_true = x**2 * np.sin(2 * np.pi * y) + np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) + 0.3*x
    z = z_true + 0.05 * np.random.randn(n_points)  # 叠加少量高斯噪声
    
    data = MultiDimData({0: x, 1: y})
    
    # 构造离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # 预测
    z_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # 对照实验:直和(加法)模型 —— AdditiveBasis.build 生成形如 f(x)+g(y) 的基,
    # 结构上无法表示 X(x)Y(y) 这类交互项,因此对含乘积项的真函数拟合误差会更大
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
    print(f"Additive (direct-sum) MSE: {mse_add:.6f}  <- cannot represent interactions")
    
    # 构造 40×40 的规则网格,用于绘制曲面
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # 把网格坐标展平并拼成 (1600, 2) 的数组,再转成 MultiDimData 格式
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # 在网格点上做预测,再还原成网格形状便于绘图
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # 在网格上计算真函数值(与 z_true 同一公式)
    z_grid_true = x_grid**2 * np.sin(2 * np.pi * y_grid) + np.exp(-((x_grid-0.5)**2 + (y_grid-0.5)**2) / 0.02) + 0.3*x_grid
    
    # 可视化
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 原始数据点
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. 真函数曲面
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. 拟合曲面
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Mixed basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. 误差曲面
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. 系数柱状图:混合基的系数分布
    ax5 = fig.add_subplot(235)
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('Basis index')
    ax5.set_ylabel('Coefficient')
    ax5.set_title('Mixed basis coefficients')
    ax5.set_xticks(indices)
    ax5.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 等高线对比:填充色为真函数,红色等高线为拟合结果,白点为数据
    ax6 = fig.add_subplot(236)
    contour_true = ax6.contourf(x_grid, y_grid, z_grid_true, levels=20, cmap='viridis')
    ax6.contour(x_grid, y_grid, z_grid_pred, levels=20, colors='red', alpha=0.5, linewidths=0.5)
    ax6.scatter(x, y, c='white', s=20, alpha=0.7, edgecolors='black')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('True function (filled) vs fit (contours)')
    plt.colorbar(contour_true, ax=ax6, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_2d_mixed_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_gram_matrix_analysis():
    """Gram matrix analysis demo"""
    print("\n" + "=" * 60)
    print("Gram matrix analysis demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器
    basis_set = BasisSet()
    
    # 添加 0..4 阶多项式基函数
    print("Building basis set...")
    for order in range(5):  # 多项式阶数 0..4
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
    
    # 生成一维测试数据:x 在 [0,1] 上取 50 个点
    x = np.linspace(0, 1, 50)
    data = MultiDimData({0: x})
    
    # 构造离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 求出设计矩阵 Φ:行 = 数据点,列 = 基函数
    Phi = basis_set.evaluate_all(data)
    print(f"Basis matrix shape: {Phi.shape}")
    
    # 计算 Gram 矩阵 G = ΦᵀΦ,其元素为基函数之间的内积
    G = inner_product.compute_gram_matrix(Phi, data)
    print(f"Gram matrix shape: {G.shape}")
    
    # 分析 Gram 矩阵:特征值、条件数、数值秩,评估基函数的"正交程度"与数值稳定性
    eigenvalues = np.linalg.eigvalsh(G)  # 用 eigvalsh 求特征值(对称矩阵专用的高效算法)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # 降序排列,便于观察谱的衰减
    
    condition_number = np.linalg.cond(G)
    rank = np.linalg.matrix_rank(G)
    
    print(f"\nGram matrix analysis:")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Numerical rank: {rank}")
    print(f"  Eigenvalue range: {eigenvalues_sorted[0]:.2e} to {eigenvalues_sorted[-1]:.2e}")
    print(f"  Eigenvalue ratio: {eigenvalues_sorted[0]/eigenvalues_sorted[-1]:.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Gram 矩阵热图:对角占优、非对角接近 0 = 基函数接近正交
    ax1 = axes[0, 0]
    im1 = ax1.imshow(G, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Basis index')
    ax1.set_ylabel('Basis index')
    ax1.set_title('Gram matrix')
    plt.colorbar(im1, ax=ax1, shrink=0.7, aspect=10)
    
    # 2. 特征值谱(对数坐标):特征值快速衰减说明基函数之间有冗余/近似线性相关
    ax2 = axes[0, 1]
    indices = np.arange(1, len(eigenvalues_sorted) + 1)
    ax2.semilogy(indices, eigenvalues_sorted, 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Threshold: 1e-10')
    ax2.set_xlabel('Eigenvalue index')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title('Gram matrix eigenvalue spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 设计矩阵 Φ 热图:展示每个数据点上各基函数的取值
    ax3 = axes[0, 2]
    im3 = ax3.imshow(Phi, cmap='plasma', aspect='auto')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Data point index')
    ax3.set_title('Basis matrix Φ')
    plt.colorbar(im3, ax=ax3, shrink=0.7, aspect=10)
    
    # 4. 基函数曲线:直观看到各阶多项式的形状
    ax4 = axes[1, 0]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax4.plot(x_plot, y_basis, label=f'Order {i}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('φ(x)')
    ax4.set_title('Polynomial basis functions')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. 奇异值分解:设计矩阵 Φ 的奇异值谱(与特征值谱类似,反映矩阵的数值稳定性)
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    ax5 = axes[1, 1]
    indices_svd = np.arange(1, len(s) + 1)
    ax5.semilogy(indices_svd, s, 'go-', linewidth=2, markersize=6)
    ax5.set_xlabel('Singular value index')
    ax5.set_ylabel('Singular value (log scale)')
    ax5.set_title('Basis matrix singular values')
    ax5.grid(True, alpha=0.3)
    
    # 6. 条件数与基函数数量的关系:基越多,列越接近线性相关,条件数通常越大
    ax6 = axes[1, 2]
    cond_numbers = []
    n_basis_range = range(2, 8)
    
    for n_basis in n_basis_range:
        # 只取前 n_basis 个基函数列,观察条件数随基数量的变化
        Phi_partial = Phi[:, :n_basis]
        G_partial = inner_product.compute_gram_matrix(Phi_partial, data)
        cond = np.linalg.cond(G_partial)
        cond_numbers.append(cond)
    
    ax6.plot(list(n_basis_range), cond_numbers, 'ro-', linewidth=2, markersize=8)
    ax6.set_xlabel('Number of bases')
    ax6.set_ylabel('Condition number (log scale)')
    ax6.set_title('Condition number vs number of bases')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_gram_matrix_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return G, eigenvalues_sorted, condition_number

if __name__ == "__main__":
    print("2D Gram Matrix Demo")
    print("=" * 60)
    
    # 依次运行全部演示,并把每个方法的 MSE 记录到字典中
    results = {}
    
    # 1. 二维多项式张量积基
    coeff_2d_poly, mse_2d_poly = demo_2d_polynomial_basis()
    results['2D polynomial'] = mse_2d_poly
    
    # 2. 二维傅里叶张量积基
    coeff_2d_fourier, mse_2d_fourier = demo_2d_fourier_basis()
    results['2D Fourier'] = mse_2d_fourier
    
    # 3. 二维混合张量积基
    coeff_2d_mixed, mse_2d_mixed = demo_2d_mixed_basis()
    results['2D mixed'] = mse_2d_mixed
    
    # 4. Gram 矩阵分析(记录条件数)
    G, eigenvalues, cond_number = demo_gram_matrix_analysis()
    results['Condition number'] = cond_number
    
    # 打印各方法的结果对比
    print("\n" + "=" * 60)
    print("Results comparison")
    print("=" * 60)
    for method, value in results.items():
        if method == 'Condition number':
            print(f"{method:10}: {value:.2e}")
        else:
            print(f"{method:10} basis: MSE = {value:.6f}")
    
    print("\nDemo complete!")
