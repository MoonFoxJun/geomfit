"""RKHS核函数示例"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functional_solver.core.data_container import MultiDimData
from functional_solver.kernel.rbf import RBFKernel
from functional_solver.kernel.polynomial import PolynomialKernel
from functional_solver.kernel.matern import MaternKernel
from functional_solver.kernel.composite import CompositeKernel
from functional_solver.solver.functional_solver import FunctionalSolver
from functional_solver.solver.kernel_solver import KernelSolver

def demo_rbf_kernel():
    """RBF核函数示例"""
    print("=" * 60)
    print("RBF核函数示例")
    print("=" * 60)
    
    # 创建RBF核函数
    sigma = 0.2
    kernel = RBFKernel(sigma=sigma)
    print(f"创建RBF核函数，sigma={sigma}")
    
    # 创建测试数据
    n_points = 30
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)  # 混合正弦波
    y = y_true + 0.1 * np.random.randn(n_points)  # 添加噪声
    
    data = MultiDimData({0: x})
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数数量: {len(coefficients)}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 计算核矩阵
    K = kernel.compute_matrix(data)
    print(f"核矩阵形状: {K.shape}")
    print(f"核矩阵条件数: {np.linalg.cond(K):.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='RBF核拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'RBF核函数拟合 (sigma={sigma})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('数据点索引')
    ax2.set_ylabel('数据点索引')
    ax2.set_title('RBF核矩阵')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 核函数值
    ax3 = axes[0, 2]
    x_test = 0.5  # 测试点
    x_range = np.linspace(0, 1, 100)
    kernel_values = np.zeros_like(x_range)
    
    for i, x_val in enumerate(x_range):
        kernel_values[i] = kernel({0: x_test}, {0: x_val})
    
    ax3.plot(x_range, kernel_values, 'b-', linewidth=2)
    ax3.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'中心点 x={x_test}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('核函数值 k(x_test, x)')
    ax3.set_title(f'RBF核函数 (中心在 x={x_test})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 系数
    ax4 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax4.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('数据点索引')
    ax4.set_ylabel('系数值')
    ax4.set_title('RBF核系数')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 残差
    ax5 = axes[1, 1]
    residuals = y - y_pred
    ax5.scatter(x, residuals, alpha=0.6, color='purple', s=20)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('残差')
    ax5.set_title('残差图')
    ax5.grid(True, alpha=0.3)
    
    # 6. 特征值谱
    ax6 = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(K)  # 对称矩阵，使用eigvalsh
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    indices_eig = np.arange(1, len(eigenvalues_sorted) + 1)
    
    ax6.semilogy(indices_eig, eigenvalues_sorted, 'bo-', linewidth=2, markersize=6)
    ax6.set_xlabel('特征值索引')
    ax6.set_ylabel('特征值（对数尺度）')
    ax6.set_title('核矩阵特征值谱')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_rbf_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse, K

def demo_polynomial_kernel():
    """多项式核函数示例"""
    print("\n" + "=" * 60)
    print("多项式核函数示例")
    print("=" * 60)
    
    # 创建多项式核函数
    degree = 3
    c = 1.0
    kernel = PolynomialKernel(degree=degree, c=c)
    print(f"创建多项式核函数，degree={degree}, c={c}")
    
    # 创建测试数据
    n_points = 25
    x = np.linspace(-1, 1, n_points)
    y_true = x**3 - 2*x**2 + 0.5*x + 1  # 三次多项式
    y = y_true + 0.05 * np.random.randn(n_points)  # 添加少量噪声
    
    data = MultiDimData({0: x})
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数数量: {len(coefficients)}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 计算核矩阵
    K = kernel.compute_matrix(data)
    print(f"核矩阵形状: {K.shape}")
    print(f"核矩阵条件数: {np.linalg.cond(K):.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='多项式核拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'多项式核函数拟合 (degree={degree})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='plasma', aspect='auto')
    ax2.set_xlabel('数据点索引')
    ax2.set_ylabel('数据点索引')
    ax2.set_title('多项式核矩阵')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 核函数值
    ax3 = axes[0, 2]
    x_test = 0.0  # 测试点
    x_range = np.linspace(-1, 1, 100)
    kernel_values = np.zeros_like(x_range)
    
    for i, x_val in enumerate(x_range):
        kernel_values[i] = kernel({0: x_test}, {0: x_val})
    
    ax3.plot(x_range, kernel_values, 'b-', linewidth=2)
    ax3.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'中心点 x={x_test}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('核函数值 k(x_test, x)')
    ax3.set_title(f'多项式核函数 (中心在 x={x_test})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不同阶数的比较
    ax4 = axes[1, 0]
    degrees = [1, 2, 3, 4, 5]
    mse_values = []
    
    for d in degrees:
        kernel_temp = PolynomialKernel(degree=d, c=c)
        solver_temp = FunctionalSolver()
        solver_temp.set_kernel(kernel_temp)
        solver_temp.load_data(data, y)
        coeff_temp = solver_temp.solve()
        y_pred_temp = solver_temp.predict(data)
        mse_temp = np.mean((y_pred_temp - y)**2)
        mse_values.append(mse_temp)
    
    ax4.plot(degrees, mse_values, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('多项式阶数')
    ax4.set_ylabel('均方误差 (MSE)')
    ax4.set_title('不同阶数多项式核的拟合误差')
    ax4.grid(True, alpha=0.3)
    
    # 5. 系数分布
    ax5 = axes[1, 1]
    indices = np.arange(len(coefficients))
    ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('数据点索引')
    ax5.set_ylabel('系数值')
    ax5.set_title('多项式核系数')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 特征值谱
    ax6 = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    indices_eig = np.arange(1, len(eigenvalues_sorted) + 1)
    
    ax6.semilogy(indices_eig, eigenvalues_sorted, 'go-', linewidth=2, markersize=6)
    ax6.set_xlabel('特征值索引')
    ax6.set_ylabel('特征值（对数尺度）')
    ax6.set_title('多项式核矩阵特征值谱')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_polynomial_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse, K

def demo_matern_kernel():
    """Matern核函数示例"""
    print("\n" + "=" * 60)
    print("Matern核函数示例")
    print("=" * 60)
    
    # 创建不同nu值的Matern核函数
    kernels = []
    nu_values = [0.5, 1.5, 2.5, float('inf')]  # 对应不同平滑度
    kernel_names = ['Matern ν=1/2', 'Matern ν=3/2', 'Matern ν=5/2', 'RBF (ν→∞)']
    
    for nu in nu_values:
        if nu == float('inf'):
            kernel = RBFKernel(sigma=0.3)  # RBF是Matern在ν→∞时的极限
        else:
            kernel = MaternKernel(sigma=0.3, nu=nu)
        kernels.append(kernel)
    
    # 创建测试数据
    n_points = 40
    x = np.linspace(0, 2, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(n_points)  # 带噪声的正弦波
    
    data = MultiDimData({0: x})
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\n使用{name}核函数...")
        
        # 创建解算器
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y_true)
        
        # 求解
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # 计算误差
        mse = np.mean((y_pred - y_true)**2)
        results.append((name, mse, y_pred))
        print(f"  均方误差 (MSE): {mse:.6f}")
        
        # 1. 不同核函数的拟合结果
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.scatter(x, y_true, alpha=0.5, label='数据', s=10)
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'{name}拟合')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}核函数拟合')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. 核函数值比较
    ax5 = axes[1, 2]
    x_test = 1.0  # 测试点
    x_range = np.linspace(0, 2, 100)
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        kernel_values = np.zeros_like(x_range)
        for i, x_val in enumerate(x_range):
            kernel_values[i] = kernel({0: x_test}, {0: x_val})
        
        ax5.plot(x_range, kernel_values, linewidth=2, label=name)
    
    ax5.axvline(x=x_test, color='k', linestyle='--', alpha=0.5, label=f'中心点 x={x_test}')
    ax5.set_xlabel('x')
    ax5.set_ylabel('核函数值 k(x_test, x)')
    ax5.set_title('不同Matern核函数比较')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 误差比较
    ax6 = axes[0, 2]
    names = [r[0] for r in results]
    mse_values = [r[1] for r in results]
    
    bars = ax6.bar(range(len(names)), mse_values, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    ax6.set_xlabel('核函数类型')
    ax6.set_ylabel('均方误差 (MSE)')
    ax6.set_title('不同核函数的拟合误差')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_matern_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def demo_composite_kernel():
    """复合核函数示例"""
    print("\n" + "=" * 60)
    print("复合核函数示例")
    print("=" * 60)
    
    # 创建基础核函数
    rbf_kernel = RBFKernel(sigma=0.3)
    poly_kernel = PolynomialKernel(degree=2, c=1.0)
    matern_kernel = MaternKernel(sigma=0.3, nu=1.5)
    
    # 创建复合核函数
    print("创建复合核函数...")
    
    # 1. 加法复合核
    additive_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='add')
    print("  - 加法复合核: RBF + 多项式")
    
    # 2. 乘法复合核
    multiplicative_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='multiply')
    print("  - 乘法复合核: RBF × 多项式")
    
    kernels = [rbf_kernel, poly_kernel, additive_kernel, multiplicative_kernel]
    kernel_names = ['RBF', '多项式', 'RBF+多项式', 'RBF×多项式']
    
    # 创建测试数据
    n_points = 30
    x = np.linspace(0, 1, n_points)
    # 复杂函数：多项式 + 周期 + 局部特征
    y_true = 0.5*x**2 + np.sin(2*np.pi*x) + np.exp(-(x-0.7)**2/0.02)
    y = y_true + 0.05 * np.random.randn(n_points)  # 添加少量噪声
    
    data = MultiDimData({0: x})
    
    # 可视化
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\n使用{name}核函数...")
        
        # 创建解算器
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y)
        
        # 求解
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # 计算误差
        mse = np.mean((y_pred - y)**2)
        results.append((name, mse, y_pred))
        print(f"  均方误差 (MSE): {mse:.6f}")
        
        # 1. 拟合结果
        ax = axes[idx]
        ax.scatter(x, y, alpha=0.5, label='数据', s=10)
        ax.plot(x, y_true, 'g-', linewidth=2, label='真实函数', alpha=0.7)
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'{name}拟合')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}核函数拟合')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 计算核矩阵
        K = kernel.compute_matrix(data)
        print(f"  核矩阵条件数: {np.linalg.cond(K):.2e}")
    
    # 6. 误差比较
    ax6 = axes[5]
    names = [r[0] for r in results]
    mse_values = [r[1] for r in results]
    
    bars = ax6.bar(range(len(names)), mse_values, alpha=0.7, 
                  color=['blue', 'green', 'red', 'purple', 'orange'])
    ax6.set_xlabel('核函数类型')
    ax6.set_ylabel('均方误差 (MSE)')
    ax6.set_title('不同核函数的拟合误差比较')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # 在柱状图上添加数值标签
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_composite_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 额外可视化：核函数值比较
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    x_test = 0.5  # 测试点
    x_range = np.linspace(0, 1, 100)
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if idx >= 6:  # 只显示前6个
            break
            
        kernel_values = np.zeros_like(x_range)
        for i, x_val in enumerate(x_range):
            kernel_values[i] = kernel({0: x_test}, {0: x_val})
        
        ax = axes2[idx]
        ax.plot(x_range, kernel_values, 'b-', linewidth=2)
        ax.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'中心点 x={x_test}')
        ax.set_xlabel('x')
        ax.set_ylabel('k(x_test, x)')
        ax.set_title(f'{name}核函数')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_kernel_functions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def demo_kernel_solver_features():
    """核解算器特性示例"""
    print("\n" + "=" * 60)
    print("核解算器特性示例")
    print("=" * 60)
    
    # 创建RBF核函数
    kernel = RBFKernel(sigma=0.2)
    
    # 创建核解算器
    solver = KernelSolver(kernel)
    print("创建核解算器...")
    
    # 创建测试数据
    n_points = 20
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(n_points)
    
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 计算核矩阵
    K = solver.compute_kernel_matrix()
    print(f"核矩阵形状: {K.shape}")
    
    # 计算特征值
    eigenvalues = solver.get_eigenvalues()
    print(f"特征值数量: {len(eigenvalues)}")
    print(f"最大特征值: {eigenvalues[0]:.4f}")
    print(f"最小特征值: {eigenvalues[-1]:.4f}")
    print(f"条件数: {solver.get_condition_number():.2e}")
    
    # 计算函数范数
    norm = solver.compute_function_norm()
    print(f"函数范数: {norm:.4f}")
    
    # 求解
    coefficients = solver.solve()
    
    # 预测
    y_pred = solver.predict(data)
    
    # 高效预测
    y_pred_efficient = solver.predict_efficient(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    mse_efficient = np.mean((y_pred_efficient - y)**2)
    print(f"标准预测MSE: {mse:.6f}")
    print(f"高效预测MSE: {mse_efficient:.6f}")
    print(f"预测一致性: {np.allclose(y_pred, y_pred_efficient, rtol=1e-10)}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数', alpha=0.7)
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='核方法拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('核方法拟合结果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('数据点索引')
    ax2.set_ylabel('数据点索引')
    ax2.set_title('核矩阵')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 特征值谱
    ax3 = axes[0, 2]
    indices = np.arange(1, len(eigenvalues) + 1)
    ax3.semilogy(indices, eigenvalues, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('特征值索引')
    ax3.set_ylabel('特征值（对数尺度）')
    ax3.set_title('核矩阵特征值谱')
    ax3.grid(True, alpha=0.3)
    
    # 4. 系数
    ax4 = axes[1, 0]
    indices_coeff = np.arange(len(coefficients))
    ax4.bar(indices_coeff, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('数据点索引')
    ax4.set_ylabel('系数值')
    ax4.set_title('核方法系数')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 留一法误差估计
    ax5 = axes[1, 1]
    loo_errors = []
    
    # 计算每个数据点的留一法误差
    for i in range(min(10, n_points)):  # 只计算前10个点以节省时间
        # 创建不包含第i个点的数据
        mask = np.ones(n_points, dtype=bool)
        mask[i] = False
        
        x_train = x[mask]
        y_train = y[mask]
        x_test = x[i]
        y_test = y[i]
        
        train_data = MultiDimData({0: x_train})
        test_data = MultiDimData({0: np.array([x_test])})
        
        # 训练新解算器
        temp_solver = KernelSolver(kernel)
        temp_solver.load_data(train_data, y_train)
        temp_solver.solve()
        
        # 预测测试点
        y_pred_test = temp_solver.predict(test_data)[0]
        loo_errors.append(abs(y_pred_test - y_test))
    
    ax5.plot(range(len(loo_errors)), loo_errors, 'ro-', linewidth=2, markersize=6)
    ax5.set_xlabel('数据点索引')
    ax5.set_ylabel('留一法误差')
    ax5.set_title('留一法误差估计（前10个点）')
    ax5.grid(True, alpha=0.3)
    
    # 6. 正则化效果
    ax6 = axes[1, 2]
    alphas = np.logspace(-6, 0, 10)  # 正则化参数
    norms = []
    mses = []
    
    for alpha in alphas:
        solver_temp = KernelSolver(kernel)
        solver_temp.load_data(data, y)
        coeff_temp = solver_temp.solve(regularization={"alpha": alpha})
        y_pred_temp = solver_temp.predict(data)
        
        norms.append(np.linalg.norm(coeff_temp))
        mses.append(np.mean((y_pred_temp - y)**2))
    
    ax6a = ax6.twinx()
    line1, = ax6.plot(alphas, norms, 'b-', linewidth=2, label='系数范数')
    line2, = ax6a.plot(alphas, mses, 'r--', linewidth=2, label='MSE')
    
    ax6.set_xlabel('正则化参数 α')
    ax6.set_ylabel('系数范数', color='b')
    ax6a.set_ylabel('均方误差', color='r')
    ax6.set_title('正则化效果')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6a.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_kernel_solver_features.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return solver, coefficients, eigenvalues

if __name__ == "__main__":
    print("RKHS核函数示例")
    print("=" * 60)
    
    # 运行所有示例
    results_summary = {}
    
    # 1. RBF核函数
    print("\n1. RBF核函数示例")
    coeff_rbf, mse_rbf, K_rbf = demo_rbf_kernel()
    results_summary['RBF'] = mse_rbf
    
    # 2. 多项式核函数
    print("\n2. 多项式核函数示例")
    coeff_poly, mse_poly, K_poly = demo_polynomial_kernel()
    results_summary['多项式'] = mse_poly
    
    # 3. Matern核函数
    print("\n3. Matern核函数示例")
    results_matern = demo_matern_kernel()
    for name, mse, _ in results_matern:
        results_summary[name] = mse
    
    # 4. 复合核函数
    print("\n4. 复合核函数示例")
    results_composite = demo_composite_kernel()
    for name, mse, _ in results_composite:
        results_summary[name] = mse
    
    # 5. 核解算器特性
    print("\n5. 核解算器特性示例")
    solver, coeff_kernel, eigenvalues = demo_kernel_solver_features()
    results_summary['核解算器'] = np.mean((solver.predict(MultiDimData({0: np.linspace(0, 1, 20)})) - 
                                          np.sin(2*np.pi*np.linspace(0, 1, 20)))**2)
    
    # 打印结果总结
    print("\n" + "=" * 60)
    print("结果总结")
    print("=" * 60)
    for method, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        print(f"{method:15}: MSE = {mse:.6f}")
    
    print("\n示例完成！")
