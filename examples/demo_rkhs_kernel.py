"""RKHS kernel demo"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # 把标准输出重设为 UTF-8 编码,确保数学符号(如 ν、α)能在任何控制台正常打印(例如 Windows 的 GBK 代码页)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geomfit.core.data_container import MultiDimData
from geomfit.kernel.rbf import RBFKernel
from geomfit.kernel.polynomial import PolynomialKernel
from geomfit.kernel.matern import MaternKernel
from geomfit.kernel.composite import CompositeKernel
from geomfit.solver.functional_solver import FunctionalSolver
from geomfit.solver.kernel_solver import KernelSolver

def demo_rbf_kernel():
    """RBF kernel demo"""
    print("=" * 60)
    print("RBF kernel demo")
    print("=" * 60)
    
    # 创建一个 RBF(高斯)核:sigma 控制核的宽度,sigma 越小核越"尖"
    sigma = 0.2
    kernel = RBFKernel(sigma=sigma)
    print(f"Created RBF kernel with sigma={sigma}")
    
    # 生成一维测试数据:在 [0, 1] 上取 30 个点
    n_points = 30
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)  # 真实函数:正弦 + 余弦的混合波
    y = y_true + 0.1 * np.random.randn(n_points)  # 叠加高斯噪声(标准差 0.1)
    
    data = MultiDimData({0: x})
    
    # 创建求解器并装配核与数据(核方法不需要显式基函数)
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # 求解:核方法拟合解线性方程组 (K + αI)α = y,得到核系数(表示定理:解是核函数的线性组合)
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 计算核矩阵 K:元素 K[i,j] = k(xᵢ, xⱼ),衡量两两数据点之间的相似度
    K = kernel.compute_matrix(data)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='RBF fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'RBF kernel fit (sigma={sigma})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图:沿对角线对称,近对角元素大(邻近点相似度高)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('RBF kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 核函数形状:固定测试点 x_test=0.5,画出 k(x_test, x) 随 x 的变化
    ax3 = axes[0, 2]
    x_test = 0.5  # 选定的测试点(核的中心)
    x_range = np.linspace(0, 1, 100)
    kernel_values = np.zeros_like(x_range)
    
    for i, x_val in enumerate(x_range):
        kernel_values[i] = kernel({0: x_test}, {0: x_val})
    
    ax3.plot(x_range, kernel_values, 'b-', linewidth=2)
    ax3.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'Center x={x_test}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Kernel value k(x_test, x)')
    ax3.set_title(f'RBF kernel (center x={x_test})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 核系数:每个数据点对应一个系数(表示定理:解位于数据点的核函数张成空间中)
    ax4 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax4.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('Data point index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('RBF kernel coefficients')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 残差图
    ax5 = axes[1, 1]
    residuals = y - y_pred
    ax5.scatter(x, residuals, alpha=0.6, color='purple', s=20)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('Residual')
    ax5.set_title('Residuals')
    ax5.grid(True, alpha=0.3)
    
    # 6. 核矩阵特征值谱:高斯核的特征值指数级衰减,反映其对应函数空间(再生核希尔伯特空间)的容量
    ax6 = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(K)  # 核矩阵对称,用 eigvalsh 求特征值
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    indices_eig = np.arange(1, len(eigenvalues_sorted) + 1)
    
    ax6.semilogy(indices_eig, eigenvalues_sorted, 'bo-', linewidth=2, markersize=6)
    ax6.set_xlabel('Eigenvalue index')
    ax6.set_ylabel('Eigenvalue (log scale)')
    ax6.set_title('Kernel matrix eigenvalue spectrum')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_rbf_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse, K

def demo_polynomial_kernel():
    """Polynomial kernel demo"""
    print("\n" + "=" * 60)
    print("Polynomial kernel demo")
    print("=" * 60)
    
    # 创建多项式核:degree 为次数,c 为偏移常数,对应特征空间中的多项式特征
    degree = 3
    c = 1.0
    kernel = PolynomialKernel(degree=degree, c=c)
    print(f"Created polynomial kernel with degree={degree}, c={c}")
    
    # 生成测试数据:在 [-1, 1] 上取 25 个点
    n_points = 25
    x = np.linspace(-1, 1, n_points)
    y_true = x**3 - 2*x**2 + 0.5*x + 1  # 真实函数:三次多项式
    y = y_true + 0.05 * np.random.randn(n_points)  # 叠加少量噪声(标准差 0.05)
    
    data = MultiDimData({0: x})
    
    # 创建求解器并装配核与数据
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # 求解核系数
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 计算核矩阵
    K = kernel.compute_matrix(data)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Polynomial kernel fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Polynomial kernel fit (degree={degree})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='plasma', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('Polynomial kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 核函数形状:固定 x_test=0.0,观察多项式核的取值
    ax3 = axes[0, 2]
    x_test = 0.0  # 选定的测试点
    x_range = np.linspace(-1, 1, 100)
    kernel_values = np.zeros_like(x_range)
    
    for i, x_val in enumerate(x_range):
        kernel_values[i] = kernel({0: x_test}, {0: x_val})
    
    ax3.plot(x_range, kernel_values, 'b-', linewidth=2)
    ax3.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'Center x={x_test}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Kernel value k(x_test, x)')
    ax3.set_title(f'Polynomial kernel (center x={x_test})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 次数对比:1 到 5 次多项式核的拟合误差,观察欠拟合 → 拟合 → 过拟合的变化
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
    ax4.set_xlabel('Polynomial degree')
    ax4.set_ylabel('Mean squared error (MSE)')
    ax4.set_title('Polynomial kernel fit error vs degree')
    ax4.grid(True, alpha=0.3)
    
    # 5. 系数分布:每个数据点对应一个核系数
    ax5 = axes[1, 1]
    indices = np.arange(len(coefficients))
    ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('Data point index')
    ax5.set_ylabel('Coefficient')
    ax5.set_title('Polynomial kernel coefficients')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 特征值谱
    ax6 = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    indices_eig = np.arange(1, len(eigenvalues_sorted) + 1)
    
    ax6.semilogy(indices_eig, eigenvalues_sorted, 'go-', linewidth=2, markersize=6)
    ax6.set_xlabel('Eigenvalue index')
    ax6.set_ylabel('Eigenvalue (log scale)')
    ax6.set_title('Polynomial kernel matrix eigenvalue spectrum')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_polynomial_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse, K

def demo_matern_kernel():
    """Matérn kernel demo"""
    print("\n" + "=" * 60)
    print("Matérn kernel demo")
    print("=" * 60)
    
    # 创建不同平滑参数 ν 的 Matérn 核
    kernels = []
    nu_values = [0.5, 1.5, 2.5, float('inf')]  # 不同的平滑程度:ν 越小,样本函数越粗糙(高频成分越多)
    kernel_names = ['Matern ν=1/2', 'Matern ν=3/2', 'Matern ν=5/2', 'RBF (ν→∞)']
    
    for nu in nu_values:
        if nu == float('inf'):
            kernel = RBFKernel(sigma=0.3)  # RBF 核是 ν → ∞ 时 Matérn 核的极限
        else:
            kernel = MaternKernel(sigma=0.3, nu=nu)
        kernels.append(kernel)
    
    # 生成测试数据:在 [0, 2] 上取 40 个点
    n_points = 40
    x = np.linspace(0, 2, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(n_points)  # 带噪声的正弦波(噪声直接加在真函数上)
    
    data = MultiDimData({0: x})
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\nFitting with {name} kernel...")
        
        # 创建求解器并装配当前核与数据
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y_true)
        
        # 求解并预测
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # 计算均方误差
        mse = np.mean((y_pred - y_true)**2)
        results.append((name, mse, y_pred))
        print(f"  Mean squared error (MSE): {mse:.6f}")
        
        # 1. 不同核的拟合曲线:前四个子图分别展示 ν=1/2、3/2、5/2 与 RBF 的拟合
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.scatter(x, y_true, alpha=0.5, label='Data', s=10)
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'{name} fit')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} kernel fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. 核函数形状对比:固定 x_test=1.0,比较不同 ν 下核随距离的衰减速度
    ax5 = axes[1, 2]
    x_test = 1.0  # 选定的测试点
    x_range = np.linspace(0, 2, 100)
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        kernel_values = np.zeros_like(x_range)
        for i, x_val in enumerate(x_range):
            kernel_values[i] = kernel({0: x_test}, {0: x_val})
        
        ax5.plot(x_range, kernel_values, linewidth=2, label=name)
    
    ax5.axvline(x=x_test, color='k', linestyle='--', alpha=0.5, label=f'Center x={x_test}')
    ax5.set_xlabel('x')
    ax5.set_ylabel('Kernel value k(x_test, x)')
    ax5.set_title('Matérn kernel comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 误差对比柱状图:比较各核的拟合 MSE
    ax6 = axes[0, 2]
    names = [r[0] for r in results]
    mse_values = [r[1] for r in results]
    
    bars = ax6.bar(range(len(names)), mse_values, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    ax6.set_xlabel('Kernel')
    ax6.set_ylabel('Mean squared error (MSE)')
    ax6.set_title('Fit error by kernel')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图顶部标注具体的 MSE 数值
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_matern_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def demo_composite_kernel():
    """Composite kernel demo"""
    print("\n" + "=" * 60)
    print("Composite kernel demo")
    print("=" * 60)
    
    # 基础核:作为复合核的成分,各自擅长刻画不同类型的结构
    rbf_kernel = RBFKernel(sigma=0.3)
    poly_kernel = PolynomialKernel(degree=2, c=1.0)
    matern_kernel = MaternKernel(sigma=0.3, nu=1.5)
    
    # 构造复合核:对基础核做加/乘运算
    print("Building composite kernels...")
    
    # 1. 加法复合核:k_add(x,x') = k_rbf(x,x') + k_poly(x,x'),对应特征空间的直和(可同时表达两套特征)
    additive_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='add')
    print("  - Additive: RBF + Polynomial")
    
    # 2. 乘法复合核:k_mul(x,x') = k_rbf(x,x') · k_poly(x,x'),对应特征空间的张量积(交互式结构)
    multiplicative_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='multiply')
    print("  - Multiplicative: RBF × Polynomial")
    
    kernels = [rbf_kernel, poly_kernel, additive_kernel, multiplicative_kernel]
    kernel_names = ['RBF', 'Polynomial', 'RBF+Polynomial', 'RBF×Polynomial']
    
    # 生成测试数据:在 [0, 1] 上取 30 个点
    n_points = 30
    x = np.linspace(0, 1, n_points)
    # 目标函数:多项式趋势 + 周期分量 + 局部尖峰,需要多种核配合才能刻画
    y_true = 0.5*x**2 + np.sin(2*np.pi*x) + np.exp(-(x-0.7)**2/0.02)
    y = y_true + 0.05 * np.random.randn(n_points)  # 叠加少量噪声(标准差 0.05)
    
    data = MultiDimData({0: x})
    
    # 可视化
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\nFitting with {name} kernel...")
        
        # 创建求解器并装配当前核与数据
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y)
        
        # 求解并预测
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # 计算均方误差
        mse = np.mean((y_pred - y)**2)
        results.append((name, mse, y_pred))
        print(f"  Mean squared error (MSE): {mse:.6f}")
        
        # 1. 拟合结果:每个核一个子图,对比各核的表达能力
        ax = axes[idx]
        ax.scatter(x, y, alpha=0.5, label='Data', s=10)
        ax.plot(x, y_true, 'g-', linewidth=2, label='True function', alpha=0.7)
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'{name} fit')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} kernel fit')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 计算当前核的核矩阵,打印条件数观察数值稳定性
        K = kernel.compute_matrix(data)
        print(f"  Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # 6. 误差对比(对数坐标):复合核通常能同时覆盖多种结构,误差更低
    ax6 = axes[5]
    names = [r[0] for r in results]
    mse_values = [r[1] for r in results]
    
    bars = ax6.bar(range(len(names)), mse_values, alpha=0.7, 
                  color=['blue', 'green', 'red', 'purple', 'orange'])
    ax6.set_xlabel('Kernel')
    ax6.set_ylabel('Mean squared error (MSE)')
    ax6.set_title('Fit error by kernel')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # 在柱状图顶部标注 MSE 数值
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_composite_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 附加图:逐一展示各核函数 k(x_test, x) 的形状
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    x_test = 0.5  # 选定的测试点
    x_range = np.linspace(0, 1, 100)
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if idx >= 6:  # 最多显示 6 个子图
            break
            
        kernel_values = np.zeros_like(x_range)
        for i, x_val in enumerate(x_range):
            kernel_values[i] = kernel({0: x_test}, {0: x_val})
        
        ax = axes2[idx]
        ax.plot(x_range, kernel_values, 'b-', linewidth=2)
        ax.axvline(x=x_test, color='r', linestyle='--', alpha=0.5, label=f'Center x={x_test}')
        ax.set_xlabel('x')
        ax.set_ylabel('k(x_test, x)')
        ax.set_title(f'{name} kernel')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_kernel_functions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

def demo_kernel_solver_features():
    """Kernel solver features demo"""
    print("\n" + "=" * 60)
    print("Kernel solver features demo")
    print("=" * 60)
    
    # RBF 核
    kernel = RBFKernel(sigma=0.2)
    
    # 核求解器:KernelSolver 封装了核矩阵计算与特征分解等操作
    solver = KernelSolver(kernel)
    print("Creating kernel solver...")
    
    # 生成测试数据:在 [0, 1] 上取 20 个点
    n_points = 20
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(n_points)
    
    data = MultiDimData({0: x})
    
    # 载入数据
    solver.load_data(data, y)
    
    # 计算核矩阵
    K = solver.compute_kernel_matrix()
    print(f"Kernel matrix shape: {K.shape}")
    
    # 特征值:核矩阵的特征值可用于估计函数范数与条件数
    eigenvalues = solver.get_eigenvalues()
    print(f"Number of eigenvalues: {len(eigenvalues)}")
    print(f"Largest eigenvalue: {eigenvalues[0]:.4f}")
    print(f"Smallest eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Condition number: {solver.get_condition_number():.2e}")
    
    # 函数范数:‖f‖_H² = αᵀKα,衡量拟合函数在再生核希尔伯特空间中的复杂度
    norm = solver.compute_function_norm()
    print(f"Function norm: {norm:.4f}")
    
    # 求解核岭回归
    coefficients = solver.solve()
    
    # 标准预测
    y_pred = solver.predict(data)
    
    # 高效预测:利用预计算的分解加速
    y_pred_efficient = solver.predict_efficient(data)
    
    # 比较两种预测的误差,并确认它们数值上一致
    mse = np.mean((y_pred - y)**2)
    mse_efficient = np.mean((y_pred_efficient - y)**2)
    print(f"Standard prediction MSE: {mse:.6f}")
    print(f"Efficient prediction MSE: {mse_efficient:.6f}")
    print(f"Predictions agree: {np.allclose(y_pred, y_pred_efficient, rtol=1e-10)}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function', alpha=0.7)
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Kernel method fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Kernel method fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核矩阵热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('Kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. 特征值谱(对数坐标)
    ax3 = axes[0, 2]
    indices = np.arange(1, len(eigenvalues) + 1)
    ax3.semilogy(indices, eigenvalues, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Eigenvalue index')
    ax3.set_ylabel('Eigenvalue (log scale)')
    ax3.set_title('Kernel matrix eigenvalue spectrum')
    ax3.grid(True, alpha=0.3)
    
    # 4. 核系数柱状图
    ax4 = axes[1, 0]
    indices_coeff = np.arange(len(coefficients))
    ax4.bar(indices_coeff, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('Data point index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('Kernel method coefficients')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 留一法(LOO)误差估计:每次拿掉一个点训练,再预测该点
    ax5 = axes[1, 1]
    loo_errors = []
    
    # 对每个点计算留一误差
    for i in range(min(10, n_points)):  # 只算前 10 个点以节省时间
        # 构造除第 i 个点以外的训练集(用布尔掩码去掉第 i 行)
        mask = np.ones(n_points, dtype=bool)
        mask[i] = False
        
        x_train = x[mask]
        y_train = y[mask]
        x_test = x[i]
        y_test = y[i]
        
        train_data = MultiDimData({0: x_train})
        test_data = MultiDimData({0: np.array([x_test])})
        
        # 重新训练一个求解器(不含第 i 个点)
        temp_solver = KernelSolver(kernel)
        temp_solver.load_data(train_data, y_train)
        temp_solver.solve()
        
        # 预测被留出的那个点,记录绝对误差
        y_pred_test = temp_solver.predict(test_data)[0]
        loo_errors.append(abs(y_pred_test - y_test))
    
    ax5.plot(range(len(loo_errors)), loo_errors, 'ro-', linewidth=2, markersize=6)
    ax5.set_xlabel('Data point index')
    ax5.set_ylabel('Leave-one-out error')
    ax5.set_title('Leave-one-out errors (first 10 points)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 正则化强度 α 的影响:观察系数范数与 MSE 之间的权衡
    ax6 = axes[1, 2]
    alphas = np.logspace(-6, 0, 10)  # 正则化参数 α 从 1e-6 到 1(对数均匀取值)
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
    line1, = ax6.plot(alphas, norms, 'b-', linewidth=2, label='Coefficient norm')
    line2, = ax6a.plot(alphas, mses, 'r--', linewidth=2, label='MSE')
    
    ax6.set_xlabel('Regularization parameter α')
    ax6.set_ylabel('Coefficient norm', color='b')
    ax6a.set_ylabel('Mean squared error', color='r')
    ax6.set_title('Effect of regularization')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6a.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # 合并双 y 轴的两条曲线到同一个图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_kernel_solver_features.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return solver, coefficients, eigenvalues

if __name__ == "__main__":
    print("RKHS Kernel Demo")
    print("=" * 60)
    
    # 依次运行全部演示,并把各方法的 MSE 汇总到字典中
    results_summary = {}
    
    # 1. RBF 核演示
    print("\n1. RBF kernel demo")
    coeff_rbf, mse_rbf, K_rbf = demo_rbf_kernel()
    results_summary['RBF'] = mse_rbf
    
    # 2. 多项式核演示
    print("\n2. Polynomial kernel demo")
    coeff_poly, mse_poly, K_poly = demo_polynomial_kernel()
    results_summary['Polynomial'] = mse_poly
    
    # 3. Matérn 核演示(一个核对应一个 MSE)
    print("\n3. Matérn kernel demo")
    results_matern = demo_matern_kernel()
    for name, mse, _ in results_matern:
        results_summary[name] = mse
    
    # 4. 复合核演示
    print("\n4. Composite kernel demo")
    results_composite = demo_composite_kernel()
    for name, mse, _ in results_composite:
        results_summary[name] = mse
    
    # 5. 核求解器特性演示(最后再算一次在均匀网格上的拟合 MSE)
    print("\n5. Kernel solver features demo")
    solver, coeff_kernel, eigenvalues = demo_kernel_solver_features()
    results_summary['Kernel solver'] = np.mean((solver.predict(MultiDimData({0: np.linspace(0, 1, 20)})) - 
                                          np.sin(2*np.pi*np.linspace(0, 1, 20)))**2)
    
    # 打印按 MSE 升序排列的汇总表
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    for method, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        print(f"{method:15}: MSE = {mse:.6f}")
    
    print("\nDemo complete!")
