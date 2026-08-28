"""一维基函数示例"""

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

def demo_polynomial_basis():
    """多项式基函数示例"""
    print("=" * 60)
    print("多项式基函数示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 添加多项式基函数
    print("添加多项式基函数:")
    for order in range(5):  # 0到4阶多项式
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 创建测试数据
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * x)  # 真实函数：正弦波
    y = y_true + 0.1 * np.random.randn(len(x))  # 添加噪声
    
    data = MultiDimData({0: x})
    
    # 创建内积（离散内积）
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数: {coefficients}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('多项式基函数拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=f'阶数 {i}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('多项式基函数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('基函数索引')
    ax3.set_ylabel('系数值')
    ax3.set_title('基函数系数')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 残差
    ax4 = axes[1, 1]
    residuals = y - y_pred
    ax4.scatter(x, residuals, alpha=0.6, color='purple', s=10)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('残差')
    ax4.set_title('残差图')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_polynomial_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_fourier_basis():
    """傅里叶基函数示例"""
    print("\n" + "=" * 60)
    print("傅里叶基函数示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 添加傅里叶基函数
    print("添加傅里叶基函数:")
    # 常数项
    bases = BasisFactory.fourier(dim=0, freq=0, L=1.0)
    for basis in bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 频率1-3的傅里叶基
    for freq in range(1, 4):
        bases = BasisFactory.fourier(dim=0, freq=freq, L=1.0)
        for basis in bases:
            basis_set.add_basis(basis)
            print(f"  - {basis.name}")
    
    # 创建测试数据（周期函数）
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * 2 * x) + 0.5 * np.cos(2 * np.pi * 3 * x)  # 混合正弦波
    y = y_true + 0.05 * np.random.randn(len(x))  # 添加少量噪声
    
    data = MultiDimData({0: x})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('傅里叶基函数拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数（显示前几个）
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases[:6]):  # 只显示前6个
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=basis.name)
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('傅里叶基函数（前6个）')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('基函数索引')
    ax3.set_ylabel('系数值')
    ax3.set_title('傅里叶基函数系数')
    ax3.set_xticks(indices)
    ax3.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 频谱
    ax4 = axes[1, 1]
    # 提取频率和系数大小
    freqs = []
    coeff_mags = []
    for i, basis in enumerate(basis_set.bases):
        if 'cos' in basis.name:
            freq = int(basis.name.split('cos')[1])
        elif 'sin' in basis.name:
            freq = int(basis.name.split('sin')[1])
        else:  # 常数项
            freq = 0
        freqs.append(freq)
        coeff_mags.append(abs(coefficients[i]))
    
    ax4.stem(freqs, coeff_mags, basefmt=" ")
    ax4.set_xlabel('频率')
    ax4.set_ylabel('系数绝对值')
    ax4.set_title('频谱')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_fourier_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_legendre_basis():
    """勒让德多项式基函数示例"""
    print("\n" + "=" * 60)
    print("勒让德多项式基函数示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 添加勒让德多项式基函数
    print("添加勒让德多项式基函数:")
    for order in range(6):  # 0到5阶勒让德多项式
        basis = BasisFactory.legendre(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 创建测试数据（在[-1,1]区间）
    x = np.linspace(-1, 1, 100)
    y_true = np.exp(-x**2)  # 高斯函数
    y = y_true + 0.05 * np.random.randn(len(x))  # 添加少量噪声
    
    data = MultiDimData({0: x})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('勒让德多项式基函数拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数
    ax2 = axes[0, 1]
    x_plot = np.linspace(-1, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=f'P{i}(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P_n(x)')
    ax2.set_title('勒让德多项式')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('多项式阶数')
    ax3.set_ylabel('系数值')
    ax3.set_title('勒让德多项式系数')
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f'n={i}' for i in indices])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 正交性检查
    ax4 = axes[1, 1]
    # 计算Gram矩阵
    Phi = basis_set.evaluate_all(data)
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 绘制Gram矩阵的热图
    im = ax4.imshow(G, cmap='viridis', aspect='auto')
    ax4.set_xlabel('基函数索引')
    ax4.set_ylabel('基函数索引')
    ax4.set_title('Gram矩阵（近似正交）')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_legendre_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_custom_basis():
    """自定义基函数示例"""
    print("\n" + "=" * 60)
    print("自定义基函数示例")
    print("=" * 60)
    
    # 创建基函数集合
    basis_set = BasisSet()
    
    # 定义自定义基函数
    print("添加自定义基函数:")
    
    # 1. 高斯函数
    def gaussian_func(x, center=0.0, sigma=1.0):
        return np.exp(-(x - center)**2 / (2 * sigma**2))
    
    gaussian_basis = BasisFactory.custom(
        dim=0, 
        name="Gaussian_center0_sigma0.2", 
        func=gaussian_func, 
        params={"center": 0.0, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis)
    print(f"  - {gaussian_basis.name}")
    
    # 2. 另一个高斯函数（不同中心）
    gaussian_basis2 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center0.5_sigma0.2",
        func=gaussian_func,
        params={"center": 0.5, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis2)
    print(f"  - {gaussian_basis2.name}")
    
    # 3. 另一个高斯函数（不同中心）
    gaussian_basis3 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center1.0_sigma0.2",
        func=gaussian_func,
        params={"center": 1.0, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis3)
    print(f"  - {gaussian_basis3.name}")
    
    # 4. 线性函数
    def linear_func(x, slope=1.0, intercept=0.0):
        return slope * x + intercept
    
    linear_basis = BasisFactory.custom(
        dim=0,
        name="Linear_slope1",
        func=linear_func,
        params={"slope": 1.0, "intercept": 0.0}
    )
    basis_set.add_basis(linear_basis)
    print(f"  - {linear_basis.name}")
    
    # 5. 常数函数
    def constant_func(x, value=1.0):
        return value
    
    constant_basis = BasisFactory.custom(
        dim=0,
        name="Constant",
        func=constant_func,
        params={"value": 1.0}
    )
    basis_set.add_basis(constant_basis)
    print(f"  - {constant_basis.name}")
    
    # 创建测试数据
    x = np.linspace(0, 1, 100)
    y_true = np.exp(-(x - 0.5)**2 / 0.1)  # 中心在0.5的高斯函数
    y = y_true + 0.05 * np.random.randn(len(x))  # 添加少量噪声
    
    data = MultiDimData({0: x})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建解算器
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解
    print("\n求解中...")
    coefficients = solver.solve()
    print(f"系数: {coefficients}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算误差
    mse = np.mean((y_pred - y)**2)
    print(f"均方误差 (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='数据', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='真实函数')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='拟合')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('自定义基函数拟合')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=basis.name)
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('自定义基函数')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('基函数索引')
    ax3.set_ylabel('系数值')
    ax3.set_title('自定义基函数系数')
    ax3.set_xticks(indices)
    ax3.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 基函数贡献
    ax4 = axes[1, 1]
    x_plot = np.linspace(0, 1, 200)
    y_total = np.zeros_like(x_plot)
    
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        y_contrib = coefficients[i] * y_basis
        ax4.plot(x_plot, y_contrib, '--', alpha=0.7, label=f'{basis.name}贡献')
        y_total += y_contrib
    
    ax4.plot(x_plot, y_total, 'r-', linewidth=2, label='总拟合')
    ax4.set_xlabel('x')
    ax4.set_ylabel('贡献值')
    ax4.set_title('各基函数贡献')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_custom_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

if __name__ == "__main__":
    print("一维基函数示例")
    print("=" * 60)
    
    # 运行所有示例
    results = {}
    
    # 1. 多项式基函数
    coeff_poly, mse_poly = demo_polynomial_basis()
    results['多项式'] = mse_poly
    
    # 2. 傅里叶基函数
    coeff_fourier, mse_fourier = demo_fourier_basis()
    results['傅里叶'] = mse_fourier
    
    # 3. 勒让德多项式
    coeff_legendre, mse_legendre = demo_legendre_basis()
    results['勒让德'] = mse_legendre
    
    # 4. 自定义基函数
    coeff_custom, mse_custom = demo_custom_basis()
    results['自定义'] = mse_custom
    
    # 打印结果比较
    print("\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)
    for method, mse in results.items():
        print(f"{method:10} 基函数: MSE = {mse:.6f}")
    
    print("\n示例完成！")
