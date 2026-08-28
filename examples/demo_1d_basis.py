"""1D basis function demo"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # 把标准输出重设为 UTF-8 编码,确保数学符号(如 φ、Σ)能在任何控制台正常打印(例如 Windows 的 GBK 代码页)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver

def demo_polynomial_basis():
    """Polynomial basis function demo"""
    print("=" * 60)
    print("Polynomial basis function demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器,后续把各个多项式基依次加入
    basis_set = BasisSet()
    
    # 添加 0 到 4 阶的多项式基函数
    print("Adding polynomial basis functions:")
    for order in range(5):  # 循环生成 0..4 阶多项式基:常数、x、x²、x³、x⁴
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 生成一维测试数据:x 在 [0, 1] 上均匀取 100 个点
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * x)  # 真实函数:一个正弦波 sin(2πx)
    y = y_true + 0.1 * np.random.randn(len(x))  # 叠加高斯噪声(标准差 0.1),模拟带噪声的观测
    
    data = MultiDimData({0: x})
    
    # 构造离散内积:在数据点上求和而非连续积分(is_continuous=False)
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建函数求解器,并依次装配基函数、内积与数据
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解:得到使 ‖y − Φc‖² 最小的组合系数 c(其中 Φ 是设计矩阵)
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Coefficients: {coefficients}")
    
    # 预测:用求得的系数在数据点处计算拟合值 ŷ
    y_pred = solver.predict(data)
    
    # 计算均方误差(MSE),衡量拟合质量
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 可视化:2×2 共 4 个子图,分别展示拟合、基函数、系数与残差
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合对比:散点为带噪数据,绿线为真实函数,红线为多项式基拟合结果
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数形状:在 [0,1] 上逐点计算每个多项式基函数并画成曲线
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=f'Order {i}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Polynomial basis functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数柱状图:展示各基函数被赋予的权重(若高阶项系数接近 0,说明数据不需要那么高阶的项)
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Basis coefficients')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 残差图:y − ŷ,理想情况下残差应围绕 0 随机分布、不呈现系统性结构
    ax4 = axes[1, 1]
    residuals = y - y_pred
    ax4.scatter(x, residuals, alpha=0.6, color='purple', s=10)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_polynomial_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_fourier_basis():
    """Fourier basis function demo"""
    print("\n" + "=" * 60)
    print("Fourier basis function demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器
    basis_set = BasisSet()
    
    # 添加傅里叶基函数:常数项 + 不同频率的正弦/余弦
    print("Adding Fourier basis functions:")
    # 常数项:频率 freq=0 的傅里叶基就是常数函数,对应信号中的直流分量
    bases = BasisFactory.fourier(dim=0, freq=0, L=1.0)
    for basis in bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 依次添加频率 1 到 3 的傅里叶基(每个频率包含 sin 和 cos 两个分量)
    for freq in range(1, 4):
        bases = BasisFactory.fourier(dim=0, freq=freq, L=1.0)
        for basis in bases:
            basis_set.add_basis(basis)
            print(f"  - {basis.name}")
    
    # 生成周期性测试数据:真实函数是两个周期分量的叠加,正好落在傅里叶基张成的空间内
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * 2 * x) + 0.5 * np.cos(2 * np.pi * 3 * x)  # 真实函数:2 倍频正弦 + 3 倍频余弦的叠加
    y = y_true + 0.05 * np.random.randn(len(x))  # 叠加少量噪声(标准差 0.05)
    
    data = MultiDimData({0: x})
    
    # 构造离散内积(基于数据点求和)
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配基函数、内积与数据
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Fourier basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数形状:只画前 6 个基函数,避免图例过密
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases[:6]):  # 只取前 6 个基函数用于绘图
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=basis.name)
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Fourier basis functions (first 6)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数柱状图:每个基函数(按 sin/cos 与频率命名)对应一个系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Fourier basis coefficients')
    ax3.set_xticks(indices)
    ax3.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 频谱图:把系数按频率归类,观察信号能量在各频率上的分布
    ax4 = axes[1, 1]
    # 从基函数名称中解析出频率,并取系数的绝对值作为该频率分量的幅度
    freqs = []
    coeff_mags = []
    for i, basis in enumerate(basis_set.bases):
        if 'cos' in basis.name:
            freq = int(basis.name.split('cos')[1])
        elif 'sin' in basis.name:
            freq = int(basis.name.split('sin')[1])
        else:  # 常数项:名称中不含 sin/cos,频率记为 0
            freq = 0
        freqs.append(freq)
        coeff_mags.append(abs(coefficients[i]))
    
    ax4.stem(freqs, coeff_mags, basefmt=" ")
    ax4.set_xlabel('Frequency')
    ax4.set_ylabel('|Coefficient|')
    ax4.set_title('Spectrum')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_fourier_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_legendre_basis():
    """Legendre polynomial basis demo"""
    print("\n" + "=" * 60)
    print("Legendre polynomial basis demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器
    basis_set = BasisSet()
    
    # 添加 0 到 5 阶的勒让德多项式基函数
    print("Adding Legendre polynomial basis functions:")
    for order in range(6):  # 勒让德多项式阶数 0..5
        basis = BasisFactory.legendre(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # 测试数据定义在 [-1, 1] 上:这是勒让德多项式的自然定义域,正交性在该区间上成立
    x = np.linspace(-1, 1, 100)
    y_true = np.exp(-x**2)  # 真实函数:高斯钟形曲线 exp(−x²)
    y = y_true + 0.05 * np.random.randn(len(x))  # 叠加少量噪声(标准差 0.05)
    
    data = MultiDimData({0: x})
    
    # 构造离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Legendre polynomial basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数形状:勒让德多项式 P₀(x)..P₅(x) 的曲线
    ax2 = axes[0, 1]
    x_plot = np.linspace(-1, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=f'P{i}(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P_n(x)')
    ax2.set_title('Legendre polynomials')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数柱状图:按多项式阶数 n 展示各系数
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Polynomial order')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Legendre polynomial coefficients')
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f'n={i}' for i in indices])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 正交性检验:计算 Gram 矩阵并用热图展示
    ax4 = axes[1, 1]
    # 先求设计矩阵 Φ(每列是一个基函数在所有数据点上的取值),再计算 Gram 矩阵 G = ΦᵀΦ
    Phi = basis_set.evaluate_all(data)
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Gram 矩阵热图:若基函数两两正交,非对角元应接近 0,矩阵近似为对角阵
    im = ax4.imshow(G, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Basis index')
    ax4.set_ylabel('Basis index')
    ax4.set_title('Gram matrix (near-orthogonal)')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_legendre_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

def demo_custom_basis():
    """Custom basis function demo"""
    print("\n" + "=" * 60)
    print("Custom basis function demo")
    print("=" * 60)
    
    # 创建一个空的基函数集合容器
    basis_set = BasisSet()
    
    # 定义并添加自定义基函数:高斯型、线性、常数等,演示工厂方法的灵活性
    print("Adding custom basis functions:")
    
    # 1. 高斯基函数:以 center 为中心、sigma 为宽度的高斯钟形曲线
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
    
    # 2. 中心移到 0.5 处的高斯基函数
    gaussian_basis2 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center0.5_sigma0.2",
        func=gaussian_func,
        params={"center": 0.5, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis2)
    print(f"  - {gaussian_basis2.name}")
    
    # 3. 中心移到 1.0 处的高斯基函数(三个高斯基共同覆盖 [0, 1])
    gaussian_basis3 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center1.0_sigma0.2",
        func=gaussian_func,
        params={"center": 1.0, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis3)
    print(f"  - {gaussian_basis3.name}")
    
    # 4. 线性基函数:slope·x + intercept
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
    
    # 5. 常数基函数:恒等于 value,用于拟合截距/偏置
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
    
    # 生成测试数据:真实函数是中心在 0.5 的窄高斯峰,与三个高斯基的形状匹配
    x = np.linspace(0, 1, 100)
    y_true = np.exp(-(x - 0.5)**2 / 0.1)  # 真实函数:中心在 0.5 的高斯峰
    y = y_true + 0.05 * np.random.randn(len(x))  # 叠加少量噪声(标准差 0.05)
    
    data = MultiDimData({0: x})
    
    # 构造离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建求解器并装配
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # 求解组合系数
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Coefficients: {coefficients}")
    
    # 预测
    y_pred = solver.predict(data)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 数据拟合对比
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Custom basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 基函数形状:展示所有自定义基函数(三个高斯 + 线性 + 常数)
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=basis.name)
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Custom basis functions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 系数柱状图:观察每个自定义基函数被赋予的权重
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    basis_names = [basis.name for basis in basis_set.bases]
    bars = ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Custom basis coefficients')
    ax3.set_xticks(indices)
    ax3.set_xticklabels(basis_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 各基函数贡献:绘制每一项 cᵢ·φᵢ(x) 的虚线,以及它们的和(总拟合)
    ax4 = axes[1, 1]
    x_plot = np.linspace(0, 1, 200)
    y_total = np.zeros_like(x_plot)
    
    for i, basis in enumerate(basis_set.bases):
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        y_contrib = coefficients[i] * y_basis
        ax4.plot(x_plot, y_contrib, '--', alpha=0.7, label=f'{basis.name} contribution')
        y_total += y_contrib
    
    ax4.plot(x_plot, y_total, 'r-', linewidth=2, label='Total fit')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Contribution')
    ax4.set_title('Individual basis contributions')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_custom_basis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return coefficients, mse

if __name__ == "__main__":
    print("1D Basis Function Demo")
    print("=" * 60)
    
    # 依次运行全部四个演示,并把每个方法的 MSE 记录到字典中,便于最后对比
    results = {}
    
    # 1. 多项式基演示
    coeff_poly, mse_poly = demo_polynomial_basis()
    results['Polynomial'] = mse_poly
    
    # 2. 傅里叶基演示
    coeff_fourier, mse_fourier = demo_fourier_basis()
    results['Fourier'] = mse_fourier
    
    # 3. 勒让德多项式演示
    coeff_legendre, mse_legendre = demo_legendre_basis()
    results['Legendre'] = mse_legendre
    
    # 4. 自定义基演示
    coeff_custom, mse_custom = demo_custom_basis()
    results['Custom'] = mse_custom
    
    # 打印各基函数方法的 MSE 对比表
    print("\n" + "=" * 60)
    print("Results comparison")
    print("=" * 60)
    for method, mse in results.items():
        print(f"{method:10} basis: MSE = {mse:.6f}")
    
    print("\nDemo complete!")
