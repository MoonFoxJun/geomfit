"""Real-world data demo (simplified)"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # 把标准输出重设为 UTF-8 编码,确保中文/数学符号能在任何控制台正常打印(例如 Windows 的 GBK 代码页)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_friedman1, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.kernel.rbf import RBFKernel
from geomfit.kernel.polynomial import PolynomialKernel
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver

def demo_synthetic_regression():
    """Synthetic regression data demo"""
    print("=" * 60)
    print("Synthetic regression data demo")
    print("=" * 60)
    
    # 生成合成回归数据:100 个样本、3 个特征、噪声强度 10
    n_samples = 100
    n_features = 3
    noise = 10.0
    
    print(f"Generating synthetic regression data...")
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 按 8:2 划分训练/测试集(固定随机种子保证可复现)
    
    # 标准化数据:在训练集上拟合均值/方差,测试集用同一组参数变换(避免数据泄漏)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 目标值同样做标准化(便于数值稳定),预测后再反变换回原始量纲
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 把特征数组转成 {维度: 取值数组} 的 MultiDimData 格式
    train_data_dict = {i: X_train_scaled[:, i] for i in range(n_features)}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test_scaled[:, i] for i in range(n_features)}
    test_data = MultiDimData(test_data_dict)
    
    # 方法 1:多项式基 —— 每个维度加 0/1/2 阶多项式基(即常数 + 线性 + 二次项)
    print("\nMethod 1: Polynomial basis")
    basis_set_poly = BasisSet()
    for dim in range(n_features):
        basis_set_poly.add_basis(BasisFactory.polynomial(dim=dim, order=0))
        basis_set_poly.add_basis(BasisFactory.polynomial(dim=dim, order=1))
        basis_set_poly.add_basis(BasisFactory.polynomial(dim=dim, order=2))
    
    inner_product = InnerProduct(is_continuous=False)
    solver_poly = FunctionalSolver()
    solver_poly.set_basis(basis_set_poly)
    solver_poly.set_inner_product(inner_product)
    solver_poly.load_data(train_data, y_train_scaled)
    
    coeff_poly = solver_poly.solve()
    y_train_pred_poly = solver_poly.predict(train_data)
    y_test_pred_poly = solver_poly.predict(test_data)
    
    # 把标准化尺度上的预测反变换回原始量纲,再计算 MSE(这样数值可直接与数据对比)
    y_train_pred_poly_orig = scaler_y.inverse_transform(y_train_pred_poly.reshape(-1, 1)).flatten()
    y_test_pred_poly_orig = scaler_y.inverse_transform(y_test_pred_poly.reshape(-1, 1)).flatten()
    
    mse_train_poly = np.mean((y_train_pred_poly_orig - y_train)**2)
    mse_test_poly = np.mean((y_test_pred_poly_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_poly:.4f}")
    print(f"  Test MSE: {mse_test_poly:.4f}")
    
    # 方法 2:RBF 核方法(核方法不需要显式基函数,σ=1.0 控制核宽度)
    print("\nMethod 2: RBF kernel")
    kernel_rbf = RBFKernel(sigma=1.0)
    solver_rbf = FunctionalSolver()
    solver_rbf.set_kernel(kernel_rbf)
    solver_rbf.load_data(train_data, y_train_scaled)
    
    coeff_rbf = solver_rbf.solve()
    y_train_pred_rbf = solver_rbf.predict(train_data)
    y_test_pred_rbf = solver_rbf.predict(test_data)
    
    y_train_pred_rbf_orig = scaler_y.inverse_transform(y_train_pred_rbf.reshape(-1, 1)).flatten()
    y_test_pred_rbf_orig = scaler_y.inverse_transform(y_test_pred_rbf.reshape(-1, 1)).flatten()
    
    mse_train_rbf = np.mean((y_train_pred_rbf_orig - y_train)**2)
    mse_test_rbf = np.mean((y_test_pred_rbf_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_rbf:.4f}")
    print(f"  Test MSE: {mse_test_rbf:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 多项式基:测试集上预测值 vs 真值(红色虚线为完美预测 y=x)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_poly_orig, alpha=0.6, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Polynomial basis - test set')
    ax1.grid(True, alpha=0.3)
    
    # 2. RBF 核:同样画预测值 vs 真值
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_test_pred_rbf_orig, alpha=0.6, s=30)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('True value')
    ax2.set_ylabel('Predicted value')
    ax2.set_title('RBF kernel - test set')
    ax2.grid(True, alpha=0.3)
    
    # 3. 测试集误差对比柱状图
    ax3 = axes[1, 0]
    methods = ['Polynomial basis', 'RBF kernel']
    test_mses = [mse_test_poly, mse_test_rbf]
    bars = ax3.bar(methods, test_mses, alpha=0.7, color=['blue', 'green'])
    ax3.set_ylabel('Test MSE')
    ax3.set_title('Test error by method')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, mse in zip(bars, test_mses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{mse:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 4. 残差分布直方图:两种方法的残差都应围绕 0 分布
    ax4 = axes[1, 1]
    residuals_poly = y_test - y_test_pred_poly_orig
    residuals_rbf = y_test - y_test_pred_rbf_orig
    ax4.hist(residuals_poly, bins=20, alpha=0.5, color='blue', label='Polynomial basis', edgecolor='black')
    ax4.hist(residuals_rbf, bins=20, alpha=0.5, color='green', label='RBF kernel', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Residual')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual distribution comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_synthetic_regression.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'Polynomial basis': mse_test_poly, 'RBF kernel': mse_test_rbf}

def demo_friedman_dataset():
    """Friedman dataset demo"""
    print("\n" + "=" * 60)
    print("Friedman dataset demo")
    print("=" * 60)
    
    # 生成 Friedman #1 数据集:200 个样本、10 个特征,但真函数只依赖其中 5 个特征
    n_samples = 200
    n_features = 10
    noise = 1.0
    
    print(f"Generating Friedman #1 dataset...")
    X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 8:2 划分训练/测试集
    
    # 标准化数据(特征与目标各自用训练集统计量)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 转成 MultiDimData 格式
    train_data_dict = {i: X_train_scaled[:, i] for i in range(n_features)}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test_scaled[:, i] for i in range(n_features)}
    test_data = MultiDimData(test_data_dict)
    
    # 方法 1:多项式基(每维只取 0/1 阶,即线性模型)
    print("\nMethod 1: Polynomial basis")
    basis_set_poly = BasisSet()
    for dim in range(n_features):
        basis_set_poly.add_basis(BasisFactory.polynomial(dim=dim, order=0))
        basis_set_poly.add_basis(BasisFactory.polynomial(dim=dim, order=1))
    
    inner_product = InnerProduct(is_continuous=False)
    solver_poly = FunctionalSolver()
    solver_poly.set_basis(basis_set_poly)
    solver_poly.set_inner_product(inner_product)
    solver_poly.load_data(train_data, y_train_scaled)
    
    coeff_poly = solver_poly.solve()
    y_train_pred_poly = solver_poly.predict(train_data)
    y_test_pred_poly = solver_poly.predict(test_data)
    
    # 反变换回原始量纲
    y_train_pred_poly_orig = scaler_y.inverse_transform(y_train_pred_poly.reshape(-1, 1)).flatten()
    y_test_pred_poly_orig = scaler_y.inverse_transform(y_test_pred_poly.reshape(-1, 1)).flatten()
    
    mse_train_poly = np.mean((y_train_pred_poly_orig - y_train)**2)
    mse_test_poly = np.mean((y_test_pred_poly_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_poly:.4f}")
    print(f"  Test MSE: {mse_test_poly:.4f}")
    
    # 方法 2:RBF 核
    print("\nMethod 2: RBF kernel")
    kernel_rbf = RBFKernel(sigma=1.0)
    solver_rbf = FunctionalSolver()
    solver_rbf.set_kernel(kernel_rbf)
    solver_rbf.load_data(train_data, y_train_scaled)
    
    coeff_rbf = solver_rbf.solve()
    y_train_pred_rbf = solver_rbf.predict(train_data)
    y_test_pred_rbf = solver_rbf.predict(test_data)
    
    y_train_pred_rbf_orig = scaler_y.inverse_transform(y_train_pred_rbf.reshape(-1, 1)).flatten()
    y_test_pred_rbf_orig = scaler_y.inverse_transform(y_test_pred_rbf.reshape(-1, 1)).flatten()
    
    mse_train_rbf = np.mean((y_train_pred_rbf_orig - y_train)**2)
    mse_test_rbf = np.mean((y_test_pred_rbf_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_rbf:.4f}")
    print(f"  Test MSE: {mse_test_rbf:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 测试集预测值 vs 真值:两种方法画在同一张图上对比
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_poly_orig, alpha=0.6, s=30, label='Polynomial basis')
    ax1.scatter(y_test, y_test_pred_rbf_orig, alpha=0.6, s=30, label='RBF kernel')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Friedman dataset - test set predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差对比柱状图
    ax2 = axes[0, 1]
    methods = ['Polynomial basis', 'RBF kernel']
    test_mses = [mse_test_poly, mse_test_rbf]
    bars = ax2.bar(methods, test_mses, alpha=0.7, color=['blue', 'green'])
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Friedman dataset - error comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mse in zip(bars, test_mses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{mse:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 特征重要性:取一阶(线性)基的系数绝对值作为各特征重要性的代理指标
    ax3 = axes[1, 0]
    first_order_coeffs = []
    for i, basis in enumerate(basis_set_poly.bases):
        if 'order1' in basis.name and 'Polynomial' in basis.name:
            first_order_coeffs.append(abs(coeff_poly[i]))
    
    feature_names = [f'x{i+1}' for i in range(min(10, len(first_order_coeffs)))]
    first_order_coeffs = first_order_coeffs[:10]
    
    ax3.bar(feature_names, first_order_coeffs, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Feature')
    ax3.set_ylabel('|Coefficient|')
    ax3.set_title('Polynomial basis feature importance')
    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. RBF 核参数调优:σ 从 0.1 到 10 对数取值,观察测试误差的变化(存在最优 σ)
    ax4 = axes[1, 1]
    sigma_values = np.logspace(-1, 1, 10)
    test_mses_sigma = []
    
    for sigma in sigma_values:
        kernel_temp = RBFKernel(sigma=sigma)
        solver_temp = FunctionalSolver()
        solver_temp.set_kernel(kernel_temp)
        solver_temp.load_data(train_data, y_train_scaled)
        solver_temp.solve()
        
        y_test_pred_temp = solver_temp.predict(test_data)
        y_test_pred_temp_orig = scaler_y.inverse_transform(y_test_pred_temp.reshape(-1, 1)).flatten()
        test_mses_sigma.append(np.mean((y_test_pred_temp_orig - y_test)**2))
    
    ax4.semilogx(sigma_values, test_mses_sigma, 'ro-', linewidth=2, markersize=6)
    ax4.set_xlabel('RBF kernel parameter sigma')
    ax4.set_ylabel('Test MSE')
    ax4.set_title('RBF kernel parameter tuning')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_friedman_dataset.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'Polynomial basis': mse_test_poly, 'RBF kernel': mse_test_rbf}

def demo_diabetes_dataset():
    """Diabetes dataset demo"""
    print("\n" + "=" * 60)
    print("Diabetes dataset demo")
    print("=" * 60)
    
    # 载入 sklearn 内置的糖尿病数据集(442 个样本、10 个特征,目标为病情进展指标)
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    print(f"Loading diabetes dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 8:2 划分训练/测试集
    
    # 转成 MultiDimData 格式(该数据集本身已标准化,无需再缩放)
    train_data_dict = {i: X_train[:, i] for i in range(X.shape[1])}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test[:, i] for i in range(X.shape[1])}
    test_data = MultiDimData(test_data_dict)
    
    # 方法 1:线性基(每个维度加入常数项 + 一阶项)
    print("\nMethod 1: Linear basis")
    basis_set_linear = BasisSet()
    for dim in range(X.shape[1]):
        basis_set_linear.add_basis(BasisFactory.polynomial(dim=dim, order=0))
        basis_set_linear.add_basis(BasisFactory.polynomial(dim=dim, order=1))
    
    inner_product = InnerProduct(is_continuous=False)
    solver_linear = FunctionalSolver()
    solver_linear.set_basis(basis_set_linear)
    solver_linear.set_inner_product(inner_product)
    solver_linear.load_data(train_data, y_train)
    
    coeff_linear = solver_linear.solve()
    y_train_pred_linear = solver_linear.predict(train_data)
    y_test_pred_linear = solver_linear.predict(test_data)
    
    mse_train_linear = np.mean((y_train_pred_linear - y_train)**2)
    mse_test_linear = np.mean((y_test_pred_linear - y_test)**2)
    
    print(f"  Train MSE: {mse_train_linear:.4f}")
    print(f"  Test MSE: {mse_test_linear:.4f}")
    
    # 方法 2:RBF 核(σ=0.5)
    print("\nMethod 2: RBF kernel")
    kernel_rbf = RBFKernel(sigma=0.5)
    solver_rbf = FunctionalSolver()
    solver_rbf.set_kernel(kernel_rbf)
    solver_rbf.load_data(train_data, y_train)
    
    coeff_rbf = solver_rbf.solve()
    y_train_pred_rbf = solver_rbf.predict(train_data)
    y_test_pred_rbf = solver_rbf.predict(test_data)
    
    mse_train_rbf = np.mean((y_train_pred_rbf - y_train)**2)
    mse_test_rbf = np.mean((y_test_pred_rbf - y_test)**2)
    
    print(f"  Train MSE: {mse_train_rbf:.4f}")
    print(f"  Test MSE: {mse_test_rbf:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 测试集预测值 vs 真值
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_linear, alpha=0.6, s=30, label='Linear basis')
    ax1.scatter(y_test, y_test_pred_rbf, alpha=0.6, s=30, label='RBF kernel')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Diabetes dataset - test set predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 误差对比柱状图
    ax2 = axes[0, 1]
    methods = ['Linear basis', 'RBF kernel']
    test_mses = [mse_test_linear, mse_test_rbf]
    bars = ax2.bar(methods, test_mses, alpha=0.7, color=['blue', 'green'])
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Diabetes dataset - error comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mse in zip(bars, test_mses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{mse:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 线性基系数分布:正负系数对应各特征的正/负影响
    ax3 = axes[1, 0]
    indices = np.arange(len(coeff_linear))
    ax3.bar(indices, coeff_linear, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Linear basis coefficients')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 残差分布直方图
    ax4 = axes[1, 1]
    residuals_linear = y_test - y_test_pred_linear
    residuals_rbf = y_test - y_test_pred_rbf
    ax4.hist(residuals_linear, bins=20, alpha=0.5, color='blue', label='Linear basis', edgecolor='black')
    ax4.hist(residuals_rbf, bins=20, alpha=0.5, color='green', label='RBF kernel', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Residual')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual distribution comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_diabetes_dataset.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'Linear basis': mse_test_linear, 'RBF kernel': mse_test_rbf}

if __name__ == "__main__":
    print("Real-World Data Demo")
    print("=" * 60)
    
    # 依次运行三个数据集演示,并把各方法的测试 MSE 汇总到字典中
    results_summary = {}
    
    # 1. 合成回归数据
    print("\n1. Synthetic regression demo")
    results_synth = demo_synthetic_regression()
    results_summary.update(results_synth)
    
    # 2. Friedman 数据集
    print("\n2. Friedman dataset demo")
    results_friedman = demo_friedman_dataset()
    results_summary.update(results_friedman)
    
    # 3. 糖尿病数据集
    print("\n3. Diabetes dataset demo")
    results_diabetes = demo_diabetes_dataset()
    results_summary.update(results_diabetes)
    
    # 按测试 MSE 升序打印汇总
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    for method, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        print(f"{method:10}: test MSE = {mse:.4f}")
    
    print("\nDemo complete!")
