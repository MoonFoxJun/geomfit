"""Real-world data demo (simplified)"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # print Unicode math on any console (e.g. GBK Windows)
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
    
    # Generate synthetic regression data
    n_samples = 100
    n_features = 3
    noise = 10.0
    
    print(f"Generating synthetic regression data...")
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to MultiDimData format
    train_data_dict = {i: X_train_scaled[:, i] for i in range(n_features)}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test_scaled[:, i] for i in range(n_features)}
    test_data = MultiDimData(test_data_dict)
    
    # Method 1: polynomial basis
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
    
    # Inverse-transform predictions back to the original scale
    y_train_pred_poly_orig = scaler_y.inverse_transform(y_train_pred_poly.reshape(-1, 1)).flatten()
    y_test_pred_poly_orig = scaler_y.inverse_transform(y_test_pred_poly.reshape(-1, 1)).flatten()
    
    mse_train_poly = np.mean((y_train_pred_poly_orig - y_train)**2)
    mse_test_poly = np.mean((y_test_pred_poly_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_poly:.4f}")
    print(f"  Test MSE: {mse_test_poly:.4f}")
    
    # Method 2: RBF kernel
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
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Test predictions vs true values (polynomial basis)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_poly_orig, alpha=0.6, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Polynomial basis - test set')
    ax1.grid(True, alpha=0.3)
    
    # 2. Test predictions vs true values (RBF kernel)
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_test_pred_rbf_orig, alpha=0.6, s=30)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('True value')
    ax2.set_ylabel('Predicted value')
    ax2.set_title('RBF kernel - test set')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error comparison
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
    
    # 4. Residual distribution
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
    
    # Generate the Friedman #1 dataset
    n_samples = 200
    n_features = 10
    noise = 1.0
    
    print(f"Generating Friedman #1 dataset...")
    X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to MultiDimData format
    train_data_dict = {i: X_train_scaled[:, i] for i in range(n_features)}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test_scaled[:, i] for i in range(n_features)}
    test_data = MultiDimData(test_data_dict)
    
    # Method 1: polynomial basis
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
    
    y_train_pred_poly_orig = scaler_y.inverse_transform(y_train_pred_poly.reshape(-1, 1)).flatten()
    y_test_pred_poly_orig = scaler_y.inverse_transform(y_test_pred_poly.reshape(-1, 1)).flatten()
    
    mse_train_poly = np.mean((y_train_pred_poly_orig - y_train)**2)
    mse_test_poly = np.mean((y_test_pred_poly_orig - y_test)**2)
    
    print(f"  Train MSE: {mse_train_poly:.4f}")
    print(f"  Test MSE: {mse_test_poly:.4f}")
    
    # Method 2: RBF kernel
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
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Test predictions vs true values
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_poly_orig, alpha=0.6, s=30, label='Polynomial basis')
    ax1.scatter(y_test, y_test_pred_rbf_orig, alpha=0.6, s=30, label='RBF kernel')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Friedman dataset - test set predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error comparison
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
    
    # 3. Feature importance
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
    
    # 4. RBF kernel performance vs sigma
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
    
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    print(f"Loading diabetes dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to MultiDimData format
    train_data_dict = {i: X_train[:, i] for i in range(X.shape[1])}
    train_data = MultiDimData(train_data_dict)
    
    test_data_dict = {i: X_test[:, i] for i in range(X.shape[1])}
    test_data = MultiDimData(test_data_dict)
    
    # Method 1: linear basis
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
    
    # Method 2: RBF kernel
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
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Test predictions vs true values
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred_linear, alpha=0.6, s=30, label='Linear basis')
    ax1.scatter(y_test, y_test_pred_rbf, alpha=0.6, s=30, label='RBF kernel')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title('Diabetes dataset - test set predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error comparison
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
    
    # 3. Coefficient distribution
    ax3 = axes[1, 0]
    indices = np.arange(len(coeff_linear))
    ax3.bar(indices, coeff_linear, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Linear basis coefficients')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Residual distribution
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
    
    # Run all demos
    results_summary = {}
    
    # 1. Synthetic regression data
    print("\n1. Synthetic regression demo")
    results_synth = demo_synthetic_regression()
    results_summary.update(results_synth)
    
    # 2. Friedman dataset
    print("\n2. Friedman dataset demo")
    results_friedman = demo_friedman_dataset()
    results_summary.update(results_friedman)
    
    # 3. Diabetes dataset
    print("\n3. Diabetes dataset demo")
    results_diabetes = demo_diabetes_dataset()
    results_summary.update(results_diabetes)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    for method, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        print(f"{method:10}: test MSE = {mse:.4f}")
    
    print("\nDemo complete!")
