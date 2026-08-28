"""RKHS kernel demo"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    
    # Create an RBF kernel
    sigma = 0.2
    kernel = RBFKernel(sigma=sigma)
    print(f"Created RBF kernel with sigma={sigma}")
    
    # Generate test data
    n_points = 30
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)  # mixed sine/cosine waves
    y = y_true + 0.1 * np.random.randn(n_points)  # add noise
    
    data = MultiDimData({0: x})
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Compute the kernel matrix
    K = kernel.compute_matrix(data)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='RBF fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'RBF kernel fit (sigma={sigma})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kernel matrix heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('RBF kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. Kernel values
    ax3 = axes[0, 2]
    x_test = 0.5  # test point
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
    
    # 4. Coefficients
    ax4 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax4.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('Data point index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('RBF kernel coefficients')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Residuals
    ax5 = axes[1, 1]
    residuals = y - y_pred
    ax5.scatter(x, residuals, alpha=0.6, color='purple', s=20)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('Residual')
    ax5.set_title('Residuals')
    ax5.grid(True, alpha=0.3)
    
    # 6. Eigenvalue spectrum
    ax6 = axes[1, 2]
    eigenvalues = np.linalg.eigvalsh(K)  # symmetric matrix: use eigvalsh
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
    
    # Create a polynomial kernel
    degree = 3
    c = 1.0
    kernel = PolynomialKernel(degree=degree, c=c)
    print(f"Created polynomial kernel with degree={degree}, c={c}")
    
    # Generate test data
    n_points = 25
    x = np.linspace(-1, 1, n_points)
    y_true = x**3 - 2*x**2 + 0.5*x + 1  # cubic polynomial
    y = y_true + 0.05 * np.random.randn(n_points)  # add a small amount of noise
    
    data = MultiDimData({0: x})
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_kernel(kernel)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Compute the kernel matrix
    K = kernel.compute_matrix(data)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Polynomial kernel fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Polynomial kernel fit (degree={degree})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kernel matrix heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='plasma', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('Polynomial kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. Kernel values
    ax3 = axes[0, 2]
    x_test = 0.0  # test point
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
    
    # 4. Comparison across degrees
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
    
    # 5. Coefficient distribution
    ax5 = axes[1, 1]
    indices = np.arange(len(coefficients))
    ax5.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax5.set_xlabel('Data point index')
    ax5.set_ylabel('Coefficient')
    ax5.set_title('Polynomial kernel coefficients')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Eigenvalue spectrum
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
    
    # Create Matérn kernels with different smoothness parameters nu
    kernels = []
    nu_values = [0.5, 1.5, 2.5, float('inf')]  # different smoothness levels
    kernel_names = ['Matern ν=1/2', 'Matern ν=3/2', 'Matern ν=5/2', 'RBF (ν→∞)']
    
    for nu in nu_values:
        if nu == float('inf'):
            kernel = RBFKernel(sigma=0.3)  # RBF is the Matérn limit as ν → ∞
        else:
            kernel = MaternKernel(sigma=0.3, nu=nu)
        kernels.append(kernel)
    
    # Generate test data
    n_points = 40
    x = np.linspace(0, 2, n_points)
    y_true = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(n_points)  # noisy sine wave
    
    data = MultiDimData({0: x})
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\nFitting with {name} kernel...")
        
        # Create a solver
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y_true)
        
        # Solve
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # Compute the error
        mse = np.mean((y_pred - y_true)**2)
        results.append((name, mse, y_pred))
        print(f"  Mean squared error (MSE): {mse:.6f}")
        
        # 1. Fits from different kernels
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
    
    # 5. Kernel value comparison
    ax5 = axes[1, 2]
    x_test = 1.0  # test point
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
    
    # 6. Error comparison
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
    
    # Annotate bars with values
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
    
    # Base kernels
    rbf_kernel = RBFKernel(sigma=0.3)
    poly_kernel = PolynomialKernel(degree=2, c=1.0)
    matern_kernel = MaternKernel(sigma=0.3, nu=1.5)
    
    # Build composite kernels
    print("Building composite kernels...")
    
    # 1. Additive composite kernel
    additive_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='add')
    print("  - Additive: RBF + Polynomial")
    
    # 2. Multiplicative composite kernel
    multiplicative_kernel = CompositeKernel([rbf_kernel, poly_kernel], operation='multiply')
    print("  - Multiplicative: RBF × Polynomial")
    
    kernels = [rbf_kernel, poly_kernel, additive_kernel, multiplicative_kernel]
    kernel_names = ['RBF', 'Polynomial', 'RBF+Polynomial', 'RBF×Polynomial']
    
    # Generate test data
    n_points = 30
    x = np.linspace(0, 1, n_points)
    # Target: polynomial + periodic + localized features
    y_true = 0.5*x**2 + np.sin(2*np.pi*x) + np.exp(-(x-0.7)**2/0.02)
    y = y_true + 0.05 * np.random.randn(n_points)  # add a small amount of noise
    
    data = MultiDimData({0: x})
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    results = []
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        print(f"\nFitting with {name} kernel...")
        
        # Create a solver
        solver = FunctionalSolver()
        solver.set_kernel(kernel)
        solver.load_data(data, y)
        
        # Solve
        coefficients = solver.solve()
        y_pred = solver.predict(data)
        
        # Compute the error
        mse = np.mean((y_pred - y)**2)
        results.append((name, mse, y_pred))
        print(f"  Mean squared error (MSE): {mse:.6f}")
        
        # 1. Fit result
        ax = axes[idx]
        ax.scatter(x, y, alpha=0.5, label='Data', s=10)
        ax.plot(x, y_true, 'g-', linewidth=2, label='True function', alpha=0.7)
        ax.plot(x, y_pred, 'r-', linewidth=2, label=f'{name} fit')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name} kernel fit')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Kernel matrix
        K = kernel.compute_matrix(data)
        print(f"  Kernel matrix condition number: {np.linalg.cond(K):.2e}")
    
    # 6. Error comparison
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
    
    # Annotate bars with values
    for bar, mse in zip(bars, mse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_composite_kernel.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional plot: kernel value comparison
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    x_test = 0.5  # test point
    x_range = np.linspace(0, 1, 100)
    
    for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if idx >= 6:  # show at most 6 panels
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
    
    # RBF kernel
    kernel = RBFKernel(sigma=0.2)
    
    # Kernel solver
    solver = KernelSolver(kernel)
    print("Creating kernel solver...")
    
    # Generate test data
    n_points = 20
    x = np.linspace(0, 1, n_points)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(n_points)
    
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Kernel matrix
    K = solver.compute_kernel_matrix()
    print(f"Kernel matrix shape: {K.shape}")
    
    # Eigenvalues
    eigenvalues = solver.get_eigenvalues()
    print(f"Number of eigenvalues: {len(eigenvalues)}")
    print(f"Largest eigenvalue: {eigenvalues[0]:.4f}")
    print(f"Smallest eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Condition number: {solver.get_condition_number():.2e}")
    
    # Function norm
    norm = solver.compute_function_norm()
    print(f"Function norm: {norm:.4f}")
    
    # Solve
    coefficients = solver.solve()
    
    # Predict
    y_pred = solver.predict(data)
    
    # Efficient prediction
    y_pred_efficient = solver.predict_efficient(data)
    
    # Errors
    mse = np.mean((y_pred - y)**2)
    mse_efficient = np.mean((y_pred_efficient - y)**2)
    print(f"Standard prediction MSE: {mse:.6f}")
    print(f"Efficient prediction MSE: {mse_efficient:.6f}")
    print(f"Predictions agree: {np.allclose(y_pred, y_pred_efficient, rtol=1e-10)}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=20)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function', alpha=0.7)
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Kernel method fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Kernel method fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kernel matrix heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(K, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Data point index')
    ax2.set_ylabel('Data point index')
    ax2.set_title('Kernel matrix')
    plt.colorbar(im2, ax=ax2, shrink=0.7, aspect=10)
    
    # 3. Eigenvalue spectrum
    ax3 = axes[0, 2]
    indices = np.arange(1, len(eigenvalues) + 1)
    ax3.semilogy(indices, eigenvalues, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Eigenvalue index')
    ax3.set_ylabel('Eigenvalue (log scale)')
    ax3.set_title('Kernel matrix eigenvalue spectrum')
    ax3.grid(True, alpha=0.3)
    
    # 4. Coefficients
    ax4 = axes[1, 0]
    indices_coeff = np.arange(len(coefficients))
    ax4.bar(indices_coeff, coefficients, alpha=0.7, color='steelblue')
    ax4.set_xlabel('Data point index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('Kernel method coefficients')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Leave-one-out error estimate
    ax5 = axes[1, 1]
    loo_errors = []
    
    # Leave-one-out error for each point
    for i in range(min(10, n_points)):  # first 10 points to save time
        # Training data without point i
        mask = np.ones(n_points, dtype=bool)
        mask[i] = False
        
        x_train = x[mask]
        y_train = y[mask]
        x_test = x[i]
        y_test = y[i]
        
        train_data = MultiDimData({0: x_train})
        test_data = MultiDimData({0: np.array([x_test])})
        
        # Train a fresh solver
        temp_solver = KernelSolver(kernel)
        temp_solver.load_data(train_data, y_train)
        temp_solver.solve()
        
        # Predict the held-out point
        y_pred_test = temp_solver.predict(test_data)[0]
        loo_errors.append(abs(y_pred_test - y_test))
    
    ax5.plot(range(len(loo_errors)), loo_errors, 'ro-', linewidth=2, markersize=6)
    ax5.set_xlabel('Data point index')
    ax5.set_ylabel('Leave-one-out error')
    ax5.set_title('Leave-one-out errors (first 10 points)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Effect of regularization
    ax6 = axes[1, 2]
    alphas = np.logspace(-6, 0, 10)  # regularization parameter
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
    
    # Merged legend
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
    
    # Run all demos
    results_summary = {}
    
    # 1. RBF kernel
    print("\n1. RBF kernel demo")
    coeff_rbf, mse_rbf, K_rbf = demo_rbf_kernel()
    results_summary['RBF'] = mse_rbf
    
    # 2. Polynomial kernel
    print("\n2. Polynomial kernel demo")
    coeff_poly, mse_poly, K_poly = demo_polynomial_kernel()
    results_summary['Polynomial'] = mse_poly
    
    # 3. Matérn kernel
    print("\n3. Matérn kernel demo")
    results_matern = demo_matern_kernel()
    for name, mse, _ in results_matern:
        results_summary[name] = mse
    
    # 4. Composite kernel
    print("\n4. Composite kernel demo")
    results_composite = demo_composite_kernel()
    for name, mse, _ in results_composite:
        results_summary[name] = mse
    
    # 5. Kernel solver features
    print("\n5. Kernel solver features demo")
    solver, coeff_kernel, eigenvalues = demo_kernel_solver_features()
    results_summary['Kernel solver'] = np.mean((solver.predict(MultiDimData({0: np.linspace(0, 1, 20)})) - 
                                          np.sin(2*np.pi*np.linspace(0, 1, 20)))**2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    for method, mse in sorted(results_summary.items(), key=lambda x: x[1]):
        print(f"{method:15}: MSE = {mse:.6f}")
    
    print("\nDemo complete!")
