"""1D basis function demo"""

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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # Add polynomial basis functions
    print("Adding polynomial basis functions:")
    for order in range(5):  # polynomials of order 0..4
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # Generate test data
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * x)  # true function: sine wave
    y = y_true + 0.1 * np.random.randn(len(x))  # add noise
    
    data = MultiDimData({0: x})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Coefficients: {coefficients}")
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Basis functions
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
    
    # 3. Coefficients
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Basis coefficients')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Residuals
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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # Add Fourier basis functions
    print("Adding Fourier basis functions:")
    # Constant term
    bases = BasisFactory.fourier(dim=0, freq=0, L=1.0)
    for basis in bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # Fourier bases for frequencies 1..3
    for freq in range(1, 4):
        bases = BasisFactory.fourier(dim=0, freq=freq, L=1.0)
        for basis in bases:
            basis_set.add_basis(basis)
            print(f"  - {basis.name}")
    
    # Periodic test data
    x = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * 2 * x) + 0.5 * np.cos(2 * np.pi * 3 * x)  # mixed sine/cosine waves
    y = y_true + 0.05 * np.random.randn(len(x))  # add a small amount of noise
    
    data = MultiDimData({0: x})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Fourier basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Basis functions (first few)
    ax2 = axes[0, 1]
    x_plot = np.linspace(0, 1, 200)
    for i, basis in enumerate(basis_set.bases[:6]):  # show the first 6 only
        y_basis = np.zeros_like(x_plot)
        for j, x_val in enumerate(x_plot):
            y_basis[j] = basis.func(x_val, **basis.params)
        ax2.plot(x_plot, y_basis, label=basis.name)
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Fourier basis functions (first 6)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficients
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
    
    # 4. Spectrum
    ax4 = axes[1, 1]
    # Extract frequencies and coefficient magnitudes
    freqs = []
    coeff_mags = []
    for i, basis in enumerate(basis_set.bases):
        if 'cos' in basis.name:
            freq = int(basis.name.split('cos')[1])
        elif 'sin' in basis.name:
            freq = int(basis.name.split('sin')[1])
        else:  # constant term
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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # Add Legendre polynomial basis functions
    print("Adding Legendre polynomial basis functions:")
    for order in range(6):  # Legendre polynomials of order 0..5
        basis = BasisFactory.legendre(dim=0, order=order)
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # Test data on [-1, 1]
    x = np.linspace(-1, 1, 100)
    y_true = np.exp(-x**2)  # Gaussian function
    y = y_true + 0.05 * np.random.randn(len(x))  # add a small amount of noise
    
    data = MultiDimData({0: x})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Legendre polynomial basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Basis functions
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
    
    # 3. Coefficients
    ax3 = axes[1, 0]
    indices = np.arange(len(coefficients))
    ax3.bar(indices, coefficients, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Polynomial order')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Legendre polynomial coefficients')
    ax3.set_xticks(indices)
    ax3.set_xticklabels([f'n={i}' for i in indices])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Orthogonality check
    ax4 = axes[1, 1]
    # Compute the Gram matrix
    Phi = basis_set.evaluate_all(data)
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Heatmap of the Gram matrix
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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # Define custom basis functions
    print("Adding custom basis functions:")
    
    # 1. Gaussian function
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
    
    # 2. Gaussian at a different center
    gaussian_basis2 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center0.5_sigma0.2",
        func=gaussian_func,
        params={"center": 0.5, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis2)
    print(f"  - {gaussian_basis2.name}")
    
    # 3. Gaussian at a different center
    gaussian_basis3 = BasisFactory.custom(
        dim=0,
        name="Gaussian_center1.0_sigma0.2",
        func=gaussian_func,
        params={"center": 1.0, "sigma": 0.2}
    )
    basis_set.add_basis(gaussian_basis3)
    print(f"  - {gaussian_basis3.name}")
    
    # 4. Linear function
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
    
    # 5. Constant function
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
    
    # Generate test data
    x = np.linspace(0, 1, 100)
    y_true = np.exp(-(x - 0.5)**2 / 0.1)  # Gaussian centered at 0.5
    y = y_true + 0.05 * np.random.randn(len(x))  # add a small amount of noise
    
    data = MultiDimData({0: x})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, y)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Coefficients: {coefficients}")
    
    # Predict
    y_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((y_pred - y)**2)
    print(f"Mean squared error (MSE): {mse:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data fit
    ax1 = axes[0, 0]
    ax1.scatter(x, y, alpha=0.5, label='Data', s=10)
    ax1.plot(x, y_true, 'g-', linewidth=2, label='True function')
    ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Custom basis fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Basis functions
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
    
    # 3. Coefficients
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
    
    # 4. Basis function contributions
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
    
    # Run all demos
    results = {}
    
    # 1. Polynomial basis
    coeff_poly, mse_poly = demo_polynomial_basis()
    results['Polynomial'] = mse_poly
    
    # 2. Fourier basis
    coeff_fourier, mse_fourier = demo_fourier_basis()
    results['Fourier'] = mse_fourier
    
    # 3. Legendre polynomials
    coeff_legendre, mse_legendre = demo_legendre_basis()
    results['Legendre'] = mse_legendre
    
    # 4. Custom basis
    coeff_custom, mse_custom = demo_custom_basis()
    results['Custom'] = mse_custom
    
    # Print comparison of results
    print("\n" + "=" * 60)
    print("Results comparison")
    print("=" * 60)
    for method, mse in results.items():
        print(f"{method:10} basis: MSE = {mse:.6f}")
    
    print("\nDemo complete!")
