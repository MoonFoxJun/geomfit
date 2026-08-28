"""2D Gram matrix demo"""

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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # 2D tensor-product basis: f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y)
    print("Building 2D tensor-product polynomial basis f(x,y) = Σ cᵢⱼ Xᵢ(x)Yⱼ(y):")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]  # x: orders 0..2
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]  # y: orders 0..2
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})   # 3×3 = 9 product bases
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # Generate 2D test data
    n_points = 20
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # True function: polynomial with interaction terms (exactly representable by the
    # tensor-product basis, but not by an additive model)
    z_true = 1.0 + 2.0*x - y + 3.0*x*y + x**2 * y
    z = z_true + 0.05 * np.random.randn(n_points)  # add a small amount of noise
    
    data = MultiDimData({0: x, 1: y})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Predict
    z_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # Comparison: direct-sum (additive) model — AdditiveBasis.build is structurally
    # unable to represent X(x)Y(y) interaction terms
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
    
    # Build a grid for visualization
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    
    # Convert grid points to MultiDimData format
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # Predict on the grid
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # Evaluate the true function on the grid
    z_grid_true = 1.0 + 2.0*x_grid - y_grid + 3.0*x_grid*y_grid + x_grid**2 * y_grid
    
    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Raw data points
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8, label='Data points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. True function surface
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function: 1 + 2x − y + 3xy + x²y')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. Fitted surface
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Polynomial basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. Error surface
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. Coefficients
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
    
    # 6. Residual scatter
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
    
    # 2D tensor-product Fourier basis (frequencies 0..1 per dimension -> 3×3 = 9 product bases)
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
    
    # Periodic 2D test data
    n_points = 25
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # True function f(x,y) = sin(2πx)cos(2πy) is itself a product X(x)Y(y),
    # exactly representable by the tensor-product basis
    z_true = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    z = z_true + 0.05 * np.random.randn(n_points)  # add a small amount of noise
    
    data = MultiDimData({0: x, 1: y})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Predict
    z_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # Comparison: direct-sum (additive) model — AdditiveBasis.build is structurally
    # unable to represent X(x)Y(y) interaction terms
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
    
    # Build a grid for visualization
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # Convert grid points to MultiDimData format
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # Predict on the grid
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # Evaluate the true function on the grid
    z_grid_true = np.sin(2 * np.pi * x_grid) * np.cos(2 * np.pi * y_grid)
    
    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Raw data points
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. True function surface
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function: sin(2πx)cos(2πy)')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. Fitted surface
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Fourier basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. Error surface
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. Coefficients
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
    
    # 6. Coefficient magnitudes
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
    
    # Mixed tensor product: polynomial in x, Fourier in y, plus a Gaussian RBF per dimension.
    # A 2D Gaussian = 1D Gaussian(x) ⊗ 1D Gaussian(y), i.e. a tensor-product structure.
    print("Building 2D mixed tensor-product basis:")
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    x_bases.append(BasisFactory.gaussian_rbf(dim=0, center=0.5, sigma=0.1))
    y_bases = []
    for freq in range(2):  # y: Fourier bases for frequencies 0..1
        y_bases.extend(BasisFactory.fourier(dim=1, freq=freq, L=1.0))
    y_bases.append(BasisFactory.gaussian_rbf(dim=1, center=0.5, sigma=0.1))
    tensor_bases = BasisFactory.tensor_product({0: x_bases, 1: y_bases})  # 4×4 = 16 product bases
    
    basis_set = BasisSet()
    for basis in tensor_bases:
        basis_set.add_basis(basis)
        print(f"  - {basis.name}")
    
    # Generate 2D test data
    n_points = 30
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # True function: x²·sin(2πy) (= x² ⊗ sin) + 2D Gaussian (= gauss ⊗ gauss) + 0.3x (= x ⊗ 1)
    z_true = x**2 * np.sin(2 * np.pi * y) + np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) + 0.3*x
    z = z_true + 0.05 * np.random.randn(n_points)  # add a small amount of noise
    
    data = MultiDimData({0: x, 1: y})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a solver
    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(inner_product)
    solver.load_data(data, z)
    
    # Solve
    print("\nSolving...")
    coefficients = solver.solve()
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Predict
    z_pred = solver.predict(data)
    
    # Compute the error
    mse = np.mean((z_pred - z)**2)
    print(f"Tensor-product MSE: {mse:.6f}")

    # Comparison: direct-sum (additive) model — AdditiveBasis.build is structurally
    # unable to represent X(x)Y(y) interaction terms
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
    
    # Build a grid for visualization
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    
    # Convert grid points to MultiDimData format
    grid_points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
    grid_data = MultiDimData({0: grid_points[:, 0], 1: grid_points[:, 1]})
    
    # Predict on the grid
    z_grid_pred = solver.predict(grid_data).reshape(x_grid.shape)
    
    # Evaluate the true function on the grid
    z_grid_true = x_grid**2 * np.sin(2 * np.pi * y_grid) + np.exp(-((x_grid-0.5)**2 + (y_grid-0.5)**2) / 0.02) + 0.3*x_grid
    
    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Raw data points
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Raw data points')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. True function surface
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, z_grid_true, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('True function')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 3. Fitted surface
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, z_grid_pred, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title('Mixed basis fit')
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. Error surface
    ax4 = fig.add_subplot(234, projection='3d')
    error = z_grid_pred - z_grid_true
    surf4 = ax4.plot_surface(x_grid, y_grid, error, cmap='RdBu', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Error')
    ax4.set_title('Fit error')
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    # 5. Coefficients
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
    
    # 6. Contour comparison
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
    
    # Create a basis set
    basis_set = BasisSet()
    
    # Add polynomial basis functions
    print("Building basis set...")
    for order in range(5):  # polynomials of order 0..4
        basis = BasisFactory.polynomial(dim=0, order=order)
        basis_set.add_basis(basis)
    
    # Generate test data
    x = np.linspace(0, 1, 50)
    data = MultiDimData({0: x})
    
    # Discrete inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Evaluate the basis matrix
    Phi = basis_set.evaluate_all(data)
    print(f"Basis matrix shape: {Phi.shape}")
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    print(f"Gram matrix shape: {G.shape}")
    
    # Analyze the Gram matrix
    eigenvalues = np.linalg.eigvalsh(G)  # eigenvalues (symmetric matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # sort in descending order
    
    condition_number = np.linalg.cond(G)
    rank = np.linalg.matrix_rank(G)
    
    print(f"\nGram matrix analysis:")
    print(f"  Condition number: {condition_number:.2e}")
    print(f"  Numerical rank: {rank}")
    print(f"  Eigenvalue range: {eigenvalues_sorted[0]:.2e} to {eigenvalues_sorted[-1]:.2e}")
    print(f"  Eigenvalue ratio: {eigenvalues_sorted[0]/eigenvalues_sorted[-1]:.2e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Gram matrix heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(G, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Basis index')
    ax1.set_ylabel('Basis index')
    ax1.set_title('Gram matrix')
    plt.colorbar(im1, ax=ax1, shrink=0.7, aspect=10)
    
    # 2. Eigenvalue spectrum
    ax2 = axes[0, 1]
    indices = np.arange(1, len(eigenvalues_sorted) + 1)
    ax2.semilogy(indices, eigenvalues_sorted, 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Threshold: 1e-10')
    ax2.set_xlabel('Eigenvalue index')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title('Gram matrix eigenvalue spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Basis matrix
    ax3 = axes[0, 2]
    im3 = ax3.imshow(Phi, cmap='plasma', aspect='auto')
    ax3.set_xlabel('Basis index')
    ax3.set_ylabel('Data point index')
    ax3.set_title('Basis matrix Φ')
    plt.colorbar(im3, ax=ax3, shrink=0.7, aspect=10)
    
    # 4. Basis functions
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
    
    # 5. Singular value decomposition
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    ax5 = axes[1, 1]
    indices_svd = np.arange(1, len(s) + 1)
    ax5.semilogy(indices_svd, s, 'go-', linewidth=2, markersize=6)
    ax5.set_xlabel('Singular value index')
    ax5.set_ylabel('Singular value (log scale)')
    ax5.set_title('Basis matrix singular values')
    ax5.grid(True, alpha=0.3)
    
    # 6. Condition number vs number of basis functions
    ax6 = axes[1, 2]
    cond_numbers = []
    n_basis_range = range(2, 8)
    
    for n_basis in n_basis_range:
        # First n_basis basis functions only
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
    
    # Run all demos
    results = {}
    
    # 1. 2D polynomial basis
    coeff_2d_poly, mse_2d_poly = demo_2d_polynomial_basis()
    results['2D polynomial'] = mse_2d_poly
    
    # 2. 2D Fourier basis
    coeff_2d_fourier, mse_2d_fourier = demo_2d_fourier_basis()
    results['2D Fourier'] = mse_2d_fourier
    
    # 3. 2D mixed basis
    coeff_2d_mixed, mse_2d_mixed = demo_2d_mixed_basis()
    results['2D mixed'] = mse_2d_mixed
    
    # 4. Gram matrix analysis
    G, eigenvalues, cond_number = demo_gram_matrix_analysis()
    results['Condition number'] = cond_number
    
    # Print comparison of results
    print("\n" + "=" * 60)
    print("Results comparison")
    print("=" * 60)
    for method, value in results.items():
        if method == 'Condition number':
            print(f"{method:10}: {value:.2e}")
        else:
            print(f"{method:10} basis: MSE = {value:.6f}")
    
    print("\nDemo complete!")
