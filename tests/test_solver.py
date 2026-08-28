"""Tests for the solvers."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver
from geomfit.solver.gram_solver import GramSolver
from geomfit.solver.kernel_solver import KernelSolver
from geomfit.kernel.rbf import RBFKernel

def test_functional_solver_basis_method():
    """Test the FunctionalSolver basis method."""
    # Create a solver
    solver = FunctionalSolver()
    
    # Create a basis set
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # Set the basis
    solver.set_basis(basis_set)
    
    # Create an inner product
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # Create data
    x = np.linspace(0, 1, 50)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # add noise
    
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Solve (no regularization)
    coefficients = solver.solve()
    assert coefficients.shape == (5,)
    
    # Predict
    y_pred = solver.predict(data)
    assert y_pred.shape == (50,)
    
    # Compute the mean squared error
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.05  # the error should be small
    
    # Retrieve debug information
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "basis"

def test_functional_solver_kernel_method():
    """Test the FunctionalSolver kernel method."""
    # Create a solver
    solver = FunctionalSolver()
    
    # Create a kernel
    kernel = RBFKernel(sigma=1.0)
    solver.set_kernel(kernel)
    
    # Create data
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Solve
    coefficients = solver.solve()
    assert coefficients.shape == (30,)  # kernel coefficients equal the number of data points
    
    # Predict
    y_pred = solver.predict(data)
    assert y_pred.shape == (30,)
    
    # Compute the mean squared error
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.1  # the kernel method should fit well
    
    # Retrieve debug information
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "kernel"

def test_functional_solver_regularization():
    """Test FunctionalSolver regularization."""
    # Create a solver
    solver = FunctionalSolver()
    
    # Create a basis set of high-order polynomials (prone to overfitting)
    basis_set = BasisSet()
    for order in range(15):  # high-order polynomials
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    solver.set_basis(basis_set)
    
    # Create an inner product
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # Create data
    x = np.linspace(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.2 * np.random.randn(len(x))  # larger noise
    
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # Solve without regularization
    coeff_no_reg = solver.solve()
    
    # Solve with regularization
    regularization = {"method": "tikhonov", "alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # The regularized coefficient norm should be smaller
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_functional_solver_reset():
    """Test the FunctionalSolver reset."""
    solver = FunctionalSolver()
    
    # Set up some state
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    solver.set_basis(basis_set)
    
    inner_product = InnerProduct()
    solver.set_inner_product(inner_product)
    
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # Solve
    solver.solve()
    
    # Reset
    solver.reset()
    
    # Verify that the state has been reset
    assert solver.basis_set is None
    assert solver.inner_product is None
    assert solver.data is None
    assert solver.target is None
    assert solver.coefficients is None
    assert solver.debug_info == {}

def test_gram_solver_basic():
    """Test basic GramSolver functionality."""
    # Create a basis set
    basis_set = BasisSet()
    for order in range(3):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # Create an inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Create a Gram solver
    solver = GramSolver(basis_set, inner_product)
    
    # Create data
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Compute the Gram matrix
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    assert np.allclose(G, G.T)  # symmetric
    
    # Compute the right-hand side
    b = solver.compute_rhs()
    assert b.shape == (3,)
    
    # Solve
    coefficients = solver.solve()
    assert coefficients.shape == (3,)
    
    # Predict
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # Compute the condition number
    cond = solver.get_condition_number()
    assert cond > 0
    
    # Compute the singular values
    sv = solver.get_singular_values()
    assert len(sv) == 3
    assert np.all(sv >= 0)

def test_kernel_solver_basic():
    """Test basic KernelSolver functionality."""
    # Create a kernel
    kernel = RBFKernel(sigma=1.0)
    
    # Create a kernel solver
    solver = KernelSolver(kernel)
    
    # Create data
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Compute the kernel matrix
    K = solver.compute_kernel_matrix()
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T)  # symmetric
    
    # Solve
    coefficients = solver.solve()
    assert coefficients.shape == (20,)
    
    # Predict
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # Efficient prediction
    y_pred_eff = solver.predict_efficient(data)
    assert y_pred_eff.shape == (20,)
    assert np.allclose(y_pred, y_pred_eff, rtol=1e-10)
    
    # Compute the condition number
    cond = solver.get_condition_number()
    assert cond > 0
    
    # Compute the eigenvalues
    eigenvalues = solver.get_eigenvalues()
    assert len(eigenvalues) == 20
    assert np.all(eigenvalues >= 0)  # the kernel matrix is positive semi-definite
    
    # Compute the function norm
    norm = solver.compute_function_norm()
    assert norm >= 0

def test_kernel_solver_regularization():
    """Test KernelSolver regularization."""
    # Create a kernel
    kernel = RBFKernel(sigma=0.5)
    
    # Create a kernel solver
    solver = KernelSolver(kernel)
    
    # Create data
    x = np.linspace(0, 1, 15)
    y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # Solve without regularization
    coeff_no_reg = solver.solve()
    
    # Solve with regularization
    regularization = {"alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # The regularized coefficient norm should be smaller
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_solver_comparison():
    """Test comparison across solvers."""
    # Create data
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # Solver 1: FunctionalSolver with basis
    solver1 = FunctionalSolver()
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    solver1.set_basis(basis_set)
    solver1.set_inner_product(InnerProduct(is_continuous=False))
    solver1.load_data(data, y)
    coeff1 = solver1.solve()
    y_pred1 = solver1.predict(data)
    mse1 = np.mean((y_pred1 - y)**2)
    
    # Solver 2: FunctionalSolver with kernel
    solver2 = FunctionalSolver()
    kernel = RBFKernel(sigma=1.0)
    solver2.set_kernel(kernel)
    solver2.load_data(data, y)
    coeff2 = solver2.solve()
    y_pred2 = solver2.predict(data)
    mse2 = np.mean((y_pred2 - y)**2)
    
    # Solver 3: GramSolver
    solver3 = GramSolver(basis_set, InnerProduct(is_continuous=False))
    solver3.load_data(data, y)
    coeff3 = solver3.solve()
    y_pred3 = solver3.predict(data)
    mse3 = np.mean((y_pred3 - y)**2)
    
    # Solver 4: KernelSolver
    solver4 = KernelSolver(kernel)
    solver4.load_data(data, y)
    coeff4 = solver4.solve()
    y_pred4 = solver4.predict(data)
    mse4 = np.mean((y_pred4 - y)**2)
    
    # All solvers should produce reasonable results
    assert mse1 < 0.05
    assert mse2 < 0.05
    assert mse3 < 0.05
    assert mse4 < 0.05
    
    # FunctionalSolver with basis and GramSolver should agree
    assert np.allclose(coeff1, coeff3, rtol=1e-10)
    assert np.allclose(y_pred1, y_pred3, rtol=1e-10)
    
    # FunctionalSolver with kernel and KernelSolver should agree
    assert np.allclose(coeff2, coeff4, rtol=1e-10)
    assert np.allclose(y_pred2, y_pred4, rtol=1e-10)

def test_solver_with_new_data():
    """Test prediction on new data."""
    # Training data
    x_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + 0.1 * np.random.randn(len(x_train))
    train_data = MultiDimData({0: x_train})
    
    # Test data
    x_test = np.linspace(0.1, 0.9, 15)
    y_test_true = np.sin(2 * np.pi * x_test)
    test_data = MultiDimData({0: x_test})
    
    # Use FunctionalSolver with basis
    solver = FunctionalSolver()
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(train_data, y_train)
    solver.solve()
    
    # Predict on test data
    y_test_pred = solver.predict(test_data)
    
    # Verify the prediction shape
    assert y_test_pred.shape == (15,)
    
    # Compute the test error
    test_mse = np.mean((y_test_pred - y_test_true)**2)
    assert test_mse < 0.2  # the test error should be reasonable

if __name__ == "__main__":
    test_functional_solver_basis_method()
    print("✓ test_functional_solver_basis_method passed")
    
    test_functional_solver_kernel_method()
    print("✓ test_functional_solver_kernel_method passed")
    
    test_functional_solver_regularization()
    print("✓ test_functional_solver_regularization passed")
    
    test_functional_solver_reset()
    print("✓ test_functional_solver_reset passed")
    
    test_gram_solver_basic()
    print("✓ test_gram_solver_basic passed")
    
    test_kernel_solver_basic()
    print("✓ test_kernel_solver_basic passed")
    
    test_kernel_solver_regularization()
    print("✓ test_kernel_solver_regularization passed")
    
    test_solver_comparison()
    print("✓ test_solver_comparison passed")
    
    test_solver_with_new_data()
    print("✓ test_solver_with_new_data passed")
    
    print("\nAll solver tests passed!")
