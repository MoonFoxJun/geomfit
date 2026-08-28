"""Tests for the Gram matrix."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct

def test_gram_matrix_1d():
    """Test Gram matrix computation in 1D."""
    # Create a basis set
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=2))
    
    # Create data
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # Create an inner product (discrete)
    inner_product = InnerProduct(is_continuous=False)
    
    # Evaluate the design matrix
    Phi = basis_set.evaluate_all(data)
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Verify Gram matrix properties
    assert G.shape == (3, 3)  # 3 basis functions
    assert np.allclose(G, G.T)  # symmetric
    
    # Verify the Gram matrix entries
    # For a discrete inner product, G[i,j] = Σ φ_i(x_k) φ_j(x_k)
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = np.sum(x**i * x**j)
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_continuous():
    """Test the Gram matrix with a continuous inner product."""
    # Create a basis set
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # Create data
    x = np.linspace(0, 1, 100)  # more points for better quadrature accuracy
    data = MultiDimData({0: x})
    
    # Create a continuous inner product
    inner_product = InnerProduct(is_continuous=True)
    
    # Evaluate the design matrix
    Phi = basis_set.evaluate_all(data)
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Verify Gram matrix properties
    assert G.shape == (2, 2)
    assert np.allclose(G, G.T)  # symmetric
    
    # Verify Gram matrix entries (approximate integrals)
    # ∫₀¹ 1 dx = 1
    # ∫₀¹ x dx = 0.5
    # ∫₀¹ x² dx = 1/3
    expected = np.array([[1.0, 0.5], [0.5, 1.0/3]])
    
    assert np.allclose(G, expected, rtol=1e-2)  # allow 2% error

def test_weighted_inner_product():
    """Test the weighted inner product."""
    # Create a basis set
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # Create data
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # Create a weighted inner product (weight function w(x) = x)
    def weight_func(x):
        return x
    
    inner_product = InnerProduct(weight_func=weight_func, is_continuous=False)
    
    # Evaluate the design matrix
    Phi = basis_set.evaluate_all(data)
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Verify the Gram matrix entries
    # For a weighted discrete inner product, G[i,j] = Σ w(x_k) φ_i(x_k) φ_j(x_k)
    expected = np.zeros((2, 2))
    w = weight_func(x)
    expected[0, 0] = np.sum(w * x**0 * x**0)  # Σ w(x) * 1 * 1
    expected[0, 1] = np.sum(w * x**0 * x**1)  # Σ w(x) * 1 * x
    expected[1, 0] = expected[0, 1]  # symmetric
    expected[1, 1] = np.sum(w * x**1 * x**1)  # Σ w(x) * x * x
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_orthogonal_basis():
    """Test the Gram matrix of orthogonal basis functions."""
    # Use Legendre polynomials (orthogonal on [-1, 1])
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=0))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=1))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=2))
    
    # Create data on the interval [-1, 1]
    x = np.linspace(-1, 1, 100)
    data = MultiDimData({0: x})
    
    # Create an inner product (discrete approximation)
    inner_product = InnerProduct(is_continuous=False)
    
    # Evaluate the design matrix
    Phi = basis_set.evaluate_all(data)
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # The Gram matrix should be nearly diagonal (Legendre polynomials
    # are nearly orthogonal on discrete points)
    # Extract the diagonal
    diag = np.diag(G)
    
    # Off-diagonal entries should be relatively small
    G_off_diag = G.copy()
    np.fill_diagonal(G_off_diag, 0)
    off_diag_norm = np.linalg.norm(G_off_diag, 'fro')
    
    # The off-diagonal norm should be much smaller than the diagonal norm
    assert off_diag_norm < 0.1 * np.linalg.norm(diag)

def test_gram_matrix_singularity():
    """Test singularity detection for the Gram matrix."""
    from geomfit.inner_product.regularization import Regularization
    
    # Create linearly dependent basis functions
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # constant 1
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # another constant 1 (linearly dependent)
    
    # Create data
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # Create an inner product
    inner_product = InnerProduct(is_continuous=False)
    
    # Evaluate the design matrix
    Phi = basis_set.evaluate_all(data)
    
    # Compute the Gram matrix
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # Detect singularity
    singularity_info = Regularization.check_singularity(G, threshold=1e-10)
    
    # The Gram matrix should be singular (linearly dependent basis)
    assert singularity_info["is_singular"] == True
    assert singularity_info["rank"] < G.shape[0]

def test_gram_matrix_regularization():
    """Test Gram matrix regularization."""
    from geomfit.inner_product.regularization import Regularization
    
    # Create a nearly singular Gram matrix
    G = np.array([[1.0, 0.999999], [0.999999, 1.0]])
    
    # Apply Tikhonov regularization
    G_reg = Regularization.tikhonov(G, alpha=1e-6)
    
    # Verify the regularized matrix
    assert G_reg.shape == G.shape
    assert np.allclose(G_reg, G_reg.T)  # still symmetric
    
    # The diagonal is increased by alpha
    assert np.allclose(np.diag(G_reg), np.diag(G) + 1e-6)
    
    # The condition number should improve
    cond_original = np.linalg.cond(G)
    cond_regularized = np.linalg.cond(G_reg)
    assert cond_regularized < cond_original

def test_gram_solver():
    """Test the GramSolver."""
    from geomfit.solver.gram_solver import GramSolver
    
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
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # add noise
    
    data = MultiDimData({0: x})
    
    # Load data
    solver.load_data(data, y)
    
    # Compute the Gram matrix
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    
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
    singular_values = solver.get_singular_values()
    assert len(singular_values) == 3

if __name__ == "__main__":
    test_gram_matrix_1d()
    print("✓ test_gram_matrix_1d passed")
    
    test_gram_matrix_continuous()
    print("✓ test_gram_matrix_continuous passed")
    
    test_weighted_inner_product()
    print("✓ test_weighted_inner_product passed")
    
    test_gram_matrix_orthogonal_basis()
    print("✓ test_gram_matrix_orthogonal_basis passed")
    
    test_gram_matrix_singularity()
    print("✓ test_gram_matrix_singularity passed")
    
    test_gram_matrix_regularization()
    print("✓ test_gram_matrix_regularization passed")
    
    test_gram_solver()
    print("✓ test_gram_solver passed")
    
    print("\nAll Gram matrix tests passed!")
