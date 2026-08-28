"""Tests for the basis function module."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory

def test_polynomial_basis():
    """Test the polynomial basis."""
    basis = BasisFactory.polynomial(dim=0, order=3)
    assert basis.name == "Polynomial_order3"
    assert basis.dim == 0
    
    # Evaluate the basis function
    x = 2.0
    val = basis.func(x, order=3)
    assert val == 8.0

def test_basis_set_evaluation():
    """Test evaluation of a basis set."""
    basis_set = BasisSet()
    
    # Add a few basis functions
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # Create test data
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0])
    })
    
    # Evaluate all basis functions
    Phi = basis_set.evaluate_all(data)
    
    # Verify the design matrix
    expected = np.array([[1, 1], [1, 2], [1, 3]])
    assert np.allclose(Phi, expected)

def test_fourier_basis():
    """Test the Fourier basis."""
    bases = BasisFactory.fourier(dim=0, freq=1, L=2*np.pi)
    
    # A frequency of 1 should return two basis functions (cosine and sine)
    assert len(bases) == 2
    
    # Test the cosine basis function
    cos_basis = bases[0]
    x = np.pi/2
    val = cos_basis.func(x, freq=1, L=2*np.pi)
    expected = np.cos(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)
    
    # Test the sine basis function
    sin_basis = bases[1]
    val = sin_basis.func(x, freq=1, L=2*np.pi)
    expected = np.sin(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)

def test_legendre_basis():
    """Test the Legendre polynomial basis."""
    basis = BasisFactory.legendre(dim=0, order=2)
    assert basis.name == "Legendre_order2"
    assert basis.dim == 0
    
    # Evaluate the basis function
    x = 0.5
    val = basis.func(x, order=2)
    # Second-order Legendre polynomial: P2(x) = (3x^2 - 1)/2
    expected = (3*x**2 - 1)/2
    assert np.allclose(val, expected)

def test_custom_basis():
    """Test a custom basis function."""
    def custom_func(x, scale=1.0):
        return np.exp(-scale * x**2)
    
    basis = BasisFactory.custom(dim=0, name="Gaussian", func=custom_func, params={"scale": 0.5})
    assert basis.name == "Gaussian"
    assert basis.dim == 0
    assert "scale" in basis.params
    
    # Evaluate the basis function
    x = 1.0
    val = basis.func(x, scale=0.5)
    expected = np.exp(-0.5 * x**2)
    assert np.allclose(val, expected)

def test_basis_set_dimension_organization():
    """Test dimension organization in a basis set."""
    basis_set = BasisSet()
    
    # Add basis functions for different dimensions
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # Check the per-dimension organization
    assert len(basis_set) == 4
    assert 0 in basis_set.by_dim
    assert 1 in basis_set.by_dim
    assert len(basis_set.by_dim[0]) == 2
    assert len(basis_set.by_dim[1]) == 2
    
    # Check retrieval of basis functions for a specific dimension
    dim0_bases = basis_set.get_bases_for_dim(0)
    assert len(dim0_bases) == 2
    for basis in dim0_bases:
        assert basis.dim == 0

def test_basis_set_evaluate_at_point():
    """Test basis set evaluation at a single point."""
    basis_set = BasisSet()
    
    # Add basis functions
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # Evaluate at a single point
    point = {0: 2.0, 1: 3.0}
    values = basis_set.evaluate_at_point(point)
    
    # Expected values: [1, 2, 1, 3]
    expected = np.array([1.0, 2.0, 1.0, 3.0])
    assert np.allclose(values, expected)

def test_rbf_basis():
    """Test the radial basis function (RBF)."""
    from geomfit.basis.rbf import RBFBasis
    
    # Create a Gaussian RBF
    rbf = RBFBasis.gaussian(dim=0, center=0.0, sigma=1.0)
    assert rbf.name.startswith("Gaussian_RBF")
    assert rbf.dim == 0
    
    # Evaluate the basis function
    x = 1.0
    val = rbf.func(x, center=0.0, sigma=1.0)
    expected = np.exp(-(x - 0.0)**2 / (2 * 1.0**2))
    assert np.allclose(val, expected)

def test_wavelet_basis():
    """Test the wavelet basis function."""
    from geomfit.basis.wavelet import WaveletBasis
    
    # Create a Mexican hat wavelet
    wavelet = WaveletBasis.mexican_hat(dim=0, center=0.0, scale=1.0)
    assert wavelet.name.startswith("MexicanHat")
    assert wavelet.dim == 0
    
    # Evaluate the basis function (verify it is callable)
    x = 0.5
    val = wavelet.func(x, center=0.0, scale=1.0)
    # Mexican hat wavelet formula: (1 - x^2) * exp(-x^2/2)
    x_norm = (x - 0.0) / 1.0
    expected = (1 - x_norm**2) * np.exp(-x_norm**2 / 2)
    assert np.allclose(val, expected)

if __name__ == "__main__":
    test_polynomial_basis()
    test_basis_set_evaluation()
    test_fourier_basis()
    test_legendre_basis()
    test_custom_basis()
    test_basis_set_dimension_organization()
    test_basis_set_evaluate_at_point()
    test_rbf_basis()
    test_wavelet_basis()
    print("All tests passed!")
