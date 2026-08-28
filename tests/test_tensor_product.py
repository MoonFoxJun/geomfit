"""Tests for tensor-product bases and multi-dimensional fitting."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver


def test_tensor_product_evaluation():
    """The value of a tensor-product basis equals the product of its factor values in each dimension."""
    x_bases = [BasisFactory.polynomial(dim=0, order=1)]   # X₁(x) = x
    y_bases = [BasisFactory.polynomial(dim=1, order=2)]   # Y₂(y) = y²
    products = BasisFactory.tensor_product({0: x_bases, 1: y_bases})
    assert len(products) == 1

    phi = products[0]
    point = {0: 3.0, 1: 4.0}
    assert phi.evaluate(point) == 3.0 * 16.0 == 48.0

    # The tensor-product basis should register in every involved dimension
    assert set(phi.dims) == {0, 1}


def test_tensor_product_count():
    """The number of tensor-product bases equals the product of per-dimension basis counts."""
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(2)]
    z_bases = [BasisFactory.polynomial(dim=2, order=o) for o in range(2)]
    products = BasisFactory.tensor_product({0: x_bases, 1: y_bases, 2: z_bases})
    assert len(products) == 3 * 2 * 2 == 12


def test_tensor_product_fits_interaction():
    """A tensor-product basis fits the interaction function f(x,y)=1+2x-y+3xy exactly, while an additive model cannot."""
    rng = np.random.default_rng(42)
    n = 40
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z_true = 1.0 + 2.0 * x - y + 3.0 * x * y   # contains the interaction term x*y
    z = z_true + 0.01 * rng.standard_normal(n)
    data = MultiDimData({0: x, 1: y})

    # Tensor product: 9 bases (x^0..2 ⊗ y^0..2)
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
    basis_set = BasisSet()
    for b in BasisFactory.tensor_product({0: x_bases, 1: y_bases}):
        basis_set.add_basis(b)

    solver = FunctionalSolver()
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    solver.solve()
    mse_tensor = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse_tensor < 1e-3

    # Additive model (direct sum): only 1, x, x², y, y² (with a duplicated constant basis)
    add_set = BasisSet()
    for o in range(3):
        add_set.add_basis(BasisFactory.polynomial(dim=0, order=o))
        add_set.add_basis(BasisFactory.polynomial(dim=1, order=o))
    solver_add = FunctionalSolver()
    solver_add.set_basis(add_set)
    solver_add.set_inner_product(InnerProduct(is_continuous=False))
    solver_add.load_data(data, z)
    solver_add.solve()
    mse_add = np.mean((solver_add.predict(data) - z_true) ** 2)
    # An additive model cannot represent x*y structurally, so the error should be significantly larger
    assert mse_add > 10 * mse_tensor


def test_continuous_multidim_gram():
    """Multi-dimensional continuous inner product (tensor-product grid): ∫∫1 = 1, ∫∫x²y² = 1/9, Gram matrix symmetric."""
    n = 81
    t = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(t, t)
    data = MultiDimData({0: xv.ravel(), 1: yv.ravel()})
    inner = InnerProduct(is_continuous=True)

    ones = np.ones(data.n_points)
    assert abs(inner(ones, ones, data) - 1.0) < 1e-3

    xy2 = xv.ravel() ** 2 * yv.ravel() ** 2
    assert abs(inner(xy2, ones, data) - 1.0 / 9.0) < 1e-3

    Phi = np.column_stack([ones, xv.ravel()])
    G = inner.compute_gram_matrix(Phi, data)
    assert np.allclose(G, G.T)


if __name__ == "__main__":
    test_tensor_product_evaluation()
    print("✓ test_tensor_product_evaluation passed")
    test_tensor_product_count()
    print("✓ test_tensor_product_count passed")
    test_tensor_product_fits_interaction()
    print("✓ test_tensor_product_fits_interaction passed")
    test_continuous_multidim_gram()
    print("✓ test_continuous_multidim_gram passed")
    print("\nAll tensor-product tests passed!")
