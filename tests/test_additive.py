"""Tests for the direct-sum (additive model) basis construction (legacy module)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.basis.factory import BasisFactory
from functional_solver.basis.additive import AdditiveBasis
from functional_solver.inner_product.base import InnerProduct
from functional_solver.solver.functional_solver import FunctionalSolver


def test_additive_constant_dedup():
    """The direct-sum construction should deduplicate constant bases repeated across dimensions."""
    x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
    y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]

    # Without deduplication: 3 + 3 = 6 bases (including two constant bases 1)
    raw = AdditiveBasis.build({0: x_bases, 1: y_bases}, deduplicate_constants=False)
    assert len(raw) == 6

    # With deduplication: a single constant remains → 5 bases
    bs = AdditiveBasis.build({0: x_bases, 1: y_bases})
    assert len(bs) == 5

    # The design matrix should contain exactly one all-ones column
    data = MultiDimData({0: np.linspace(0, 1, 10), 1: np.linspace(0, 1, 10)})
    Phi = bs.evaluate_all(data)
    ones_cols = int(np.sum(np.all(np.abs(Phi - 1.0) < 1e-12, axis=0)))
    assert ones_cols == 1


def test_additive_fits_additive_function():
    """The direct sum can fit the purely additive function f = 1 + 2x - y exactly."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 40)
    y = rng.uniform(0, 1, 40)
    z_true = 1 + 2.0 * x - y
    z = z_true + 0.01 * rng.standard_normal(40)
    data = MultiDimData({0: x, 1: y})

    bs = AdditiveBasis.build({
        0: [BasisFactory.polynomial(dim=0, order=o) for o in range(2)],
        1: [BasisFactory.polynomial(dim=1, order=o) for o in range(2)],
    })
    solver = FunctionalSolver()
    solver.set_basis(bs)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(data, z)
    solver.solve()

    mse = np.mean((solver.predict(data) - z_true) ** 2)
    assert mse < 1e-3


if __name__ == "__main__":
    test_additive_constant_dedup()
    print("✓ test_additive_constant_dedup passed")
    test_additive_fits_additive_function()
    print("✓ test_additive_fits_additive_function passed")
    print("\nAll direct-sum basis tests passed!")
