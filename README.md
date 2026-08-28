# Functional Solver

**Functional Solver** is a Python library for function approximation and functional optimization through basis expansion and kernel methods. Given a set of multi-dimensional data points with target values, it reconstructs the underlying function using either a linear combination of basis functions or a kernel-based representation, with configurable inner products and regularization.

## Features

- **Multi-dimensional data**: A `MultiDimData` container that stores data points of arbitrary dimensionality, with slicing, masking, concatenation, statistics, and serialization support.
- **Tensor-product bases**: Build product bases for interaction modeling,

  f(x, y, z) = Σᵢ Σⱼ Σₖ cᵢⱼₖ Xᵢ(x) Yⱼ(y) Zₖ(z),

  which capture multiplicative structure between dimensions (unlike a plain per-dimension direct sum). `BasisFactory.tensor_product` generates the full product set.
- **Direct-sum (additive) model**: A preserved legacy module, `AdditiveBasis`, concatenates per-dimension bases into an additive model and deduplicates constant bases repeated across dimensions. Useful for high-dimensional data or when dimensions contribute independently.
- **Kernel methods**: RBF, Matern, polynomial, and composite kernels, plus a `MahalanobisKernel` that absorbs data correlation into the distance metric. Kernels are naturally multi-dimensional.
- **Configurable inner product**: Weighted inner products, discrete and continuous (with volume elements for multi-dimensional quadrature) inner products, and Gram matrix computation.
- **Regularization**: Tikhonov, truncated SVD, Lasso, and elastic net strategies to stabilize ill-conditioned problems.
- **Solvers**: `FunctionalSolver`, `GramSolver`, `KernelSolver`, `GradientSolver`, and `AdaptiveSolver`, covering basis, Gram, and kernel formulations.
- **PCA preprocessing**: `pca_rotate` / `pca_transform` decouple strongly correlated dimensions before basis expansion, dramatically improving conditioning.

## Installation

```bash
# Clone the repository
git clone https://github.com/MoonFoxJun/geomfit.git
cd geomfit

# (Recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows; Linux/macOS: source .venv/bin/activate

# Install dependencies and the package in development mode
pip install -e .
```

Runtime dependencies are `numpy`, `scipy`, `matplotlib`, and `pyyaml`. For development, additionally install `pytest`:

```bash
pip install -e .[dev]
```

## Quick Start

### One-dimensional fitting with a polynomial basis

```python
import numpy as np
from geomfit import MultiDimData, BasisSet, BasisFactory, InnerProduct, FunctionalSolver

# 1. Prepare data
x = np.linspace(0, 1, 50)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.1 * np.random.randn(50)

data = MultiDimData({0: x})

# 2. Build a basis set
basis_set = BasisSet()
for order in range(5):
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))

# 3. Configure the inner product
inner = InnerProduct(weight_func=lambda x: np.exp(-x))

# 4. Solve
solver = FunctionalSolver()
solver.set_basis(basis_set)
solver.set_inner_product(inner)
solver.load_data(data, y)
coeff = solver.solve(regularization={"method": "tikhonov", "alpha": 1e-6})

# 5. Predict
y_pred = solver.predict(data)
```

### Multi-dimensional tensor-product bases (recommended)

For multi-dimensional problems, use `BasisFactory.tensor_product` to build product bases instead of stacking per-dimension bases into one set (the latter is a direct sum and cannot represent interaction terms such as x·y):

```python
import numpy as np
from geomfit import MultiDimData, BasisSet, BasisFactory, InnerProduct, FunctionalSolver

# 1. Two-dimensional data (the true function contains the interaction term x*y)
x = np.random.uniform(0, 1, 50)
y = np.random.uniform(0, 1, 50)
z = 1 + 2*x - y + 3*x*y + 0.1 * np.random.randn(50)
data = MultiDimData({0: x, 1: y})

# 2. Tensor-product basis: 3×3 = 9 products Xᵢ(x)Yⱼ(y)
x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
basis_set = BasisSet()
for b in BasisFactory.tensor_product({0: x_bases, 1: y_bases}):
    basis_set.add_basis(b)

# 3. Solve (identical workflow to the 1D case)
solver = FunctionalSolver()
solver.set_basis(basis_set)
solver.set_inner_product(InnerProduct(is_continuous=False))
solver.load_data(data, z)
coeff = solver.solve()
z_pred = solver.predict(data)
```

### Handling strongly correlated dimensions

Tensor-product bases assume orthogonal coordinates. When data is concentrated on a narrow band (for example y ≈ x, strongly correlated), the basis columns become nearly collinear and the Gram matrix is ill-conditioned. Two built-in remedies:

```python
from geomfit.utils.preprocess import pca_rotate, pca_transform
from geomfit.kernel.mahalanobis import MahalanobisKernel

# 1) PCA rotation (optionally dropping near-zero-variance directions) before expansion
rotated, info = pca_rotate(data, n_components=1)   # leading component variance share ≈ 1
# ... build a tensor-product basis on `rotated` and solve ...
new_rotated = pca_transform(new_data, info)        # transform new data with the same mapping

# 2) Or use a Mahalanobis kernel directly (correlation absorbed into the metric;
#    pair kernel methods with ridge regularization)
kernel = MahalanobisKernel(sigma=1.0)              # metric is estimated from the data automatically
solver.set_kernel(kernel)
solver.solve(regularization={"alpha": 1e-6})
```

A quantitative comparison is available in [`examples/demo_correlated_dims.py`](examples/demo_correlated_dims.py): the Gram condition number drops from ≈1e9 in original coordinates to ≈8.6 after PCA decoupling, and the coefficient norm from 140 to 1.

## Project Structure

```
geomfit/
├── geomfit/       # the package (import geomfit.*)
│   ├── core/                # core data structures (MultiDimData, BasisSet)
│   ├── basis/               # basis functions (polynomial, Fourier, wavelet, RBF; tensor_product, additive)
│   ├── kernel/              # kernels (RBF, Matern, polynomial, composite, Mahalanobis)
│   ├── inner_product/       # inner products and regularization
│   ├── solver/              # solvers (FunctionalSolver, GramSolver, KernelSolver, GradientSolver, AdaptiveSolver)
│   └── utils/               # numerical, visualization, and logging utilities
├── examples/                # runnable example scripts
├── tests/                   # pytest test suite
├── docs/                    # documentation
└── output/                  # figures generated by examples (gitignored)
```

## Documentation

- [docs/kernel_convolution_attention.md](docs/kernel_convolution_attention.md) — the mathematical relationship between kernels, convolution, and attention, and how it maps onto this library.
- Companion demo: [`examples/demo_attention_as_kernel.py`](examples/demo_attention_as_kernel.py) — an exact identity between QKᵀ and a kernel matrix.

## Testing

```bash
# Run the full test suite
pytest tests/

# Run a specific test file
pytest tests/test_basis.py
```

## License

Distributed under the [MIT License](LICENSE).
