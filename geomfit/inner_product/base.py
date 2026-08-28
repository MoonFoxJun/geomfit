"""Inner product base class for the functional solver."""

from typing import Callable, Dict, Any, Optional
import numpy as np
from ..core.data_container import MultiDimData


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Per-point weights for one-dimensional composite trapezoid quadrature.

    For sorted, deduplicated coordinates v₀ < v₁ < ... < v_{m-1}, the weights are
        endpoints h₀ = (v₁-v₀)/2, h_{m-1} = (v_{m-1}-v_{m-2})/2,
        interior hᵢ = (v_{i+1} - v_{i-1})/2,
    so that Σᵢ f(vᵢ)·hᵢ is the composite trapezoid rule ∫f dv. Points sharing a
    repeated coordinate value receive the weight of their unique value (works
    for both scattered and gridded data).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.ones(n)

    order = np.argsort(x, kind="stable")
    xs = x[order]
    unique, inverse = np.unique(xs, return_inverse=True)
    m = len(unique)

    if m == 1:
        h = np.ones(m)          # single coordinate value: degenerate to no volume element
    elif m == 2:
        d = unique[1] - unique[0]
        h = np.array([d / 2.0, d / 2.0])
    else:
        h = np.empty(m)
        h[0] = (unique[1] - unique[0]) / 2.0
        h[-1] = (unique[-1] - unique[-2]) / 2.0
        h[1:-1] = (unique[2:] - unique[:-2]) / 2.0
    h = np.maximum(h, 0.0)

    w = h[inverse]
    back = np.empty(n)
    back[order] = w
    return back


class InnerProduct:
    """Definition of an inner product on a function space."""

    def __init__(self, weight_func: Optional[Callable] = None,
                 is_continuous: bool = True):
        """
        Initialize the inner product.

        Parameters
        ----------
        weight_func : Callable, optional
            Weight function w(x) of the weighted inner product.
        is_continuous : bool, default=True
            Whether the inner product is defined on continuous functions
            (True, via numerical integration) or on discrete data
            (False, via summation).
        """
        self.weight_func = weight_func
        self.is_continuous = is_continuous

    def __call__(self, f: np.ndarray, g: np.ndarray,
                 data: Optional[MultiDimData] = None) -> float:
        """
        Compute the inner product of two function vectors.

        Parameters
        ----------
        f : np.ndarray
            Values of the first function at the data points.
        g : np.ndarray
            Values of the second function at the data points.
        data : MultiDimData, optional
            Data container (required for continuous inner products).

        Returns
        -------
        float
            Inner product ⟨f, g⟩.
        """
        if self.is_continuous:
            if data is None:
                raise ValueError("data is required for continuous inner product")
            return self._continuous_inner_product(f, g, data)
        else:
            return self._discrete_inner_product(f, g, data)

    # ---------- quadrature utilities ----------

    def _volume_weights(self, data: MultiDimData) -> np.ndarray:
        """Multi-dimensional volume-element weights: per-dimension trapezoid
        weights multiplied pointwise (multi-dimensional composite trapezoid rule).

        For tensor-product grids (e.g. flattened meshgrid data) this is the
        standard multi-dimensional trapezoid quadrature; for scattered data it
        is a reasonable approximation (each dimension weighted by the spacing
        of its own coordinates).
        """
        vw = np.ones(data.n_points)
        for dim in data.dims:
            vw = vw * _trapezoid_weights(data.get_dim(dim))
        return vw

    def _point_weights(self, data: MultiDimData) -> np.ndarray:
        """Evaluate the weight function w(x) at the data points."""
        if self.weight_func is None:
            return np.ones(data.n_points)
        if data.n_dims == 1:
            return np.asarray(self.weight_func(data.get_dim(data.dims[0])), dtype=float)
        # Multi-dimensional: evaluate pointwise as dicts (the exponential,
        # polynomial, etc. WeightFunction helpers accept dict input)
        w = np.empty(data.n_points)
        points = data.get_all_points()
        for i, p in enumerate(points):
            wi = self.weight_func(p)
            w[i] = float(np.asarray(wi).ravel()[0])
        return w

    def _continuous_inner_product(self, f: np.ndarray, g: np.ndarray,
                                  data: MultiDimData) -> float:
        """Continuous inner product: ⟨f,g⟩ = ∫ f(x)g(x)w(x)dV, discretized by
        the multi-dimensional composite trapezoid rule."""
        integrand = f * g * self._volume_weights(data)
        if self.weight_func is not None:
            integrand = integrand * self._point_weights(data)
        return float(np.sum(integrand))

    def _discrete_inner_product(self, f: np.ndarray, g: np.ndarray,
                                data: Optional[MultiDimData] = None) -> float:
        """Discrete inner product: ⟨f,g⟩ = Σᵢ fᵢgᵢ (optionally weighted)."""
        if self.weight_func is not None:
            if data is not None and len(data.dims) > 0:
                w = self._point_weights(data)
            else:
                w = np.asarray(self.weight_func(np.arange(len(f))), dtype=float)
            return float(np.sum(f * g * w))
        return float(np.dot(f, g))

    def compute_gram_matrix(self, Phi: np.ndarray,
                            data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        Compute the Gram matrix of basis functions.

        Parameters
        ----------
        Phi : np.ndarray
            Basis function matrix of shape (n_points, n_basis).
        data : MultiDimData, optional
            Data container (required for continuous inner products).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (n_basis, n_basis), with
            G_ij = ⟨φᵢ, φⱼ⟩.
        """
        n_basis = Phi.shape[1]
        G = np.zeros((n_basis, n_basis))

        for i in range(n_basis):
            for j in range(i, n_basis):
                g_ij = self(Phi[:, i], Phi[:, j], data)
                G[i, j] = g_ij
                if i != j:
                    G[j, i] = g_ij

        return G

    def rhs_vector(self, Phi: np.ndarray, target: np.ndarray,
                   data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        Compute the right-hand-side vector of the normal equations:
        bᵢ = ⟨φᵢ, target⟩.

        Uses the same inner product as the Gram matrix, so that Gc = b is the
        orthogonal projection equation in the same inner product space
        (consistency: in one dimension bᵢ = ∫φᵢ(x)y(x)w(x)dx).
        """
        n_basis = Phi.shape[1]
        return np.array([self(Phi[:, i], target, data) for i in range(n_basis)])

    def norm(self, f: np.ndarray, data: Optional[MultiDimData] = None) -> float:
        """
        Compute the norm of a function.

        Parameters
        ----------
        f : np.ndarray
            Function values.
        data : MultiDimData, optional
            Data container.

        Returns
        -------
        float
            Norm ||f|| = sqrt(⟨f, f⟩).
        """
        return np.sqrt(self(f, f, data))

    def __repr__(self) -> str:
        return f"InnerProduct(weight_func={self.weight_func is not None}, is_continuous={self.is_continuous})"

    def __str__(self) -> str:
        return self.__repr__()
