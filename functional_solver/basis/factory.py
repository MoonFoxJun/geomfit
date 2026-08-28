"""Factory for predefined single-factor and tensor-product basis functions."""

import numpy as np
from typing import Callable, Dict, List
from ..core.basis_container import BasisInfo

class BasisFactory:
    """Factory providing predefined basis functions."""
    
    @staticmethod 
    def polynomial(dim: int, order: int):
        """Monomial basis φ(x) = x^order.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        order : int
            Polynomial order.

        Returns
        -------
        BasisInfo
            Single-factor monomial basis.
        """
        def func(x, order):
            return x ** order
        return BasisInfo(
            name=f"Polynomial_order{order}",
            dim=dim,
            params={"order": order},
            func=func
        )
    
    @staticmethod 
    def legendre(dim: int, order: int):
        """Legendre polynomial basis P_order(x) (numerically more stable than monomials).

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        order : int
            Polynomial order.

        Returns
        -------
        BasisInfo
            Single-factor Legendre basis, evaluated with numpy's Legendre routine.
        """
        from numpy.polynomial.legendre import legval
        def func(x, order):
            return legval(x, [0]*order + [1])
        return BasisInfo(
            name=f"Legendre_order{order}",
            dim=dim,
            params={"order": order},
            func=func
        )
    
    @staticmethod 
    def fourier(dim: int, freq: int, L: float = 2*np.pi):
        """Fourier basis: cos(2πf x/L) and sin(2πf x/L).

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        freq : int
            Frequency; for freq=0 only the constant basis φ ≡ 1 is returned.
        L : float, default=2π
            Period of the trigonometric functions.

        Returns
        -------
        List[BasisInfo]
            The constant basis (freq=0) or the cosine and sine bases (freq>0).
        """
        def func_cos(x, freq, L):
            return np.cos(2 * np.pi * freq * x / L)
        
        def func_sin(x, freq, L):
            return np.sin(2 * np.pi * freq * x / L)
        
        bases = []
        if freq == 0:
            bases.append(BasisInfo(
                name="Fourier_const",
                dim=dim,
                params={"freq": 0, "L": L},
                func=lambda x, **_: 1.0
            ))
        else:
            bases.append(BasisInfo(
                name=f"Fourier_cos{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=func_cos
            ))
            bases.append(BasisInfo(
                name=f"Fourier_sin{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=func_sin
            ))
        return bases
    
    @staticmethod 
    def laurent(dim: int, order: int):
        """Laurent basis: x^order and x^−order (for x ≠ 0).

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        order : int
            Laurent order; for order=0 only the constant basis φ ≡ 1 is returned.

        Returns
        -------
        List[BasisInfo]
            The constant basis (order=0) or the positive- and negative-power
            bases (order>0). The negative-power branch evaluates to 0 at x = 0.
        """
        def func_positive(x, order):
            return x ** order
        
        def func_negative(x, order):
            return x ** (-order) if x != 0 else 0.0
        
        bases = []
        if order == 0:
            bases.append(BasisInfo(
                name="Laurent_const",
                dim=dim,
                params={"order": 0},
                func=lambda x, **_: 1.0
            ))
        else:
            bases.append(BasisInfo(
                name=f"Laurent_positive{order}",
                dim=dim,
                params={"order": order},
                func=func_positive
            ))
            bases.append(BasisInfo(
                name=f"Laurent_negative{order}",
                dim=dim,
                params={"order": order},
                func=func_negative
            ))
        return bases
    
    @staticmethod 
    def gaussian_rbf(dim: int, center: float, sigma: float = 1.0):
        """Gaussian radial basis function φ(x) = exp(−(x − center)²/(2σ²)).

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the Gaussian.
        sigma : float, default=1.0
            Width of the Gaussian.

        Returns
        -------
        BasisInfo
            Single-factor Gaussian RBF basis.
        """
        def func(x, center, sigma):
            return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        
        return BasisInfo(
            name=f"GaussianRBF_center{center}_sigma{sigma}",
            dim=dim,
            params={"center": center, "sigma": sigma},
            func=func
        )
    
    @staticmethod 
    def wavelet(dim: int, scale: float, translation: float, wavelet_type: str = "mexican_hat"):
        """Wavelet basis of the requested type, with t = (x − translation)/scale.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        scale : float
            Wavelet scale.
        translation : float
            Wavelet translation.
        wavelet_type : str, default="mexican_hat"
            One of "mexican_hat" or "morlet".

        Returns
        -------
        BasisInfo
            Single-factor wavelet basis.
        """
        if wavelet_type == "mexican_hat":
            def func(x, scale, translation):
                t = (x - translation) / scale
                return (1 - t ** 2) * np.exp(-t ** 2 / 2)
        elif wavelet_type == "morlet":
            def func(x, scale, translation):
                t = (x - translation) / scale
                return np.cos(5 * t) * np.exp(-t ** 2 / 2)
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")
        
        return BasisInfo(
            name=f"Wavelet_{wavelet_type}_scale{scale}_trans{translation}",
            dim=dim,
            params={"scale": scale, "translation": translation, "wavelet_type": wavelet_type},
            func=func
        )
    
    @staticmethod
    def tensor_product(dim_bases: Dict[int, List[BasisInfo]]) -> List[BasisInfo]:
        """
        Multi-dimensional tensor-product basis: build every product combination
        of the per-dimension basis lists.

        For example, dim_bases = {0: [X₁, X₂], 1: [Y₁, Y₂]} generates
        X₁⊗Y₁, X₁⊗Y₂, X₂⊗Y₁, X₂⊗Y₂, i.e. the model
        f(x, y) = Σᵢ Σⱼ cᵢⱼ Xᵢ(x)Yⱼ(y).

        Parameters
        ----------
        dim_bases : Dict[int, List[BasisInfo]]
            Dimension index -> list of bases for that dimension.

        Returns
        -------
        List[BasisInfo]
            All product combinations; each returned basis is a multi-factor
            tensor-product basis.
        """
        import itertools

        if not dim_bases:
            raise ValueError("dim_bases must contain at least one dimension")
        dims = sorted(dim_bases.keys())
        lists = [dim_bases[d] for d in dims]

        products = []
        for combo in itertools.product(*lists):
            # Flatten nested tensor-product factors into a single factor list
            factors = []
            for b in combo:
                if b.factors is not None:
                    factors.extend(b.factors)
                else:
                    factors.append((b.dim, b.func, b.params))
            name = " ⊗ ".join(b.name for b in combo)
            products.append(BasisInfo(
                name=name,
                dim=combo[0].dim,
                params={},
                func=None,
                factors=factors,
            ))
        return products

    @staticmethod 
    def custom(dim: int, name: str, func: Callable, params: Dict = None):
        """Fully custom single-factor basis from an arbitrary callable.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        name : str
            Name of the basis.
        func : Callable
            Evaluation function func(x, **params).
        params : Dict, optional
            Parameters passed to `func`.

        Returns
        -------
        BasisInfo
            Single-factor custom basis.
        """
        return BasisInfo(
            name=name,
            dim=dim,
            params=params or {},
            func=func
        )
