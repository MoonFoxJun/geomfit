"""Fourier basis functions over a single dimension: real trigonometric, complex exponential, and DCT families."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class FourierBasis:
    """Fourier basis functions over a single dimension."""
    
    @staticmethod
    def create_basis(dim: int, max_freq: int, L: float = 2*np.pi) -> List[BasisInfo]:
        """Create the real Fourier basis on [0, L]: the constant φ₀ ≡ 1 plus
        cos(2πf x/L) and sin(2πf x/L) for f = 1, ..., max_freq.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_freq : int
            Highest frequency (inclusive).
        L : float, default=2π
            Period of the trigonometric functions.

        Returns
        -------
        List[BasisInfo]
            The constant basis followed by a cosine and a sine basis per frequency.
        """
        bases = []
        
        # Constant term (frequency 0)
        bases.append(BasisInfo(
            name="Fourier_const",
            dim=dim,
            params={"freq": 0, "L": L},
            func=lambda x, **_: 1.0
        ))
        
        # Cosine and sine terms for each frequency
        for freq in range(1, max_freq + 1):
            # Cosine term
            bases.append(BasisInfo(
                name=f"Fourier_cos{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=lambda x, freq=freq, L=L: np.cos(2 * np.pi * freq * x / L)
            ))
            
            # Sine term
            bases.append(BasisInfo(
                name=f"Fourier_sin{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=lambda x, freq=freq, L=L: np.sin(2 * np.pi * freq * x / L)
            ))
        
        return bases
    
    @staticmethod
    def complex_fourier_basis(dim: int, max_freq: int, L: float = 2*np.pi) -> List[BasisInfo]:
        """Create the complex Fourier basis φ_f(x) = exp(2πi f x / L) for
        f = -max_freq, ..., max_freq.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_freq : int
            Highest frequency magnitude (inclusive).
        L : float, default=2π
            Period of the exponentials.

        Returns
        -------
        List[BasisInfo]
            One complex exponential basis per frequency from -max_freq to max_freq.
        """
        bases = []
        
        # Negative frequencies
        for freq in range(-max_freq, max_freq + 1):
            bases.append(BasisInfo(
                name=f"ComplexFourier_freq{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=lambda x, freq=freq, L=L: np.exp(2j * np.pi * freq * x / L)
            ))
        
        return bases
    
    @staticmethod
    def discrete_cosine_transform(dim: int, max_order: int, type_: int = 2) -> List[BasisInfo]:
        """Create discrete cosine transform (DCT) basis functions.

        DCT-II: φ_k(x) = cos(πk(x + 0.5)/N); DCT-IV: φ_k(x) = cos(π(k + 0.5)(x + 0.5)/N),
        for k = 0, ..., max_order and N = max_order. For N = 0 every basis
        degenerates to the constant 1.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        max_order : int
            Highest basis index (inclusive); also the normalization N.
        type_ : int, default=2
            DCT variant: 2 (DCT-II) or 4 (DCT-IV).

        Returns
        -------
        List[BasisInfo]
            One DCT basis per index from 0 to max_order (empty if type_ is
            neither 2 nor 4).
        """
        bases = []
        
        if type_ == 2:  # DCT-II (most common)
            for k in range(max_order + 1):
                bases.append(BasisInfo(
                    name=f"DCT-II_order{k}",
                    dim=dim,
                    params={"k": k, "type": 2},
                    func=lambda x, k=k: np.cos(np.pi * k * (x + 0.5) / max_order) if max_order > 0 else 1.0
                ))
        elif type_ == 4:  # DCT-IV
            for k in range(max_order + 1):
                bases.append(BasisInfo(
                    name=f"DCT-IV_order{k}",
                    dim=dim,
                    params={"k": k, "type": 4},
                    func=lambda x, k=k: np.cos(np.pi * (k + 0.5) * (x + 0.5) / max_order) if max_order > 0 else 1.0
                ))
        
        return bases
