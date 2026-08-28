"""Fourier basis functions."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class FourierBasis:
    """Fourier basis functions."""
    
    @staticmethod
    def create_basis(dim: int, max_freq: int, L: float = 2*np.pi) -> List[BasisInfo]:
        """Create Fourier basis functions up to specified frequency."""
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
        """Create complex Fourier basis functions."""
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
        """Create Discrete Cosine Transform (DCT) basis functions."""
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
