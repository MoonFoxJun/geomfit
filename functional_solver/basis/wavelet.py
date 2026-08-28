"""Wavelet basis functions over a single dimension: Mexican hat, Morlet, and Haar families."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class WaveletBasis:
    """Wavelet basis functions over a single dimension."""
    
    @staticmethod
    def mexican_hat(dim: int, center: float, scale: float) -> BasisInfo:
        """Mexican-hat (Ricker) wavelet: φ(x) = (1 − t²)exp(−t²/2), t = (x − center)/scale.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Wavelet center (translation).
        scale : float
            Wavelet scale.

        Returns
        -------
        BasisInfo
            Single-factor Mexican-hat wavelet basis.
        """
        def func(x, center, scale):
            t = (x - center) / scale
            return (1 - t ** 2) * np.exp(-t ** 2 / 2)
        
        return BasisInfo(
            name=f"MexicanHat_scale{scale}_trans{center}",
            dim=dim,
            params={"center": center, "scale": scale},
            func=func
        )
    
    @staticmethod
    def morlet(dim: int, scale: float, translation: float, omega0: float = 5.0) -> BasisInfo:
        """Morlet wavelet: φ(x) = cos(ω₀t)exp(−t²/2), t = (x − translation)/scale.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        scale : float
            Wavelet scale.
        translation : float
            Wavelet translation.
        omega0 : float, default=5.0
            Central angular frequency of the carrier wave.

        Returns
        -------
        BasisInfo
            Single-factor Morlet wavelet basis.
        """
        def func(x, scale, translation, omega0):
            t = (x - translation) / scale
            return np.cos(omega0 * t) * np.exp(-t ** 2 / 2)
        
        return BasisInfo(
            name=f"Morlet_scale{scale}_trans{translation}_omega{omega0}",
            dim=dim,
            params={"scale": scale, "translation": translation, "omega0": omega0},
            func=func
        )
    
    @staticmethod
    def haar(dim: int, scale: float, translation: float) -> BasisInfo:
        """Haar wavelet: 1 for 0 ≤ t < 0.5, −1 for 0.5 ≤ t < 1, 0 otherwise,
        t = (x − translation)/scale.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        scale : float
            Wavelet scale.
        translation : float
            Wavelet translation.

        Returns
        -------
        BasisInfo
            Single-factor Haar wavelet basis.
        """
        def func(x, scale, translation):
            t = (x - translation) / scale
            if 0 <= t < 0.5:
                return 1.0
            elif 0.5 <= t < 1.0:
                return -1.0
            else:
                return 0.0
        
        return BasisInfo(
            name=f"Haar_scale{scale}_trans{translation}",
            dim=dim,
            params={"scale": scale, "translation": translation},
            func=func
        )
    
    @staticmethod
    def create_wavelet_basis(dim: int, scales: List[float], translations: List[float], 
                            wavelet_type: str = "mexican_hat", **kwargs) -> List[BasisInfo]:
        """Create wavelet bases for every (scale, translation) combination.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        scales : List[float]
            Wavelet scales.
        translations : List[float]
            Wavelet translations.
        wavelet_type : str, default="mexican_hat"
            One of "mexican_hat", "morlet" or "haar".
        **kwargs
            Type-specific parameters, e.g. omega0 (morlet).

        Returns
        -------
        List[BasisInfo]
            One wavelet basis per (scale, translation) combination.
        """
        bases = []
        
        for scale in scales:
            for translation in translations:
                if wavelet_type == "mexican_hat":
                    bases.append(WaveletBasis.mexican_hat(dim, center=translation, scale=scale))
                elif wavelet_type == "morlet":
                    omega0 = kwargs.get("omega0", 5.0)
                    bases.append(WaveletBasis.morlet(dim, scale, translation, omega0))
                elif wavelet_type == "haar":
                    bases.append(WaveletBasis.haar(dim, scale, translation))
                else:
                    raise ValueError(f"Unknown wavelet type: {wavelet_type}")
        
        return bases
