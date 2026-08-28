"""小波基函数。"""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class WaveletBasis:
    """小波基函数。"""
    
    @staticmethod
    def mexican_hat(dim: int, center: float, scale: float) -> BasisInfo:
        """墨西哥帽小波（Ricker 小波）。"""
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
        """Morlet 小波。"""
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
        """Haar 小波。"""
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
        """创建包含多个尺度和平移的小波基函数。"""
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
