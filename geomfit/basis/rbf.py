"""Radial basis functions (RBFs) over a single dimension, centered at c."""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class RBFBasis:
    """Radial basis functions φ(|x − c|) over a single dimension."""
    
    @staticmethod
    def gaussian(dim: int, center: float, sigma: float = 1.0) -> BasisInfo:
        """Gaussian RBF: φ(x) = exp(−(x − center)²/(2σ²)).

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
            name=f"Gaussian_RBF_center{center}_sigma{sigma}",
            dim=dim,
            params={"center": center, "sigma": sigma},
            func=func
        )
    
    @staticmethod
    def multiquadric(dim: int, center: float, epsilon: float = 1.0) -> BasisInfo:
        """Multiquadric RBF: φ(x) = √(1 + (ε(x − center))²).

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the RBF.
        epsilon : float, default=1.0
            Shape parameter.

        Returns
        -------
        BasisInfo
            Single-factor multiquadric RBF basis.
        """
        def func(x, center, epsilon):
            return np.sqrt(1 + (epsilon * (x - center)) ** 2)
        
        return BasisInfo(
            name=f"MultiquadricRBF_center{center}_epsilon{epsilon}",
            dim=dim,
            params={"center": center, "epsilon": epsilon},
            func=func
        )
    
    @staticmethod
    def inverse_multiquadric(dim: int, center: float, epsilon: float = 1.0) -> BasisInfo:
        """Inverse multiquadric RBF: φ(x) = 1/√(1 + (ε(x − center))²).

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the RBF.
        epsilon : float, default=1.0
            Shape parameter.

        Returns
        -------
        BasisInfo
            Single-factor inverse-multiquadric RBF basis.
        """
        def func(x, center, epsilon):
            return 1.0 / np.sqrt(1 + (epsilon * (x - center)) ** 2)
        
        return BasisInfo(
            name=f"InverseMultiquadricRBF_center{center}_epsilon{epsilon}",
            dim=dim,
            params={"center": center, "epsilon": epsilon},
            func=func
        )
    
    @staticmethod
    def thin_plate_spline(dim: int, center: float) -> BasisInfo:
        """Thin-plate-spline RBF: φ(x) = r² log(r), r = |x − center|.

        A small epsilon (1e-10) inside the logarithm avoids log(0) at r = 0.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the RBF.

        Returns
        -------
        BasisInfo
            Single-factor thin-plate-spline basis.
        """
        def func(x, center):
            r = np.abs(x - center)
            return r ** 2 * np.log(r + 1e-10)  # small epsilon avoids log(0)
        
        return BasisInfo(
            name=f"ThinPlateSplineRBF_center{center}",
            dim=dim,
            params={"center": center},
            func=func
        )
    
    @staticmethod
    def cubic(dim: int, center: float) -> BasisInfo:
        """Cubic RBF: φ(x) = r³, r = |x − center|.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the RBF.

        Returns
        -------
        BasisInfo
            Single-factor cubic RBF basis.
        """
        def func(x, center):
            r = np.abs(x - center)
            return r ** 3
        
        return BasisInfo(
            name=f"CubicRBF_center{center}",
            dim=dim,
            params={"center": center},
            func=func
        )
    
    @staticmethod
    def linear(dim: int, center: float) -> BasisInfo:
        """Linear RBF: φ(x) = r, r = |x − center|.

        Parameters
        ----------
        dim : int
            Dimension index the basis acts on.
        center : float
            Center of the RBF.

        Returns
        -------
        BasisInfo
            Single-factor linear RBF basis.
        """
        def func(x, center):
            r = np.abs(x - center)
            return r
        
        return BasisInfo(
            name=f"LinearRBF_center{center}",
            dim=dim,
            params={"center": center},
            func=func
        )
    
    @staticmethod
    def create_rbf_basis(dim: int, centers: List[float], rbf_type: str = "gaussian", 
                        **kwargs) -> List[BasisInfo]:
        """Create one RBF basis per center in `centers`.

        Parameters
        ----------
        dim : int
            Dimension index the bases act on.
        centers : List[float]
            RBF centers.
        rbf_type : str, default="gaussian"
            One of "gaussian", "multiquadric", "inverse_multiquadric",
            "thin_plate_spline", "cubic" or "linear".
        **kwargs
            Type-specific parameters, e.g. sigma (gaussian) or epsilon
            (multiquadric, inverse_multiquadric).

        Returns
        -------
        List[BasisInfo]
            One RBF basis per center.
        """
        bases = []
        
        for center in centers:
            if rbf_type == "gaussian":
                sigma = kwargs.get("sigma", 1.0)
                bases.append(RBFBasis.gaussian(dim, center, sigma))
            elif rbf_type == "multiquadric":
                epsilon = kwargs.get("epsilon", 1.0)
                bases.append(RBFBasis.multiquadric(dim, center, epsilon))
            elif rbf_type == "inverse_multiquadric":
                epsilon = kwargs.get("epsilon", 1.0)
                bases.append(RBFBasis.inverse_multiquadric(dim, center, epsilon))
            elif rbf_type == "thin_plate_spline":
                bases.append(RBFBasis.thin_plate_spline(dim, center))
            elif rbf_type == "cubic":
                bases.append(RBFBasis.cubic(dim, center))
            elif rbf_type == "linear":
                bases.append(RBFBasis.linear(dim, center))
            else:
                raise ValueError(f"Unknown RBF type: {rbf_type}")
        
        return bases
