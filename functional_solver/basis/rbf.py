"""径向基函数（RBF）。"""

import numpy as np
from typing import List, Dict, Any
from ..core.basis_container import BasisInfo

class RBFBasis:
    """径向基函数。"""
    
    @staticmethod
    def gaussian(dim: int, center: float, sigma: float = 1.0) -> BasisInfo:
        """高斯 RBF。"""
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
        """多重二次 RBF。"""
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
        """逆多重二次 RBF。"""
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
        """薄板样条 RBF。"""
        def func(x, center):
            r = np.abs(x - center)
            return r ** 2 * np.log(r + 1e-10)  # 加一个小 epsilon 以避免 log(0)
        
        return BasisInfo(
            name=f"ThinPlateSplineRBF_center{center}",
            dim=dim,
            params={"center": center},
            func=func
        )
    
    @staticmethod
    def cubic(dim: int, center: float) -> BasisInfo:
        """三次 RBF。"""
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
        """线性 RBF。"""
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
        """创建包含多个中心的 RBF 基函数。"""
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
