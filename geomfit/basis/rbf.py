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
            # 高斯 RBF：φ(x) = exp(−(x − center)²/(2σ²))。距离 |x − center| 越远
            # 取值越小（无限支撑、光滑），σ 控制核宽度：σ 越小峰值越尖、影响范围越窄
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
            # 多重二次（multiquadric）RBF：φ(x) = √(1 + (ε(x − center))²)，
            # 完全正定且全局支撑；ε 是形状参数，控制函数随距离增长的弯曲程度
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
            # 逆多重二次 RBF：φ(x) = 1/√(1 + (ε(x − center))²)，是多重二次的倒数形式，
            # 正定、有界且在 x = center 处取最大值 1，常用于保证插值矩阵可逆
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
            # 薄板样条 RBF：φ(x) = r² log r（r = |x − center|），是条件正定核，
            # 常用于曲面插值。数学上 r²log r 在 r → 0 时趋于 0，但直接算 log(0)
            # 会报错/得 -inf，所以给 r 加一个极小量 1e-10 来数值上避开这个奇点
            r = np.abs(x - center)
            return r ** 2 * np.log(r + 1e-10)  # 加极小量 ε 避免 log(0)
        
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
            # 三次 RBF：φ(x) = r³（r = |x − center|），条件正定核；
            # 实际使用时常与低阶多项式（如常数/线性）联合以保证插值问题适定
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
            # 线性 RBF：φ(x) = r = |x − center|，即到中心的距离本身
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
        
        # 对每个中心点生成一个同类型的 RBF 基；rbf_type 选择径向函数族，
        # 各类型自己的形状参数（sigma / epsilon）从 kwargs 读取，缺省取 1.0
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
