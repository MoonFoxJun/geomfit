"""基函数工厂 - 提供各种预定义基函数"""

import numpy as np
from typing import Callable, Dict, List
from ..core.basis_container import BasisInfo

class BasisFactory:
    """基函数工厂，提供各种预定义基"""
    
    @staticmethod 
    def polynomial(dim: int, order: int):
        """多项式基：x^n"""
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
        """勒让德多项式（更数值稳定）"""
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
        """傅里叶基：cos(2πf x/L) 和 sin(2πf x/L)"""
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
        """洛朗级数基：x^n 和 x^{-n} (x ≠ 0)"""
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
        """高斯径向基函数"""
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
        """小波基函数"""
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
        多维张量积基：给定每个维度的基函数列表，生成所有乘积组合。

        例如 dim_bases = {0: [X₁, X₂], 1: [Y₁, Y₂]} 生成 X₁⊗Y₁, X₁⊗Y₂, X₂⊗Y₁, X₂⊗Y₂，
        数学形式 f(x,y) = Σᵢ Σⱼ cᵢⱼ Xᵢ(x)Yⱼ(y)。

        参数
        ----
        dim_bases : Dict[int, List[BasisInfo]]
            维度索引 -> 该维度的基函数列表

        返回
        ----
        List[BasisInfo]
            所有乘积组合的基函数（每个都是多因子张量积基）
        """
        import itertools

        if not dim_bases:
            raise ValueError("dim_bases must contain at least one dimension")
        dims = sorted(dim_bases.keys())
        lists = [dim_bases[d] for d in dims]

        products = []
        for combo in itertools.product(*lists):
            # 展开每个因子的因子（若因子本身是张量积基则递归展开）
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
        """完全自定义的基函数"""
        return BasisInfo(
            name=name,
            dim=dim,
            params=params or {},
            func=func
        )
