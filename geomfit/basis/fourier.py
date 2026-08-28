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
        
        # 常数项（频率 0）：函数恒等于 1，是傅里叶展开的直流分量（平均值项）
        bases.append(BasisInfo(
            name="Fourier_const",
            dim=dim,
            params={"freq": 0, "L": L},
            func=lambda x, **_: 1.0
        ))
        
        # 对每个频率 f 生成一对基：cos 项 + sin 项（实傅里叶基，总数 = 2·max_freq + 1）
        for freq in range(1, max_freq + 1):
            # 余弦项 cos(2πf x/L)：在 [0, L] 上完成 f 个完整周期；
            # lambda 用默认参数 freq=freq, L=L 冻结循环变量，避免闭包陷阱
            bases.append(BasisInfo(
                name=f"Fourier_cos{freq}",
                dim=dim,
                params={"freq": freq, "L": L},
                func=lambda x, freq=freq, L=L: np.cos(2 * np.pi * freq * x / L)
            ))
            
            # 正弦项 sin(2πf x/L)：与余弦项正交，共同张成周期函数空间
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
        
        # 遍历 -max_freq 到 max_freq 的所有整数频率，每个频率生成一个复指数基
        # φ_f(x) = exp(2πi f x / L)；由欧拉公式 exp(iθ) = cosθ + i·sinθ 可知，
        # 复指数基与实傅里叶基一一对应（cos/sin 分别是复指数的实部与虚部），
        # 负频率项使复数展开比实数展开更紧凑（每对 ±f 只需一个复系数）
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
        
        if type_ == 2:  # DCT-II（最常用的变体）
            # DCT-II：φ_k(x) = cos(πk(x + 0.5)/N)，k = 0..max_order。
            # 与 DFT 相比只用余弦（实数输出），且 (x + 0.5) 的半格点偏移使能量更集中、
            # 去相关能力更强（图像/音频压缩中大量使用）。注意 N = max_order：
            # 当 max_order = 0 时分母为 0，所有基退化为常数 1
            for k in range(max_order + 1):
                bases.append(BasisInfo(
                    name=f"DCT-II_order{k}",
                    dim=dim,
                    params={"k": k, "type": 2},
                    func=lambda x, k=k: np.cos(np.pi * k * (x + 0.5) / max_order) if max_order > 0 else 1.0
                ))
        elif type_ == 4:  # DCT-IV
            # DCT-IV：φ_k(x) = cos(π(k + 0.5)(x + 0.5)/N)，相比 DCT-II 多了
            # (k + 0.5) 的频移，边界条件不同（偶对称中心位于两点中点），
            # 常用于 MDCT 等需要"时间域重叠"的变换；N = max_order 为 0 时同样退化为常数 1
            for k in range(max_order + 1):
                bases.append(BasisInfo(
                    name=f"DCT-IV_order{k}",
                    dim=dim,
                    params={"k": k, "type": 4},
                    func=lambda x, k=k: np.cos(np.pi * (k + 0.5) * (x + 0.5) / max_order) if max_order > 0 else 1.0
                ))
        
        return bases
