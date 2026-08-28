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
            # 单项式基 φ(x) = x^order：直接求 x 的 order 次幂
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
            # 系数向量 [0]*order + [1] 表示"除第 order 个系数为 1 外其余全为 0"，
            # 因此 legval 求值结果恰为 P_order(x)；numpy 内部用三递推公式计算，
            # 比直接算 x^n 的幂级数在数值上更稳定（避免高阶幂带来的灾难性消去）
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
            # 余弦基 cos(2πf x / L)：在区间 [0, L] 上恰好完成 f 个完整周期，
            # 与正弦项一起构成 L-周期函数空间的正交三角函数族
            return np.cos(2 * np.pi * freq * x / L)
        
        def func_sin(x, freq, L):
            # 正弦基 sin(2πf x / L)：与余弦项同为傅里叶展开的频率分量
            return np.sin(2 * np.pi * freq * x / L)
        
        bases = []
        if freq == 0:
            # 频率 0 时 cos/sin 都退化为常数 1，只保留一个常数基（直流分量）
            bases.append(BasisInfo(
                name="Fourier_const",
                dim=dim,
                params={"freq": 0, "L": L},
                func=lambda x, **_: 1.0
            ))
        else:
            # 频率 > 0 时每个频率生成两个基：余弦项 + 正弦项
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
            # 正幂项 x^order：对应洛朗级数展开中的正则（解析）部分
            return x ** order
        
        def func_negative(x, order):
            # 负幂项 x^(-order)：对应洛朗级数的主部；x = 0 处负幂无定义，
            # 这里约定返回 0.0 以保持函数可求值
            return x ** (-order) if x != 0 else 0.0
        
        bases = []
        if order == 0:
            # order = 0 时正幂与负幂重合，只返回一个常数基 1
            bases.append(BasisInfo(
                name="Laurent_const",
                dim=dim,
                params={"order": 0},
                func=lambda x, **_: 1.0
            ))
        else:
            # 每个阶数生成两个基：x^order（正幂）与 x^(-order)（负幂）
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
            # 高斯 RBF：φ(x) = exp(−(x − center)²/(2σ²))，取值随离中心距离增大而
            # 指数衰减；center 决定峰值位置，σ（宽度参数）决定衰减快慢
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
                # 归一化坐标变换：t = (x − translation)/scale（先平移、再伸缩）
                t = (x - translation) / scale
                # 墨西哥帽（Ricker）小波：(1 − t²)exp(−t²/2) 是高斯函数二阶导数的相反数，
                # 波形像一顶帽子，在 t = 0 处取最大值 1，整体积分为 0（满足小波容许条件）
                return (1 - t ** 2) * np.exp(-t ** 2 / 2)
        elif wavelet_type == "morlet":
            def func(x, scale, translation):
                t = (x - translation) / scale
                # Morlet 小波：余弦载波 cos(5t) × 高斯包络 exp(−t²/2)；
                # 载波提供振荡，包络保证波形在时域上快速衰减，从而兼具时频局部化
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
        # 取出每个维度对应的基列表，按维度升序排列，供笛卡尔积使用
        lists = [dim_bases[d] for d in dims]

        products = []
        # itertools.product(*lists) 生成笛卡尔积：从每个维度的基列表中各取一个基，
        # 遍历所有组合。例如 dim_bases = {0: [X₁, X₂], 1: [Y₁, Y₂]} 会得到
        # (X₁,Y₁)、(X₁,Y₂)、(X₂,Y₁)、(X₂,Y₂) 四种组合，对应展开式
        # f(x, y) = Σᵢ Σⱼ cᵢⱼ Xᵢ(x) Yⱼ(y) 中的每个乘积项 Xᵢ(x)Yⱼ(y)，
        # 系数 cᵢⱼ 由之后的最小二乘拟合确定
        for combo in itertools.product(*lists):
            # 把嵌套的张量积基展开成单个因子列表：若某因子本身是张量积基
            # （b.factors 非空），直接把它内部的 factors 拼进来，
            # 避免出现"张量积里再套张量积"的结构
            factors = []
            for b in combo:
                if b.factors is not None:
                    factors.extend(b.factors)
                else:
                    factors.append((b.dim, b.func, b.params))
            # 用 ⊗ 连接各因子的名字，例如 "X₁ ⊗ Y₂"
            name = " ⊗ ".join(b.name for b in combo)
            # 每个组合生成一个多因子张量积基：func 置空、只保留 factors，
            # 主维度取第一个因子的维度（保持与 BasisInfo 单因子约定兼容）
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
            # params 缺省（None）时统一转成空字典，保证求值时 **(params or {}) 解包安全
            params=params or {},
            func=func
        )
