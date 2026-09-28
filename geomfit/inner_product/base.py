"""Inner product base class for the functional solver."""

from typing import Callable, Dict, Any, Optional
import numpy as np
from ..core.data_container import MultiDimData


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Per-point weights for one-dimensional composite trapezoid quadrature.

    For sorted, deduplicated coordinates v₀ < v₁ < ... < v_{m-1}, the weights are
        endpoints h₀ = (v₁-v₀)/2, h_{m-1} = (v_{m-1}-v_{m-2})/2,
        interior hᵢ = (v_{i+1} - v_{i-1})/2,
    so that Σᵢ f(vᵢ)·hᵢ is the composite trapezoid rule ∫f dv. Points sharing a
    repeated coordinate value receive the weight of their unique value (works
    for both scattered and gridded data).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.ones(n)  # 点数不足 2 时无法构成积分区间，权重取 1（退化为平凡求和，无体积元）

    order = np.argsort(x, kind="stable")  # 稳定排序：按坐标值排序，便于后续找重复值与相邻间距
    xs = x[order]
    unique, inverse = np.unique(xs, return_inverse=True)  # unique 为去重后的有序坐标，inverse 把每个原值映射回唯一值下标
    m = len(unique)  # 去重后的坐标个数

    if m == 1:
        # 所有点共享同一坐标值：区间退化，没有长度可言，权重取 1
        h = np.ones(m)
    elif m == 2:
        d = unique[1] - unique[0]
        h = np.array([d / 2.0, d / 2.0])  # 只有两个点：梯形法则退化为两端点各占 h/2
    else:
        h = np.empty(m)
        h[0] = (unique[1] - unique[0]) / 2.0        # 左端点权重：第一段间距的一半
        h[-1] = (unique[-1] - unique[-2]) / 2.0     # 右端点权重：最后一段间距的一半
        h[1:-1] = (unique[2:] - unique[:-2]) / 2.0  # 内部点权重：(v_{i+1} - v_{i-1})/2，即左右相邻间距之和的一半
    h = np.maximum(h, 0.0)  # 坐标未严格递增时数值上可能算出负权重，一律截断为 0

    w = h[inverse]     # 重复坐标的点共享其唯一坐标值的权重（"重复坐标共享权重"的关键一步）
    back = np.empty(n)
    back[order] = w    # 把按排序顺序算出的权重还原回原始输入顺序
    return back


# 公共别名：分离式（网格）求解器需要复用同一条一维求积规则，
# 以便把多维体积元拆成各维权重之积。
trapezoid_weights = _trapezoid_weights


class InnerProduct:
    """Definition of an inner product on a function space."""

    def __init__(self, weight_func: Optional[Callable] = None,
                 is_continuous: bool = True):
        """
        Initialize the inner product.

        Parameters
        ----------
        weight_func : Callable, optional
            Weight function w(x) of the weighted inner product.
        is_continuous : bool, default=True
            Whether the inner product is defined on continuous functions
            (True, via numerical integration) or on discrete data
            (False, via summation).
        """
        self.weight_func = weight_func
        self.is_continuous = is_continuous

    def __call__(self, f: np.ndarray, g: np.ndarray,
                 data: Optional[MultiDimData] = None) -> float:
        """
        Compute the inner product of two function vectors.

        Parameters
        ----------
        f : np.ndarray
            Values of the first function at the data points.
        g : np.ndarray
            Values of the second function at the data points.
        data : MultiDimData, optional
            Data container (required for continuous inner products).

        Returns
        -------
        float
            Inner product ⟨f, g⟩.
        """
        if self.is_continuous:
            if data is None:
                raise ValueError("data is required for continuous inner product")
            return self._continuous_inner_product(f, g, data)
        else:
            return self._discrete_inner_product(f, g, data)

    # ---------- 数值求积工具 ----------

    def _volume_weights(self, data: MultiDimData) -> np.ndarray:
        """Multi-dimensional volume-element weights: per-dimension trapezoid
        weights multiplied pointwise (multi-dimensional composite trapezoid rule).

        For tensor-product grids (e.g. flattened meshgrid data) this is the
        standard multi-dimensional trapezoid quadrature; for scattered data it
        is a reasonable approximation (each dimension weighted by the spacing
        of its own coordinates).
        """
        # 多维体积元 dV = dv₁·dv₂·...·dv_d：把各维的梯形权重逐点相乘。
        # 对张量积网格（如展平的 meshgrid 数据）这是标准的多维复合梯形法则；
        # 对散点数据则是合理近似（每个维度只按自身坐标间距加权）
        vw = np.ones(data.n_points)
        for dim in data.dims:
            vw = vw * _trapezoid_weights(data.get_dim(dim))
        return vw

    def _point_weights(self, data: MultiDimData) -> np.ndarray:
        """Evaluate the weight function w(x) at the data points."""
        if self.weight_func is None:
            return np.ones(data.n_points)  # 未定义权重函数：权重恒为 1（普通内积）
        if data.n_dims == 1:
            return np.asarray(self.weight_func(data.get_dim(data.dims[0])), dtype=float)  # 一维：权重函数可直接作用于整段坐标数组（一次向量化求值）
        # 多维：权重函数以"逐点传字典"的方式求值（exponential、polynomial 等
        # WeightFunction 辅助函数都接受字典输入），故此处逐点循环
        w = np.empty(data.n_points)
        points = data.get_all_points()
        for i, p in enumerate(points):
            wi = self.weight_func(p)                # 每个点单独求权重
            w[i] = float(np.asarray(wi).ravel()[0])  # 返回值可能是标量或数组，统一展平取第一个元素
        return w

    def _continuous_inner_product(self, f: np.ndarray, g: np.ndarray,
                                  data: MultiDimData) -> float:
        """Continuous inner product: ⟨f,g⟩ = ∫ f(x)g(x)w(x)dV, discretized by
        the multi-dimensional composite trapezoid rule."""
        # 连续内积 ⟨f, g⟩ = ∫ f(x)g(x)w(x)dV 的数值离散化：
        # 先逐点相乘 f·g，再乘上多维体积元权重 dV（复合梯形法则的"dx"）
        integrand = f * g * self._volume_weights(data)
        if self.weight_func is not None:
            integrand = integrand * self._point_weights(data)  # 若定义了权重函数 w(x)，再逐点乘上 w(x_i)
        return float(np.sum(integrand))  # 加权求和 Σᵢ fᵢgᵢ·wᵢ·dVᵢ 即梯形法则的最终积分近似

    def _discrete_inner_product(self, f: np.ndarray, g: np.ndarray,
                                data: Optional[MultiDimData] = None) -> float:
        """Discrete inner product: ⟨f,g⟩ = Σᵢ fᵢgᵢ (optionally weighted)."""
        # 离散内积 ⟨f, g⟩ = Σᵢ fᵢgᵢ：纯求和、无体积元；可选地乘上权重函数
        if self.weight_func is not None:
            if data is not None and len(data.dims) > 0:
                w = self._point_weights(data)  # 有数据上下文：在数据点上求权重
            else:
                w = np.asarray(self.weight_func(np.arange(len(f))), dtype=float)  # 无数据：按索引 0..n-1 求权重
            return float(np.sum(f * g * w))  # 加权离散内积 Σᵢ fᵢgᵢwᵢ
        return float(np.dot(f, g))  # 无权重函数：退化为标准点积

    def compute_gram_matrix(self, Phi: np.ndarray,
                            data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        Compute the Gram matrix of basis functions.

        Parameters
        ----------
        Phi : np.ndarray
            Basis function matrix of shape (n_points, n_basis).
        data : MultiDimData, optional
            Data container (required for continuous inner products).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (n_basis, n_basis), with
            G_ij = ⟨φᵢ, φⱼ⟩.
        """
        n_basis = Phi.shape[1]
        G = np.zeros((n_basis, n_basis))

        # Gram 矩阵元素 G_ij = ⟨φᵢ, φⱼ⟩。内积具有对称性 G_ij = G_ji，
        # 因此只计算上三角（含对角线，j ≥ i），再镜像填充下三角，省一半内积求值
        for i in range(n_basis):
            for j in range(i, n_basis):
                g_ij = self(Phi[:, i], Phi[:, j], data)  # 第 i、j 个基函数列向量在数据点上的内积
                G[i, j] = g_ij
                if i != j:
                    G[j, i] = g_ij  # 对称填充：G_ji = G_ij

        return G

    def rhs_vector(self, Phi: np.ndarray, target: np.ndarray,
                   data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        Compute the right-hand-side vector of the normal equations:
        bᵢ = ⟨φᵢ, target⟩.

        Uses the same inner product as the Gram matrix, so that Gc = b is the
        orthogonal projection equation in the same inner product space
        (consistency: in one dimension bᵢ = ∫φᵢ(x)y(x)w(x)dx).
        """
        n_basis = Phi.shape[1]
        # 右端项 b_i = ⟨φᵢ, target⟩：与 Gram 矩阵使用完全相同的内积，保证法方程
        # Gc = b 是同一内积空间中的正交投影方程，从而自洽
        # （一维情形即 b_i = ∫φᵢ(x)y(x)w(x)dx，与 G_ij = ∫φᵢφⱼw dx 配套使用）
        return np.array([self(Phi[:, i], target, data) for i in range(n_basis)])

    def norm(self, f: np.ndarray, data: Optional[MultiDimData] = None) -> float:
        """
        Compute the norm of a function.

        Parameters
        ----------
        f : np.ndarray
            Function values.
        data : MultiDimData, optional
            Data container.

        Returns
        -------
        float
            Norm ||f|| = sqrt(⟨f, f⟩).
        """
        return np.sqrt(self(f, f, data))  # 范数定义 ‖f‖ = √⟨f, f⟩（内积诱导的范数）

    def __repr__(self) -> str:
        return f"InnerProduct(weight_func={self.weight_func is not None}, is_continuous={self.is_continuous})"

    def __str__(self) -> str:
        return self.__repr__()
