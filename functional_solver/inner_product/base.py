"""函数求解器的内积基类。"""

from typing import Callable, Dict, Any, Optional
import numpy as np
from ..core.data_container import MultiDimData


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """一维复合梯形求积的逐点权重。

    对排序去重后的坐标 v₀ < v₁ < ... < v_{m-1}，权重为：
        端点 h₀ = (v₁-v₀)/2, h_{m-1} = (v_{m-1}-v_{m-2})/2,
        内部 hᵢ = (v_{i+1} - v_{i-1})/2，
    于是 Σᵢ f(vᵢ)·hᵢ 就是复合梯形法则 ∫f dv。
    坐标值重复出现的点共享其唯一值的权重（散点/网格数据均适用）。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.ones(n)

    order = np.argsort(x, kind="stable")
    xs = x[order]
    unique, inverse = np.unique(xs, return_inverse=True)
    m = len(unique)

    if m == 1:
        h = np.ones(m)          # 只有一个坐标值：退化为不加体积元
    elif m == 2:
        d = unique[1] - unique[0]
        h = np.array([d / 2.0, d / 2.0])
    else:
        h = np.empty(m)
        h[0] = (unique[1] - unique[0]) / 2.0
        h[-1] = (unique[-1] - unique[-2]) / 2.0
        h[1:-1] = (unique[2:] - unique[:-2]) / 2.0
    h = np.maximum(h, 0.0)

    w = h[inverse]
    back = np.empty(n)
    back[order] = w
    return back


class InnerProduct:
    """函数空间上的内积定义。"""

    def __init__(self, weight_func: Optional[Callable] = None,
                 is_continuous: bool = True):
        """
        初始化内积。

        参数
        ----
        weight_func : Callable, 可选
            加权内积中的权重函数 w(x)
        is_continuous : bool, 默认=True
            内积是定义在连续函数上（True，用数值积分）还是离散数据上（False，求和）
        """
        self.weight_func = weight_func
        self.is_continuous = is_continuous

    def __call__(self, f: np.ndarray, g: np.ndarray,
                 data: Optional[MultiDimData] = None) -> float:
        """
        计算两个函数向量之间的内积。

        参数
        ----
        f : np.ndarray
            第一个函数在数据点上的取值
        g : np.ndarray
            第二个函数在数据点上的取值
        data : MultiDimData, 可选
            数据容器（连续内积必需）

        返回
        ----
        float
            内积 ⟨f, g⟩
        """
        if self.is_continuous:
            if data is None:
                raise ValueError("data is required for continuous inner product")
            return self._continuous_inner_product(f, g, data)
        else:
            return self._discrete_inner_product(f, g, data)

    # ---------- 求积工具 ----------

    def _volume_weights(self, data: MultiDimData) -> np.ndarray:
        """多维体积元权重：各维梯形权重的逐点乘积（多维复合梯形法则）。

        对张量积网格（如 meshgrid 展平的数据）这就是标准的多维梯形求积；
        对散点数据是合理近似（每一维按该维坐标的间距近似）。
        """
        vw = np.ones(data.n_points)
        for dim in data.dims:
            vw = vw * _trapezoid_weights(data.get_dim(dim))
        return vw

    def _point_weights(self, data: MultiDimData) -> np.ndarray:
        """权重函数 w(x) 在数据点上的取值。"""
        if self.weight_func is None:
            return np.ones(data.n_points)
        if data.n_dims == 1:
            return np.asarray(self.weight_func(data.get_dim(data.dims[0])), dtype=float)
        # 多维：逐点以字典形式求值（WeightFunction 的指数/多项式等支持字典输入）
        w = np.empty(data.n_points)
        points = data.get_all_points()
        for i, p in enumerate(points):
            wi = self.weight_func(p)
            w[i] = float(np.asarray(wi).ravel()[0])
        return w

    def _continuous_inner_product(self, f: np.ndarray, g: np.ndarray,
                                  data: MultiDimData) -> float:
        """连续内积：⟨f,g⟩ = ∫ f(x)g(x)w(x)dV，用多维复合梯形法则离散。"""
        integrand = f * g * self._volume_weights(data)
        if self.weight_func is not None:
            integrand = integrand * self._point_weights(data)
        return float(np.sum(integrand))

    def _discrete_inner_product(self, f: np.ndarray, g: np.ndarray,
                                data: Optional[MultiDimData] = None) -> float:
        """离散内积：⟨f,g⟩ = Σᵢ fᵢgᵢ（可加权）。"""
        if self.weight_func is not None:
            if data is not None and len(data.dims) > 0:
                w = self._point_weights(data)
            else:
                w = np.asarray(self.weight_func(np.arange(len(f))), dtype=float)
            return float(np.sum(f * g * w))
        return float(np.dot(f, g))

    def compute_gram_matrix(self, Phi: np.ndarray,
                            data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        计算基函数的 Gram 矩阵。

        参数
        ----
        Phi : np.ndarray
            基函数矩阵，形状为 (n_points, n_basis)
        data : MultiDimData, 可选
            数据容器（连续内积必需）

        返回
        ----
        np.ndarray
            Gram 矩阵，形状为 (n_basis, n_basis)，Gᵢⱼ = ⟨φᵢ, φⱼ⟩
        """
        n_basis = Phi.shape[1]
        G = np.zeros((n_basis, n_basis))

        for i in range(n_basis):
            for j in range(i, n_basis):
                g_ij = self(Phi[:, i], Phi[:, j], data)
                G[i, j] = g_ij
                if i != j:
                    G[j, i] = g_ij

        return G

    def rhs_vector(self, Phi: np.ndarray, target: np.ndarray,
                   data: Optional[MultiDimData] = None) -> np.ndarray:
        """
        计算法方程的右端向量：bᵢ = ⟨φᵢ, target⟩。

        与 Gram 矩阵使用同一内积定义，保证 Gc = b 是同一内积空间下的
        正交投影方程（一致性：一维下 bᵢ = ∫φᵢ(x)y(x)w(x)dx）。
        """
        n_basis = Phi.shape[1]
        return np.array([self(Phi[:, i], target, data) for i in range(n_basis)])

    def norm(self, f: np.ndarray, data: Optional[MultiDimData] = None) -> float:
        """
        计算函数的范数。

        参数
        ----
        f : np.ndarray
            函数取值
        data : MultiDimData, 可选
            数据容器

        返回
        ----
        float
            范数 ||f|| = sqrt(⟨f, f⟩)
        """
        return np.sqrt(self(f, f, data))

    def __repr__(self) -> str:
        return f"InnerProduct(weight_func={self.weight_func is not None}, is_continuous={self.is_continuous})"

    def __str__(self) -> str:
        return self.__repr__()
