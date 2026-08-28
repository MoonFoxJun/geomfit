"""马氏距离 RBF 核（把维度间相关性吸收进距离度量）。

数学形式
--------
    k(x, y) = σ² · exp(-½ (x-y)ᵀ M (x-y))

其中 M 为半正定度量矩阵（精度矩阵）。当 M = Σ⁻¹（协方差的逆）时：

- 强相关方向上的距离被压缩、弱相关方向被拉伸 → 核形状沿数据主轴“拉长”，
  与“PCA 白化 + 普通 RBF”完全等价（M = V diag(1/λ) Vᵀ 就是白化变换）；
- 避免了“用方形直积核去拟合窄带数据”造成的病态与浪费。

用法
----
    kernel = MahalanobisKernel(sigma=1.0)          # metric 自动从数据估计
    kernel = MahalanobisKernel(sigma=1.0, metric=...)  # 或显式给出度量矩阵

注意：metric=None 时，度量在首次 compute_matrix（即 solve 阶段）从训练数据
自动估计；单独调用 kernel(x, y)（尚无数据上下文）时退化为欧氏距离核。
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import Kernel


class MahalanobisKernel(Kernel):
    """马氏距离高斯核。"""

    def __init__(self, sigma: float = 1.0,
                 metric: Optional[np.ndarray] = None,
                 alpha: float = 1e-6):
        """
        初始化马氏核。

        参数
        ----
        sigma : float, 默认=1.0
            输出尺度
        metric : np.ndarray, 可选
            度量矩阵 M（半正定）。为 None 时在 compute_matrix 时
            从数据自动估计正则化精度矩阵 (Σ + αI)⁻¹
        alpha : float, 默认=1e-6
            估计度量时的正则化参数（防止近零方差方向爆炸）
        """
        super().__init__(sigma=sigma, metric=metric, alpha=alpha)
        self.sigma = sigma
        self.metric = metric
        self.alpha = alpha
        self._effective_metric = None

    # ---------- 度量 ----------

    def _estimate_metric(self, X: np.ndarray) -> np.ndarray:
        """从数据估计正则化精度矩阵 M = (Σ + αI)⁻¹。"""
        n, d = X.shape
        if n < 2:
            return np.eye(d)
        Xc = X - X.mean(axis=0)
        cov = (Xc.T @ Xc) / (n - 1)
        eigval, eigvec = np.linalg.eigh(cov)
        reg = np.maximum(eigval, self.alpha)          # 抑制近零方差方向
        return (eigvec / reg) @ eigvec.T              # V diag(1/reg) Vᵀ

    def _get_metric(self, data: Any) -> np.ndarray:
        if self.metric is not None:
            return np.asarray(self.metric, dtype=float)
        if self._effective_metric is None:
            X = data.get_coordinate_matrix()
            self._effective_metric = self._estimate_metric(X)
        return self._effective_metric

    # ---------- 求值 ----------

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的马氏核值。

        要求坐标维度为 0..d-1（PCA 旋转后的坐标满足这一点）。
        """
        M = self._effective_metric if self._effective_metric is not None \
            else (np.asarray(self.metric, dtype=float) if self.metric is not None else None)

        if M is None:
            # 尚无数据上下文：退化为欧氏距离 RBF
            d2 = 0.0
            for dim in set(x.keys()) | set(y.keys()):
                d2 += (x.get(dim, 0.0) - y.get(dim, 0.0)) ** 2
            return self.sigma ** 2 * np.exp(-0.5 * d2)

        d = M.shape[0]
        diff = np.array([x.get(i, 0.0) - y.get(i, 0.0) for i in range(d)])
        d2 = float(diff @ M @ diff)
        return self.sigma ** 2 * np.exp(-0.5 * d2)

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        高效计算马氏核矩阵。

        d² = (x-y)ᵀM(x-y) = xᵀMx + yᵀMy - 2xᵀMy，向量化实现。
        """
        from ..core.data_container import MultiDimData

        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")

        X = data.get_coordinate_matrix()
        M = self._get_metric(data)

        XM = X @ M
        quad = np.sum(XM * X, axis=1)                 # diag(X M Xᵀ)
        d2 = quad[:, None] + quad[None, :] - 2.0 * (XM @ X.T)
        d2 = np.maximum(d2, 0.0)                      # 消除浮点误差
        return self.sigma ** 2 * np.exp(-0.5 * d2)
