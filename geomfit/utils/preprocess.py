"""Data preprocessing: PCA coordinate rotation (decorrelating strongly
correlated dimensions).

Background
----------
Tensor-product bases implicitly assume that data lie on a Cartesian product
domain (a hypercube) with orthogonal coordinates. When data are concentrated
along an oblique direction (e.g. a narrow band y ≈ x):

- basis functions become almost linearly dependent on the data, which makes
  the Gram matrix ill-conditioned and the coefficients huge;
- most basis functions waste degrees of freedom on regions without data.

PCA rotates the coordinates onto the orthogonal principal axes of the data
variance, making the principal components mutually uncorrelated:

- after the rotation, near-collinearity of the basis columns disappears;
- dropping near-zero-variance components (n_components < d) reduces dimension.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..core.data_container import MultiDimData


def pca_rotate(data: MultiDimData,
               n_components: Optional[int] = None,
               whiten: bool = False) -> Tuple[MultiDimData, Dict[str, Any]]:
    """
    Rotate (and optionally truncate or whiten) the coordinates of data via PCA.

    The rotation is obtained from the SVD of the centered coordinate matrix,
    Xc = U S Vᵀ, so that the columns of V are the principal directions.

    Parameters
    ----------
    data : MultiDimData
        Input data.
    n_components : int, optional
        Number of principal components to keep (default: keep all).
    whiten : bool, default=False
        If True, scale each principal component by its singular value so that
        the rotated axes have unit variance.

    Returns
    -------
    rotated : MultiDimData
        Rotated data with dimension indices 0..k-1, optionally truncated or
        whitened.
    info : Dict
        Dictionary with "components" (V, the d x k rotation matrix), "mean"
        (centering vector), "singular_values", "explained_variance_ratio",
        and "whiten". Pass it to pca_transform / pca_rotate_back to process
        new data.
    """
    X = data.get_coordinate_matrix()          # 坐标矩阵，形状 (n, d)：n 个数据点 × d 个维度
    n, d = X.shape
    mean = X.mean(axis=0)                     # 按列求均值 → 中心化向量（各维度的数据中心）
    Xc = X - mean                             # 中心化：每个维度减去均值，使数据中心移到原点

    # PCA 核心：对中心化矩阵做 SVD 分解 Xc = U S Vᵀ。
    # 数学事实：XcᵀXc / n 是协方差矩阵，而 V 的列正是协方差矩阵的
    # 特征向量（主方向），S 的对角元 s 是奇异值，s² 正比于特征值（方差）。
    # 因此 V 的列 = 数据方差最大的正交方向（主轴），
    # 把数据旋转到这些主轴上就实现了去相关。
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    if n_components is None:
        k = d                                  # 默认保留全部主成分（不降维）
    else:
        k = int(n_components)
        # 把 k 限制在 [1, d] 之间：至少保留 1 个、至多保留 d 个主成分。
        # 降维的本质是丢弃方差接近 0 的方向（那些方向上数据几乎没有
        # 变化，保留它们只会增加基函数之间的近似线性相关性）。
        k = min(max(k, 1), d)

    V = Vt[:k].T                             # 旋转矩阵 (d, k)：取 Vᵀ 的前 k 行再转置，得到前 k 个主方向
    Y = Xc @ V                                # 旋转：把中心化数据投影到主方向上 → 新坐标 Y（主成分得分）
    if whiten:
        # 白化：把每个主成分除以对应的奇异值，使各主轴方向的方差都变成 1。
        # 除以 np.maximum(s[:k], 1e-12) 是为了防止奇异值为 0 时除零
        #（1e-12 是数值稳定下限）。白化后各分量单位方差、互不相关。
        Y = Y / np.maximum(s[:k], 1e-12)

    # 解释方差比例：第 i 个主成分的方差占总体方差的比例 = s_i² / Σs²。
    # （因为 ‖Xc‖_F² = Σs² 正好是中心化数据的总平方和，即总方差。）
    total_var = float(np.sum(s ** 2))
    ratio = (s ** 2) / total_var if total_var > 0 else np.zeros_like(s)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values             # 目标值跟随数据点移动，坐标旋转不改变目标值本身

    info = {
        "components": V,
        "mean": mean,
        "singular_values": s,
        "explained_variance_ratio": ratio,
        "whiten": whiten,
    }
    return rotated, info


def pca_transform(data: MultiDimData, info: Dict[str, Any]) -> MultiDimData:
    """
    Transform new data with PCA information fitted by pca_rotate.

    New data (test set or grid) must use the same rotation fitted on the
    training data, otherwise the coordinate systems are inconsistent.
    """
    X = data.get_coordinate_matrix()
    V = info["components"]
    mean = info["mean"]
    k = V.shape[1]

    # 关键点：新数据必须使用 pca_rotate 在训练数据上拟合出的同一套
    # 变换（同一中心 mean、同一旋转矩阵 V、同一白化缩放），
    # 否则训练与测试处于不同的坐标系，预测结果没有意义。
    Y = (X - mean) @ V
    if info.get("whiten"):
        # 若训练时做了白化，新数据也要用训练时的奇异值做同样的缩放
        s = info["singular_values"]
        Y = Y / np.maximum(s[:k], 1e-12)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values
    return rotated


def pca_rotate_back(point: Dict[int, float], info: Dict[str, Any]) -> Dict[int, float]:
    """
    Map a single point from the rotated coordinates back to the original
    coordinates (e.g. for interpretation or plotting).

    Parameters
    ----------
    point : Dict[int, float]
        Point in the rotated coordinates (keys 0..k-1).
    info : Dict
        Information returned by pca_rotate.

    Returns
    -------
    Dict[int, float]
        Point in the original coordinates.
    """
    k = len(point)
    # 把旋转坐标系下的点组装成向量 y（按键排序保证分量顺序一致）
    y = np.array([point[i] for i in sorted(point.keys())], dtype=float)
    V = info["components"]
    # 还原变换：旋转是正交变换，逆变换就是 Vᵀ 的右乘；
    # 再加回中心化时减掉的均值 mean，即 x = y·Vᵀ + mean。
    # 注意白化过的坐标要先乘回奇异值才能还原，本函数不处理白化逆缩放。
    x = y @ V.T + info["mean"]
    return {i: float(xi) for i, xi in enumerate(x)}
