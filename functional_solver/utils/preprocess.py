"""数据预处理：PCA 坐标旋转（解耦强相关维度）。

背景
----
张量积基天然假设数据落在笛卡尔直积域（超立方体）上，且各坐标正交。
当数据集中在某个斜方向上（如窄带 y ≈ x）时：
- 各维基函数在数据上几乎线性相关 → Gram 矩阵病态、系数巨大；
- 大部分基函数在为“没有数据的虚空”分配自由度（浪费）。
PCA 把坐标旋转到数据方差最大的正交主轴上，使各主成分互不相关：
- 旋转后再做张量积，基列近似共线的问题消失；
- 丢弃近零方差的主成分（n_components < d）可顺带降维。
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..core.data_container import MultiDimData


def pca_rotate(data: MultiDimData,
               n_components: Optional[int] = None,
               whiten: bool = False) -> Tuple[MultiDimData, Dict[str, Any]]:
    """
    对 MultiDimData 做 PCA 坐标旋转（解耦/白化）。

    参数
    ----
    data : MultiDimData
        原始数据
    n_components : int, 可选
        保留的主成分个数（默认保留全部）
    whiten : bool, 默认=False
        是否按奇异值缩放（白化）：旋转后各主轴方差归一

    返回
    ----
    (rotated, info)
    rotated : MultiDimData
        旋转（可选降维/白化）后的数据，维度索引为 0..k-1
    info : Dict
        {"components": V (d×k 旋转矩阵), "mean": 中心向量,
         "singular_values": 奇异值, "explained_variance_ratio": 各主轴方差占比,
         "whiten": 是否白化}
        可用 pca_transform / pca_rotate_back 复用该信息处理新数据。
    """
    X = data.get_coordinate_matrix()          # (n, d)
    n, d = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    # SVD: Xc = U S Vᵀ，V 的列是协方差矩阵的特征向量（主成分方向）
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    if n_components is None:
        k = d
    else:
        k = int(n_components)
        k = min(max(k, 1), d)

    V = Vt[:k].T                             # (d, k)
    Y = Xc @ V
    if whiten:
        Y = Y / np.maximum(s[:k], 1e-12)

    total_var = float(np.sum(s ** 2))
    ratio = (s ** 2) / total_var if total_var > 0 else np.zeros_like(s)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values             # 目标值跟随数据点，不随坐标变换

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
    用已拟合的 PCA 信息（pca_rotate 返回的 info）变换新数据。

    新数据（测试集/网格）必须使用训练时拟合的同一旋转，否则坐标系不一致。
    """
    X = data.get_coordinate_matrix()
    V = info["components"]
    mean = info["mean"]
    k = V.shape[1]

    Y = (X - mean) @ V
    if info.get("whiten"):
        s = info["singular_values"]
        Y = Y / np.maximum(s[:k], 1e-12)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values
    return rotated


def pca_rotate_back(point: Dict[int, float], info: Dict[str, Any]) -> Dict[int, float]:
    """
    把旋转坐标系下的单个点还原到原始坐标系（用于解释/画图）。

    参数
    ----
    point : Dict[int, float]
        旋转坐标系下的点（键 0..k-1）
    info : Dict
        pca_rotate 返回的信息

    返回
    ----
    Dict[int, float]
        原始坐标系下的点
    """
    k = len(point)
    y = np.array([point[i] for i in sorted(point.keys())], dtype=float)
    V = info["components"]
    x = y @ V.T + info["mean"]
    return {i: float(xi) for i, xi in enumerate(x)}
