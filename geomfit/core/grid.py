"""Grid detection utilities for separable (tensor-product) solvers.

A data set is a *full Cartesian grid* when its points are exactly the product of
the per-dimension sample values, i.e. every combination (x_a, y_b, z_c, ...) is
present exactly once. Only then does a sum over the points factorise into
per-dimension sums (Fubini at the discrete level), which is the structural
precondition of the separable "sandwich" solve.
"""

from typing import Any, Dict, Optional
import numpy as np

from .data_container import MultiDimData


def detect_grid(data: MultiDimData, decimals: int = 12) -> Optional[Dict[str, Any]]:
    """Detect whether the stored points form a full Cartesian grid.

    Parameters
    ----------
    data : MultiDimData
        Data container to inspect.
    decimals : int, default=12
        Coordinates are rounded to this many decimals before matching against
        the per-dimension axes (floating-point tolerance).

    Returns
    -------
    dict or None
        None when the points are scattered (or contain duplicate cells).
        Otherwise a dictionary with:

        - ``dims``: dimension indices in sorted order;
        - ``axes``: per-dimension unique coordinates (the grid axes);
        - ``shape``: grid shape ``(n_1, ..., n_D)``;
        - ``flat_index``: row-major multi-index of every stored point, so that a
          data vector can be scattered into the grid tensor with
          ``tensor.flat[flat_index] = values``.
    """
    dims = data.dims
    axes = []
    index_maps = []

    for dim in dims:
        vals = np.asarray(data.get_dim(dim), dtype=float)
        rounded = np.round(vals, decimals)
        uniq = np.unique(rounded)  # 该维度上所有的网格坐标（去重后的轴）
        pos = np.searchsorted(uniq, rounded)  # 每个点落在轴的哪个位置上
        pos = np.clip(pos, 0, len(uniq) - 1)
        # 坐标必须精确落在某条轴上；否则说明该维度不是规则采样，直接判定为散点
        if not np.allclose(uniq[pos], vals, rtol=0.0, atol=10.0 ** (-decimals)):
            return None
        axes.append(uniq)
        index_maps.append(pos)

    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != data.n_points:
        # 点数不等于各轴长度的乘积 → 不可能填满整个笛卡尔积
        return None

    multi = np.ravel_multi_index(index_maps, shape)  # C 序（行优先）多重指标
    if len(np.unique(multi)) != data.n_points:
        # 有多个点落到同一个格子 → 是重复采样，而不是完整网格
        return None

    return {
        "dims": dims,
        "axes": axes,
        "shape": shape,
        "flat_index": multi,
    }
