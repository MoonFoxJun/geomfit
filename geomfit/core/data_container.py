"""Multi-dimensional container for the functional solver's data points."""

from typing import Dict, List, Any, Optional, Union
import numpy as np

class MultiDimData:
    """Container for multi-dimensional data points."""
    
    def __init__(self, data_dict: Dict[int, np.ndarray]):
        """
        Initialize from a dictionary mapping dimension indices to coordinate arrays.

        Parameters
        ----------
        data_dict : Dict[int, np.ndarray]
            Keys are dimension indices and values are the corresponding coordinate
            arrays. All arrays must have the same length (the number of points).
        """
        if not data_dict:
            raise ValueError("data_dict must contain at least one dimension")
        
        # 逐维检查：要求坐标数组必须是 numpy 数组，以便后续做向量化运算与花式索引
        for dim, arr in data_dict.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError("All coordinate arrays must be numpy arrays")
        
        # 原样保存坐标字典，并把维度索引升序排序，保证后续遍历 dims 的顺序恒定
        self.data = data_dict
        self.dims = sorted(data_dict.keys())
        
        # 校验所有维度的坐标数组长度是否一致：每一维都必须含有相同数量的点，
        # 否则"第 i 个点"在不同维度上无法对齐，也就没法组装后续的设计矩阵
        lengths = [len(arr) for arr in data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All coordinate arrays must have the same length")
        
        self.n_points = lengths[0]  # 数据点总数（上面已保证各长度相等，取第一个即可）
        self.n_dims = len(self.dims)  # 维度个数
        self._values = None  # 目标值 y 暂为空，后续通过 values 属性赋值（拟合/回归场景使用）
    
    @property
    def values(self) -> Optional[np.ndarray]:
        """Target values associated with the data points."""
        return self._values
    
    @values.setter
    def values(self, values: np.ndarray):
        """Set the target values, validating their length against n_points."""
        # 每个数据点必须恰好对应一个目标值；长度不匹配说明点与标签对不齐，直接报错
        if values is not None and len(values) != self.n_points:
            raise ValueError("Values must have the same length as the data points")
        self._values = values
    
    def get_dim(self, dim: int) -> np.ndarray:
        """Return the coordinate array of the given dimension."""
        return self.data[dim]
    
    def get_point(self, index: int) -> Dict[int, float]:
        """Return the coordinates of one point as a dict keyed by dimension."""
        # 取第 index 个点在各维上的坐标，组装成 {维度索引: 坐标} 的字典
        return {dim: self.data[dim][index] for dim in self.dims}
    
    def get_all_points(self) -> List[Dict[int, float]]:
        """Return all points as a list of dicts keyed by dimension."""
        # 依次取出每个点的字典表示，得到长度为 n_points 的列表
        return [self.get_point(i) for i in range(self.n_points)]
    
    def get_coordinate_matrix(self) -> np.ndarray:
        """Arrange the coordinates into a matrix of shape (n_points, n_dims)."""
        # 按 dims 升序把各维坐标数组"按列"拼接成矩阵：np.column_stack 把每个一维数组
        # 作为一列，最终形状为 (n_points, n_dims)
        return np.column_stack([self.data[dim] for dim in self.dims])
    
    def slice(self, indices) -> 'MultiDimData':
        """Return a new MultiDimData containing only the given indices."""
        # 先把索引转成 numpy 数组，再对每一维做"花式索引"（fancy indexing），
        # 只保留 indices 中指定位置的点，返回一个新的 MultiDimData
        indices = np.asarray(indices)
        return MultiDimData({dim: self.data[dim][indices] for dim in self.dims})
    
    def mask(self, mask) -> 'MultiDimData':
        """Return a new MultiDimData filtered by a boolean mask."""
        # 布尔掩码筛选：把 mask 强制转成布尔数组后，每个维度只保留 mask 为 True 的位置
        mask = np.asarray(mask, dtype=bool)
        return MultiDimData({dim: self.data[dim][mask] for dim in self.dims})
    
    def mean(self) -> np.ndarray:
        """Return the per-dimension mean, in ascending `dims` order."""
        # 逐维求均值，返回长度 = n_dims 的向量，分量顺序与 dims 一致
        return np.array([self.data[dim].mean() for dim in self.dims])
    
    def std(self) -> np.ndarray:
        """Return the per-dimension standard deviation, in ascending `dims` order."""
        # 逐维求标准差（numpy 默认总体标准差 ddof=0）
        return np.array([self.data[dim].std() for dim in self.dims])
    
    def min(self) -> np.ndarray:
        """Return the per-dimension minimum, in ascending `dims` order."""
        # 逐维求最小值，可用于快速查看各维数据的取值范围
        return np.array([self.data[dim].min() for dim in self.dims])
    
    def max(self) -> np.ndarray:
        """Return the per-dimension maximum, in ascending `dims` order."""
        # 逐维求最大值
        return np.array([self.data[dim].max() for dim in self.dims])
    
    def to_array(self, dims: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert the coordinates into a matrix of shape (n_points, n_dims).

        Parameters
        ----------
        dims : List[int], optional
            Column order; defaults to the ascending dimension order of the data.

        Returns
        -------
        np.ndarray
            Coordinate matrix with one column per requested dimension.
        """
        # order 决定列的排列顺序：未指定时用数据自身的升序维度；指定时按用户给定顺序取列
        order = dims if dims is not None else self.dims
        return np.column_stack([self.data[dim] for dim in order])
    
    def __add__(self, other: 'MultiDimData') -> 'MultiDimData':
        """Elementwise addition of two MultiDimData objects."""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        # 逐元素相加：按 self.dims 逐维做向量加法，结果仍是一个 MultiDimData
        return MultiDimData({dim: self.data[dim] + other.data[dim] for dim in self.dims})
    
    def __sub__(self, other: 'MultiDimData') -> 'MultiDimData':
        """Elementwise subtraction of two MultiDimData objects."""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        # 逐元素相减，语义与 __add__ 对称
        return MultiDimData({dim: self.data[dim] - other.data[dim] for dim in self.dims})
    
    def __mul__(self, scalar: float) -> 'MultiDimData':
        """Scalar multiplication."""
        # 标量乘法：所有维度的坐标都乘以同一个标量（整体缩放数据）
        return MultiDimData({dim: self.data[dim] * scalar for dim in self.dims})
    
    def __rmul__(self, scalar: float) -> 'MultiDimData':
        """Scalar multiplication (scalar on the left)."""
        # 标量写在左边（scalar * self）时触发，直接复用 __mul__ 即可
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'MultiDimData':
        """Scalar division."""
        # 标量除法：所有维度的坐标都除以同一个标量
        return MultiDimData({dim: self.data[dim] / scalar for dim in self.dims})
    
    @classmethod
    def concatenate(cls, data_list: List['MultiDimData'], axis: int = 0) -> 'MultiDimData':
        """
        Concatenate several MultiDimData objects.

        Parameters
        ----------
        data_list : List[MultiDimData]
            Objects to concatenate.
        axis : int, default=0
            If 0: concatenate along the point axis (all objects must share the
            same dims).
            If 1: concatenate along the dimension axis (all objects must have the
            same n_points).

        Returns
        -------
        MultiDimData
            The concatenated container.
        """
        if not data_list:
            raise ValueError("data_list must not be empty")
        
        if axis == 0:
            # 沿"点"的方向拼接：所有对象必须具有相同的维度集合，
            # 然后对每个维度分别用 np.concatenate 纵向堆叠各对象的坐标
            dims = data_list[0].dims
            for d in data_list[1:]:
                if d.dims != dims:
                    raise ValueError("All data must have the same dims when axis=0")
            return cls({dim: np.concatenate([d.get_dim(dim) for d in data_list])
                        for dim in dims})
        
        elif axis == 1:
            # 沿"维度"的方向拼接：所有对象必须含有相同数量的点（同一批点、不同坐标维度），
            # 直接把各对象的 data 字典合并（update 按维度索引去重），等价于给点补上新维度
            n_points = data_list[0].n_points
            for d in data_list[1:]:
                if d.n_points != n_points:
                    raise ValueError("All data must have the same number of points when axis=1")
            result = {}
            for d in data_list:
                result.update(d.data)
            return cls(result)
        
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")
    
    @classmethod
    def from_array(cls, array: np.ndarray, dim: Optional[int] = None,
                   dims: Optional[List[int]] = None) -> 'MultiDimData':
        """
        Create a MultiDimData from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            A 1D array becomes a single dimension; each column of a 2D array
            becomes a dimension.
        dim : int, optional
            Dimension index for a 1D array.
        dims : List[int], optional
            Dimension indices for the columns of a 2D array (default: 0..n_cols-1).

        Returns
        -------
        MultiDimData
            The constructed container.
        """
        # 统一转成 numpy 数组，方便后续按数组维度数分支处理
        arr = np.asarray(array)
        
        if arr.ndim == 1:
            # 一维数组：只有一条坐标轴，整个数组就是该维的坐标；维度索引未指定时默认取 0
            d = 0 if dim is None else dim
            return cls({d: arr})
        
        elif arr.ndim == 2:
            # 二维数组：每一列视为一个维度，列数即维度数
            n_cols = arr.shape[1]
            if dims is None:
                # 未指定列名时，自动用 0, 1, ..., n_cols-1 作为各维度的索引
                dims = list(range(n_cols))
            if len(dims) != n_cols:
                raise ValueError("dims must have one entry per column")
            # 逐列提取：arr[:, i] 取第 i 列作为维度 dims[i] 的坐标数组
            return cls({d: arr[:, i] for i, d in enumerate(dims)})
        
        else:
            raise ValueError(f"array must be 1D or 2D, got {arr.ndim}D")
    
    def __len__(self) -> int:
        """Number of data points."""
        return self.n_points
    
    def __repr__(self) -> str:
        """Machine-readable representation of the container."""
        return f"MultiDimData(n_points={self.n_points}, dims={self.dims})"
    
    def __str__(self) -> str:
        """String representation identical to repr."""
        return self.__repr__()
