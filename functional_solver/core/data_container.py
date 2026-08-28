"""功能求解器的多维数据容器。"""

from typing import Dict, List, Any, Optional, Union
import numpy as np

class MultiDimData:
    """多维数据点的容器。"""
    
    def __init__(self, data_dict: Dict[int, np.ndarray]):
        """
        用一个字典初始化，字典把维度索引映射到坐标数组。

        参数
        ----
        data_dict : Dict[int, np.ndarray]
            字典的键是维度索引，值是对应的坐标数组。
            所有数组的长度必须相同（即点数）。
        """
        if not data_dict:
            raise ValueError("data_dict must contain at least one dimension")
        
        for dim, arr in data_dict.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError("All coordinate arrays must be numpy arrays")
        
        self.data = data_dict
        self.dims = sorted(data_dict.keys())
        
        # 验证所有数组的长度是否相同
        lengths = [len(arr) for arr in data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All coordinate arrays must have the same length")
        
        self.n_points = lengths[0]
        self.n_dims = len(self.dims)
        self._values = None
    
    @property
    def values(self) -> Optional[np.ndarray]:
        """获取与数据点关联的目标值。"""
        return self._values
    
    @values.setter
    def values(self, values: np.ndarray):
        """设置目标值，并校验其长度与 n_points 一致。"""
        if values is not None and len(values) != self.n_points:
            raise ValueError("Values must have the same length as the data points")
        self._values = values
    
    def get_dim(self, dim: int) -> np.ndarray:
        """获取指定维度的坐标数组。"""
        return self.data[dim]
    
    def get_point(self, index: int) -> Dict[int, float]:
        """以字典形式获取某个点的坐标。"""
        return {dim: self.data[dim][index] for dim in self.dims}
    
    def get_all_points(self) -> List[Dict[int, float]]:
        """以字典列表的形式获取所有点。"""
        return [self.get_point(i) for i in range(self.n_points)]
    
    def get_coordinate_matrix(self) -> np.ndarray:
        """把坐标整理成形状为 (n_points, n_dims) 的矩阵。"""
        return np.column_stack([self.data[dim] for dim in self.dims])
    
    def slice(self, indices) -> 'MultiDimData':
        """返回只包含指定索引的新 MultiDimData。"""
        indices = np.asarray(indices)
        return MultiDimData({dim: self.data[dim][indices] for dim in self.dims})
    
    def mask(self, mask) -> 'MultiDimData':
        """返回由布尔掩码筛选出的新 MultiDimData。"""
        mask = np.asarray(mask, dtype=bool)
        return MultiDimData({dim: self.data[dim][mask] for dim in self.dims})
    
    def mean(self) -> np.ndarray:
        """按 dims 顺序返回每个维度的均值。"""
        return np.array([self.data[dim].mean() for dim in self.dims])
    
    def std(self) -> np.ndarray:
        """按 dims 顺序返回每个维度的标准差。"""
        return np.array([self.data[dim].std() for dim in self.dims])
    
    def min(self) -> np.ndarray:
        """按 dims 顺序返回每个维度的最小值。"""
        return np.array([self.data[dim].min() for dim in self.dims])
    
    def max(self) -> np.ndarray:
        """按 dims 顺序返回每个维度的最大值。"""
        return np.array([self.data[dim].max() for dim in self.dims])
    
    def to_array(self, dims: Optional[List[int]] = None) -> np.ndarray:
        """把坐标转换成形状为 (n_points, n_dims) 的矩阵。"""
        order = dims if dims is not None else self.dims
        return np.column_stack([self.data[dim] for dim in order])
    
    def __add__(self, other: 'MultiDimData') -> 'MultiDimData':
        """两个 MultiDimData 对象的逐元素相加。"""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        return MultiDimData({dim: self.data[dim] + other.data[dim] for dim in self.dims})
    
    def __sub__(self, other: 'MultiDimData') -> 'MultiDimData':
        """两个 MultiDimData 对象的逐元素相减。"""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        return MultiDimData({dim: self.data[dim] - other.data[dim] for dim in self.dims})
    
    def __mul__(self, scalar: float) -> 'MultiDimData':
        """标量乘法。"""
        return MultiDimData({dim: self.data[dim] * scalar for dim in self.dims})
    
    def __rmul__(self, scalar: float) -> 'MultiDimData':
        """标量乘法（标量在左边）。"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'MultiDimData':
        """标量除法。"""
        return MultiDimData({dim: self.data[dim] / scalar for dim in self.dims})
    
    @classmethod
    def concatenate(cls, data_list: List['MultiDimData'], axis: int = 0) -> 'MultiDimData':
        """
        拼接多个 MultiDimData 对象。

        参数
        ----
        data_list : List[MultiDimData]
            要拼接的对象
        axis : int, 默认=0
            若为 0：沿点数方向拼接（各对象的 dims 必须一致）。
            若为 1：沿维度方向拼接（各对象的 n_points 必须一致）。
        """
        if not data_list:
            raise ValueError("data_list must not be empty")
        
        if axis == 0:
            dims = data_list[0].dims
            for d in data_list[1:]:
                if d.dims != dims:
                    raise ValueError("All data must have the same dims when axis=0")
            return cls({dim: np.concatenate([d.get_dim(dim) for d in data_list])
                        for dim in dims})
        
        elif axis == 1:
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
        从 numpy 数组创建 MultiDimData。

        参数
        ----
        array : np.ndarray
            一维数组变成一个维度；二维数组的每一列成为一个维度。
        dim : int, 可选
            一维数组对应的维度索引。
        dims : List[int], 可选
            二维数组各列对应的维度索引（默认是 0..n_cols-1）。
        """
        arr = np.asarray(array)
        
        if arr.ndim == 1:
            d = 0 if dim is None else dim
            return cls({d: arr})
        
        elif arr.ndim == 2:
            n_cols = arr.shape[1]
            if dims is None:
                dims = list(range(n_cols))
            if len(dims) != n_cols:
                raise ValueError("dims must have one entry per column")
            return cls({d: arr[:, i] for i, d in enumerate(dims)})
        
        else:
            raise ValueError(f"array must be 1D or 2D, got {arr.ndim}D")
    
    def __len__(self) -> int:
        return self.n_points
    
    def __repr__(self) -> str:
        return f"MultiDimData(n_points={self.n_points}, dims={self.dims})"
    
    def __str__(self) -> str:
        return self.__repr__()
