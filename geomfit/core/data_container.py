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
        
        for dim, arr in data_dict.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError("All coordinate arrays must be numpy arrays")
        
        self.data = data_dict
        self.dims = sorted(data_dict.keys())
        
        # Verify that all coordinate arrays share the same length
        lengths = [len(arr) for arr in data_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All coordinate arrays must have the same length")
        
        self.n_points = lengths[0]
        self.n_dims = len(self.dims)
        self._values = None
    
    @property
    def values(self) -> Optional[np.ndarray]:
        """Target values associated with the data points."""
        return self._values
    
    @values.setter
    def values(self, values: np.ndarray):
        """Set the target values, validating their length against n_points."""
        if values is not None and len(values) != self.n_points:
            raise ValueError("Values must have the same length as the data points")
        self._values = values
    
    def get_dim(self, dim: int) -> np.ndarray:
        """Return the coordinate array of the given dimension."""
        return self.data[dim]
    
    def get_point(self, index: int) -> Dict[int, float]:
        """Return the coordinates of one point as a dict keyed by dimension."""
        return {dim: self.data[dim][index] for dim in self.dims}
    
    def get_all_points(self) -> List[Dict[int, float]]:
        """Return all points as a list of dicts keyed by dimension."""
        return [self.get_point(i) for i in range(self.n_points)]
    
    def get_coordinate_matrix(self) -> np.ndarray:
        """Arrange the coordinates into a matrix of shape (n_points, n_dims)."""
        return np.column_stack([self.data[dim] for dim in self.dims])
    
    def slice(self, indices) -> 'MultiDimData':
        """Return a new MultiDimData containing only the given indices."""
        indices = np.asarray(indices)
        return MultiDimData({dim: self.data[dim][indices] for dim in self.dims})
    
    def mask(self, mask) -> 'MultiDimData':
        """Return a new MultiDimData filtered by a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        return MultiDimData({dim: self.data[dim][mask] for dim in self.dims})
    
    def mean(self) -> np.ndarray:
        """Return the per-dimension mean, in ascending `dims` order."""
        return np.array([self.data[dim].mean() for dim in self.dims])
    
    def std(self) -> np.ndarray:
        """Return the per-dimension standard deviation, in ascending `dims` order."""
        return np.array([self.data[dim].std() for dim in self.dims])
    
    def min(self) -> np.ndarray:
        """Return the per-dimension minimum, in ascending `dims` order."""
        return np.array([self.data[dim].min() for dim in self.dims])
    
    def max(self) -> np.ndarray:
        """Return the per-dimension maximum, in ascending `dims` order."""
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
        order = dims if dims is not None else self.dims
        return np.column_stack([self.data[dim] for dim in order])
    
    def __add__(self, other: 'MultiDimData') -> 'MultiDimData':
        """Elementwise addition of two MultiDimData objects."""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        return MultiDimData({dim: self.data[dim] + other.data[dim] for dim in self.dims})
    
    def __sub__(self, other: 'MultiDimData') -> 'MultiDimData':
        """Elementwise subtraction of two MultiDimData objects."""
        if not isinstance(other, MultiDimData):
            return NotImplemented
        return MultiDimData({dim: self.data[dim] - other.data[dim] for dim in self.dims})
    
    def __mul__(self, scalar: float) -> 'MultiDimData':
        """Scalar multiplication."""
        return MultiDimData({dim: self.data[dim] * scalar for dim in self.dims})
    
    def __rmul__(self, scalar: float) -> 'MultiDimData':
        """Scalar multiplication (scalar on the left)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'MultiDimData':
        """Scalar division."""
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
        """Number of data points."""
        return self.n_points
    
    def __repr__(self) -> str:
        """Machine-readable representation of the container."""
        return f"MultiDimData(n_points={self.n_points}, dims={self.dims})"
    
    def __str__(self) -> str:
        """String representation identical to repr."""
        return self.__repr__()
