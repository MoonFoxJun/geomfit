"""Abstract base class for kernel functions."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from ..core.data_container import MultiDimData

class Kernel(ABC):
    """Abstract base class for kernel functions."""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the kernel between two points.

        Parameters
        ----------
        x : Dict[int, float]
            Coordinates of the first point.
        y : Dict[int, float]
            Coordinates of the second point.

        Returns
        -------
        float
            The kernel value k(x, y).
        """
        pass

    def compute_matrix(self, data: MultiDimData) -> np.ndarray:
        """
        Compute the kernel matrix for all points in ``data``.

        Parameters
        ----------
        data : MultiDimData
            Data container.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_points, n_points), where
            K_ij = k(x_i, x_j).
        """
        n_points = data.n_points
        K = np.zeros((n_points, n_points))

        # 核矩阵对称：K_ij = k(x_i, x_j) = k(x_j, x_i) = K_ji，
        # 因此只需计算上三角（含对角线，j ≥ i），再镜像填充下三角，省一半核函数求值开销
        for i in range(n_points):
            x_i = data.get_point(i)
            for j in range(i, n_points):
                x_j = data.get_point(j)
                k_val = self(x_i, x_j)
                K[i, j] = k_val
                if i != j:
                    K[j, i] = k_val  # 利用对称性 K_ji = K_ij，镜像填充下三角

        return K

    def compute_cross_matrix(self, data1: MultiDimData, data2: MultiDimData) -> np.ndarray:
        """
        Compute the cross kernel matrix between two data sets.

        Parameters
        ----------
        data1 : MultiDimData
            First data set.
        data2 : MultiDimData
            Second data set.

        Returns
        -------
        np.ndarray
            Cross kernel matrix of shape (n_points1, n_points2), where
            K_ij = k(x_i, y_j).
        """
        n_points1 = data1.n_points
        n_points2 = data2.n_points
        K = np.zeros((n_points1, n_points2))

        # 交叉核矩阵：K_ij = k(x_i, y_j)，对第一组每个点与第二组每个点两两求核。
        # 注意交叉矩阵一般不对称（n_points1 ≠ n_points2 时甚至不是方阵），必须算满全部元素
        for i in range(n_points1):
            x_i = data1.get_point(i)
            for j in range(n_points2):
                x_j = data2.get_point(j)
                K[i, j] = self(x_i, x_j)

        return K

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"

    def __str__(self) -> str:
        return self.__repr__()
