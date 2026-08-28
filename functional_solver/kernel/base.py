"""功能求解器的核基类。"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from ..core.data_container import MultiDimData

class Kernel(ABC):
    """核函数的抽象基类。"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的核值。
        
        参数
        ----
        x : Dict[int, float]
            第一个点的坐标
        y : Dict[int, float]
            第二个点的坐标
            
        返回
        ----
        float
            核值 k(x, y)
        """
        pass
    
    def compute_matrix(self, data: MultiDimData) -> np.ndarray:
        """
        计算 data 中所有点的核矩阵。
        
        参数
        ----
        data : MultiDimData
            数据容器
            
        返回
        ----
        np.ndarray
            形状为 (n_points, n_points) 的核矩阵
        """
        n_points = data.n_points
        K = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            x_i = data.get_point(i)
            for j in range(i, n_points):
                x_j = data.get_point(j)
                k_val = self(x_i, x_j)
                K[i, j] = k_val
                if i != j:
                    K[j, i] = k_val  # 对称
        
        return K
    
    def compute_cross_matrix(self, data1: MultiDimData, data2: MultiDimData) -> np.ndarray:
        """
        计算两个数据集之间的交叉核矩阵。
        
        参数
        ----
        data1 : MultiDimData
            第一个数据集
        data2 : MultiDimData
            第二个数据集
            
        返回
        ----
        np.ndarray
            形状为 (n_points1, n_points2) 的交叉核矩阵
        """
        n_points1 = data1.n_points
        n_points2 = data2.n_points
        K = np.zeros((n_points1, n_points2))
        
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
