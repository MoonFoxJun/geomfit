"""复合核（加法、乘法等）。"""

import numpy as np
from typing import List, Dict, Any
from .base import Kernel

class CompositeKernel(Kernel):
    """由多个核组合而成的复合核。"""
    
    def __init__(self, kernels: List[Kernel], operation: str = "add"):
        """
        初始化复合核。
        
        参数
        ----
        kernels : List[Kernel]
            要组合的核函数列表
        operation : str, default="add"
            组合核的方式："add"（相加）或 "multiply"（相乘）
        """
        super().__init__(kernels=kernels, operation=operation)
        self.kernels = kernels
        self.operation = operation
        
        # 校验操作类型
        valid_operations = ["add", "multiply"]
        if operation not in valid_operations:
            raise ValueError(f"operation must be one of {valid_operations}")
    
    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        计算两个点之间的复合核值。
        
        参数
        ----
        x : Dict[int, float]
            第一个点的坐标
        y : Dict[int, float]
            第二个点的坐标
            
        返回
        ----
        float
            复合核值
        """
        if self.operation == "add":
            # 各核之和
            result = 0.0
            for kernel in self.kernels:
                result += kernel(x, y)
            return result
        
        elif self.operation == "multiply":
            # 各核之积
            result = 1.0
            for kernel in self.kernels:
                result *= kernel(x, y)
            return result
    
    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        计算复合核矩阵。
        
        参数
        ----
        data : MultiDimData
            数据容器
            
        返回
        ----
        np.ndarray
            复合核矩阵
        """
        from ..core.data_container import MultiDimData
        
        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")
        
        if self.operation == "add":
            # 各核矩阵之和
            K = np.zeros((data.n_points, data.n_points))
            for kernel in self.kernels:
                K += kernel.compute_matrix(data)
            return K
        
        elif self.operation == "multiply":
            # 各核矩阵逐元素相乘
            K = None
            for kernel in self.kernels:
                K_k = kernel.compute_matrix(data)
                if K is None:
                    K = K_k
                else:
                    K *= K_k
            return K if K is not None else np.ones((data.n_points, data.n_points))
    
    def add_kernel(self, kernel: Kernel):
        """向复合核中添加一个核。"""
        self.kernels.append(kernel)
    
    def remove_kernel(self, index: int):
        """按索引从复合核中移除一个核。"""
        if 0 <= index < len(self.kernels):
            del self.kernels[index]
    
    def __len__(self) -> int:
        return len(self.kernels)
