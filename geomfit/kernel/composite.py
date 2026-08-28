"""Composite kernel (sum and product combinations)."""

import numpy as np
from typing import List, Dict, Any
from .base import Kernel

class CompositeKernel(Kernel):
    """Kernel formed by combining multiple kernels."""

    def __init__(self, kernels: List[Kernel], operation: str = "add"):
        """
        Initialize the composite kernel.

        Parameters
        ----------
        kernels : List[Kernel]
            List of kernels to combine.
        operation : str, default="add"
            Combination operation: "add" (sum) or "multiply" (product).
        """
        super().__init__(kernels=kernels, operation=operation)
        self.kernels = kernels
        self.operation = operation

        # 校验组合方式：只允许 "add"（加法）与 "multiply"（乘法）。
        # 两者都是合法的核运算——正定核的逐点和与逐点乘积仍是正定核（Mercer 核的封闭性）
        valid_operations = ["add", "multiply"]
        if operation not in valid_operations:
            raise ValueError(f"operation must be one of {valid_operations}")

    def __call__(self, x: Dict[int, float], y: Dict[int, float]) -> float:
        """
        Evaluate the composite kernel between two points.

        Parameters
        ----------
        x : Dict[int, float]
            Coordinates of the first point.
        y : Dict[int, float]
            Coordinates of the second point.

        Returns
        -------
        float
            Composite kernel value.
        """
        if self.operation == "add":
            # 加法组合：k(x, y) = Σᵢ kᵢ(x, y)，从 0 开始累加（加法单位元）
            result = 0.0
            for kernel in self.kernels:
                result += kernel(x, y)
            return result

        elif self.operation == "multiply":
            # 乘法组合：k(x, y) = Πᵢ kᵢ(x, y)，从 1 开始连乘（乘法单位元）
            result = 1.0
            for kernel in self.kernels:
                result *= kernel(x, y)
            return result

    def compute_matrix(self, data: Any) -> np.ndarray:
        """
        Compute the composite kernel matrix.

        Parameters
        ----------
        data : MultiDimData
            Data container.

        Returns
        -------
        np.ndarray
            Composite kernel matrix of shape (n_points, n_points).
        """
        from ..core.data_container import MultiDimData

        if not isinstance(data, MultiDimData):
            raise TypeError("data must be a MultiDimData instance")

        if self.operation == "add":
            # 加法组合：各核矩阵逐元素相加（等价于核函数求和后的整矩阵形式）
            K = np.zeros((data.n_points, data.n_points))
            for kernel in self.kernels:
                K += kernel.compute_matrix(data)
            return K

        elif self.operation == "multiply":
            # 乘法组合：各核矩阵逐元素相乘（Hadamard 乘积），对应核函数的逐点乘积
            K = None
            for kernel in self.kernels:
                K_k = kernel.compute_matrix(data)
                if K is None:
                    K = K_k  # 第一个核矩阵直接作为初值
                else:
                    K *= K_k  # 之后逐个与当前结果逐元素相乘
            # 空组合（kernels 为空列表）时退化为全 1 矩阵，即核值恒为 1（乘法的单位元）
            return K if K is not None else np.ones((data.n_points, data.n_points))

    def add_kernel(self, kernel: Kernel):
        """Append a kernel to the composite."""
        self.kernels.append(kernel)

    def remove_kernel(self, index: int):
        """Remove the kernel at the given index from the composite."""
        if 0 <= index < len(self.kernels):
            del self.kernels[index]

    def __len__(self) -> int:
        return len(self.kernels)
