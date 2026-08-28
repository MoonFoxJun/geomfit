"""用于函数逼近的核方法求解器。"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.data_container import MultiDimData
from ..kernel.base import Kernel

class KernelSolver:
    """使用核方法的求解器。"""
    
    def __init__(self, kernel: Kernel):
        """
        初始化核求解器。
        
        参数
        ----------
        kernel : Kernel
            核函数
        """
        self.kernel = kernel
        self.data = None
        self.target = None
        self.coefficients = None
        self.kernel_matrix = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """加载用于求解的数据。"""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_kernel_matrix(self) -> np.ndarray:
        """计算数据的核矩阵。"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        self.kernel_matrix = self.kernel.compute_matrix(self.data)
        return self.kernel_matrix
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """使用核方法求解系数。"""
        K = self.compute_kernel_matrix()
        
        # 如果指定了正则化则应用之
        if regularization and "alpha" in regularization:
            alpha = regularization["alpha"]
            K = K + alpha * np.eye(len(K))
        
        # 求解线性方程组
        try:
            self.coefficients = np.linalg.solve(K, self.target)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(K, self.target, rcond=None)[0]
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """预测新数据的值。"""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        if self.data is None:
            raise ValueError("Training data not loaded")
        
        n_train = self.data.n_points
        n_test = new_data.n_points
        
        # 计算新数据与训练数据之间的核矩阵
        K_pred = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            x_i = new_data.get_point(i)
            for j in range(n_train):
                x_j = self.data.get_point(j)
                K_pred[i, j] = self.kernel(x_i, x_j)
        
        return K_pred @ self.coefficients
    
    def predict_efficient(self, new_data: MultiDimData) -> np.ndarray:
        """利用矩阵运算进行高效预测。"""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        if self.data is None:
            raise ValueError("Training data not loaded")
        
        # 如果可用，则使用核的交叉矩阵计算
        K_pred = self.kernel.compute_cross_matrix(new_data, self.data)
        return K_pred @ self.coefficients
    
    def get_condition_number(self) -> float:
        """获取核矩阵的条件数。"""
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        return np.linalg.cond(self.kernel_matrix)
    
    def get_eigenvalues(self) -> np.ndarray:
        """获取核矩阵的特征值。"""
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        # 核矩阵是对称的，使用 eigh 以提高效率
        eigenvalues = np.linalg.eigvalsh(self.kernel_matrix)
        # 将浮点舍入误差产生的小负值裁剪为 0，
        # 使（理论上为半正定的）核矩阵保持非负
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return np.sort(eigenvalues)[::-1]  # 按降序排序
    
    def compute_representer_weights(self) -> np.ndarray:
        """
        计算表示定理形式下的权重。
        
        返回
        -------
        np.ndarray
            权重 α，使得 f(x) = Σ α_i k(x, x_i)
        """
        if self.coefficients is None:
            self.solve()
        
        return self.coefficients
    
    def compute_function_norm(self) -> float:
        """
        计算函数在 RKHS 中的范数。
        
        返回
        -------
        float
            RKHS 范数 ||f||_H = sqrt(α^T K α)
        """
        if self.coefficients is None:
            self.solve()
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        return np.sqrt(self.coefficients.T @ self.kernel_matrix @ self.coefficients)
    
    def leave_one_out_error(self) -> np.ndarray:
        """
        计算留一法交叉验证误差。
        
        返回
        -------
        np.ndarray
            每个数据点的留一法交叉验证（LOOCV）误差
        """
        if self.coefficients is None:
            self.solve()
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        n = len(self.coefficients)
        errors = np.zeros(n)
        
        # 高效计算核矩阵逆矩阵的对角线
        try:
            K_inv = np.linalg.inv(self.kernel_matrix)
            diag_K_inv = np.diag(K_inv)
            
            # 留一法公式：e_i = α_i / (K^{-1})_{ii}
            errors = self.coefficients / diag_K_inv
        except np.linalg.LinAlgError:
            # 回退方案：逐个计算每个留一法误差
            for i in range(n):
                # 移除第 i 个数据点
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                
                K_reduced = self.kernel_matrix[mask, :][:, mask]
                target_reduced = self.target[mask]
                
                # 求解缩减后的方程组
                try:
                    alpha_reduced = np.linalg.solve(K_reduced, target_reduced)
                except np.linalg.LinAlgError:
                    alpha_reduced = np.linalg.lstsq(K_reduced, target_reduced, rcond=None)[0]
                
                # 在移除的点上进行预测
                k_pred = self.kernel_matrix[i, mask]
                pred = k_pred @ alpha_reduced
                errors[i] = self.target[i] - pred
        
        return errors
