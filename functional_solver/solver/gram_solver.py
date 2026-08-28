"""用于函数逼近的 Gram 矩阵求解器。"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct

class GramSolver:
    """使用 Gram 矩阵方法的求解器。"""
    
    def __init__(self, basis_set: BasisSet, inner_product: InnerProduct):
        """
        初始化 Gram 求解器。
        
        参数
        ----------
        basis_set : BasisSet
            基函数集合
        inner_product : InnerProduct
            内积定义
        """
        self.basis_set = basis_set
        self.inner_product = inner_product
        self.data = None
        self.target = None
        self.coefficients = None
        self.gram_matrix = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """加载用于求解的数据。"""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_gram_matrix(self) -> np.ndarray:
        """计算基集合的 Gram 矩阵。"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        Phi = self.basis_set.evaluate_all(self.data)
        self.gram_matrix = self.inner_product.compute_gram_matrix(Phi, self.data)
        return self.gram_matrix
    
    def compute_rhs(self) -> np.ndarray:
        """计算右端向量（与 Gram 矩阵同一内积定义）。"""
        if self.data is None or self.target is None:
            raise ValueError("Data and target not loaded")
        
        Phi = self.basis_set.evaluate_all(self.data)
        return self.inner_product.rhs_vector(Phi, self.target, self.data)
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """求解系数。"""
        from ..inner_product.regularization import Regularization
        
        # 计算 Gram 矩阵和右端向量
        G = self.compute_gram_matrix()
        b = self.compute_rhs()
        
        # 如有需要则应用正则化
        if regularization:
            method = regularization.get("method", "tikhonov")
            if method == "tikhonov":
                alpha = regularization.get("alpha", 1e-6)
                G = Regularization.tikhonov(G, alpha)
            elif method == "svd":
                threshold = regularization.get("threshold", 1e-10)
                U, s, Vh = Regularization.truncated_svd(G, threshold)
                s_inv = 1.0 / s
                self.coefficients = Vh.T @ (s_inv * (U.T @ b))
                return self.coefficients
        
        # 求解线性方程组
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(G, b, rcond=None)[0]
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """预测新数据的值。"""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def get_condition_number(self) -> float:
        """获取 Gram 矩阵的条件数。"""
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        return np.linalg.cond(self.gram_matrix)
    
    def get_singular_values(self) -> np.ndarray:
        """获取 Gram 矩阵的奇异值。"""
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        return np.linalg.svd(self.gram_matrix, compute_uv=False)
    
    def check_orthogonality(self) -> np.ndarray:
        """
        检查基函数的正交性。
        
        返回
        -------
        np.ndarray
            基函数之间的内积矩阵
        """
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        # 按对角线归一化，得到类似相关矩阵的结果
        diag = np.diag(self.gram_matrix)
        diag_sqrt = np.sqrt(diag)
        diag_inv_sqrt = 1.0 / diag_sqrt
        
        # 计算归一化的 Gram 矩阵
        G_norm = self.gram_matrix * diag_inv_sqrt[:, None] * diag_inv_sqrt[None, :]
        
        return G_norm
