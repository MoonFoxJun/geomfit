"""主解算器 - 整合所有模块"""

import numpy as np
from typing import Optional, Dict, Any

from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct
from ..kernel.base import Kernel

class FunctionalSolver:
    """泛函解算器主类"""
    
    def __init__(self):
        self.basis_set = None
        self.inner_product = None
        self.kernel = None
        self.data = None
        self.target = None
        self.coefficients = None
        self.debug_info = {}
    
    def set_basis(self, basis_set: BasisSet):
        """设置基函数集合"""
        self.basis_set = basis_set
        self.kernel = None  # 互斥
    
    def set_kernel(self, kernel: Kernel):
        """设置核函数（覆盖基函数方式）"""
        self.kernel = kernel
        self.basis_set = None  # 互斥
    
    def set_inner_product(self, inner_product: InnerProduct):
        """设置内积定义"""
        self.inner_product = inner_product
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """加载数据"""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "数据点数量与目标值数量不匹配"
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """求解最优系数"""
        if self.data is None or self.target is None:
            raise ValueError("请先加载数据")
        
        if self.kernel is not None:
            # 核方法
            return self._solve_kernel(regularization)
        elif self.basis_set is not None:
            # 基函数方法
            return self._solve_basis(regularization)
        else:
            raise ValueError("请先设置基函数集合或核函数")
    
    def _solve_basis(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """基函数方法求解"""
        from ..inner_product.regularization import Regularization
        
        Phi = self.basis_set.evaluate_all(self.data)
        G = self.inner_product.compute_gram_matrix(Phi, self.data)

        # 右端项：bᵢ = ⟨φᵢ, y⟩（与 Gram 矩阵使用同一内积定义，保证 Gc=b 自洽）
        b = self.inner_product.rhs_vector(Phi, self.target, self.data)
        
        # 奇异性检测
        singularity_info = Regularization.check_singularity(G)
        self.debug_info["singularity"] = singularity_info
        
        if singularity_info["is_singular"] and regularization:
            method = regularization.get("method", "tikhonov")
            if method == "tikhonov":
                alpha = regularization.get("alpha", 1e-6)
                G = Regularization.tikhonov(G, alpha)
            elif method == "svd":
                threshold = regularization.get("threshold", 1e-10)
                U, s, Vh = Regularization.truncated_svd(G, threshold)
                s_inv = 1.0 / s
                self.coefficients = Vh.T @ (s_inv * (U.T @ b))
                self.debug_info["method"] = "basis_svd"
                return self.coefficients
        
        # 解方程
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(G, b, rcond=None)[0]
        
        self.debug_info["method"] = "basis"
        return self.coefficients
    
    def _solve_kernel(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """核方法求解"""
        K = self.kernel.compute_matrix(self.data)
        
        if regularization and "alpha" in regularization:
            K = K + regularization["alpha"] * np.eye(len(K))
        
        self.coefficients = np.linalg.solve(K, self.target)
        self.debug_info["method"] = "kernel"
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """对新数据进行预测"""
        if self.coefficients is None:
            raise ValueError("请先调用solve()")
        
        if self.kernel is not None:
            return self._predict_kernel(new_data)
        else:
            return self._predict_basis(new_data)
    
    def _predict_basis(self, new_data: MultiDimData) -> np.ndarray:
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def _predict_kernel(self, new_data: MultiDimData) -> np.ndarray:
        n_train = self.data.n_points
        n_test = new_data.n_points
        K_pred = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            for j in range(n_train):
                K_pred[i, j] = self.kernel(
                    new_data.get_point(i),
                    self.data.get_point(j)
                )
        
        return K_pred @ self.coefficients
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return self.debug_info
    
    def reset(self):
        """重置解算器状态"""
        self.basis_set = None
        self.inner_product = None
        self.kernel = None
        self.data = None
        self.target = None
        self.coefficients = None
        self.debug_info = {}
