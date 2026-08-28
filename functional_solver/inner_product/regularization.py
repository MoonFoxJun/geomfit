"""适用于不适定问题的正则化方法。"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

class Regularization:
    """用于求解不适定问题的正则化方法。"""
    
    @staticmethod
    def tikhonov(G: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """
        应用 Tikhonov 正则化。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        alpha : float, 默认=1e-6
            正则化参数
            
        返回
        ----
        np.ndarray
            正则化后的矩阵 G + αI
        """
        n = G.shape[0]
        return G + alpha * np.eye(n)
    
    @staticmethod
    def ridge(G: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """Tikhonov 正则化的别名。"""
        return Regularization.tikhonov(G, alpha)
    
    @staticmethod
    def truncated_svd(G: np.ndarray, threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算截断 SVD 正则化。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        threshold : float, 默认=1e-10
            奇异值阈值
            
        返回
        ----
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            去除小奇异值后的 U、S、Vh 矩阵
        """
        U, s, Vh = np.linalg.svd(G, full_matrices=False)
        
        # 截断较小的奇异值
        mask = s > threshold
        U_trunc = U[:, mask]
        s_trunc = s[mask]
        Vh_trunc = Vh[mask, :]
        
        return U_trunc, s_trunc, Vh_trunc
    
    @staticmethod
    def solve_regularized(G: np.ndarray, b: np.ndarray, method: str = "tikhonov", 
                         **kwargs) -> np.ndarray:
        """
        求解正则化线性方程组 Gx = b。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        b : np.ndarray
            右端向量
        method : str, 默认="tikhonov"
            正则化方法："tikhonov"、"svd" 或 "lstsq"
        **kwargs
            正则化的附加参数
            
        返回
        ----
        np.ndarray
            解向量 x
        """
        if method == "tikhonov":
            alpha = kwargs.get("alpha", 1e-6)
            G_reg = Regularization.tikhonov(G, alpha)
            return np.linalg.solve(G_reg, b)
        
        elif method == "svd":
            threshold = kwargs.get("threshold", 1e-10)
            U, s, Vh = Regularization.truncated_svd(G, threshold)
            
            # 使用截断 SVD 求解
            s_inv = 1.0 / s
            x = Vh.T @ (s_inv * (U.T @ b))
            return x
        
        elif method == "lstsq":
            rcond = kwargs.get("rcond", None)
            x, residuals, rank, s = np.linalg.lstsq(G, b, rcond=rcond)
            return x
        
        else:
            raise ValueError(f"Unknown regularization method: {method}")
    
    @staticmethod
    def check_singularity(G: np.ndarray, threshold: float = 1e-10) -> Dict[str, Any]:
        """
        检查 Gram 矩阵是否奇异或病态。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        threshold : float, 默认=1e-10
            奇异值阈值
            
        返回
        ----
        Dict[str, Any]
            奇异性信息
        """
        # 计算条件数
        cond = np.linalg.cond(G)
        
        # 计算奇异值
        s = np.linalg.svd(G, compute_uv=False)
        
        # 检查奇异性
        min_sv = np.min(s)
        max_sv = np.max(s)
        is_singular = min_sv < threshold
        
        # 计算秩
        rank = np.sum(s > threshold)
        
        return {
            "is_singular": is_singular,
            "condition_number": cond,
            "min_singular_value": min_sv,
            "max_singular_value": max_sv,
            "rank": rank,
            "size": G.shape[0],
            "threshold": threshold
        }
    
    @staticmethod
    def lasso_regularization(G: np.ndarray, b: np.ndarray, alpha: float = 1e-3, 
                            max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        使用坐标下降应用 LASSO 正则化（L1 惩罚）。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        b : np.ndarray
            右端向量
        alpha : float, 默认=1e-3
            正则化参数
        max_iter : int, 默认=1000
            最大迭代次数
        tol : float, 默认=1e-6
            收敛容差
            
        返回
        ----
        np.ndarray
            解向量 x
        """
        n = G.shape[0]
        x = np.zeros(n)
        
        # 预计算 G 的对角线
        diag_G = np.diag(G)
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # 坐标下降
            for j in range(n):
                # 计算去掉第 j 个分量后的残差
                r_j = b - G @ x + G[:, j] * x[j]
                
                # 更新第 j 个分量
                numerator = np.dot(G[:, j], r_j)
                denominator = diag_G[j]
                
                # 软阈值
                if numerator > alpha:
                    x[j] = (numerator - alpha) / denominator
                elif numerator < -alpha:
                    x[j] = (numerator + alpha) / denominator
                else:
                    x[j] = 0.0
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < tol:
                break
        
        return x
    
    @staticmethod
    def elastic_net(G: np.ndarray, b: np.ndarray, alpha: float = 1e-3, 
                   l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        应用弹性网正则化（L1 + L2 惩罚）。
        
        参数
        ----
        G : np.ndarray
            Gram 矩阵
        b : np.ndarray
            右端向量
        alpha : float, 默认=1e-3
            总正则化参数
        l1_ratio : float, 默认=0.5
            L1 惩罚所占比例（0 为 ridge，1 为 lasso）
        max_iter : int, 默认=1000
            最大迭代次数
        tol : float, 默认=1e-6
            收敛容差
            
        返回
        ----
        np.ndarray
            解向量 x
        """
        n = G.shape[0]
        x = np.zeros(n)
        
        # 将 alpha 拆分为 L1 与 L2 两部分
        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)
        
        # 预计算用于 L2 正则化的修正 Gram 矩阵
        G_mod = G + alpha_l2 * np.eye(n)
        diag_G_mod = np.diag(G_mod)
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # 坐标下降
            for j in range(n):
                # 计算去掉第 j 个分量后的残差
                r_j = b - G_mod @ x + G_mod[:, j] * x[j]
                
                # 使用软阈值更新第 j 个分量
                numerator = np.dot(G_mod[:, j], r_j)
                denominator = diag_G_mod[j]
                
                # 对 L1 惩罚做软阈值处理
                if numerator > alpha_l1:
                    x[j] = (numerator - alpha_l1) / denominator
                elif numerator < -alpha_l1:
                    x[j] = (numerator + alpha_l1) / denominator
                else:
                    x[j] = 0.0
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < tol:
                break
        
        return x
