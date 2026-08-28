"""Gram-matrix solver for functional approximation."""

import numpy as np
from typing import Optional, Dict, Any
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct

class GramSolver:
    """Solver based on the Gram-matrix formulation."""
    
    def __init__(self, basis_set: BasisSet, inner_product: InnerProduct):
        """
        Initialize the Gram solver.
        
        Parameters
        ----------
        basis_set : BasisSet
            Basis-function set.
        inner_product : InnerProduct
            Inner-product definition.
        """
        self.basis_set = basis_set
        self.inner_product = inner_product
        self.data = None
        self.target = None
        self.coefficients = None
        self.gram_matrix = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the data used for solving."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_gram_matrix(self) -> np.ndarray:
        """Compute the Gram matrix G_ij = <phi_i, phi_j> of the basis set."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # 先组装设计矩阵 Φ（行 = 数据点，列 = 基函数），
        # 再用用户定义的内积算出 Gram 矩阵 G_ij = ⟨φ_i, φ_j⟩。
        # Gram 矩阵编码了基函数之间的"几何关系"：对角元是各基函数
        # 的范数平方，非对角元反映基函数之间的重叠/相关性。
        Phi = self.basis_set.evaluate_all(self.data)
        self.gram_matrix = self.inner_product.compute_gram_matrix(Phi, self.data)
        return self.gram_matrix
    
    def compute_rhs(self) -> np.ndarray:
        """Compute the right-hand side vector b_i = <phi_i, y>, using the same
        inner product as the Gram matrix."""
        if self.data is None or self.target is None:
            raise ValueError("Data and target not loaded")
        
        # 右端项 b_i = ⟨φ_i, y⟩：把目标函数 y 与每个基函数做内积，
        # 得到法方程 Gc = b 的右端。几何意义是 y 在基函数方向上的投影分量；
        # 必须与 Gram 矩阵使用同一内积，法方程才自洽。
        Phi = self.basis_set.evaluate_all(self.data)
        return self.inner_product.rhs_vector(Phi, self.target, self.data)
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve the normal equations Gc = b for the coefficients."""
        from ..inner_product.regularization import Regularization
        
        # 求解法方程 Gc = b：先准备好 Gram 矩阵 G 与右端项 b
        G = self.compute_gram_matrix()
        b = self.compute_rhs()
        
        # 按需正则化：如果 Gram 矩阵病态（例如基函数近似线性相关），
        # 直接求解会放大浮点误差，需先做 Tikhonov（岭）或截断 SVD 处理。
        if regularization:
            method = regularization.get("method", "tikhonov")
            if method == "tikhonov":
                # 岭正则化：G ← G + αI，抬高小特征值，改善条件数
                alpha = regularization.get("alpha", 1e-6)
                G = Regularization.tikhonov(G, alpha)
            elif method == "svd":
                # 截断 SVD：分解 G = U·diag(s)·Vᵀ，截掉过小的奇异值后
                # 用伪逆形式 c = V·diag(1/s)·Uᵀ·b 求解，避免 1/s 爆炸
                threshold = regularization.get("threshold", 1e-10)
                U, s, Vh = Regularization.truncated_svd(G, threshold)
                s_inv = 1.0 / s
                self.coefficients = Vh.T @ (s_inv * (U.T @ b))
                return self.coefficients
        
        # 解线性方程组 Gc = b（G 对称正定时数值上最稳定）
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            # 兜底：若 G 奇异导致 solve 失败，退回 lstsq（等价于伪逆最小二乘解）
            self.coefficients = np.linalg.lstsq(G, b, rcond=None)[0]
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict values for new data."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def get_condition_number(self) -> float:
        """Return the condition number of the Gram matrix."""
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        # 条件数 κ(G) = σ_max / σ_min（最大奇异值 / 最小奇异值）。
        # 它衡量方程组 Gc = b 对右端扰动的敏感程度：条件数越大，
        # 数值求解越不稳定，说明基函数越接近线性相关（病态）。
        return np.linalg.cond(self.gram_matrix)
    
    def get_singular_values(self) -> np.ndarray:
        """Return the singular values of the Gram matrix."""
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        # 只计算奇异值（compute_uv=False 跳过 U、V 分解，节省内存）。
        # 由于 G 是（半）正定的，奇异值即特征值；观察奇异值谱可以判断
        # 基函数是否退化——出现接近 0 的奇异值说明存在冗余/近似线性相关。
        return np.linalg.svd(self.gram_matrix, compute_uv=False)
    
    def check_orthogonality(self) -> np.ndarray:
        """
        Check the orthogonality of the basis functions.
        
        Returns
        -------
        np.ndarray
            Inner-product matrix between basis functions, normalized by the
            diagonal to give a correlation-like matrix.
        """
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
        # 正交性检查：把 Gram 矩阵按对角线归一化，得到"相关系数矩阵"。
        # 归一化公式 G_norm_ij = G_ij / (√G_ii · √G_jj)（即用基函数范数
        # 的乘积去除），使得对角元恒为 1、非对角元落在 [-1, 1]。
        # 若基函数两两正交，G_norm 应近似等于单位阵；
        # 若某两个基函数高度相关，对应元素接近 ±1。
        diag = np.diag(self.gram_matrix)
        diag_sqrt = np.sqrt(diag)
        diag_inv_sqrt = 1.0 / diag_sqrt
        
        # 归一化 Gram 矩阵：左乘 diag_inv_sqrt（按列缩放）、
        # 右乘 diag_inv_sqrt（按行缩放），等价于每个元素除以对应范数之积。
        G_norm = self.gram_matrix * diag_inv_sqrt[:, None] * diag_inv_sqrt[None, :]
        
        return G_norm
