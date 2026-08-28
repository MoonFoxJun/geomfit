"""Kernel-method solver for functional approximation."""

import numpy as np
from typing import Optional, Dict, Any
from ..core.data_container import MultiDimData
from ..kernel.base import Kernel

class KernelSolver:
    """Solver based on the kernel method."""
    
    def __init__(self, kernel: Kernel):
        """
        Initialize the kernel solver.
        
        Parameters
        ----------
        kernel : Kernel
            Kernel function.
        """
        self.kernel = kernel
        self.data = None
        self.target = None
        self.coefficients = None
        self.kernel_matrix = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the data used for solving."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_kernel_matrix(self) -> np.ndarray:
        """Compute the kernel matrix K_ij = k(x_i, x_j) of the data."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # 核矩阵 K_ij = k(x_i, x_j) 是 n×n 的对称（半）正定矩阵，
        # 其作用等价于把数据映射到 RKHS 后两两做内积（核技巧）：
        # k(x_i, x_j) = ⟨φ(x_i), φ(x_j)⟩_H，从而无需显式构造高维特征 φ。
        self.kernel_matrix = self.kernel.compute_matrix(self.data)
        return self.kernel_matrix
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve for the representer coefficients alpha of (K + alpha I) alpha = y."""
        # 求解表示定理系数 α：根据表示定理，最优解形如
        #   f(x) = Σᵢ αᵢ k(x, xᵢ)，
        # 即只依赖训练点处的核函数值。在平方损失 + L2 惩罚（核岭回归）下，
        # α 满足 (K + αI)α = y。先计算核矩阵 K
        K = self.compute_kernel_matrix()
        
        # 若请求了岭正则化，就在对角线上加 αI：K ← K + α·I。
        # 这保证方程组可解（即使 K 奇异），并抑制过拟合。
        if regularization and "alpha" in regularization:
            alpha = regularization["alpha"]
            K = K + alpha * np.eye(len(K))
        
        # 解线性方程组 (K + αI)α = y 得到表示系数 α
        try:
            self.coefficients = np.linalg.solve(K, self.target)
        except np.linalg.LinAlgError:
            # 兜底：矩阵仍奇异时退回最小二乘（伪逆意义下的最小范数解）
            self.coefficients = np.linalg.lstsq(K, self.target, rcond=None)[0]
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict values for new data."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        if self.data is None:
            raise ValueError("Training data not loaded")
        
        n_train = self.data.n_points
        n_test = new_data.n_points
        
        # 预测核矩阵：行 = 新测试点，列 = 训练点，
        # K_pred[i, j] = k(x_i^new, x_j^train)。因为表示定理给出的
        # 解只与训练点有关，新点的预测 = Σᵢ αᵢ k(x_new, x_i^train)。
        K_pred = np.zeros((n_test, n_train))
        
        # 逐元素填充核矩阵（核函数一般无法向量化，只能双重循环）
        for i in range(n_test):
            x_i = new_data.get_point(i)
            for j in range(n_train):
                x_j = self.data.get_point(j)
                K_pred[i, j] = self.kernel(x_i, x_j)
        
        # 预测值 = K_pred · α
        return K_pred @ self.coefficients
    
    def predict_efficient(self, new_data: MultiDimData) -> np.ndarray:
        """Predict using the kernel's vectorized cross-matrix computation."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        if self.data is None:
            raise ValueError("Training data not loaded")
        
        # 优先使用核对象自带的向量化交叉矩阵方法（如果实现的话），
        # 它通常比这里手动逐元素双重循环快得多（利用 numpy 批量计算）。
        K_pred = self.kernel.compute_cross_matrix(new_data, self.data)
        return K_pred @ self.coefficients
    
    def get_condition_number(self) -> float:
        """Return the condition number of the kernel matrix."""
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        return np.linalg.cond(self.kernel_matrix)
    
    def get_eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of the kernel matrix in descending order."""
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        # 核矩阵是对称矩阵，用 eigvalsh（专门针对对称矩阵的特征值分解）
        # 比通用 eig 更快更稳定，且返回的特征值保证为实数。
        eigenvalues = np.linalg.eigvalsh(self.kernel_matrix)
        # 理论上核矩阵是半正定的（特征值 ≥ 0），但浮点舍入误差可能
        # 产生极小的负特征值；把它们裁剪到 0，保证"谱"非负，
        # 便于后续解释方差占比或画特征值谱。
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return np.sort(eigenvalues)[::-1]  # 降序排列：最大的特征值在前
    
    def compute_representer_weights(self) -> np.ndarray:
        """
        Return the representer-theorem weights.
        
        Returns
        -------
        np.ndarray
            Weights alpha such that f(x) = Σ alpha_i k(x, x_i).
        """
        # 表示定理的权重：就是求解得到的系数 α，满足
        #   f(x) = Σᵢ αᵢ k(x, xᵢ)。
        # 换句话说，拟合函数完全由训练点处的核函数线性组合确定。
        if self.coefficients is None:
            self.solve()
        
        return self.coefficients
    
    def compute_function_norm(self) -> float:
        """
        Compute the RKHS norm of the fitted function.
        
        Returns
        -------
        float
            RKHS norm ||f||_H = sqrt(α^T K α).
        """
        if self.coefficients is None:
            self.solve()
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        # RKHS 范数：‖f‖_H = √(αᵀ K α)。
        # 由表示定理 f(x) = Σᵢ αᵢ k(x, xᵢ)，再利用再生核性质
        # ⟨f, f⟩_H = αᵀKα，即可算出 f 在再生核希尔伯特空间中的范数。
        # 这个范数度量了拟合函数的"复杂度/光滑性"，常用作模型选择依据。
        return np.sqrt(self.coefficients.T @ self.kernel_matrix @ self.coefficients)
    
    def leave_one_out_error(self) -> np.ndarray:
        """
        Compute the leave-one-out cross-validation errors.
        
        Returns
        -------
        np.ndarray
            Leave-one-out cross-validation (LOOCV) error for each data point.
        """
        if self.coefficients is None:
            self.solve()
        if self.kernel_matrix is None:
            self.compute_kernel_matrix()
        
        n = len(self.coefficients)
        errors = np.zeros(n)
        
        # LOOCV（留一交叉验证）：对每个样本 i，用其余 n-1 个样本重新
        # 拟合后在样本 i 处预测，记录误差。朴素做法要解 n 次线性方程组，
        # 这里利用逆矩阵对角线给出闭式公式 e_i = α_i / (K⁻¹)_ii，
        # 一次求逆即可得到全部 n 个留一误差，代价大幅降低。
        # 该公式成立的前提是正则化参数为 0（或固定 K + αI 后再求逆，
        # 此时 α 也要换成对应正则化方程的解）。
        # 先一次性求逆，取出对角线 (K⁻¹)_ii
        try:
            K_inv = np.linalg.inv(self.kernel_matrix)
            diag_K_inv = np.diag(K_inv)
            
            # LOOCV 公式：e_i = α_i / (K^{-1})_ii
            # （α_i 是全量拟合的表示系数，除以逆矩阵对角线即得第 i 个留一误差）
            errors = self.coefficients / diag_K_inv
        except np.linalg.LinAlgError:
            # 兜底：如果核矩阵奇异无法求逆，就逐个样本朴素计算——
            # 对每个 i 删掉第 i 个样本，在缩小的核矩阵上重新求解。
            for i in range(n):
                # 构造掩码：把第 i 个样本排除在外（置 False），
                # 其余样本保留（True），用来从矩阵/向量中"挖掉"一行一列。
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                
                # 取子矩阵：K_reduced = K[除 i 外, 除 i 外]（n-1 阶），
                # 目标向量也去掉第 i 个分量。
                K_reduced = self.kernel_matrix[mask, :][:, mask]
                target_reduced = self.target[mask]
                
                # 在缩小的系统上解出系数 α_reduced
                try:
                    alpha_reduced = np.linalg.solve(K_reduced, target_reduced)
                except np.linalg.LinAlgError:
                    alpha_reduced = np.linalg.lstsq(K_reduced, target_reduced, rcond=None)[0]
                
                # 用被"挖掉"的第 i 行核值做预测：
                # f(x_i) = Σⱼ α_reduced_j · k(x_i, x_j)，j 遍历保留的样本
                k_pred = self.kernel_matrix[i, mask]
                pred = k_pred @ alpha_reduced
                errors[i] = self.target[i] - pred
        
        return errors
