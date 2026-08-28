"""Main solver integrating the basis-set and kernel formulations."""

import numpy as np
from typing import Optional, Dict, Any

from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct
from ..kernel.base import Kernel

class FunctionalSolver:
    """Main functional solver supporting both the basis-set and kernel paths."""
    
    def __init__(self):
        self.basis_set = None
        self.inner_product = None
        self.kernel = None
        self.data = None
        self.target = None
        self.coefficients = None
        self.debug_info = {}
    
    def set_basis(self, basis_set: BasisSet):
        """Set the basis-set formulation."""
        self.basis_set = basis_set
        # 基函数路径与核路径互斥：一旦设置了基函数，就必须清空 kernel，
        # 否则 solve() 里无法判断该走哪条路径（solve() 优先检查 kernel）。
        self.kernel = None  # 与核路径互斥：设置了基函数后不能再保留核函数
    
    def set_kernel(self, kernel: Kernel):
        """Set the kernel formulation (overrides the basis-set formulation)."""
        self.kernel = kernel
        # 同理，设置核函数后要把 basis_set 清空，保证两条路径二选一，
        # 不会同时携带两套参数导致 solve() 行为歧义。
        self.basis_set = None  # 与基函数路径互斥：设置了核函数后不能再保留基函数集合
    
    def set_inner_product(self, inner_product: InnerProduct):
        """Set the inner-product definition."""
        self.inner_product = inner_product
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the training data."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve for the optimal coefficients."""
        if self.data is None or self.target is None:
            raise ValueError("Load data first")
        
        if self.kernel is not None:
            # 核路径：用核函数隐式地把数据映射到高维（甚至无穷维）特征空间，
            # 通过表示定理把求解转化为核矩阵上的线性方程组，见 _solve_kernel。
            return self._solve_kernel(regularization)
        elif self.basis_set is not None:
            # 基函数路径：显式地构造基函数矩阵 Φ，再解法方程 Gc = b，
            # 见 _solve_basis。
            return self._solve_basis(regularization)
        else:
            raise ValueError("Set a basis set or a kernel first")
    
    def _solve_basis(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve via the basis-set formulation: the normal equations Gc = b."""
        from ..inner_product.regularization import Regularization
        
        # 基函数路径的核心流程（正交投影 / 最小二乘）：
        #   1) 组装设计矩阵 Φ：每一列是某个基函数 φ_i 在所有数据点上的取值；
        #   2) 用内积计算 Gram 矩阵 G_ij = ⟨φ_i, φ_j⟩；
        #   3) 计算右端项 b_i = ⟨φ_i, y⟩；
        #   4) 解法方程 Gc = b，得到系数 c。
        # 几何上，解 Gc = b 等价于把目标 y 正交投影到基函数张成的子空间上，
        # 投影坐标正是最小二乘意义下的最优系数。
        Phi = self.basis_set.evaluate_all(self.data)
        G = self.inner_product.compute_gram_matrix(Phi, self.data)

        # 右端项 b_i = ⟨φ_i, y⟩：必须与 Gram 矩阵使用同一个内积定义，
        # 否则法方程 Gc = b 的两边不在同一坐标系下，解出来没有最小二乘意义。
        # （这里内积通常是对数据点求和的离散内积，也可能是积分内积。）
        b = self.inner_product.rhs_vector(Phi, self.target, self.data)
        
        # 奇异性检查：若 Gram 矩阵接近奇异（病态），直接求解会放大数值误差，
        # 需要先做正则化再求解；检查结果（条件数/最小奇异值等）存入 debug_info。
        singularity_info = Regularization.check_singularity(G)
        self.debug_info["singularity"] = singularity_info
        
        if singularity_info["is_singular"] and regularization:
            method = regularization.get("method", "tikhonov")
            if method == "tikhonov":
                # Tikhonov（岭）正则化：把 G 替换成 G + αI，等价于在系数上
                # 加一个 L2 惩罚项，给对角线加小的正值能抬高特征值、
                # 改善条件数，从而稳定求解。
                alpha = regularization.get("alpha", 1e-6)
                G = Regularization.tikhonov(G, alpha)
            elif method == "svd":
                # 截断 SVD 正则化：对 G 做 SVD 分解 G = U·diag(s)·Vᵀ，
                # 把小于阈值的奇异值直接截断（置零），再用
                # G⁻¹ ≈ V·diag(1/s)·Uᵀ 回代到法方程，避免除以接近 0 的奇异值。
                threshold = regularization.get("threshold", 1e-10)
                U, s, Vh = Regularization.truncated_svd(G, threshold)
                s_inv = 1.0 / s
                # 求解：c = V·diag(1/s)·Uᵀ·b，其中 Vh.T = V（Vh 是 V 的共轭转置）
                self.coefficients = Vh.T @ (s_inv * (U.T @ b))
                self.debug_info["method"] = "basis_svd"
                return self.coefficients
        
        # 求解法方程 Gc = b（对正定 G 用 Cholesky/LU 分解，速度快且稳定）
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            # 兜底：如果矩阵奇异导致直接求解失败（例如上面未启用正则化），
            # 退回最小二乘 lstsq，它在数值上等价于用伪逆求最小范数解。
            self.coefficients = np.linalg.lstsq(G, b, rcond=None)[0]
        
        self.debug_info["method"] = "basis"
        return self.coefficients
    
    def _solve_kernel(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve the ridge-regularized kernel system (K + alpha I) alpha = y."""
        # 核路径：根据表示定理，最优拟合函数可以写成训练点核函数的线性组合
        #   f(x) = Σᵢ αᵢ k(x, xᵢ)，
        # 因此只需求解系数 α。在最小二乘 + L2 惩罚（核岭回归）下，
        # α 满足线性方程组 (K + αI)α = y，其中 K 是核矩阵 K_ij = k(x_i, x_j)。
        K = self.kernel.compute_matrix(self.data)
        
        if regularization and "alpha" in regularization:
            # 岭正则化：在核矩阵对角线上加 αI。由于核矩阵可能病态，
            # 加一个小的正对角项能保证方程组可解，并抑制过拟合
            # （等价于在 RKHS 范数 ‖f‖_H² = αᵀKα 上加了惩罚）。
            K = K + regularization["alpha"] * np.eye(len(K))
        
        # 直接解线性方程组得到表示系数 α；理论上 (K + αI) 正定，求解稳定
        self.coefficients = np.linalg.solve(K, self.target)
        self.debug_info["method"] = "kernel"
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict on new data."""
        if self.coefficients is None:
            raise ValueError("Call solve() first")
        
        if self.kernel is not None:
            return self._predict_kernel(new_data)
        else:
            return self._predict_basis(new_data)
    
    def _predict_basis(self, new_data: MultiDimData) -> np.ndarray:
        # 基函数路径预测：在新的数据点上重新组装设计矩阵 Φ_new，
        # 预测值 = Φ_new · c（把系数 c 当作基函数的线性组合系数直接加权求和）。
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def _predict_kernel(self, new_data: MultiDimData) -> np.ndarray:
        # 核路径预测：根据表示定理 f(x) = Σᵢ αᵢ k(x, xᵢ)，
        # 需要逐个计算新点 x_i 与每个训练点 x_j 之间的核值 k(x_i, x_j)，
        # 组成预测核矩阵 K_pred，再与系数 α 做矩阵乘法。
        n_train = self.data.n_points
        n_test = new_data.n_points
        K_pred = np.zeros((n_test, n_train))
        
        # 双重循环逐元素填充核矩阵：行对应测试点，列对应训练点
        for i in range(n_test):
            for j in range(n_train):
                K_pred[i, j] = self.kernel(
                    new_data.get_point(i),
                    self.data.get_point(j)
                )
        
        # 预测值 = K_pred · α（K_pred 的每一行是某个新点与全部训练点的核值）
        return K_pred @ self.coefficients
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Return the debug information dictionary."""
        return self.debug_info
    
    def reset(self):
        """Reset the solver state."""
        self.basis_set = None
        self.inner_product = None
        self.kernel = None
        self.data = None
        self.target = None
        self.coefficients = None
        self.debug_info = {}
