"""Gradient-descent solver for functional approximation."""

import numpy as np
from typing import Optional, Dict, Any, Callable
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct

class GradientSolver:
    """Gradient-descent solver for the least-squares fitting problem."""
    
    def __init__(self, basis_set: BasisSet, inner_product: InnerProduct):
        """
        Initialize the gradient-descent solver.
        
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
        self.loss_history = []
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the data used for solving."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_loss(self, coefficients: np.ndarray) -> float:
        """Compute the least-squares loss 0.5 * ||Phi c - y||^2."""
        # 组装设计矩阵 Φ（行 = 数据点，列 = 基函数），
        # 预测值 = Φc，残差 = Φc - y
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # 最小二乘损失取 ½‖Φc − y‖²：前面的 1/2 是为了让损失对 c 求导时
        # 恰好抵消掉平方项的系数 2，得到干净的梯度表达式 Φᵀ(Φc − y)。
        loss = 0.5 * np.sum(residuals ** 2)
        
        return loss
    
    def compute_gradient(self, coefficients: np.ndarray) -> np.ndarray:
        """Compute the loss gradient Phi^T (Phi c - y)."""
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # 梯度：∇L = Φᵀ(Φc − y)（链式法则：d(½‖r‖²)/dc = Φᵀr）。
        # 几何上这是损失函数在 c 处的下降方向；梯度下降沿负梯度走。
        gradient = Phi.T @ residuals
        
        return gradient
    
    def solve(self, learning_rate: float = 0.01, max_iter: int = 1000,
              tol: float = 1e-6, regularization: Optional[Dict] = None) -> np.ndarray:
        """
        Solve by plain gradient descent on the least-squares loss.
        
        Parameters
        ----------
        learning_rate : float, default=0.01
            Learning rate of the gradient-descent update.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance on the change in loss.
        regularization : Dict, optional
            Regularization parameters; supported keys are "lambda" (strength)
            and "type" ("l1" or "l2").
            
        Returns
        -------
        np.ndarray
            Optimal coefficients.
        """
        # 初始化：系数从全零开始，损失历史清空（用于画学习曲线）
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        self.loss_history = []
        
        # 正则化参数：默认不启用（λ=0），类型默认 L2。
        # L2 惩罚是 ½λ‖c‖²（平滑、不稀疏），L1 惩罚是 λ‖c‖₁（可产生稀疏解）。
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for iteration in range(max_iter):
            loss = self.compute_loss(self.coefficients)
            
            # 把正则项加到损失上（只影响数值，不参与梯度之外的逻辑）
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # 把正则项的梯度加到数据拟合梯度上：
            #   L2 → 梯度加 λc（对 c 求导）；
            #   L1 → 梯度加 λ·sign(c)（|c| 在 c≠0 处导数为 ±1）。
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 梯度下降更新：c ← c − η·∇L（沿负梯度方向迈出一步，步长 η）
            self.coefficients -= learning_rate * gradient
            
            # 收敛判据：相邻两次迭代的损失变化小于 tol 就提前停止
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def solve_with_momentum(self, learning_rate: float = 0.01, momentum: float = 0.9,
                           max_iter: int = 1000, tol: float = 1e-6,
                           regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve by gradient descent with momentum."""
        # 动量法：在普通梯度下降基础上引入"速度" v，让更新方向融合历史
        # 梯度的惯性（momentum 通常取 0.9）。好处：在损失面较陡的方向上
        # 加速、在来回震荡的方向上相互抵消，收敛更快更稳。
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        velocity = np.zeros(n_basis)
        self.loss_history = []
        
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for iteration in range(max_iter):
            loss = self.compute_loss(self.coefficients)
            
            # 正则项计入损失（与 solve 中一致）
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # 正则项的梯度（L2 → λc；L1 → λ·sign(c)）
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 动量更新：v ← μ·v − η·∇L；c ← c + v
            # 即速度 = 上一时刻速度按 μ 衰减 + 当前负梯度贡献，
            # 系数沿速度方向移动。
            velocity = momentum * velocity - learning_rate * gradient
            self.coefficients += velocity
            
            # 损失变化小于 tol 时提前收敛
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def solve_adam(self, learning_rate: float = 0.001, beta1: float = 0.9,
                  beta2: float = 0.999, epsilon: float = 1e-8,
                  max_iter: int = 1000, tol: float = 1e-6,
                  regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve using the Adam optimizer."""
        # Adam = 自适应矩估计：同时维护一阶矩 m（梯度均值，方向）和
        # 二阶矩 v（梯度平方均值，尺度），并对每个参数单独归一化学习率，
        # 因此对学习率不敏感、收敛稳定，是深度学习事实标准。
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        m = np.zeros(n_basis)  # 一阶矩估计（梯度的指数滑动平均）
        v = np.zeros(n_basis)  # 二阶原始矩估计（梯度平方的指数滑动平均）
        self.loss_history = []
        
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        # t 从 1 开始编号，因为偏差修正公式里要用到 beta^t
        for t in range(1, max_iter + 1):
            loss = self.compute_loss(self.coefficients)
            
            # 正则项计入损失
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # 正则项的梯度
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 有偏一阶矩估计：m_t = β₁·m_{t−1} + (1−β₁)·g
            # （对历史梯度做指数加权平均，β₁=0.9 时约等效于最近 10 步的均值）
            m = beta1 * m + (1 - beta1) * gradient
            
            # 有偏二阶原始矩估计：v_t = β₂·v_{t−1} + (1−β₂)·g²
            # （对梯度平方做指数加权平均，反映每个参数方向的梯度幅度）
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # 偏差修正的一阶矩估计：除以 (1 − β₁ᵗ)。因为 m、v 从 0 起步，
            # 早期估计偏向 0，除以这个因子可以把偏置纠正回来（t 越大越接近 1）。
            m_hat = m / (1 - beta1 ** t)
            
            # 偏差修正的二阶原始矩估计
            v_hat = v / (1 - beta2 ** t)
            
            # 参数更新：c ← c − η·m̂/(√v̂ + ε)。
            # 分母里的 √v̂ 起到逐参数自适应学习率的作用（梯度大的方向走得慢），
            # ε 是数值稳定性小量，防止除以 0。
            self.coefficients -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # 损失变化小于 tol 时提前收敛
            if t > 1:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict values for new data."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def get_loss_history(self) -> np.ndarray:
        """Return the loss history over training iterations."""
        return np.array(self.loss_history)
    
    def compute_optimal_learning_rate(self) -> float:
        """
        Estimate the optimal learning rate for the quadratic least-squares loss.
        
        The optimal rate for a quadratic objective is 1 / lambda_max, where
        lambda_max is the largest eigenvalue of the Hessian approximation
        H = Phi^T Phi.
        
        Returns
        -------
        float
            Optimal learning rate, or a default of 0.01 when H is singular.
        """
        # 先保证系数存在（缺省为零向量），并预计算一次梯度、组装 Φ。
        # 注意：最优学习率的推导只依赖 Φ，与当前系数取值无关（二次损失）。
        if self.coefficients is None:
            self.coefficients = np.zeros(len(self.basis_set))
        
        gradient = self.compute_gradient(self.coefficients)
        Phi = self.basis_set.evaluate_all(self.data)
        
        # Hessian 近似：H = ΦᵀΦ。二次损失 L = ½‖Φc − y‖² 的精确 Hessian
        # 就是 ΦᵀΦ（与 c 无关的常数矩阵），它刻画了损失面的曲率。
        H = Phi.T @ Phi
        
        # 二次目标的最优学习率：1 / λ_max(H)。
        # 直觉：沿 Hessian 最大特征值 λ_max 对应的方向，曲率最大，
        # 步长超过 1/λ_max 就会发散（震荡），等于它则一步收敛到该方向的最小值。
        eigenvalues = np.linalg.eigvalsh(H)
        max_eigenvalue = np.max(eigenvalues)
        
        # 若 H 奇异（λ_max ≤ 0），退回默认学习率 0.01
        return 1.0 / max_eigenvalue if max_eigenvalue > 0 else 0.01
