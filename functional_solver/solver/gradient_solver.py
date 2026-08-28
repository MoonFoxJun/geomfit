"""用于函数逼近的梯度下降求解器。"""

import numpy as np
from typing import Optional, Dict, Any, Callable
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet
from ..inner_product.base import InnerProduct

class GradientSolver:
    """用于函数优化的梯度下降求解器。"""
    
    def __init__(self, basis_set: BasisSet, inner_product: InnerProduct):
        """
        初始化梯度下降求解器。
        
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
        self.loss_history = []
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """加载用于求解的数据。"""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
    
    def compute_loss(self, coefficients: np.ndarray) -> float:
        """计算损失函数值。"""
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # L2 损失
        loss = 0.5 * np.sum(residuals ** 2)
        
        return loss
    
    def compute_gradient(self, coefficients: np.ndarray) -> np.ndarray:
        """计算损失函数的梯度。"""
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # 梯度：Φ^T (Φc - y)
        gradient = Phi.T @ residuals
        
        return gradient
    
    def solve(self, learning_rate: float = 0.01, max_iter: int = 1000,
              tol: float = 1e-6, regularization: Optional[Dict] = None) -> np.ndarray:
        """
        使用梯度下降求解。
        
        参数
        ----------
        learning_rate : float, 默认=0.01
            梯度下降的学习率
        max_iter : int, 默认=1000
            最大迭代次数
        tol : float, 默认=1e-6
            收敛容差
        regularization : Dict, 可选
            正则化参数
            
        返回
        -------
        np.ndarray
            最优系数
        """
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        self.loss_history = []
        
        # 提取正则化参数
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for iteration in range(max_iter):
            # 计算损失
            loss = self.compute_loss(self.coefficients)
            
            # 加入正则化损失
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            # 计算梯度
            gradient = self.compute_gradient(self.coefficients)
            
            # 加入正则化梯度
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 更新系数
            self.coefficients -= learning_rate * gradient
            
            # 检查收敛性
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def solve_with_momentum(self, learning_rate: float = 0.01, momentum: float = 0.9,
                           max_iter: int = 1000, tol: float = 1e-6,
                           regularization: Optional[Dict] = None) -> np.ndarray:
        """使用带动量的梯度下降求解。"""
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        velocity = np.zeros(n_basis)
        self.loss_history = []
        
        # 提取正则化参数
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for iteration in range(max_iter):
            # 计算损失
            loss = self.compute_loss(self.coefficients)
            
            # 加入正则化损失
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            # 计算梯度
            gradient = self.compute_gradient(self.coefficients)
            
            # 加入正则化梯度
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 更新动量和系数
            velocity = momentum * velocity - learning_rate * gradient
            self.coefficients += velocity
            
            # 检查收敛性
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def solve_adam(self, learning_rate: float = 0.001, beta1: float = 0.9,
                  beta2: float = 0.999, epsilon: float = 1e-8,
                  max_iter: int = 1000, tol: float = 1e-6,
                  regularization: Optional[Dict] = None) -> np.ndarray:
        """使用 Adam 优化器求解。"""
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        m = np.zeros(n_basis)  # 一阶矩
        v = np.zeros(n_basis)  # 二阶矩
        self.loss_history = []
        
        # 提取正则化参数
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for t in range(1, max_iter + 1):
            # 计算损失
            loss = self.compute_loss(self.coefficients)
            
            # 加入正则化损失
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            # 计算梯度
            gradient = self.compute_gradient(self.coefficients)
            
            # 加入正则化梯度
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # 更新有偏一阶矩估计
            m = beta1 * m + (1 - beta1) * gradient
            
            # 更新有偏二阶原始矩估计
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # 计算偏差修正后的一阶矩估计
            m_hat = m / (1 - beta1 ** t)
            
            # 计算偏差修正后的二阶原始矩估计
            v_hat = v / (1 - beta2 ** t)
            
            # 更新参数
            self.coefficients -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # 检查收敛性
            if t > 1:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """预测新数据的值。"""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        
        Phi_new = self.basis_set.evaluate_all(new_data)
        return Phi_new @ self.coefficients
    
    def get_loss_history(self) -> np.ndarray:
        """获取训练过程中的损失历史。"""
        return np.array(self.loss_history)
    
    def compute_optimal_learning_rate(self) -> float:
        """
        使用线搜索计算最优学习率。
        
        返回
        -------
        float
            最优学习率
        """
        if self.coefficients is None:
            self.coefficients = np.zeros(len(self.basis_set))
        
        gradient = self.compute_gradient(self.coefficients)
        Phi = self.basis_set.evaluate_all(self.data)
        
        # 计算 Hessian 近似：Φ^T Φ
        H = Phi.T @ Phi
        
        # 二次函数的最优学习率：1 / H 的最大特征值
        eigenvalues = np.linalg.eigvalsh(H)
        max_eigenvalue = np.max(eigenvalues)
        
        return 1.0 / max_eigenvalue if max_eigenvalue > 0 else 0.01
