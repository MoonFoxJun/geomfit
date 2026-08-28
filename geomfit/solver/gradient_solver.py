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
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # Half the squared L2 residual
        loss = 0.5 * np.sum(residuals ** 2)
        
        return loss
    
    def compute_gradient(self, coefficients: np.ndarray) -> np.ndarray:
        """Compute the loss gradient Phi^T (Phi c - y)."""
        Phi = self.basis_set.evaluate_all(self.data)
        predictions = Phi @ coefficients
        residuals = predictions - self.target
        
        # Gradient: Φ^T (Φc - y)
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
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        self.loss_history = []
        
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for iteration in range(max_iter):
            loss = self.compute_loss(self.coefficients)
            
            # Add the regularization term to the loss
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # Add the regularization term to the gradient
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # Gradient-descent update: c <- c - eta * grad
            self.coefficients -= learning_rate * gradient
            
            # Stop when the loss change falls below the tolerance
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - loss)
                if loss_change < tol:
                    break
        
        return self.coefficients
    
    def solve_with_momentum(self, learning_rate: float = 0.01, momentum: float = 0.9,
                           max_iter: int = 1000, tol: float = 1e-6,
                           regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve by gradient descent with momentum."""
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
            
            # Add the regularization term to the loss
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # Add the regularization term to the gradient
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # Momentum update: v <- mu * v - eta * grad; c <- c + v
            velocity = momentum * velocity - learning_rate * gradient
            self.coefficients += velocity
            
            # Stop when the loss change falls below the tolerance
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
        n_basis = len(self.basis_set)
        self.coefficients = np.zeros(n_basis)
        m = np.zeros(n_basis)  # First moment estimate
        v = np.zeros(n_basis)  # Second raw moment estimate
        self.loss_history = []
        
        reg_lambda = 0.0
        reg_type = "l2"
        if regularization:
            reg_lambda = regularization.get("lambda", 0.0)
            reg_type = regularization.get("type", "l2")
        
        for t in range(1, max_iter + 1):
            loss = self.compute_loss(self.coefficients)
            
            # Add the regularization term to the loss
            if reg_lambda > 0:
                if reg_type == "l2":
                    loss += 0.5 * reg_lambda * np.sum(self.coefficients ** 2)
                elif reg_type == "l1":
                    loss += reg_lambda * np.sum(np.abs(self.coefficients))
            
            self.loss_history.append(loss)
            
            gradient = self.compute_gradient(self.coefficients)
            
            # Add the regularization term to the gradient
            if reg_lambda > 0:
                if reg_type == "l2":
                    gradient += reg_lambda * self.coefficients
                elif reg_type == "l1":
                    gradient += reg_lambda * np.sign(self.coefficients)
            
            # Biased first-moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g
            m = beta1 * m + (1 - beta1) * gradient
            
            # Biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Bias-corrected first-moment estimate
            m_hat = m / (1 - beta1 ** t)
            
            # Bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** t)
            
            # Parameter update with numerical-stability epsilon
            self.coefficients -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Stop when the loss change falls below the tolerance
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
        if self.coefficients is None:
            self.coefficients = np.zeros(len(self.basis_set))
        
        gradient = self.compute_gradient(self.coefficients)
        Phi = self.basis_set.evaluate_all(self.data)
        
        # Hessian approximation: H = Phi^T Phi
        H = Phi.T @ Phi
        
        # Optimal learning rate for a quadratic: 1 / lambda_max(H)
        eigenvalues = np.linalg.eigvalsh(H)
        max_eigenvalue = np.max(eigenvalues)
        
        return 1.0 / max_eigenvalue if max_eigenvalue > 0 else 0.01
