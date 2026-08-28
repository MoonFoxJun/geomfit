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
        
        self.kernel_matrix = self.kernel.compute_matrix(self.data)
        return self.kernel_matrix
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve for the representer coefficients alpha of (K + alpha I) alpha = y."""
        K = self.compute_kernel_matrix()
        
        # Apply ridge regularization if requested
        if regularization and "alpha" in regularization:
            alpha = regularization["alpha"]
            K = K + alpha * np.eye(len(K))
        
        # Solve the linear system
        try:
            self.coefficients = np.linalg.solve(K, self.target)
        except np.linalg.LinAlgError:
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
        
        # Kernel matrix between new and training data
        K_pred = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            x_i = new_data.get_point(i)
            for j in range(n_train):
                x_j = self.data.get_point(j)
                K_pred[i, j] = self.kernel(x_i, x_j)
        
        return K_pred @ self.coefficients
    
    def predict_efficient(self, new_data: MultiDimData) -> np.ndarray:
        """Predict using the kernel's vectorized cross-matrix computation."""
        if self.coefficients is None:
            raise ValueError("Must call solve() first")
        if self.data is None:
            raise ValueError("Training data not loaded")
        
        # Use the kernel's cross-matrix method when available
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
        
        # The kernel matrix is symmetric; use eigh for efficiency
        eigenvalues = np.linalg.eigvalsh(self.kernel_matrix)
        # Clip small negative eigenvalues caused by floating-point round-off so
        # that the (theoretically positive semidefinite) kernel matrix stays nonnegative
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return np.sort(eigenvalues)[::-1]  # Descending order
    
    def compute_representer_weights(self) -> np.ndarray:
        """
        Return the representer-theorem weights.
        
        Returns
        -------
        np.ndarray
            Weights alpha such that f(x) = Σ alpha_i k(x, x_i).
        """
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
        
        # Diagonal of the kernel-matrix inverse, computed in one pass
        try:
            K_inv = np.linalg.inv(self.kernel_matrix)
            diag_K_inv = np.diag(K_inv)
            
            # LOOCV formula: e_i = alpha_i / (K^{-1})_{ii}
            errors = self.coefficients / diag_K_inv
        except np.linalg.LinAlgError:
            # Fallback: compute each leave-one-out error individually
            for i in range(n):
                # Remove the i-th data point
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                
                K_reduced = self.kernel_matrix[mask, :][:, mask]
                target_reduced = self.target[mask]
                
                # Solve the reduced system
                try:
                    alpha_reduced = np.linalg.solve(K_reduced, target_reduced)
                except np.linalg.LinAlgError:
                    alpha_reduced = np.linalg.lstsq(K_reduced, target_reduced, rcond=None)[0]
                
                # Predict at the held-out point
                k_pred = self.kernel_matrix[i, mask]
                pred = k_pred @ alpha_reduced
                errors[i] = self.target[i] - pred
        
        return errors
