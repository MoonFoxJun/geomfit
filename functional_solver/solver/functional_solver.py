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
        self.kernel = None  # Mutually exclusive with the kernel formulation
    
    def set_kernel(self, kernel: Kernel):
        """Set the kernel formulation (overrides the basis-set formulation)."""
        self.kernel = kernel
        self.basis_set = None  # Mutually exclusive with the basis-set formulation
    
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
            # Kernel formulation
            return self._solve_kernel(regularization)
        elif self.basis_set is not None:
            # Basis-set formulation
            return self._solve_basis(regularization)
        else:
            raise ValueError("Set a basis set or a kernel first")
    
    def _solve_basis(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve via the basis-set formulation: the normal equations Gc = b."""
        from ..inner_product.regularization import Regularization
        
        Phi = self.basis_set.evaluate_all(self.data)
        G = self.inner_product.compute_gram_matrix(Phi, self.data)

        # Right-hand side b_i = ⟨φ_i, y⟩, using the same inner product as the
        # Gram matrix so that the normal equations Gc = b are consistent.
        b = self.inner_product.rhs_vector(Phi, self.target, self.data)
        
        # Singularity check
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
        
        # Solve the normal equations Gc = b
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(G, b, rcond=None)[0]
        
        self.debug_info["method"] = "basis"
        return self.coefficients
    
    def _solve_kernel(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve the ridge-regularized kernel system (K + alpha I) alpha = y."""
        K = self.kernel.compute_matrix(self.data)
        
        if regularization and "alpha" in regularization:
            K = K + regularization["alpha"] * np.eye(len(K))
        
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
