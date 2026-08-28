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
        
        Phi = self.basis_set.evaluate_all(self.data)
        self.gram_matrix = self.inner_product.compute_gram_matrix(Phi, self.data)
        return self.gram_matrix
    
    def compute_rhs(self) -> np.ndarray:
        """Compute the right-hand side vector b_i = <phi_i, y>, using the same
        inner product as the Gram matrix."""
        if self.data is None or self.target is None:
            raise ValueError("Data and target not loaded")
        
        Phi = self.basis_set.evaluate_all(self.data)
        return self.inner_product.rhs_vector(Phi, self.target, self.data)
    
    def solve(self, regularization: Optional[Dict] = None) -> np.ndarray:
        """Solve the normal equations Gc = b for the coefficients."""
        from ..inner_product.regularization import Regularization
        
        # Compute the Gram matrix and the right-hand side
        G = self.compute_gram_matrix()
        b = self.compute_rhs()
        
        # Apply regularization if requested
        if regularization:
            method = regularization.get("method", "tikhonov")
            if method == "tikhonov":
                alpha = regularization.get("alpha", 1e-6)
                G = Regularization.tikhonov(G, alpha)
            elif method == "svd":
                threshold = regularization.get("threshold", 1e-10)
                U, s, Vh = Regularization.truncated_svd(G, threshold)
                s_inv = 1.0 / s
                self.coefficients = Vh.T @ (s_inv * (U.T @ b))
                return self.coefficients
        
        # Solve the linear system
        try:
            self.coefficients = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
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
        
        return np.linalg.cond(self.gram_matrix)
    
    def get_singular_values(self) -> np.ndarray:
        """Return the singular values of the Gram matrix."""
        if self.gram_matrix is None:
            self.compute_gram_matrix()
        
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
        
        # Normalize by the diagonal to obtain a correlation-like matrix
        diag = np.diag(self.gram_matrix)
        diag_sqrt = np.sqrt(diag)
        diag_inv_sqrt = 1.0 / diag_sqrt
        
        # Normalized Gram matrix
        G_norm = self.gram_matrix * diag_inv_sqrt[:, None] * diag_inv_sqrt[None, :]
        
        return G_norm
