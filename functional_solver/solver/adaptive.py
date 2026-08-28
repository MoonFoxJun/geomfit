"""Adaptive basis selection for functional approximation."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet, BasisInfo
from ..inner_product.base import InnerProduct

class AdaptiveSolver:
    """Solver performing greedy adaptive selection over basis candidates."""
    
    def __init__(self, inner_product: InnerProduct):
        """
        Initialize the adaptive solver.
        
        Parameters
        ----------
        inner_product : InnerProduct
            Inner-product definition.
        """
        self.inner_product = inner_product
        self.data = None
        self.target = None
        self.basis_set = BasisSet()
        self.coefficients = None
        self.selected_indices = []
        self.residuals = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """Load the data used for solving."""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
        self.residuals = target.copy()
    
    def add_basis_candidate(self, basis_info: BasisInfo):
        """Add a single basis-function candidate."""
        self.basis_set.add_basis(basis_info)
    
    def add_basis_candidates(self, basis_list: List[BasisInfo]):
        """Add multiple basis-function candidates."""
        for basis in basis_list:
            self.basis_set.add_basis(basis)
    
    def forward_selection(self, max_basis: int = 10, tol: float = 1e-6) -> List[int]:
        """
        Perform greedy forward selection over the basis candidates.
        
        At each step, the candidate whose basis values correlate most strongly
        with the current residual is added, and the coefficients are then
        refit by least squares on the selected subset.
        
        Parameters
        ----------
        max_basis : int, default=10
            Maximum number of basis functions to select.
        tol : float, default=1e-6
            Tolerance on the residual correlation; selection stops when the best
            correlation falls below it.
            
        Returns
        -------
        List[int]
            Indices of the selected basis functions.
        """
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # Evaluate all candidate basis functions
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # Basis values at all data points
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # Correlation of the candidate with the residual
                correlation = np.abs(np.dot(self.residuals, basis_values))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            if best_index == -1 or best_reduction < tol:
                break
            
            self.selected_indices.append(best_index)
            
            # Refit the least-squares solution on the selected subset
            self._update_solution()
        
        return self.selected_indices
    
    def orthogonal_matching_pursuit(self, max_basis: int = 10, tol: float = 1e-6) -> List[int]:
        """
        Perform orthogonal matching pursuit (OMP) over the basis candidates.
        
        Unlike forward selection, each candidate is orthogonalized against the
        span of the already selected basis before its correlation with the
        residual is measured.
        
        Parameters
        ----------
        max_basis : int, default=10
            Maximum number of basis functions to select.
        tol : float, default=1e-6
            Tolerance on the residual correlation and the residual norm.
            
        Returns
        -------
        List[int]
            Indices of the selected basis functions.
        """
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # Evaluate all candidate basis functions
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # Basis values at all data points
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # Orthogonalize the candidate against the selected basis
                if self.selected_indices:
                    # Matrix of selected basis values
                    selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
                    Phi_selected = np.zeros((self.data.n_points, len(selected_bases)))
                    
                    for k, basis_k in enumerate(selected_bases):
                        for j in range(self.data.n_points):
                            point = self.data.get_point(j)
                            Phi_selected[j, k] = basis_k.evaluate(point)
                    
                    # Orthogonalize via the QR decomposition of the selected basis
                    Q, _ = np.linalg.qr(Phi_selected)
                    basis_values_orth = basis_values - Q @ (Q.T @ basis_values)
                else:
                    basis_values_orth = basis_values
                
                # Correlation of the orthogonalized candidate with the residual
                correlation = np.abs(np.dot(self.residuals, basis_values_orth))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            if best_index == -1 or best_reduction < tol:
                break
            
            self.selected_indices.append(best_index)
            
            # Update the solution using all selected basis functions
            self._update_solution_omp()
            
            # Stop when the residual norm drops below the tolerance
            residual_norm = np.linalg.norm(self.residuals)
            if residual_norm < tol:
                break
        
        return self.selected_indices
    
    def _update_solution(self):
        """Update the least-squares solution on the selected basis functions."""
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # Matrix of selected basis values
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # Least-squares solve
        try:
            self.coefficients = np.linalg.lstsq(Phi, self.target, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback: pseudoinverse
            self.coefficients = np.linalg.pinv(Phi) @ self.target
        
        # Residual update
        self.residuals = self.target - Phi @ self.coefficients
    
    def _update_solution_omp(self):
        """Update the OMP solution via orthogonal projection onto the selected basis."""
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # Matrix of selected basis values
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # Orthogonal projection via the QR decomposition
        Q, R = np.linalg.qr(Phi)
        
        # Solve the triangular system R c = Q^T y
        y = Q.T @ self.target
        self.coefficients = np.linalg.solve(R, y)
        
        # Residual after projection onto span(Q)
        self.residuals = self.target - Q @ (Q.T @ self.target)
    
    def get_selected_basis_set(self) -> BasisSet:
        """Return a BasisSet containing only the selected basis functions."""
        selected_set = BasisSet()
        for idx in self.selected_indices:
            selected_set.add_basis(self.basis_set.bases[idx])
        return selected_set
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """Predict values for new data."""
        if self.coefficients is None:
            raise ValueError("Must perform basis selection first")
        
        # Selected-basis matrix evaluated at the new data
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi_new = np.zeros((new_data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(new_data.n_points):
                point = new_data.get_point(j)
                Phi_new[j, k] = basis.evaluate(point)
        
        return Phi_new @ self.coefficients
    
    def get_residual_norm(self) -> float:
        """Return the L2 norm of the current residuals."""
        if self.residuals is None:
            return 0.0
        return np.linalg.norm(self.residuals)
    
    def get_explained_variance(self) -> float:
        """Return the fraction of target variance explained by the selection."""
        if self.target is None:
            return 0.0
        
        total_variance = np.var(self.target)
        if total_variance == 0:
            return 1.0
        
        residual_variance = np.var(self.residuals) if self.residuals is not None else total_variance
        return 1.0 - residual_variance / total_variance
    
    def cross_validate_selection(self, n_folds: int = 5, max_basis: int = 10) -> Tuple[List[int], float]:
        """
        Perform k-fold cross-validated forward selection.
        
        On each fold, forward selection is run on the training split and
        evaluated on the held-out split; the final selection is then made on
        the full data.
        
        Parameters
        ----------
        n_folds : int, default=5
            Number of cross-validation folds.
        max_basis : int, default=10
            Maximum number of basis functions.
            
        Returns
        -------
        Tuple[List[int], float]
            Indices of the selected basis functions and the mean
            cross-validation score (mean squared error).
        """
        n_points = self.data.n_points
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        fold_size = n_points // n_folds
        cv_scores = []
        
        for fold in range(n_folds):
            # Split into train and test folds
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_points
            
            test_indices = indices[test_start:test_end]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Build the training fold
            train_data_dict = {}
            for dim in self.data.dims:
                train_data_dict[dim] = self.data.get_dim(dim)[train_indices]
            
            train_data = MultiDimData(train_data_dict)
            train_target = self.target[train_indices]
            
            # Build the test fold
            test_data_dict = {}
            for dim in self.data.dims:
                test_data_dict[dim] = self.data.get_dim(dim)[test_indices]
            
            test_data = MultiDimData(test_data_dict)
            test_target = self.target[test_indices]
            
            # Fit an adaptive solver on the training fold
            train_solver = AdaptiveSolver(self.inner_product)
            train_solver.load_data(train_data, train_target)
            
            # Register the same basis candidates
            for basis in self.basis_set.bases:
                train_solver.add_basis_candidate(basis)
            
            # Run forward selection
            train_solver.forward_selection(max_basis=max_basis)
            
            # Evaluate on the held-out fold
            predictions = train_solver.predict(test_data)
            test_error = np.mean((predictions - test_target) ** 2)
            cv_scores.append(test_error)
        
        # Final selection on the full data
        self.forward_selection(max_basis=max_basis)
        
        return self.selected_indices, np.mean(cv_scores)
