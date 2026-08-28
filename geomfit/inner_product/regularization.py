"""Regularization methods for ill-posed problems."""

import numpy as np
from typing import Dict, Any, Optional, Tuple

class Regularization:
    """Regularization methods for solving ill-posed problems."""

    @staticmethod
    def tikhonov(G: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """
        Apply Tikhonov regularization.

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        alpha : float, default=1e-6
            Regularization parameter.

        Returns
        -------
        np.ndarray
            Regularized matrix G + αI.
        """
        n = G.shape[0]
        return G + alpha * np.eye(n)

    @staticmethod
    def ridge(G: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """Alias of Tikhonov regularization (ridge regression)."""
        return Regularization.tikhonov(G, alpha)

    @staticmethod
    def truncated_svd(G: np.ndarray, threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a truncated-SVD regularization.

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        threshold : float, default=1e-10
            Singular value threshold.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            U, S, Vh with small singular values removed.
        """
        U, s, Vh = np.linalg.svd(G, full_matrices=False)

        # Truncate small singular values
        mask = s > threshold
        U_trunc = U[:, mask]
        s_trunc = s[mask]
        Vh_trunc = Vh[mask, :]

        return U_trunc, s_trunc, Vh_trunc

    @staticmethod
    def solve_regularized(G: np.ndarray, b: np.ndarray, method: str = "tikhonov",
                         **kwargs) -> np.ndarray:
        """
        Solve the regularized linear system Gx = b.

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        b : np.ndarray
            Right-hand-side vector.
        method : str, default="tikhonov"
            Regularization method: "tikhonov", "svd", or "lstsq".
        **kwargs
            Additional arguments for the regularization method.

        Returns
        -------
        np.ndarray
            Solution vector x.
        """
        if method == "tikhonov":
            alpha = kwargs.get("alpha", 1e-6)
            G_reg = Regularization.tikhonov(G, alpha)
            return np.linalg.solve(G_reg, b)

        elif method == "svd":
            threshold = kwargs.get("threshold", 1e-10)
            U, s, Vh = Regularization.truncated_svd(G, threshold)

            # Solve with the truncated SVD: x = V Σ⁻¹ Uᵀ b
            s_inv = 1.0 / s
            x = Vh.T @ (s_inv * (U.T @ b))
            return x

        elif method == "lstsq":
            rcond = kwargs.get("rcond", None)
            x, residuals, rank, s = np.linalg.lstsq(G, b, rcond=rcond)
            return x

        else:
            raise ValueError(f"Unknown regularization method: {method}")

    @staticmethod
    def check_singularity(G: np.ndarray, threshold: float = 1e-10) -> Dict[str, Any]:
        """
        Check whether the Gram matrix is singular or ill-conditioned.

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        threshold : float, default=1e-10
            Singular value threshold.

        Returns
        -------
        Dict[str, Any]
            Singularity information.
        """
        cond = np.linalg.cond(G)
        s = np.linalg.svd(G, compute_uv=False)
        min_sv = np.min(s)
        max_sv = np.max(s)
        is_singular = min_sv < threshold
        rank = np.sum(s > threshold)

        return {
            "is_singular": is_singular,
            "condition_number": cond,
            "min_singular_value": min_sv,
            "max_singular_value": max_sv,
            "rank": rank,
            "size": G.shape[0],
            "threshold": threshold
        }

    @staticmethod
    def lasso_regularization(G: np.ndarray, b: np.ndarray, alpha: float = 1e-3,
                            max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Apply LASSO regularization (L1 penalty) via coordinate descent.

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        b : np.ndarray
            Right-hand-side vector.
        alpha : float, default=1e-3
            Regularization parameter.
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance.

        Returns
        -------
        np.ndarray
            Solution vector x.
        """
        n = G.shape[0]
        x = np.zeros(n)

        # Precompute the diagonal of G
        diag_G = np.diag(G)

        for iteration in range(max_iter):
            x_old = x.copy()

            # Coordinate descent
            for j in range(n):
                # Residual with the j-th component removed
                r_j = b - G @ x + G[:, j] * x[j]

                numerator = np.dot(G[:, j], r_j)
                denominator = diag_G[j]

                # Soft thresholding
                if numerator > alpha:
                    x[j] = (numerator - alpha) / denominator
                elif numerator < -alpha:
                    x[j] = (numerator + alpha) / denominator
                else:
                    x[j] = 0.0

            # Convergence check
            if np.linalg.norm(x - x_old) < tol:
                break

        return x

    @staticmethod
    def elastic_net(G: np.ndarray, b: np.ndarray, alpha: float = 1e-3,
                   l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Apply elastic net regularization (L1 + L2 penalty).

        Parameters
        ----------
        G : np.ndarray
            Gram matrix.
        b : np.ndarray
            Right-hand-side vector.
        alpha : float, default=1e-3
            Total regularization parameter.
        l1_ratio : float, default=0.5
            Proportion of the L1 penalty (0 is ridge, 1 is lasso).
        max_iter : int, default=1000
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance.

        Returns
        -------
        np.ndarray
            Solution vector x.
        """
        n = G.shape[0]
        x = np.zeros(n)

        # Split alpha into L1 and L2 parts
        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)

        # Gram matrix pre-modified by the L2 regularization
        G_mod = G + alpha_l2 * np.eye(n)
        diag_G_mod = np.diag(G_mod)

        for iteration in range(max_iter):
            x_old = x.copy()

            # Coordinate descent
            for j in range(n):
                # Residual with the j-th component removed
                r_j = b - G_mod @ x + G_mod[:, j] * x[j]

                numerator = np.dot(G_mod[:, j], r_j)
                denominator = diag_G_mod[j]

                # Soft thresholding for the L1 penalty
                if numerator > alpha_l1:
                    x[j] = (numerator - alpha_l1) / denominator
                elif numerator < -alpha_l1:
                    x[j] = (numerator + alpha_l1) / denominator
                else:
                    x[j] = 0.0

            # Convergence check
            if np.linalg.norm(x - x_old) < tol:
                break

        return x
