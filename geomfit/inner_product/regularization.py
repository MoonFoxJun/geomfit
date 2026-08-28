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
        # Tikhonov / 岭正则化：G + αI 把每个特征值都垫高 α，最小特征值从 λ_min 抬到 λ_min + α，
        # 从而压低条件数 κ = λ_max/λ_min、稳定求逆；α 越大正则化越强（解的范数越小、偏差越大）
        return G + alpha * np.eye(n)

    @staticmethod
    def ridge(G: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
        """Alias of Tikhonov regularization (ridge regression)."""
        # 岭回归别名：与 Tikhonov 完全等价（都是对 Gram 矩阵加 αI）
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
        # 对 Gram 矩阵做紧凑 SVD 分解：G = U diag(s) Vᵀ
        # （full_matrices=False 只保留非零奇异值对应的维度，矩阵更小）
        U, s, Vh = np.linalg.svd(G, full_matrices=False)

        # 截断：丢弃所有小于阈值的奇异值（对应数值上近似零的方向，即病态/秩亏方向）。
        # 截断后的三件套构成原矩阵的"低秩近似"，求逆时不会放大这些噪声方向
        mask = s > threshold
        U_trunc = U[:, mask]    # 只保留显著奇异值对应的左奇异向量
        s_trunc = s[mask]       # 保留的奇异值
        Vh_trunc = Vh[mask, :]  # 只保留显著奇异值对应的右奇异向量（Vh 的行）

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
            G_reg = Regularization.tikhonov(G, alpha)  # 先正则化：(G + αI)
            return np.linalg.solve(G_reg, b)           # 再求解正则化后的线性系统

        elif method == "svd":
            threshold = kwargs.get("threshold", 1e-10)
            U, s, Vh = Regularization.truncated_svd(G, threshold)

            # 用截断后的 SVD 求解最小范数解：x = V Σ⁻¹ Uᵀ b。
            # 由于小奇异值已被剔除，Σ⁻¹ = 1/s 不会出现天文数字级的放大
            s_inv = 1.0 / s
            x = Vh.T @ (s_inv * (U.T @ b))
            return x

        elif method == "lstsq":
            rcond = kwargs.get("rcond", None)
            # np.linalg.lstsq 内部基于 SVD 求最小二乘解；rcond 控制小奇异值的截断阈值
            # （rcond=None 时采用机器精度相关的默认值）
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
        # 条件数 κ = σ_max/σ_min：衡量矩阵对扰动的敏感程度，κ 越大矩阵越病态；
        # 对 Gram 矩阵（半正定）而言，条件数大意味着基函数之间接近线性相关
        cond = np.linalg.cond(G)
        s = np.linalg.svd(G, compute_uv=False)  # compute_uv=False：只求奇异值、不分解 U/V，更省计算
        min_sv = np.min(s)                      # 最小奇异值 σ_min
        max_sv = np.max(s)                      # 最大奇异值 σ_max
        is_singular = min_sv < threshold        # 最小奇异值低于阈值即判为"数值奇异"
        rank = np.sum(s > threshold)            # 数值秩：大于阈值的奇异值个数（矩阵的有效维数）

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
        # LASSO 正则化：在最小二乘目标上附加 L1 惩罚 α‖x‖₁，促使解稀疏（许多分量恰好等于 0）
        n = G.shape[0]
        x = np.zeros(n)  # 解从零向量出发

        # 预取 G 的对角元（每个坐标下降步的分母），避免在循环内反复取对角
        diag_G = np.diag(G)

        for iteration in range(max_iter):
            x_old = x.copy()

            # 坐标下降：每轮依次更新每个分量，更新时固定其它所有分量不变
            for j in range(n):
                # r_j = b - Gx + G[:,j]·x_j：先把当前残差 b - Gx 中"第 j 个分量贡献的部分"
                # 加回来，得到"除去第 j 个分量后"的残差（其余分量视为固定常数）
                r_j = b - G @ x + G[:, j] * x[j]

                numerator = np.dot(G[:, j], r_j)  # 分子 = G 第 j 列与 r_j 的内积（无正则化时最优解即 numerator/denominator）
                denominator = diag_G[j]           # 分母 = G_jj（G 半正定故非负）

                # 软阈值（soft thresholding）：L1 惩罚产生稀疏性的关键步骤。
                # |numerator| ≤ α 时分量直接置 0；否则向 0 方向收缩 α 后再除以分母
                if numerator > alpha:
                    x[j] = (numerator - alpha) / denominator
                elif numerator < -alpha:
                    x[j] = (numerator + alpha) / denominator
                else:
                    x[j] = 0.0

            # 收敛判据：连续两轮解的欧氏距离小于容差 tol 则提前终止迭代
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
        # 弹性网正则化 = L1（Lasso）+ L2（岭）混合惩罚：
        # l1_ratio = 1 退化为纯 Lasso，l1_ratio = 0 退化为纯岭回归，中间值兼顾稀疏性与稳定性
        n = G.shape[0]
        x = np.zeros(n)  # 解从零向量出发

        # 把总正则化强度 α 按 l1_ratio 拆成 L1 与 L2 两部分
        alpha_l1 = alpha * l1_ratio
        alpha_l2 = alpha * (1 - l1_ratio)

        # L2 部分可以提前并入 Gram 矩阵（G + α_l2·I，与 Tikhonov 正则化相同），
        # 这样坐标下降的每一步只需处理 L1 的软阈值，无需单独处理 L2 的梯度
        G_mod = G + alpha_l2 * np.eye(n)
        diag_G_mod = np.diag(G_mod)  # 预取修正后 Gram 矩阵的对角元（坐标下降的分母）

        for iteration in range(max_iter):
            x_old = x.copy()

            # 坐标下降：逐分量更新，其余分量固定
            for j in range(n):
                # 与 Lasso 同理：先移除第 j 个分量的贡献，得到其余分量固定时的残差
                r_j = b - G_mod @ x + G_mod[:, j] * x[j]

                numerator = np.dot(G_mod[:, j], r_j)  # 分子 = (G_mod 第 j 列) 与残差的内积
                denominator = diag_G_mod[j]           # 分母 = (G_mod)_jj

                # 对 L1 部分做软阈值收缩（阈值改为 alpha_l1），L2 部分已体现在 G_mod 中
                if numerator > alpha_l1:
                    x[j] = (numerator - alpha_l1) / denominator
                elif numerator < -alpha_l1:
                    x[j] = (numerator + alpha_l1) / denominator
                else:
                    x[j] = 0.0

            # 收敛检查：连续两轮解的欧氏距离小于容差即停止
            if np.linalg.norm(x - x_old) < tol:
                break

        return x
