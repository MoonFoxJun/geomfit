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
        # 初始残差 = 目标值本身（此时还没有选任何基函数，
        # 相当于用零函数拟合，残差就是全部目标）。
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
        # 前向选择（贪婪法）主循环：每一步从候选基函数里挑出
        # "与当前残差相关性最大"的一个加入选中集合，然后重新最小二乘拟合。
        # 这是贪心策略：每次只做局部最优选择，不求全局最优，但通常很有效。
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        # 最多选 max_basis 个基（也不能超过候选总数）
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # 遍历所有候选基函数，找出本轮最该加入的那个
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # 把第 i 个基函数在所有数据点上求值，得到列向量 basis_values
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # 候选基与残差的相关性：|⟨residuals, basis_values⟩|。
                # 残差是"还没被当前模型解释掉的部分"，与残差最相关的基
                # 能最大程度地降低拟合误差，因此取绝对值后挑最大的。
                correlation = np.abs(np.dot(self.residuals, basis_values))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            # 没有可选基，或最好的相关性都低于 tol（残差基本解释完了），停止
            if best_index == -1 or best_reduction < tol:
                break
            
            self.selected_indices.append(best_index)
            
            # 选完后在"已选基函数"上重新做一次最小二乘拟合，
            # 更新系数与残差（残差 = 目标 − 当前模型预测）。
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
        # OMP（正交匹配追踪）：与前向选择的区别在于，每个候选基在计算
        # 与残差的相关性之前，先对"已选基函数张成的子空间"做正交化，
        # 只保留垂直于该子空间的分量。这样已选基之间的重复信息不会被
        # 重复计入，选基更干净、残差下降更单调。
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # 遍历所有候选基函数
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # 候选基在所有数据点上的取值
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # 正交化候选基：如果已经选了基函数，就把候选基投影到
                # 已选基张成的子空间上并从自身中减去（Gram-Schmidt 思想），
                # 得到与已选子空间正交的残余方向 basis_values_orth。
                if self.selected_indices:
                    # 组装已选基函数的设计矩阵 Phi_selected（列 = 已选基）
                    selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
                    Phi_selected = np.zeros((self.data.n_points, len(selected_bases)))
                    
                    for k, basis_k in enumerate(selected_bases):
                        for j in range(self.data.n_points):
                            point = self.data.get_point(j)
                            Phi_selected[j, k] = basis_k.evaluate(point)
                    
                    # 用 QR 分解实现正交化：Q 的列张成与 Phi_selected 相同的
                    # 子空间，Q(Qᵀv) 是 v 到该子空间的正交投影；
                    # v − Q(Qᵀv) 就是 v 中垂直于子空间的部分。
                    Q, _ = np.linalg.qr(Phi_selected)
                    basis_values_orth = basis_values - Q @ (Q.T @ basis_values)
                else:
                    # 尚未选择任何基，候选基本身就是"正交"的
                    basis_values_orth = basis_values
                
                # 用正交化后的候选基与残差做相关性度量
                correlation = np.abs(np.dot(self.residuals, basis_values_orth))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            if best_index == -1 or best_reduction < tol:
                break
            
            self.selected_indices.append(best_index)
            
            # 用全部已选基函数重新求解（正交投影更新）
            self._update_solution_omp()
            
            # 残差范数降到 tol 以下说明已选基已经能很好地解释目标，提前停止
            residual_norm = np.linalg.norm(self.residuals)
            if residual_norm < tol:
                break
        
        return self.selected_indices
    
    def _update_solution(self):
        """Update the least-squares solution on the selected basis functions."""
        # 尚未选中任何基函数时，系数为空数组
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # 组装已选基函数的设计矩阵 Phi（行 = 数据点，列 = 已选基）
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # 最小二乘求解：min_c ‖Φc − y‖²，用 lstsq 一步到位
        try:
            self.coefficients = np.linalg.lstsq(Phi, self.target, rcond=None)[0]
        except np.linalg.LinAlgError:
            # 兜底：用伪逆 pinv 求最小范数最小二乘解
            self.coefficients = np.linalg.pinv(Phi) @ self.target
        
        # 更新残差：残差 = 目标 − 当前模型预测（用于下一轮选基）
        self.residuals = self.target - Phi @ self.coefficients
    
    def _update_solution_omp(self):
        """Update the OMP solution via orthogonal projection onto the selected basis."""
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # 组装已选基函数的设计矩阵 Phi
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # 正交投影：对 Phi 做 QR 分解 Phi = QR。Q 的列构成
        # 已选基张成子空间的标准正交基，因此到该子空间的正交投影
        # 就是 P = QQᵀ，系数在 Q 坐标系下的坐标是 Qᵀy。
        Q, R = np.linalg.qr(Phi)
        
        # 解上三角方程组 R c = Qᵀ y（回代法，O(n²) 高效且稳定）
        y = Q.T @ self.target
        self.coefficients = np.linalg.solve(R, y)
        
        # 投影后的残差：y 减去它在 span(Q) 上的投影 QQᵀy，
        # 残差与已选子空间正交（这是 OMP 名称中"正交"的由来）。
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
        
        # 预测：只在"选中的基函数"上组装设计矩阵并加权求和
        # （未选中的基不参与预测，这是稀疏模型的核心优势——模型更简洁）。
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
        
        # 解释方差比 = 1 − Var(残差)/Var(目标)：衡量选中的基函数
        # 能解释目标值中多大比例的波动，1.0 表示完全拟合。
        total_variance = np.var(self.target)
        if total_variance == 0:
            # 目标本身没有波动（常数），任何拟合都"完全解释"，返回 1.0
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
        # k 折交叉验证：把数据随机打乱后切成 k 份，轮流拿 1 份做验证、
        # 其余 k−1 份做训练，在每一折上重新跑前向选择并评估误差，
        # 最后用平均误差衡量"选基策略"的泛化能力，避免过拟合到某一组数据。
        n_points = self.data.n_points
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        fold_size = n_points // n_folds
        cv_scores = []
        
        for fold in range(n_folds):
            # 按块切分：第 fold 折的测试区间 [test_start, test_end)，
            # 注意最后一折要一直取到末尾（因为整除可能有余数）。
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_points
            
            test_indices = indices[test_start:test_end]
            # 训练集 = 全集 − 测试集（按索引集合差集）
            train_indices = np.setdiff1d(indices, test_indices)
            
            # 从原始数据里按 train_indices 抽取训练折的数据（每个维度分别切片）
            train_data_dict = {}
            for dim in self.data.dims:
                train_data_dict[dim] = self.data.get_dim(dim)[train_indices]
            
            train_data = MultiDimData(train_data_dict)
            train_target = self.target[train_indices]
            
            # 同样构造测试折的数据
            test_data_dict = {}
            for dim in self.data.dims:
                test_data_dict[dim] = self.data.get_dim(dim)[test_indices]
            
            test_data = MultiDimData(test_data_dict)
            test_target = self.target[test_indices]
            
            # 在训练折上新建一个独立的 AdaptiveSolver 实例，
            # 保证每一折的选基过程互不干扰（不污染 self 的状态）
            train_solver = AdaptiveSolver(self.inner_product)
            train_solver.load_data(train_data, train_target)
            
            # 注册与完整候选集相同的基函数候选
            for basis in self.basis_set.bases:
                train_solver.add_basis_candidate(basis)
            
            # 在训练折上跑前向选择
            train_solver.forward_selection(max_basis=max_basis)
            
            # 在留出的测试折上评估：均方误差 MSE = mean((pred − y)²)
            predictions = train_solver.predict(test_data)
            test_error = np.mean((predictions - test_target) ** 2)
            cv_scores.append(test_error)
        
        # 交叉验证只用于评估；最终模型仍在全部数据上做一次前向选择
        self.forward_selection(max_basis=max_basis)
        
        return self.selected_indices, np.mean(cv_scores)
