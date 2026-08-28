"""用于函数逼近的自适应基选择。"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..core.data_container import MultiDimData
from ..core.basis_container import BasisSet, BasisInfo
from ..inner_product.base import InnerProduct

class AdaptiveSolver:
    """自适应基选择求解器。"""
    
    def __init__(self, inner_product: InnerProduct):
        """
        初始化自适应求解器。
        
        参数
        ----------
        inner_product : InnerProduct
            内积定义
        """
        self.inner_product = inner_product
        self.data = None
        self.target = None
        self.basis_set = BasisSet()
        self.coefficients = None
        self.selected_indices = []
        self.residuals = None
    
    def load_data(self, data: MultiDimData, target: np.ndarray):
        """加载用于求解的数据。"""
        self.data = data
        self.target = target
        assert data.n_points == len(target), "Number of data points must match target values"
        self.residuals = target.copy()
    
    def add_basis_candidate(self, basis_info: BasisInfo):
        """添加一个基函数候选。"""
        self.basis_set.add_basis(basis_info)
    
    def add_basis_candidates(self, basis_list: List[BasisInfo]):
        """添加多个基函数候选。"""
        for basis in basis_list:
            self.basis_set.add_basis(basis)
    
    def forward_selection(self, max_basis: int = 10, tol: float = 1e-6) -> List[int]:
        """
        执行前向基选择。
        
        参数
        ----------
        max_basis : int, 默认=10
            最多选择的基函数数量
        tol : float, 默认=1e-6
            残差范数的容差
            
        返回
        -------
        List[int]
            被选中的基函数索引
        """
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # 评估所有候选基函数
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # 获取基函数值
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # 计算与残差的相关性
                correlation = np.abs(np.dot(self.residuals, basis_values))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            if best_index == -1 or best_reduction < tol:
                break
            
            # 添加选中的基
            self.selected_indices.append(best_index)
            
            # 更新解和残差
            self._update_solution()
        
        return self.selected_indices
    
    def orthogonal_matching_pursuit(self, max_basis: int = 10, tol: float = 1e-6) -> List[int]:
        """
        执行正交匹配追踪（OMP）。
        
        参数
        ----------
        max_basis : int, 默认=10
            最多选择的基函数数量
        tol : float, 默认=1e-6
            残差范数的容差
            
        返回
        -------
        List[int]
            被选中的基函数索引
        """
        n_candidates = len(self.basis_set)
        self.selected_indices = []
        self.residuals = self.target.copy()
        
        for step in range(min(max_basis, n_candidates)):
            best_index = -1
            best_reduction = -1
            
            # 评估所有候选基函数
            for i in range(n_candidates):
                if i in self.selected_indices:
                    continue
                
                # 获取基函数值
                basis = self.basis_set.bases[i]
                basis_values = np.zeros(self.data.n_points)
                
                for j in range(self.data.n_points):
                    point = self.data.get_point(j)
                    basis_values[j] = basis.evaluate(point)
                
                # 与已选基进行正交化
                if self.selected_indices:
                    # 获取已选基矩阵
                    selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
                    Phi_selected = np.zeros((self.data.n_points, len(selected_bases)))
                    
                    for k, basis_k in enumerate(selected_bases):
                        for j in range(self.data.n_points):
                            point = self.data.get_point(j)
                            Phi_selected[j, k] = basis_k.evaluate(point)
                    
                    # 使用 QR 分解进行正交化
                    Q, _ = np.linalg.qr(Phi_selected)
                    basis_values_orth = basis_values - Q @ (Q.T @ basis_values)
                else:
                    basis_values_orth = basis_values
                
                # 计算与残差的相关性
                correlation = np.abs(np.dot(self.residuals, basis_values_orth))
                
                if correlation > best_reduction:
                    best_reduction = correlation
                    best_index = i
            
            if best_index == -1 or best_reduction < tol:
                break
            
            # 添加选中的基
            self.selected_indices.append(best_index)
            
            # 使用所有已选基更新解
            self._update_solution_omp()
            
            # 检查残差范数
            residual_norm = np.linalg.norm(self.residuals)
            if residual_norm < tol:
                break
        
        return self.selected_indices
    
    def _update_solution(self):
        """使用已选基更新解。"""
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # 获取已选基矩阵
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # 求解最小二乘问题
        try:
            self.coefficients = np.linalg.lstsq(Phi, self.target, rcond=None)[0]
        except np.linalg.LinAlgError:
            # 回退方案：使用伪逆
            self.coefficients = np.linalg.pinv(Phi) @ self.target
        
        # 更新残差
        self.residuals = self.target - Phi @ self.coefficients
    
    def _update_solution_omp(self):
        """使用正交投影更新 OMP 的解。"""
        if not self.selected_indices:
            self.coefficients = np.array([])
            return
        
        # 获取已选基矩阵
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi = np.zeros((self.data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(self.data.n_points):
                point = self.data.get_point(j)
                Phi[j, k] = basis.evaluate(point)
        
        # 使用 QR 分解进行正交投影
        Q, R = np.linalg.qr(Phi)
        
        # 求解三角方程组
        y = Q.T @ self.target
        self.coefficients = np.linalg.solve(R, y)
        
        # 更新残差
        self.residuals = self.target - Q @ (Q.T @ self.target)
    
    def get_selected_basis_set(self) -> BasisSet:
        """获取只包含已选基函数的基集合。"""
        selected_set = BasisSet()
        for idx in self.selected_indices:
            selected_set.add_basis(self.basis_set.bases[idx])
        return selected_set
    
    def predict(self, new_data: MultiDimData) -> np.ndarray:
        """预测新数据的值。"""
        if self.coefficients is None:
            raise ValueError("Must perform basis selection first")
        
        # 获取新数据对应的已选基矩阵
        selected_bases = [self.basis_set.bases[idx] for idx in self.selected_indices]
        Phi_new = np.zeros((new_data.n_points, len(selected_bases)))
        
        for k, basis in enumerate(selected_bases):
            for j in range(new_data.n_points):
                point = new_data.get_point(j)
                Phi_new[j, k] = basis.evaluate(point)
        
        return Phi_new @ self.coefficients
    
    def get_residual_norm(self) -> float:
        """获取残差的范数。"""
        if self.residuals is None:
            return 0.0
        return np.linalg.norm(self.residuals)
    
    def get_explained_variance(self) -> float:
        """获取解释方差的占比。"""
        if self.target is None:
            return 0.0
        
        total_variance = np.var(self.target)
        if total_variance == 0:
            return 1.0
        
        residual_variance = np.var(self.residuals) if self.residuals is not None else total_variance
        return 1.0 - residual_variance / total_variance
    
    def cross_validate_selection(self, n_folds: int = 5, max_basis: int = 10) -> Tuple[List[int], float]:
        """
        执行交叉验证的基选择。
        
        参数
        ----------
        n_folds : int, 默认=5
            交叉验证的折数
        max_basis : int, 默认=10
            最大基函数数量
            
        返回
        -------
        Tuple[List[int], float]
            选中的索引和交叉验证得分
        """
        n_points = self.data.n_points
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        fold_size = n_points // n_folds
        cv_scores = []
        
        for fold in range(n_folds):
            # 划分数据
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_points
            
            test_indices = indices[test_start:test_end]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # 创建训练数据
            train_data_dict = {}
            for dim in self.data.dims:
                train_data_dict[dim] = self.data.get_dim(dim)[train_indices]
            
            train_data = MultiDimData(train_data_dict)
            train_target = self.target[train_indices]
            
            # 创建测试数据
            test_data_dict = {}
            for dim in self.data.dims:
                test_data_dict[dim] = self.data.get_dim(dim)[test_indices]
            
            test_data = MultiDimData(test_data_dict)
            test_target = self.target[test_indices]
            
            # 在训练数据上训练自适应求解器
            train_solver = AdaptiveSolver(self.inner_product)
            train_solver.load_data(train_data, train_target)
            
            # 添加相同的基候选
            for basis in self.basis_set.bases:
                train_solver.add_basis_candidate(basis)
            
            # 执行选择
            train_solver.forward_selection(max_basis=max_basis)
            
            # 在测试数据上评估
            predictions = train_solver.predict(test_data)
            test_error = np.mean((predictions - test_target) ** 2)
            cv_scores.append(test_error)
        
        # 在完整数据上执行最终选择
        self.forward_selection(max_basis=max_basis)
        
        return self.selected_indices, np.mean(cv_scores)
