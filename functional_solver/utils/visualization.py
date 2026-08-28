"""函数求解器的可视化工具。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Dict, Any
from ..core.data_container import MultiDimData

class Visualization:
    """函数求解器的可视化工具类。"""
    
    @staticmethod
    def plot_1d_fit(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                   title: str = "1D Function Fit", xlabel: str = "x", 
                   ylabel: str = "y", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        绘制一维函数拟合图。
        
        参数
        ----------
        x : np.ndarray
            x 值
        y_true : np.ndarray
            真实的 y 值
        y_pred : np.ndarray
            预测的 y 值
        title : str, 默认="1D Function Fit"
            图标题
        xlabel : str, 默认="x"
            x 轴标签
        ylabel : str, 默认="y"
            y 轴标签
        figsize : Tuple[int, int], 默认=(10, 6)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制真实数据
        ax.scatter(x, y_true, alpha=0.5, label='Data', color='blue')
        
        # 排序以绘制平滑曲线
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # 绘制预测曲线
        ax.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Fit')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_residuals(x: np.ndarray, residuals: np.ndarray, 
                      title: str = "Residuals", xlabel: str = "x", 
                      ylabel: str = "Residual", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        绘制残差图。
        
        参数
        ----------
        x : np.ndarray
            x 值
        residuals : np.ndarray
            残差值
        title : str, 默认="Residuals"
            图标题
        xlabel : str, 默认="x"
            x 轴标签
        ylabel : str, 默认="Residual"
            y 轴标签
        figsize : Tuple[int, int], 默认=(10, 6)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(x, residuals, alpha=0.6, color='purple')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_basis_functions(basis_set, x_range: Tuple[float, float] = (0, 1), 
                           n_points: int = 100, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        绘制基函数。
        
        参数
        ----------
        basis_set
            基集合对象
        x_range : Tuple[float, float], 默认=(0, 1)
            绘图的 x 范围
        n_points : int, 默认=100
            绘图的点数
        figsize : Tuple[int, int], 默认=(12, 8)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        n_basis = len(basis_set)
        n_cols = min(3, n_basis)
        n_rows = (n_basis + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_basis > 1 else [axes]
        
        for i, basis in enumerate(basis_set.bases):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 评估基函数（张量积基沿主维度切片，其余维度取 0）
            y = np.zeros_like(x)
            for j, x_val in enumerate(x):
                point = {d: 0.0 for d in basis.dims}
                point[basis.dim] = x_val
                y[j] = basis.evaluate(point)
            
            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_title(f"{basis.name}")
            ax.set_xlabel("x")
            ax.set_ylabel("φ(x)")
            ax.grid(True, alpha=0.3)
        
        # 隐藏未使用的子图
        for i in range(n_basis, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_2d_surface(data: MultiDimData, values: np.ndarray, 
                       title: str = "2D Surface", figsize: Tuple[int, int] = (10, 8)) -> Figure:
        """
        绘制二维曲面图。
        
        参数
        ----------
        data : MultiDimData
            包含两个维度的数据容器
        values : np.ndarray
            要绘制的值
        title : str, 默认="2D Surface"
            图标题
        figsize : Tuple[int, int], 默认=(10, 8)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        if data.n_dims != 2:
            raise ValueError("Data must have exactly 2 dimensions for 2D surface plot")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取维度数据
        dim0 = data.get_dim(0)
        dim1 = data.get_dim(1)
        
        # 创建网格
        X, Y = np.meshgrid(np.unique(dim0), np.unique(dim1))
        
        # 重塑值以匹配网格
        Z = values.reshape(X.shape)
        
        # 绘制曲面
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_zlabel('Value')
        ax.set_title(title)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig
    
    @staticmethod
    def plot_contour(data: MultiDimData, values: np.ndarray, 
                    title: str = "Contour Plot", figsize: Tuple[int, int] = (10, 8)) -> Figure:
        """
        绘制等高线图。
        
        参数
        ----------
        data : MultiDimData
            包含两个维度的数据容器
        values : np.ndarray
            要绘制的值
        title : str, 默认="Contour Plot"
            图标题
        figsize : Tuple[int, int], 默认=(10, 8)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        if data.n_dims != 2:
            raise ValueError("Data must have exactly 2 dimensions for contour plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 提取维度数据
        dim0 = data.get_dim(0)
        dim1 = data.get_dim(1)
        
        # 创建网格
        X, Y = np.meshgrid(np.unique(dim0), np.unique(dim1))
        
        # 重塑值以匹配网格
        Z = values.reshape(X.shape)
        
        # 绘制等高线
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3)
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(title)
        
        fig.colorbar(contour, ax=ax)
        
        return fig
    
    @staticmethod
    def plot_coefficients(coefficients: np.ndarray, basis_names: Optional[List[str]] = None,
                         title: str = "Basis Coefficients", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        绘制基系数柱状图。
        
        参数
        ----------
        coefficients : np.ndarray
            系数值
        basis_names : List[str], 可选
            基函数名称
        title : str, 默认="Basis Coefficients"
            图标题
        figsize : Tuple[int, int], 默认=(10, 6)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n_coeff = len(coefficients)
        indices = np.arange(n_coeff)
        
        if basis_names is None:
            basis_names = [f'Basis {i}' for i in range(n_coeff)]
        
        bars = ax.bar(indices, coefficients, alpha=0.7, color='steelblue')
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:  # 只为显著的数值添加标签
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8)
        
        ax.set_xlabel('Basis Function')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(title)
        ax.set_xticks(indices)
        ax.set_xticklabels(basis_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_learning_curve(loss_history: List[float], 
                           title: str = "Learning Curve", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        绘制学习曲线。
        
        参数
        ----------
        loss_history : List[float]
            各迭代步的损失值
        title : str, 默认="Learning Curve"
            图标题
        figsize : Tuple[int, int], 默认=(10, 6)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        iterations = np.arange(len(loss_history))
        ax.plot(iterations, loss_history, 'b-', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 使用对数刻度以便更好地可视化
        
        return fig
    
    @staticmethod
    def plot_singular_values(singular_values: np.ndarray, 
                            threshold: Optional[float] = None,
                            title: str = "Singular Values", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        绘制奇异值图。
        
        参数
        ----------
        singular_values : np.ndarray
            奇异值
        threshold : float, 可选
            奇异值的阈值
        title : str, 默认="Singular Values"
            图标题
        figsize : Tuple[int, int], 默认=(10, 6)
            图像大小
            
        返回
        -------
        Figure
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        indices = np.arange(len(singular_values)) + 1
        
        ax.semilogy(indices, singular_values, 'bo-', linewidth=2, markersize=6)
        
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                      label=f'Threshold: {threshold:.1e}')
            ax.legend()
        
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def create_dashboard(figures: List[Figure], n_cols: int = 2, 
                        figsize: Tuple[int, int] = (15, 10)) -> Figure:
        """
        由多个图形创建仪表盘。
        
        参数
        ----------
        figures : List[Figure]
            要合并的图形列表
        n_cols : int, 默认=2
            仪表盘的列数
        figsize : Tuple[int, int], 默认=(15, 10)
            图像大小
            
        返回
        -------
        Figure
            合并后的仪表盘图形
        """
        n_figures = len(figures)
        n_rows = (n_figures + n_cols - 1) // n_cols
        
        dashboard_fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_figures > 1 else [axes]
        
        # 先隐藏所有坐标轴
        for ax in axes:
            ax.axis('off')
        
        # 将每个图形复制到仪表盘
        for i, fig in enumerate(figures):
            if i >= len(axes):
                break
                
            # 获取图形的坐标轴
            fig_axes = fig.axes
            
            # 在仪表盘中创建新的坐标轴
            dashboard_ax = axes[i]
            dashboard_ax.axis('on')
            
            # 复制内容（简化处理——实际实现需要更复杂的操作）
            # 这只是占位实现——实际实现需要
            # 提取并重新绘制数据
            dashboard_ax.text(0.5, 0.5, f'Figure {i+1}', 
                            ha='center', va='center', fontsize=12)
            dashboard_ax.set_title(fig._suptitle.get_text() if fig._suptitle else f'Plot {i+1}')
        
        # 隐藏未使用的坐标轴
        for i in range(n_figures, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return dashboard_fig
    
    @staticmethod
    def save_figure(fig: Figure, filename: str, dpi: int = 300, 
                   bbox_inches: str = 'tight'):
        """
        将图形保存到文件。
        
        参数
        ----------
        fig : Figure
            Matplotlib 图形对象
        filename : str
            输出文件名
        dpi : int, 默认=300
            保存的分辨率（DPI）
        bbox_inches : str, 默认='tight'
            边界框尺寸（英寸）
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
