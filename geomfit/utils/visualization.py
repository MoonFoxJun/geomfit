"""Visualization utilities for the functional solver."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Dict, Any
from ..core.data_container import MultiDimData

class Visualization:
    """Plotting utilities for fits, basis functions, and surfaces."""
    
    @staticmethod
    def plot_1d_fit(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                   title: str = "1D Function Fit", xlabel: str = "x", 
                   ylabel: str = "y", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot a 1-D function fit.
        
        Parameters
        ----------
        x : np.ndarray
            x values.
        y_true : np.ndarray
            True y values.
        y_pred : np.ndarray
            Predicted y values.
        title : str, default="1D Function Fit"
            Plot title.
        xlabel : str, default="x"
            x-axis label.
        ylabel : str, default="y"
            y-axis label.
        figsize : Tuple[int, int], default=(10, 6)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter the observed data
        ax.scatter(x, y_true, alpha=0.5, label='Data', color='blue')
        
        # Sort x so the predicted curve is drawn as a smooth line
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Plot the fitted curve
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
        Plot the residuals.
        
        Parameters
        ----------
        x : np.ndarray
            x values.
        residuals : np.ndarray
            Residual values.
        title : str, default="Residuals"
            Plot title.
        xlabel : str, default="x"
            x-axis label.
        ylabel : str, default="Residual"
            y-axis label.
        figsize : Tuple[int, int], default=(10, 6)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
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
        Plot the basis functions.
        
        Parameters
        ----------
        basis_set
            Basis-set object.
        x_range : Tuple[float, float], default=(0, 1)
            x range of the plot.
        n_points : int, default=100
            Number of plot points.
        figsize : Tuple[int, int], default=(12, 8)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
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
            
            # Evaluate each basis: tensor-product bases are sliced along the
            # primary dimension, with all other dimensions set to zero
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
        
        # Hide unused subplots
        for i in range(n_basis, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_2d_surface(data: MultiDimData, values: np.ndarray, 
                       title: str = "2D Surface", figsize: Tuple[int, int] = (10, 8)) -> Figure:
        """
        Plot a 2-D surface.
        
        Parameters
        ----------
        data : MultiDimData
            Data container with exactly two dimensions.
        values : np.ndarray
            Values to plot.
        title : str, default="2D Surface"
            Plot title.
        figsize : Tuple[int, int], default=(10, 8)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        if data.n_dims != 2:
            raise ValueError("Data must have exactly 2 dimensions for 2D surface plot")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        dim0 = data.get_dim(0)
        dim1 = data.get_dim(1)
        
        X, Y = np.meshgrid(np.unique(dim0), np.unique(dim1))
        Z = values.reshape(X.shape)
        
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
        Plot a filled contour of a 2-D function.
        
        Parameters
        ----------
        data : MultiDimData
            Data container with exactly two dimensions.
        values : np.ndarray
            Values to plot.
        title : str, default="Contour Plot"
            Plot title.
        figsize : Tuple[int, int], default=(10, 8)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        if data.n_dims != 2:
            raise ValueError("Data must have exactly 2 dimensions for contour plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dim0 = data.get_dim(0)
        dim1 = data.get_dim(1)
        
        X, Y = np.meshgrid(np.unique(dim0), np.unique(dim1))
        Z = values.reshape(X.shape)
        
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
        Plot the basis coefficients as a bar chart.
        
        Parameters
        ----------
        coefficients : np.ndarray
            Coefficient values.
        basis_names : List[str], optional
            Names of the basis functions.
        title : str, default="Basis Coefficients"
            Plot title.
        figsize : Tuple[int, int], default=(10, 6)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n_coeff = len(coefficients)
        indices = np.arange(n_coeff)
        
        if basis_names is None:
            basis_names = [f'Basis {i}' for i in range(n_coeff)]
        
        bars = ax.bar(indices, coefficients, alpha=0.7, color='steelblue')
        
        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:  # Label only values above the display threshold
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
        Plot the loss history of an iterative solver.
        
        Parameters
        ----------
        loss_history : List[float]
            Loss value at each iteration.
        title : str, default="Learning Curve"
            Plot title.
        figsize : Tuple[int, int], default=(10, 6)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        iterations = np.arange(len(loss_history))
        ax.plot(iterations, loss_history, 'b-', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale to visualize the loss decay
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_singular_values(singular_values: np.ndarray, 
                            threshold: Optional[float] = None,
                            title: str = "Singular Values", figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot the singular values on a log scale.
        
        Parameters
        ----------
        singular_values : np.ndarray
            Singular values.
        threshold : float, optional
            Threshold line to mark on the plot.
        title : str, default="Singular Values"
            Plot title.
        figsize : Tuple[int, int], default=(10, 6)
            Figure size.
            
        Returns
        -------
        Figure
            Matplotlib figure object.
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
        Combine multiple figures into a dashboard.
        
        Parameters
        ----------
        figures : List[Figure]
            Figures to combine.
        n_cols : int, default=2
            Number of dashboard columns.
        figsize : Tuple[int, int], default=(15, 10)
            Figure size.
            
        Returns
        -------
        Figure
            Combined dashboard figure.
        """
        n_figures = len(figures)
        n_rows = (n_figures + n_cols - 1) // n_cols
        
        dashboard_fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_figures > 1 else [axes]
        
        # Hide all axes initially
        for ax in axes:
            ax.axis('off')
        
        # Copy each figure into the dashboard
        for i, fig in enumerate(figures):
            if i >= len(axes):
                break
                
            dashboard_ax = axes[i]
            dashboard_ax.axis('on')
            
            # Placeholder: a full implementation would extract and redraw each
            # figure's artists; here each panel shows a label only
            dashboard_ax.text(0.5, 0.5, f'Figure {i+1}', 
                            ha='center', va='center', fontsize=12)
            dashboard_ax.set_title(fig._suptitle.get_text() if fig._suptitle else f'Plot {i+1}')
        
        # Hide unused axes
        for i in range(n_figures, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return dashboard_fig
    
    @staticmethod
    def save_figure(fig: Figure, filename: str, dpi: int = 300, 
                   bbox_inches: str = 'tight'):
        """
        Save a figure to a file.
        
        Parameters
        ----------
        fig : Figure
            Matplotlib figure object.
        filename : str
            Output file name.
        dpi : int, default=300
            Resolution in dots per inch.
        bbox_inches : str, default='tight'
            Bounding box in inches.
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
