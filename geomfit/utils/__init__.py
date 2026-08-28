"""Utility functions for the functional solver."""

# 工具子包导出：数值计算（积分/微分/插值）、输入输出、日志、
# 可视化，以及数据预处理（PCA 坐标旋转）。
from .numerical import NumericalUtils
from .io import IOUtils
from .logger import Logger
from .visualization import Visualization
from .preprocess import pca_rotate, pca_transform, pca_rotate_back

# 显式声明 __all__，控制 from geomfit.utils import * 的导出范围
__all__ = [
    "NumericalUtils",
    "IOUtils",
    "Logger",
    "Visualization",
    "pca_rotate",
    "pca_transform",
    "pca_rotate_back",
]
