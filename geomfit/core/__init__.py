"""Core data structures of the functional solver."""

# 核心模块对外只暴露三个类：MultiDimData 存放多维数据点；
# BasisInfo 描述单个基函数；BasisSet 管理基集合并组装设计矩阵 Φ
from .data_container import MultiDimData
from .basis_container import BasisSet, BasisInfo

__all__ = ["MultiDimData", "BasisSet", "BasisInfo"]
