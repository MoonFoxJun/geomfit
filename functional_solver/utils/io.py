"""函数求解器的输入/输出工具。"""

import numpy as np
import json
import yaml
import pickle
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
from ..core.data_container import MultiDimData

class IOUtils:
    """用于数据处理的输入/输出工具。"""
    
    @staticmethod
    def load_csv(filepath: str, dim_columns: List[str] = None, 
                value_column: str = None) -> MultiDimData:
        """
        从 CSV 文件加载数据。
        
        参数
        ----------
        filepath : str
            CSV 文件路径
        dim_columns : List[str], 可选
            维度对应的列名
        value_column : str, 可选
            值对应的列名
            
        返回
        -------
        MultiDimData
            加载的数据容器
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("load_csv 需要 pandas，请先安装：pip install pandas")
        df = pd.read_csv(filepath)
        
        if dim_columns is None:
            # 假设除最后一列外都是维度列
            dim_columns = df.columns[:-1].tolist()
        
        if value_column is None:
            # 假设最后一列是值
            value_column = df.columns[-1]
        
        # 提取维度数据
        dim_data = {}
        for i, col in enumerate(dim_columns):
            dim_data[i] = df[col].values
        
        # 创建 MultiDimData
        data = MultiDimData(dim_data)
        
        # 如有需要则保存值
        if value_column in df.columns:
            data.values = df[value_column].values
        
        return data
    
    @staticmethod
    def save_csv(data: MultiDimData, filepath: str, value_column: str = "values"):
        """
        将数据保存为 CSV 文件。
        
        参数
        ----------
        data : MultiDimData
            要保存的数据容器
        filepath : str
            CSV 文件的保存路径
        value_column : str, 默认="values"
            值对应的列名
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("save_csv 需要 pandas，请先安装：pip install pandas")
        # 创建 DataFrame
        records = []
        for i in range(data.n_points):
            point = data.get_point(i)
            record = {f"dim_{dim}": val for dim, val in point.items()}
            
            if hasattr(data, 'values') and data.values is not None:
                record[value_column] = data.values[i]
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def load_npy(filepath: str) -> np.ndarray:
        """
        从 .npy 文件加载 numpy 数组。
        
        参数
        ----------
        filepath : str
            .npy 文件路径
            
        返回
        -------
        np.ndarray
            加载的数组
        """
        return np.load(filepath)
    
    @staticmethod
    def save_npy(array: np.ndarray, filepath: str):
        """
        将 numpy 数组保存为 .npy 文件。
        
        参数
        ----------
        array : np.ndarray
            要保存的数组
        filepath : str
            .npy 文件的保存路径
        """
        np.save(filepath, array)
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """
        从 .npz 文件加载数据。
        
        参数
        ----------
        filepath : str
            .npz 文件路径
            
        返回
        -------
        Dict[str, np.ndarray]
            数组字典
        """
        return dict(np.load(filepath))
    
    @staticmethod
    def save_npz(data: Dict[str, np.ndarray], filepath: str):
        """
        将数据保存为 .npz 文件。
        
        参数
        ----------
        data : Dict[str, np.ndarray]
            要保存的数组字典
        filepath : str
            .npz 文件的保存路径
        """
        np.savez(filepath, **data)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        从 JSON 文件加载数据。
        
        参数
        ----------
        filepath : str
            JSON 文件路径
            
        返回
        -------
        Dict[str, Any]
            加载的数据
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
        """
        将数据保存为 JSON 文件。
        
        参数
        ----------
        data : Dict[str, Any]
            要保存的数据
        filepath : str
            JSON 文件的保存路径
        indent : int, 默认=2
            美化输出的缩进量
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        从 YAML 文件加载数据。
        
        参数
        ----------
        filepath : str
            YAML 文件路径
            
        返回
        -------
        Dict[str, Any]
            加载的数据
        """
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: str):
        """
        将数据保存为 YAML 文件。
        
        参数
        ----------
        data : Dict[str, Any]
            要保存的数据
        filepath : str
            YAML 文件的保存路径
        """
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        从 pickle 文件加载数据。
        
        参数
        ----------
        filepath : str
            pickle 文件路径
            
        返回
        -------
        Any
            加载的对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """
        将对象保存为 pickle 文件。
        
        参数
        ----------
        obj : Any
            要保存的对象
        filepath : str
            pickle 文件的保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_basis_set(filepath: str) -> 'BasisSet':
        """
        从文件加载基集合。
        
        参数
        ----------
        filepath : str
            基集合文件路径
            
        返回
        -------
        BasisSet
            加载的基集合
        """
        from ..core.basis_container import BasisSet, BasisInfo
        
        data = IOUtils.load_pickle(filepath)
        basis_set = BasisSet()
        
        for basis_data in data:
            basis_info = BasisInfo(
                name=basis_data['name'],
                dim=basis_data['dim'],
                params=basis_data.get('params'),
                func=basis_data.get('func'),  # 注意：函数可能无法很好地序列化
                factors=basis_data.get('factors'),
            )
            basis_set.add_basis(basis_info)
        
        return basis_set
    
    @staticmethod
    def save_basis_set(basis_set: 'BasisSet', filepath: str):
        """
        将基集合保存到文件。
        
        参数
        ----------
        basis_set : BasisSet
            要保存的基集合
        filepath : str
            基集合文件的保存路径
        """
        data = []
        for basis in basis_set.bases:
            basis_data = {
                'name': basis.name,
                'dim': basis.dim,
                'params': basis.params,
                'func': basis.func,
                'factors': basis.factors,
            }
            data.append(basis_data)
        
        IOUtils.save_pickle(data, filepath)
    
    @staticmethod
    def load_solver_state(filepath: str) -> Dict[str, Any]:
        """
        从文件加载求解器状态。
        
        参数
        ----------
        filepath : str
            求解器状态文件路径
            
        返回
        -------
        Dict[str, Any]
            求解器状态
        """
        return IOUtils.load_pickle(filepath)
    
    @staticmethod
    def save_solver_state(solver: Any, filepath: str):
        """
        将求解器状态保存到文件。
        
        参数
        ----------
        solver : Any
            要保存的求解器对象
        filepath : str
            求解器状态文件的保存路径
        """
        # 提取相关状态
        state = {
            'coefficients': solver.coefficients if hasattr(solver, 'coefficients') else None,
            'basis_set': solver.basis_set if hasattr(solver, 'basis_set') else None,
            'kernel': solver.kernel if hasattr(solver, 'kernel') else None,
            'inner_product': solver.inner_product if hasattr(solver, 'inner_product') else None,
            'data': solver.data if hasattr(solver, 'data') else None,
            'target': solver.target if hasattr(solver, 'target') else None,
            'debug_info': solver.debug_info if hasattr(solver, 'debug_info') else None,
        }
        
        IOUtils.save_pickle(state, filepath)
    
    @staticmethod
    def generate_sample_data(n_points: int = 100, n_dims: int = 1, 
                            noise_std: float = 0.1) -> Tuple[MultiDimData, np.ndarray]:
        """
        生成用于测试的示例数据。
        
        参数
        ----------
        n_points : int, 默认=100
            数据点数量
        n_dims : int, 默认=1
            维度数量
        noise_std : float, 默认=0.1
            噪声的标准差
            
        返回
        -------
        Tuple[MultiDimData, np.ndarray]
            数据容器和目标值
        """
        # 生成维度数据
        dim_data = {}
        for i in range(n_dims):
            dim_data[i] = np.linspace(0, 1, n_points)
        
        data = MultiDimData(dim_data)
        
        # 生成目标值（带噪声的正弦函数）
        if n_dims == 1:
            x = dim_data[0]
            y_true = np.sin(2 * np.pi * x)
        else:
            # 多维：正弦函数的乘积
            y_true = np.ones(n_points)
            for i in range(n_dims):
                y_true *= np.sin(2 * np.pi * dim_data[i])
        
        # 添加噪声
        y = y_true + noise_std * np.random.randn(n_points)
        
        return data, y
    
    @staticmethod
    def export_to_matlab(data: MultiDimData, filepath: str):
        """
        将数据导出为 MATLAB .mat 文件。
        
        参数
        ----------
        data : MultiDimData
            要导出的数据容器
        filepath : str
            .mat 文件的保存路径
        """
        try:
            from scipy.io import savemat
            
            mat_data = {}
            for dim in data.dims:
                mat_data[f'dim_{dim}'] = data.get_dim(dim)
            
            if hasattr(data, 'values') and data.values is not None:
                mat_data['values'] = data.values
            
            savemat(filepath, mat_data)
        except ImportError:
            raise ImportError("scipy is required for MATLAB export")
    
    @staticmethod
    def import_from_matlab(filepath: str) -> MultiDimData:
        """
        从 MATLAB .mat 文件导入数据。
        
        参数
        ----------
        filepath : str
            .mat 文件路径
            
        返回
        -------
        MultiDimData
            导入的数据容器
        """
        try:
            from scipy.io import loadmat
            
            mat_data = loadmat(filepath)
            
            # 查找维度数组
            dim_data = {}
            dim_idx = 0
            
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1):
                        # 展平为一维
                        value = value.flatten()
                        dim_data[dim_idx] = value
                        dim_idx += 1
            
            return MultiDimData(dim_data)
        except ImportError:
            raise ImportError("scipy is required for MATLAB import")
