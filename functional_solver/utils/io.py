"""Input/output utilities for the functional solver."""

import numpy as np
import json
import yaml
import pickle
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
from ..core.data_container import MultiDimData

class IOUtils:
    """Input/output utilities for data, basis sets, and solver states."""
    
    @staticmethod
    def load_csv(filepath: str, dim_columns: List[str] = None, 
                value_column: str = None) -> MultiDimData:
        """
        Load data from a CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        dim_columns : List[str], optional
            Column names holding the dimension coordinates.
        value_column : str, optional
            Column name holding the values.
            
        Returns
        -------
        MultiDimData
            Loaded data container.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("load_csv requires pandas; install it with: pip install pandas")
        df = pd.read_csv(filepath)
        
        if dim_columns is None:
            # Default: all columns except the last are dimensions
            dim_columns = df.columns[:-1].tolist()
        
        if value_column is None:
            # Default: the last column holds the values
            value_column = df.columns[-1]
        
        # Extract the dimension columns
        dim_data = {}
        for i, col in enumerate(dim_columns):
            dim_data[i] = df[col].values
        
        # Build the data container
        data = MultiDimData(dim_data)
        
        # Attach the value column if present
        if value_column in df.columns:
            data.values = df[value_column].values
        
        return data
    
    @staticmethod
    def save_csv(data: MultiDimData, filepath: str, value_column: str = "values"):
        """
        Save data to a CSV file.
        
        Parameters
        ----------
        data : MultiDimData
            Data container to save.
        filepath : str
            Path of the CSV file to write.
        value_column : str, default="values"
            Column name for the values.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("save_csv requires pandas; install it with: pip install pandas")
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
        Load a numpy array from a .npy file.
        
        Parameters
        ----------
        filepath : str
            Path to the .npy file.
            
        Returns
        -------
        np.ndarray
            Loaded array.
        """
        return np.load(filepath)
    
    @staticmethod
    def save_npy(array: np.ndarray, filepath: str):
        """
        Save a numpy array to a .npy file.
        
        Parameters
        ----------
        array : np.ndarray
            Array to save.
        filepath : str
            Path of the .npy file to write.
        """
        np.save(filepath, array)
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """
        Load data from a .npz file.
        
        Parameters
        ----------
        filepath : str
            Path to the .npz file.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of arrays.
        """
        return dict(np.load(filepath))
    
    @staticmethod
    def save_npz(data: Dict[str, np.ndarray], filepath: str):
        """
        Save data to a .npz file.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary of arrays to save.
        filepath : str
            Path of the .npz file to write.
        """
        np.savez(filepath, **data)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file.
            
        Returns
        -------
        Dict[str, Any]
            Loaded data.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
        """
        Save data to a JSON file.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to save.
        filepath : str
            Path of the JSON file to write.
        indent : int, default=2
            Indentation used for pretty-printing.
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load data from a YAML file.
        
        Parameters
        ----------
        filepath : str
            Path to the YAML file.
            
        Returns
        -------
        Dict[str, Any]
            Loaded data.
        """
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: str):
        """
        Save data to a YAML file.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to save.
        filepath : str
            Path of the YAML file to write.
        """
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Load an object from a pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to the pickle file.
            
        Returns
        -------
        Any
            Loaded object.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """
        Save an object to a pickle file.
        
        Parameters
        ----------
        obj : Any
            Object to save.
        filepath : str
            Path of the pickle file to write.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_basis_set(filepath: str) -> 'BasisSet':
        """
        Load a basis set from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the basis-set file.
            
        Returns
        -------
        BasisSet
            Loaded basis set.
        """
        from ..core.basis_container import BasisSet, BasisInfo
        
        data = IOUtils.load_pickle(filepath)
        basis_set = BasisSet()
        
        for basis_data in data:
            basis_info = BasisInfo(
                name=basis_data['name'],
                dim=basis_data['dim'],
                params=basis_data.get('params'),
                func=basis_data.get('func'),  # Note: callables may not pickle reliably
                factors=basis_data.get('factors'),
            )
            basis_set.add_basis(basis_info)
        
        return basis_set
    
    @staticmethod
    def save_basis_set(basis_set: 'BasisSet', filepath: str):
        """
        Save a basis set to a file.
        
        Parameters
        ----------
        basis_set : BasisSet
            Basis set to save.
        filepath : str
            Path of the basis-set file to write.
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
        Load a solver state from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the solver-state file.
            
        Returns
        -------
        Dict[str, Any]
            Solver state.
        """
        return IOUtils.load_pickle(filepath)
    
    @staticmethod
    def save_solver_state(solver: Any, filepath: str):
        """
        Save a solver state to a file.
        
        Parameters
        ----------
        solver : Any
            Solver object whose state is saved.
        filepath : str
            Path of the solver-state file to write.
        """
        # Collect the relevant solver attributes
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
        Generate sample data for testing.
        
        Parameters
        ----------
        n_points : int, default=100
            Number of data points.
        n_dims : int, default=1
            Number of dimensions.
        noise_std : float, default=0.1
            Standard deviation of the additive noise.
            
        Returns
        -------
        Tuple[MultiDimData, np.ndarray]
            Data container and target values.
        """
        # Generate the dimension coordinates
        dim_data = {}
        for i in range(n_dims):
            dim_data[i] = np.linspace(0, 1, n_points)
        
        data = MultiDimData(dim_data)
        
        # Target: sine function with additive noise
        if n_dims == 1:
            x = dim_data[0]
            y_true = np.sin(2 * np.pi * x)
        else:
            # Multi-dimensional: product of sine functions
            y_true = np.ones(n_points)
            for i in range(n_dims):
                y_true *= np.sin(2 * np.pi * dim_data[i])
        
        y = y_true + noise_std * np.random.randn(n_points)
        
        return data, y
    
    @staticmethod
    def export_to_matlab(data: MultiDimData, filepath: str):
        """
        Export data to a MATLAB .mat file.
        
        Parameters
        ----------
        data : MultiDimData
            Data container to export.
        filepath : str
            Path of the .mat file to write.
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
        Import data from a MATLAB .mat file.
        
        Parameters
        ----------
        filepath : str
            Path to the .mat file.
            
        Returns
        -------
        MultiDimData
            Imported data container.
        """
        try:
            from scipy.io import loadmat
            
            mat_data = loadmat(filepath)
            
            # Collect 1-D arrays as dimension columns
            dim_data = {}
            dim_idx = 0
            
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.ndim == 1 or (value.ndim == 2 and value.shape[1] == 1):
                        # Flatten column vectors to 1-D
                        value = value.flatten()
                        dim_data[dim_idx] = value
                        dim_idx += 1
            
            return MultiDimData(dim_data)
        except ImportError:
            raise ImportError("scipy is required for MATLAB import")
