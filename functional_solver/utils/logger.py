"""函数求解器的日志工具。"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

class Logger:
    """函数求解器的日志工具类。"""
    
    def __init__(self, name: str = "functional_solver", 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True):
        """
        初始化日志器。
        
        参数
        ----------
        name : str, 默认="functional_solver"
            日志器名称
        log_level : str, 默认="INFO"
            日志级别：DEBUG、INFO、WARNING、ERROR、CRITICAL
        log_file : str, 可选
            日志文件路径
        console_output : bool, 默认=True
            是否输出到控制台
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.console_output = console_output
        
        # 创建日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # 移除已有的处理器
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_file:
            # 如果目录不存在则创建
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """记录调试消息。"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """记录信息消息。"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录警告消息。"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """记录错误消息。"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """记录严重错误消息。"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """使用附加上下文格式化消息。"""
        if kwargs:
            context_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} [{context_str}]"
        return message
    
    def log_solver_start(self, solver_name: str, **kwargs):
        """记录求解器开始。"""
        self.info(f"Starting {solver_name}", **kwargs)
    
    def log_solver_end(self, solver_name: str, success: bool = True, 
                      duration: Optional[float] = None, **kwargs):
        """记录求解器结束。"""
        status = "completed successfully" if success else "failed"
        message = f"{solver_name} {status}"
        
        if duration is not None:
            message += f" in {duration:.2f}s"
        
        if success:
            self.info(message, **kwargs)
        else:
            self.error(message, **kwargs)
    
    def log_data_loading(self, data_info: Dict[str, Any]):
        """记录数据加载信息。"""
        n_points = data_info.get('n_points', 'unknown')
        n_dims = data_info.get('n_dims', 'unknown')
        self.info(f"Loaded data: {n_points} points, {n_dims} dimensions", **data_info)
    
    def log_basis_info(self, basis_info: Dict[str, Any]):
        """记录基函数信息。"""
        n_basis = basis_info.get('n_basis', 'unknown')
        basis_types = basis_info.get('basis_types', [])
        self.info(f"Using {n_basis} basis functions: {basis_types}", **basis_info)
    
    def log_solution_info(self, solution_info: Dict[str, Any]):
        """记录求解信息。"""
        n_coeff = solution_info.get('n_coefficients', 'unknown')
        residual_norm = solution_info.get('residual_norm', 'unknown')
        condition_number = solution_info.get('condition_number', 'unknown')
        
        self.info(f"Solution: {n_coeff} coefficients, residual norm: {residual_norm:.2e}, "
                 f"condition number: {condition_number:.2e}", **solution_info)
    
    def log_prediction_info(self, prediction_info: Dict[str, Any]):
        """记录预测信息。"""
        n_points = prediction_info.get('n_points', 'unknown')
        mse = prediction_info.get('mse', 'unknown')
        r2 = prediction_info.get('r2', 'unknown')
        
        self.info(f"Prediction: {n_points} points, MSE: {mse:.2e}, R²: {r2:.4f}", **prediction_info)
    
    def log_exception(self, exception: Exception, context: str = ""):
        """记录带上下文的异常。"""
        self.error(f"Exception in {context}: {str(exception)}", 
                  exception_type=type(exception).__name__)
    
    def get_log_messages(self, level: Optional[str] = None) -> list:
        """
        获取已记录的消息（如果使用了内存处理器）。
        
        参数
        ----------
        level : str, 可选
            按日志级别过滤
            
        返回
        -------
        list
            日志消息列表
        """
        # 这需要预先设置一个内存处理器
        # 目前先返回空列表
        return []
    
    @staticmethod
    def create_default_logger(log_file: Optional[str] = None) -> 'Logger':
        """
        创建默认日志器。
        
        参数
        ----------
        log_file : str, 可选
            日志文件路径
            
        返回
        -------
        Logger
            默认日志器实例
        """
        return Logger(
            name="functional_solver",
            log_level="INFO",
            log_file=log_file,
            console_output=True
        )
    
    @staticmethod
    def create_debug_logger(log_file: Optional[str] = None) -> 'Logger':
        """
        创建调试日志器。
        
        参数
        ----------
        log_file : str, 可选
            日志文件路径
            
        返回
        -------
        Logger
            调试日志器实例
        """
        return Logger(
            name="functional_solver_debug",
            log_level="DEBUG",
            log_file=log_file,
            console_output=True
        )
    
    @staticmethod
    def create_quiet_logger() -> 'Logger':
        """
        创建安静模式日志器（仅写入文件）。
        
        返回
        -------
        Logger
            安静模式日志器实例
        """
        return Logger(
            name="functional_solver_quiet",
            log_level="INFO",
            log_file="functional_solver.log",
            console_output=False
        )
