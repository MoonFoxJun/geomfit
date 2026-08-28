"""Logging utilities for the functional solver."""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

class Logger:
    """Logging utility for the functional solver."""
    
    def __init__(self, name: str = "geomfit", 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True):
        """
        Initialize the logger.
        
        Parameters
        ----------
        name : str, default="geomfit"
            Logger name.
        log_level : str, default="INFO"
            Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        log_file : str, optional
            Path of the log file.
        console_output : bool, default=True
            Whether to write messages to the console.
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.console_output = console_output
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        # 清空预先存在的 handler，避免重复输出（比如多次实例化 Logger
        # 时，同一 logger 名下会堆积多个输出流）。
        self.logger.handlers = []
        
        # 共享的格式化器：统一时间戳/名称/级别/消息的排版
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出 handler：把日志写到 stdout
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出 handler：把日志同时写入文件
        if log_file:
            # 若日志文件所在的目录还不存在，先递归创建
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format the message with extra context appended as key=value pairs."""
        if kwargs:
            context_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} [{context_str}]"
        return message
    
    def log_solver_start(self, solver_name: str, **kwargs):
        """Log the start of a solver run."""
        self.info(f"Starting {solver_name}", **kwargs)
    
    def log_solver_end(self, solver_name: str, success: bool = True, 
                      duration: Optional[float] = None, **kwargs):
        """Log the end of a solver run."""
        status = "completed successfully" if success else "failed"
        message = f"{solver_name} {status}"
        
        if duration is not None:
            message += f" in {duration:.2f}s"
        
        if success:
            self.info(message, **kwargs)
        else:
            self.error(message, **kwargs)
    
    def log_data_loading(self, data_info: Dict[str, Any]):
        """Log data-loading information."""
        n_points = data_info.get('n_points', 'unknown')
        n_dims = data_info.get('n_dims', 'unknown')
        self.info(f"Loaded data: {n_points} points, {n_dims} dimensions", **data_info)
    
    def log_basis_info(self, basis_info: Dict[str, Any]):
        """Log basis-function information."""
        n_basis = basis_info.get('n_basis', 'unknown')
        basis_types = basis_info.get('basis_types', [])
        self.info(f"Using {n_basis} basis functions: {basis_types}", **basis_info)
    
    def log_solution_info(self, solution_info: Dict[str, Any]):
        """Log solution information."""
        n_coeff = solution_info.get('n_coefficients', 'unknown')
        residual_norm = solution_info.get('residual_norm', 'unknown')
        condition_number = solution_info.get('condition_number', 'unknown')
        
        self.info(f"Solution: {n_coeff} coefficients, residual norm: {residual_norm:.2e}, "
                 f"condition number: {condition_number:.2e}", **solution_info)
    
    def log_prediction_info(self, prediction_info: Dict[str, Any]):
        """Log prediction information."""
        n_points = prediction_info.get('n_points', 'unknown')
        mse = prediction_info.get('mse', 'unknown')
        r2 = prediction_info.get('r2', 'unknown')
        
        self.info(f"Prediction: {n_points} points, MSE: {mse:.2e}, R²: {r2:.4f}", **prediction_info)
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log an exception with context."""
        self.error(f"Exception in {context}: {str(exception)}", 
                  exception_type=type(exception).__name__)
    
    def get_log_messages(self, level: Optional[str] = None) -> list:
        """
        Return the recorded messages (requires a memory handler).
        
        Parameters
        ----------
        level : str, optional
            Filter messages by log level.
            
        Returns
        -------
        list
            List of log messages.
        """
        # 要返回已记录的日志消息，需要预先安装一个"内存 handler"
        # （如 logging.handlers.MemoryHandler）把消息存到内存；
        # 当前实现尚未安装，因此直接返回空列表。
        return []
    
    @staticmethod
    def create_default_logger(log_file: Optional[str] = None) -> 'Logger':
        """
        Create a default logger.
        
        Parameters
        ----------
        log_file : str, optional
            Path of the log file.
            
        Returns
        -------
        Logger
            Default logger instance.
        """
        return Logger(
            name="geomfit",
            log_level="INFO",
            log_file=log_file,
            console_output=True
        )
    
    @staticmethod
    def create_debug_logger(log_file: Optional[str] = None) -> 'Logger':
        """
        Create a debug logger.
        
        Parameters
        ----------
        log_file : str, optional
            Path of the log file.
            
        Returns
        -------
        Logger
            Debug logger instance.
        """
        return Logger(
            name="geomfit_debug",
            log_level="DEBUG",
            log_file=log_file,
            console_output=True
        )
    
    @staticmethod
    def create_quiet_logger() -> 'Logger':
        """
        Create a quiet logger that writes to a file only.
        
        Returns
        -------
        Logger
            Quiet logger instance.
        """
        return Logger(
            name="geomfit_quiet",
            log_level="INFO",
            log_file="geomfit.log",
            console_output=False
        )
