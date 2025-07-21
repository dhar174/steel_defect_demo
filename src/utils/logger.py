import logging
import sys
import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

class Logger:
    """Centralized logging configuration and utilities with performance tracking"""
    
    def __init__(self, name: str = __name__, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name (str): Logger name
            log_level (str): Logging level
            log_file (str): Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.log_level = getattr(logging, log_level.upper())
        self.process = psutil.Process()
        self.start_time = time.time()
        self.setup_logger(log_file)
    
    def setup_logger(self, log_file: Optional[str] = None) -> None:
        """
        Setup logger configuration with console and optional file handlers.
        
        Args:
            log_file (str): Optional log file path
        """
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = self.create_console_handler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = self.create_file_handler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def create_file_handler(self, log_file: str) -> logging.FileHandler:
        """
        Create file handler for logging to file.
        
        Args:
            log_file (str): Path to log file
            
        Returns:
            logging.FileHandler: File handler
        """
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(self.log_level)
        return handler
    
    def create_console_handler(self) -> logging.StreamHandler:
        """
        Create console handler for logging to stdout.
        
        Returns:
            logging.StreamHandler: Console handler
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.log_level)
        return handler
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_runtime(self) -> float:
        """Get runtime in seconds since logger initialization."""
        return time.time() - self.start_time
    
    def log_system_performance(self, component: str, latency_ms: float = None,
                             memory_usage_mb: float = None, operation: str = None) -> None:
        """
        Log system performance metrics.
        
        Args:
            component (str): System component name
            latency_ms (float): Processing latency in milliseconds
            memory_usage_mb (float): Memory usage in MB
            operation (str): Operation being performed
        """
        memory_stats = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()
        runtime = self.get_runtime()
        
        perf_info = {
            'component': component,
            'operation': operation,
            'runtime_seconds': runtime,
            'cpu_percent': cpu_usage,
            'memory_rss_mb': memory_stats['rss_mb'],
            'memory_percent': memory_stats['percent']
        }
        
        if latency_ms is not None:
            perf_info['latency_ms'] = latency_ms
        
        if memory_usage_mb is not None:
            perf_info['operation_memory_mb'] = memory_usage_mb
        
        self.logger.info(f"Performance: {json.dumps(perf_info, indent=None)}")
    
    def log_data_generation_start(self, total_casts: int, config: Dict[str, Any]) -> None:
        """
        Log data generation start.
        
        Args:
            total_casts (int): Total number of casts to generate
            config (dict): Generation configuration
        """
        self.logger.info(f"Starting data generation: {total_casts} casts")
        self.logger.debug(f"Configuration: {json.dumps(config, indent=2)}")
        self.log_system_performance("data_generator", operation="generation_start")
    
    def log_data_generation_progress(self, current_cast: int, total_casts: int, 
                                   defect_count: int, eta_seconds: float = None) -> None:
        """
        Log data generation progress.
        
        Args:
            current_cast (int): Current cast number
            total_casts (int): Total casts to generate
            defect_count (int): Number of defects generated so far
            eta_seconds (float): Estimated time to completion
        """
        progress_pct = (current_cast / total_casts) * 100
        defect_rate = defect_count / current_cast if current_cast > 0 else 0
        
        progress_info = {
            'progress_percent': progress_pct,
            'current_cast': current_cast,
            'total_casts': total_casts,
            'defect_count': defect_count,
            'defect_rate': defect_rate
        }
        
        if eta_seconds is not None:
            progress_info['eta_seconds'] = eta_seconds
            progress_info['eta_formatted'] = self._format_duration(eta_seconds)
        
        self.logger.info(f"Progress: {json.dumps(progress_info, indent=None)}")
    
    def log_data_generation_end(self, total_casts: int, defect_count: int, 
                              quality_metrics: Dict[str, Any] = None) -> None:
        """
        Log data generation completion.
        
        Args:
            total_casts (int): Total casts generated
            defect_count (int): Number of defects generated
            quality_metrics (dict): Data quality metrics
        """
        runtime = self.get_runtime()
        final_stats = {
            'total_casts': total_casts,
            'defect_count': defect_count,
            'defect_rate': defect_count / total_casts,
            'generation_time_seconds': runtime,
            'casts_per_second': total_casts / runtime if runtime > 0 else 0
        }
        
        if quality_metrics:
            final_stats['quality_metrics'] = quality_metrics
        
        self.logger.info(f"Data generation completed: {json.dumps(final_stats, indent=2)}")
        self.log_system_performance("data_generator", operation="generation_complete")
    
    def log_data_quality_issue(self, issue_type: str, details: str, severity: str = "WARNING") -> None:
        """
        Log data quality issues.
        
        Args:
            issue_type (str): Type of data quality issue
            details (str): Issue details
            severity (str): Issue severity level
        """
        quality_issue = {
            'issue_type': issue_type,
            'details': details,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        log_level = getattr(logging, severity.upper(), logging.WARNING)
        self.logger.log(log_level, f"Data Quality Issue: {json.dumps(quality_issue, indent=None)}")
    
    def log_validation_results(self, validation_type: str, results: Dict[str, Any], 
                             passed: bool) -> None:
        """
        Log validation results.
        
        Args:
            validation_type (str): Type of validation performed
            results (dict): Validation results
            passed (bool): Whether validation passed
        """
        validation_log = {
            'validation_type': validation_type,
            'passed': passed,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        level = logging.INFO if passed else logging.WARNING
        self.logger.log(level, f"Validation: {json.dumps(validation_log, indent=2)}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.
        
        Returns:
            logging.Logger: Logger instance
        """
        return self.logger

# Module-level functions for convenience
def get_logger(name: str = __name__, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name
        log_level (str): Logging level
        log_file (str): Optional log file path
        
    Returns:
        logging.Logger: Logger instance
    """
    logger_instance = Logger(name, log_level, log_file)
    return logger_instance.get_logger()

def setup_project_logging(log_dir: str = "logs", log_level: str = "INFO") -> Logger:
    """
    Setup project-wide logging configuration.
    
    Args:
        log_dir (str): Directory for log files
        log_level (str): Logging level
        
    Returns:
        Logger: Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"data_generation_{timestamp}.log"
    
    logger_instance = Logger("steel_defect_demo", log_level, str(log_file))
    return logger_instance