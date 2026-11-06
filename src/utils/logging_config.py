"""
Logging configuration for HPC chunk processing.
"""

import logging
import os
import sys
import time
import psutil
from functools import wraps
from src.config.parameters import LOGGING_PARAMS

def setup_chunk_logger(chunk_id, output_dir):
    """
    Set up logger for a specific chunk with both file and console output.
    
    Args:
        chunk_id (int): Chunk identifier
        output_dir (str): Directory to save log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger_name = f'chunk_{chunk_id}'
    logger = logging.getLogger(logger_name)
    
    # Set log level based on configuration
    log_level = LOGGING_PARAMS.get('log_level', 'DEBUG').upper()
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logging
    log_file = os.path.join(log_dir, f'chunk_{chunk_id}.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Test that debug logging is working
    logger.debug(f"DEBUG logging initialized for chunk {chunk_id}")
    logger.info(f"INFO logging initialized for chunk {chunk_id}")
    
    return logger


def get_chunk_logger(chunk_id=None):
    """
    Get the logger for the current chunk, with fallback logic.
    
    Args:
        chunk_id (int, optional): Chunk ID to get logger for
    
    Returns:
        logging.Logger or None: Logger instance or None if not available
    """
    try:
        if chunk_id is not None:
            logger_name = f'chunk_{chunk_id}'
            logger = logging.getLogger(logger_name)
            if logger.handlers:  # Logger is properly configured
                return logger
        
        # Try to get any existing chunk logger
        for logger_name in logging.Logger.manager.loggerDict:
            if logger_name.startswith('chunk_'):
                logger = logging.getLogger(logger_name)
                if logger.handlers:
                    return logger
        
        # Fallback: try to get root logger
        root_logger = logging.getLogger()
        if root_logger.handlers:
            return root_logger
            
        return None
        
    except Exception:
        return None


def setup_worker_logger(chunk_id, output_dir):
    """
    Set up logger for multiprocessing worker processes.
    This should be called in each worker process.
    """
    return setup_chunk_logger(chunk_id, output_dir)


# Global variable to store the worker logger
_worker_logger = None


def get_worker_logger():
    """Get the logger for the current worker process."""
    global _worker_logger
    return _worker_logger


def log_system_info(logger):
    """Log system information and resource usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        logger.info(f"System Info - Memory: {memory_info.rss / 1024 / 1024:.1f} MB, "
                   f"CPU: {cpu_percent:.1f}%, "
                   f"Available CPUs: {psutil.cpu_count()}")
                   
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")


class ProgressTracker:
    """Track and log progress of iterative operations."""
    
    def __init__(self, logger, total_items, operation_name, log_interval=50):
        self.logger = logger
        self.total_items = total_items
        self.operation_name = operation_name
        self.log_interval = log_interval
        self.start_time = time.time()
        self.processed_items = 0
        
    def update(self, increment=1):
        """Update progress and log if needed."""
        self.processed_items += increment
        
        if self.processed_items % self.log_interval == 0 or self.processed_items == self.total_items:
            elapsed = time.time() - self.start_time
            rate = self.processed_items / elapsed if elapsed > 0 else 0
            remaining = (self.total_items - self.processed_items) / rate if rate > 0 else 0
            
            progress_pct = (self.processed_items / self.total_items) * 100
            
            self.logger.info(f"{self.operation_name} - Progress: {self.processed_items}/{self.total_items} "
                           f"({progress_pct:.1f}%) - Rate: {rate:.2f} items/s - "
                           f"ETA: {remaining:.1f}s")
