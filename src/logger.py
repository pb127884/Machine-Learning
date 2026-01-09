"""
Logging setup with dual output (console + file).
"""
import os
import sys
import logging
from datetime import datetime
from src.config import LOG_DIR

# Generate timestamped log filename
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def setup_logging(level=logging.INFO):
    """
    Setup dual logging to console and file.
    
    Args:
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance
    """
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name for the logger
    
    Returns:
        Logger instance with module name
    """
    return logging.getLogger(name)


# Initialize logging on module import
_root_logger = setup_logging()


def log_info(message, logger_name="main"):
    """Log info level message."""
    get_logger(logger_name).info(message)


def log_debug(message, logger_name="main"):
    """Log debug level message."""
    get_logger(logger_name).debug(message)


def log_warning(message, logger_name="main"):
    """Log warning level message."""
    get_logger(logger_name).warning(message)


def log_error(message, logger_name="main"):
    """Log error level message."""
    get_logger(logger_name).error(message)
