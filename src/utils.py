"""
Utility functions for metrics and display.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import get_logger

logger = get_logger("utils")


def header(title):
    """Print a formatted section header."""
    line = "=" * 90
    logger.info("")
    logger.info(line)
    logger.info(title)
    logger.info(line)


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    """
    Evaluate predictions with multiple metrics.
    
    Returns:
        dict with rmse, mae, r2 scores
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
