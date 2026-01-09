# utils.py - helper functions for metrics
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import get_logger

log = get_logger("utils")

def header(title):
    # print section header
    log.info("")
    log.info("=" * 90)
    log.info(title)
    log.info("=" * 90)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(y_true, y_pred):
    # get all metrics at once
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
