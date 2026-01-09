"""
Data preprocessing pipelines and splitting utilities.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger

logger = get_logger("preprocessing")


def make_strat_bins(y, n_bins=10):
    """Create stratification bins from continuous target."""
    y = np.asarray(y)
    try:
        b = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        return np.asarray(b, dtype=int)
    except Exception:
        b = pd.qcut(pd.Series(y).rank(method="average"), q=n_bins, labels=False, duplicates="drop")
        return np.asarray(b, dtype=int)


def stratified_70_15_15_split(X, y, seed=42, n_bins=10):
    """
    Split data into train/val/test with 70/15/15 ratio using stratification.
    
    Args:
        X: Feature DataFrame
        y: Target array
        seed: Random seed
        n_bins: Number of bins for stratification
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Performing stratified 70/15/15 split with seed={seed}")
    
    bins = make_strat_bins(y, n_bins=n_bins)
    
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(X, bins))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y[temp_idx]
    
    bins_temp = make_strat_bins(y_temp, n_bins=n_bins)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(X_temp, bins_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp[test_idx]
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_basic_prep():
    """
    Get basic preprocessing pipeline.
    
    Returns:
        Pipeline with MedianImputer + VarianceThreshold
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])


def get_scaled_prep():
    """
    Get scaled preprocessing pipeline.
    
    Returns:
        Pipeline with MedianImputer + VarianceThreshold + StandardScaler
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])
