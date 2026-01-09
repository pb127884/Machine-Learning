# preprocessing.py - data prep and splitting
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger

log = get_logger("preprocessing")

def make_bins(y, n_bins=10):
    # bin continuous target for stratification
    y = np.asarray(y)
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except:
        bins = pd.qcut(pd.Series(y).rank(method="average"), q=n_bins, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)

def split_data(X, y, seed=42, n_bins=10):
    # 70/15/15 stratified split
    log.info(f"splitting data with seed={seed}")
    
    bins = make_bins(y, n_bins)
    
    # first split: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(X, bins))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y[temp_idx]
    
    # second split: 50/50 of temp -> 15% val, 15% test
    bins_temp = make_bins(y_temp, n_bins)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(X_temp, bins_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp[test_idx]
    
    log.info(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_basic_pipeline():
    # imputer + variance threshold
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])

def get_scaled_pipeline():
    # same as basic but with scaler
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])
