"""
Hyperparameter tuning functions.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from src.config import SEED, OUT_DIR
from src.utils import header, evaluate
from src.logger import get_logger

logger = get_logger("tuning")


def hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=SEED):
    """
    Hyper-tune HistGradientBoosting using ONLY the Validation set (no CV).
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        seed: Random seed
    
    Returns:
        best_model, best_params, tuning_dataframe
    """
    header("HYPER TUNING (NO CV): HistGradientBoostingRegressor using VALIDATION ONLY")
    
    # Preprocessing pipeline
    basic_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])
    
    base_pipe = Pipeline([
        ("prep", basic_prep),
        ("model", HistGradientBoostingRegressor(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        ))
    ])
    
    # Hyperparameter grid
    param_grid = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [3, 5, 7, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_bins": [128, 255],
        "model__max_iter": [200, 400, 800],
    }
    
    grid = list(ParameterGrid(param_grid))
    logger.info(f"Total combinations to try: {len(grid)}")
    
    best = None
    best_params = None
    best_val_rmse = np.inf
    tuning_rows = []
    
    for i, params in enumerate(grid, 1):
        m = clone(base_pipe)
        m.set_params(**params)
        m.fit(X_train, y_train)
        
        tr_pred = m.predict(X_train)
        va_pred = m.predict(X_val)
        
        tr_m = evaluate(y_train, tr_pred)
        va_m = evaluate(y_val, va_pred)
        
        tuning_rows.append({
            "i": i,
            "val_rmse": va_m["rmse"],
            "val_r2": va_m["r2"],
            "train_rmse": tr_m["rmse"],
            "train_r2": tr_m["r2"],
            "gap_rmse": va_m["rmse"] - tr_m["rmse"],
            **params
        })
        
        if va_m["rmse"] < best_val_rmse:
            best_val_rmse = va_m["rmse"]
            best = m
            best_params = params
        
        if i % 50 == 0 or i == 1 or i == len(grid):
            logger.info(f"[{i:>5}/{len(grid)}] best_val_rmse={best_val_rmse:.6f}")
    
    header("BEST HGB PARAMS (by Validation RMSE) - NO CV")
    logger.info(f"Best validation RMSE: {round(best_val_rmse, 6)}")
    logger.info("Best params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    
    # Save tuning table
    tune_df = pd.DataFrame(tuning_rows).sort_values("val_rmse")
    tune_path = os.path.join(OUT_DIR, "hgb_tuning_no_cv.csv")
    tune_df.to_csv(tune_path, index=False)
    logger.info(f"Saved tuning table: {tune_path}")
    
    return best, best_params, tune_df
