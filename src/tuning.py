# tuning.py - hyperparameter search
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

log = get_logger("tuning")

def tune_hgb(X_train, y_train, X_val, y_val, seed=SEED):
    # grid search for HistGradientBoosting
    # using validation set only (no cv for speed)
    
    header("HYPERPARAMETER TUNING")
    
    # base model
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])
    
    base = Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingRegressor(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        ))
    ])
    
    # params to try
    grid = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [3, 5, 7, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__max_bins": [128, 255],
        "model__max_iter": [200, 400, 800],
    }
    
    combos = list(ParameterGrid(grid))
    log.info(f"trying {len(combos)} combinations")
    
    best_model = None
    best_params = None
    best_rmse = np.inf
    results = []
    
    for i, params in enumerate(combos, 1):
        m = clone(base)
        m.set_params(**params)
        m.fit(X_train, y_train)
        
        tr_pred = m.predict(X_train)
        va_pred = m.predict(X_val)
        
        tr_m = evaluate(y_train, tr_pred)
        va_m = evaluate(y_val, va_pred)
        
        results.append({
            "i": i,
            "val_rmse": va_m["rmse"],
            "val_r2": va_m["r2"],
            "train_rmse": tr_m["rmse"],
            "gap": va_m["rmse"] - tr_m["rmse"],
            **params
        })
        
        if va_m["rmse"] < best_rmse:
            best_rmse = va_m["rmse"]
            best_model = m
            best_params = params
        
        # print progress
        if i % 50 == 0 or i == 1 or i == len(combos):
            log.info(f"[{i}/{len(combos)}] best rmse: {best_rmse:.6f}")
    
    header("BEST PARAMS")
    log.info(f"best rmse: {best_rmse:.6f}")
    for k, v in best_params.items():
        log.info(f"  {k}: {v}")
    
    # save results
    df = pd.DataFrame(results).sort_values("val_rmse")
    df.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)
    log.info("saved tuning results")
    
    return best_model, best_params, df
