#!/usr/bin/env python3
# main.py - ML pipeline for regression
# runs the full training pipeline

import os
import numpy as np
import pandas as pd

from src.config import SEED, DATA_X_PATH, DATA_Y_PATH, DATA_EVAL_PATH, TARGET_COL
from src.config import OUT_DIR, PLOT_DIR, MAX_FEATURE_HISTS
from src.logger import get_logger, LOG_FILE
from src.utils import header, evaluate
from src.eda import eda_summary, mean_median_std_check, duplication_checks
from src.eda import basic_data_distribution, outlier_check, correlation_analysis
from src.eda import target_distribution, leakage_overlap_check
from src.plotting import plot_feature_histograms, plot_r2_comparison
from src.preprocessing import split_data
from src.models import get_all_models
from src.tuning import tune_hgb

log = get_logger("main")


def load_data():
    header("LOAD DATA")
    
    X = pd.read_csv(DATA_X_PATH)
    y = pd.read_csv(DATA_Y_PATH)[TARGET_COL].values
    X_eval = pd.read_csv(DATA_EVAL_PATH)
    
    log.info(f"X: {X.shape}")
    log.info(f"y: {y.shape}")
    log.info(f"X_eval: {X_eval.shape}")
    
    return X, y, X_eval


def train_baseline(models, X_train, y_train, X_val, y_val):
    header("BASELINE MODEL COMPARISON")
    
    results = []
    for name, pipe in models.items():
        log.info(f"training {name}...")
        pipe.fit(X_train, y_train)
        
        tr_pred = pipe.predict(X_train)
        va_pred = pipe.predict(X_val)
        
        tr = evaluate(y_train, tr_pred)
        va = evaluate(y_val, va_pred)
        gap = va["rmse"] - tr["rmse"]
        
        results.append((name, va["rmse"], va["mae"], va["r2"], tr["rmse"], tr["r2"], gap))
        log.info(f"{name}: train rmse={tr['rmse']:.4f} val rmse={va['rmse']:.4f} gap={gap:.4f}")
    
    df = pd.DataFrame(results, columns=["model", "val_rmse", "val_mae", "val_r2", "train_rmse", "train_r2", "gap"])
    df = df.sort_values("val_rmse")
    
    header("BEST BASELINE")
    log.info(str(df.iloc[0]))
    
    df.to_csv(os.path.join(OUT_DIR, "baseline_results.csv"), index=False)
    return df


def final_eval(model, params, X_train, X_val, X_test, y_train, y_val, y_test):
    # train on train only first
    header("TUNED MODEL EVALUATION")
    model.fit(X_train, y_train)
    
    tr = evaluate(y_train, model.predict(X_train))
    va = evaluate(y_val, model.predict(X_val))
    te = evaluate(y_test, model.predict(X_test))
    
    log.info(f"train rmse: {tr['rmse']:.4f}, r2: {tr['r2']:.4f}")
    log.info(f"val rmse: {va['rmse']:.4f}, r2: {va['r2']:.4f}")
    log.info(f"test rmse: {te['rmse']:.4f}, r2: {te['r2']:.4f}")
    
    # now train on train+val for final model
    header("FINAL MODEL (train+val)")
    X_full = pd.concat([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    model.fit(X_full, y_full)
    
    tr = evaluate(y_train, model.predict(X_train))
    va = evaluate(y_val, model.predict(X_val))
    te = evaluate(y_test, model.predict(X_test))
    
    log.info(f"train rmse: {tr['rmse']:.4f}, r2: {tr['r2']:.4f}")
    log.info(f"val rmse: {va['rmse']:.4f}, r2: {va['r2']:.4f}")
    log.info(f"test rmse: {te['rmse']:.4f}, r2: {te['r2']:.4f}")
    
    plot_r2_comparison("Tuned HGB", tr["r2"], va["r2"], te["r2"])
    
    # save metrics
    metrics = pd.DataFrame([{
        "model": "TunedHGB",
        "train_rmse": tr["rmse"], "train_r2": tr["r2"],
        "val_rmse": va["rmse"], "val_r2": va["r2"],
        "test_rmse": te["rmse"], "test_r2": te["r2"],
        "params": str(params)
    }])
    metrics.to_csv(os.path.join(OUT_DIR, "final_metrics.csv"), index=False)
    
    return model


def make_predictions(model, X, y, X_eval):
    header("GENERATE PREDICTIONS")
    
    # train on all data
    model.fit(X, y)
    preds = model.predict(X_eval)
    
    out = pd.DataFrame({TARGET_COL: preds})
    
    # save predictions
    out.to_csv(os.path.join(OUT_DIR, "EVAL_target01_91.csv"), index=False)
    out.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    
    log.info(f"saved predictions")
    log.info(str(out.head()))


def main():
    log.info("=" * 80)
    log.info("ML PIPELINE")
    log.info(f"log file: {LOG_FILE}")
    log.info("=" * 80)
    
    # load data
    X, y, X_eval = load_data()
    
    # run eda checks
    mean_median_std_check(X, y)
    duplication_checks(X)
    basic_data_distribution(X)
    outlier_check(X)
    correlation_analysis(X, y)
    
    # split data
    header("SPLIT DATA")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=SEED)
    
    log.info(f"train: {X_train.shape}")
    log.info(f"val: {X_val.shape}")
    log.info(f"test: {X_test.shape}")
    
    target_distribution(y_train, y_val, y_test)
    
    # check for leakage
    header("LEAKAGE CHECK")
    leakage_overlap_check(X_train, X_val, "train", "val")
    leakage_overlap_check(X_train, X_test, "train", "test")
    leakage_overlap_check(X_val, X_test, "val", "test")
    
    # feature plots
    header("FEATURE PLOTS")
    X_num = X_train.select_dtypes(include=[np.number])
    if X_num.shape[1] > 0:
        top_cols = X_num.var().sort_values(ascending=False).head(MAX_FEATURE_HISTS).index.tolist()
        plot_feature_histograms(X_num, top_cols)
    
    # train baseline models
    models = get_all_models()
    baseline_df = train_baseline(models, X_train, y_train, X_val, y_val)
    
    # tune best model
    tuned, params, tune_df = tune_hgb(X_train, y_train, X_val, y_val, seed=SEED)
    
    # final evaluation
    tuned = final_eval(tuned, params, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # generate predictions
    make_predictions(tuned, X, y, X_eval)
    
    # summary
    eda_summary(X, y, X_eval)
    
    header("DONE")
    log.info(f"plots: {PLOT_DIR}")
    log.info(f"log: {LOG_FILE}")


if __name__ == "__main__":
    main()
