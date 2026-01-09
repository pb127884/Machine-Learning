#!/usr/bin/env python3
"""
Main ML Pipeline Entry Point

This is the main orchestration script that runs the complete ML pipeline.
All functionality is imported from modular src/ packages.

Usage:
    python main.py
    
Logs are saved to: outputs/logs/run_YYYYMMDD_HHMMSS.log
"""
import os
import numpy as np
import pandas as pd

# Import from src modules
from src.config import (
    SEED, DATA_X_PATH, DATA_Y_PATH, DATA_EVAL_PATH, TARGET_COL,
    OUT_DIR, PLOT_DIR, MAX_FEATURE_HISTS
)
from src.logger import get_logger, LOG_FILE
from src.utils import header, evaluate
from src.eda import (
    eda_summary, mean_median_std_check, duplication_checks,
    basic_data_distribution, outlier_check_report_only,
    correlation_analysis, target_distribution, leakage_overlap_check
)
from src.plotting import plot_feature_histograms, plot_r2_for_best
from src.preprocessing import stratified_70_15_15_split
from src.models import get_models
from src.tuning import hypertune_hgb_no_cv

logger = get_logger("main")


def load_data():
    """Load training and evaluation datasets."""
    header("LOAD DATA")
    
    X = pd.read_csv(DATA_X_PATH)
    y = pd.read_csv(DATA_Y_PATH)[TARGET_COL].values
    X_eval = pd.read_csv(DATA_EVAL_PATH)
    
    logger.info(f"X shape     : {X.shape}")
    logger.info(f"y shape     : {y.shape}")
    logger.info(f"X_eval shape: {X_eval.shape}")
    
    return X, y, X_eval


def run_baseline_models(models, X_train, y_train, X_val, y_val):
    """Train and evaluate baseline models."""
    header("BASELINE MODEL SELECTION USING VALIDATION SET")
    
    results = []
    for name, pipe in models.items():
        logger.info(f"Training {name}...")
        pipe.fit(X_train, y_train)
        
        tr_pred = pipe.predict(X_train)
        va_pred = pipe.predict(X_val)
        
        tr = evaluate(y_train, tr_pred)
        va = evaluate(y_val, va_pred)
        
        gap_rmse = va["rmse"] - tr["rmse"]
        results.append((name, va["rmse"], va["mae"], va["r2"], tr["rmse"], tr["r2"], gap_rmse))
        
        logger.info(
            f"{name:18s} | "
            f"TRAIN RMSE {tr['rmse']:.6f} R² {tr['r2']:.6f} || "
            f"VAL RMSE {va['rmse']:.6f} R² {va['r2']:.6f} || "
            f"GAP {gap_rmse:.6f}"
        )
    
    results_df = pd.DataFrame(
        results,
        columns=["model", "val_rmse", "val_mae", "val_r2", "train_rmse", "train_r2", "gap_rmse"]
    ).sort_values("val_rmse")
    
    header("BASELINE BEST MODEL (by Validation RMSE)")
    logger.info(str(results_df.iloc[0]))
    
    results_df.to_csv(os.path.join(OUT_DIR, "baseline_metrics_table.csv"), index=False)
    logger.info(f"Saved baseline metrics: {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")
    
    return results_df


def final_evaluation(tuned_hgb, tuned_params, X_train, X_val, X_test, y_train, y_val, y_test):
    """Perform final evaluation on test set."""
    # Evaluate tuned model (fit train only)
    header("TUNED HGB (FIT TRAIN ONLY) → TRAIN/VAL/TEST RESULTS")
    tuned_hgb.fit(X_train, y_train)
    
    pred_tr = tuned_hgb.predict(X_train)
    pred_va = tuned_hgb.predict(X_val)
    pred_te = tuned_hgb.predict(X_test)
    
    m_tr = evaluate(y_train, pred_tr)
    m_va = evaluate(y_val, pred_va)
    m_te = evaluate(y_test, pred_te)
    
    logger.info("TUNED HGB (train-only fit)")
    logger.info(f"Train RMSE: {m_tr['rmse']:.6f} | Train R²: {m_tr['r2']:.6f}")
    logger.info(f"Val   RMSE: {m_va['rmse']:.6f} | Val   R²: {m_va['r2']:.6f}")
    logger.info(f"Test  RMSE: {m_te['rmse']:.6f} | Test  R²: {m_te['r2']:.6f}")
    logger.info(f"GAP (Val-Train RMSE): {m_va['rmse'] - m_tr['rmse']:.6f}")
    
    # Final train on (train + val)
    header("FINAL EVALUATION ON TEST SET (USED ONCE) - TUNED HGB TRAINED ON (TRAIN+VAL)")
    
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = np.concatenate([y_train, y_val])
    
    tuned_hgb.fit(X_train_final, y_train_final)
    
    pred_train = tuned_hgb.predict(X_train)
    pred_val = tuned_hgb.predict(X_val)
    pred_test = tuned_hgb.predict(X_test)
    
    train_m = evaluate(y_train, pred_train)
    val_m = evaluate(y_val, pred_val)
    test_m = evaluate(y_test, pred_test)
    
    logger.info("\nBEST MODEL: Tuned HistGradientBoostingRegressor (no CV)")
    logger.info(f"Train RMSE: {train_m['rmse']:.6f} | Train R²: {train_m['r2']:.6f}")
    logger.info(f"Val   RMSE: {val_m['rmse']:.6f} | Val   R²: {val_m['r2']:.6f}")
    logger.info(f"Test  RMSE: {test_m['rmse']:.6f} | Test  R²: {test_m['r2']:.6f}")
    logger.info(f"GAP (Val-Train RMSE): {val_m['rmse'] - train_m['rmse']:.6f}")
    
    plot_r2_for_best("Tuned HistGradientBoosting (no CV)", train_m["r2"], val_m["r2"], test_m["r2"])
    
    # Save final metrics
    final_metrics = pd.DataFrame([{
        "model": "TunedHistGradientBoosting_noCV",
        "train_rmse": train_m["rmse"],
        "train_mae": train_m["mae"],
        "train_r2": train_m["r2"],
        "val_rmse": val_m["rmse"],
        "val_mae": val_m["mae"],
        "val_r2": val_m["r2"],
        "test_rmse": test_m["rmse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2"],
        "best_params": str(tuned_params),
    }])
    final_metrics.to_csv(os.path.join(OUT_DIR, "final_metrics_tuned_hgb_no_cv.csv"), index=False)
    logger.info(f"Saved final metrics: {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")
    
    return tuned_hgb


def generate_predictions(model, X, y, X_eval):
    """Generate predictions for evaluation set."""
    header("TRAIN ON FULL DATA + PREDICT EVAL (OFFICIAL SUBMISSION FILE) - TUNED HGB")
    
    model.fit(X, y)
    eval_pred = model.predict(X_eval)
    
    out = pd.DataFrame({TARGET_COL: eval_pred.astype(float)})
    
    official_path = os.path.join(OUT_DIR, "EVAL_target01_91.csv")
    out.to_csv(official_path, index=False)
    
    extra_path = os.path.join(OUT_DIR, "EVAL_target01_best.csv")
    out.to_csv(extra_path, index=False)
    
    logger.info(f"Saved OFFICIAL: {official_path}")
    logger.info(f"Saved EXTRA   : {extra_path}")
    logger.info(str(out.head()))


def main():
    """Main pipeline execution."""
    logger.info("=" * 90)
    logger.info("ML PIPELINE START")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 90)
    
    # 1) Load Data
    X, y, X_eval = load_data()
    
    # 2) EDA
    mean_median_std_check(X, y)
    duplication_checks(X)
    basic_data_distribution(X)
    outlier_check_report_only(X)
    correlation_analysis(X, y)
    
    # 3) Train/Val/Test Split
    header("TRAIN / VAL / TEST SPLIT (70/15/15) - STRATIFIED ON TARGET")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_70_15_15_split(
        X, y, seed=SEED, n_bins=10
    )
    
    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Val  : {X_val.shape}")
    logger.info(f"Test : {X_test.shape}")
    
    target_distribution(y_train, y_val, y_test)
    
    # Data leakage check
    header("DATA LEAKAGE CHECK (OVERLAP BETWEEN SPLITS)")
    leakage_overlap_check(X_train, X_val, "train", "val")
    leakage_overlap_check(X_train, X_test, "train", "test")
    leakage_overlap_check(X_val, X_test, "val", "test")
    
    # Feature plotting
    header("FEATURE PLOTTING (Top variance numeric features)")
    X_num = X_train.select_dtypes(include=[np.number])
    if X_num.shape[1] > 0:
        variances = X_num.var().sort_values(ascending=False)
        top_cols = variances.head(min(MAX_FEATURE_HISTS, len(variances))).index.tolist()
        plot_feature_histograms(X_num, top_cols, filename_prefix="hist")
    else:
        logger.info("No numeric features detected -> skipping histograms.")
    
    # 4) Define and train baseline models
    header("DEFINE MODELS (BASELINE MODELS)")
    models = get_models()
    results_df = run_baseline_models(models, X_train, y_train, X_val, y_val)
    
    # 5) Hyperparameter tuning
    tuned_hgb, tuned_params, tune_df = hypertune_hgb_no_cv(
        X_train, y_train, X_val, y_val, seed=SEED
    )
    
    # 6) Final evaluation
    tuned_hgb = final_evaluation(
        tuned_hgb, tuned_params,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
    
    # 7) Generate predictions
    generate_predictions(tuned_hgb, X, y, X_eval)
    
    # 8) EDA Summary
    eda_summary(X, y, X_eval)
    
    # Done
    header("DONE")
    logger.info(f"All plots saved in: {PLOT_DIR}")
    logger.info(f"Baseline metrics saved in : {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")
    logger.info(f"Tuning table saved in     : {os.path.join(OUT_DIR, 'hgb_tuning_no_cv.csv')}")
    logger.info(f"Run log saved in          : {LOG_FILE}")
    logger.info(f"Final metrics saved in    : {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")


if __name__ == "__main__":
    main()
