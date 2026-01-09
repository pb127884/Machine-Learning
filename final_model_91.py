#!/usr/bin/env python3
"""
final_model_91.py - Final Training Script for Task 1 (target01 prediction)

ID: 91
Target: target01

This script trains the final HistGradientBoostingRegressor model with optimized
hyperparameters and generates predictions for the evaluation dataset.

Best Model: HistGradientBoostingRegressor
- R² (test): 0.7860
- RMSE (test): 0.1102

Usage:
    python final_model_91.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
ID = 91

# Data paths
DATA_X_PATH = "data/dataset_91.csv"
DATA_Y_PATH = "data/target_91.csv"
DATA_EVAL_PATH = "data/EVAL_91.csv"
TARGET_COL = "target01"

# Output paths
OUTPUT_DIR = "outputs"
PREDICTIONS_FILE = f"EVAL_target01_{ID}.csv"

# =============================================================================
# BEST HYPERPARAMETERS (tuned for low overfitting, gap < 0.10)
# =============================================================================

BEST_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 5,              # Moderate depth
    'max_iter': 500,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 30,      # Moderate to reduce overfitting
    'l2_regularization': 0.5,    # Moderate regularization
    'max_bins': 128,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 15,
    'random_state': SEED
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data():
    """Load training and evaluation data."""
    print("Loading data...")
    X = pd.read_csv(DATA_X_PATH)
    y = pd.read_csv(DATA_Y_PATH)[TARGET_COL].values
    X_eval = pd.read_csv(DATA_EVAL_PATH)
    
    print(f"  Training data: {X.shape}")
    print(f"  Evaluation data: {X_eval.shape}")
    
    return X, y, X_eval


def create_model():
    """Create the final model pipeline."""
    
    # Preprocessing pipeline
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var_threshold", VarianceThreshold(threshold=0.0)),
    ])
    
    # Full pipeline with model
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", HistGradientBoostingRegressor(**BEST_PARAMS))
    ])
    
    return model


def evaluate_model(y_true, y_pred, split_name=""):
    """Calculate and print evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    if split_name:
        print(f"  {split_name}:")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    R²:   {r2:.6f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def split_data(X, y, seed=SEED):
    """Split data into train/val/test sets."""
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print(f"FINAL MODEL TRAINING - Task 1 (target01)")
    print(f"ID: {ID}")
    print("=" * 70)
    
    # Load data
    X, y, X_eval = load_data()
    
    # Split for validation
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Create and train model on train set only (for validation)
    print("\n" + "=" * 70)
    print("TRAINING MODEL (for validation)")
    print("=" * 70)
    
    print("\nModel: HistGradientBoostingRegressor")
    print("Hyperparameters:")
    for k, v in BEST_PARAMS.items():
        print(f"  {k}: {v}")
    
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate on all splits
    print("\nEvaluation (train set only):")
    train_metrics = evaluate_model(y_train, model.predict(X_train), "Train")
    val_metrics = evaluate_model(y_val, model.predict(X_val), "Validation")
    test_metrics = evaluate_model(y_test, model.predict(X_test), "Test")
    
    # Retrain on train+val for final model
    print("\n" + "=" * 70)
    print("FINAL MODEL (train + validation)")
    print("=" * 70)
    
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    final_model = create_model()
    final_model.fit(X_trainval, y_trainval)
    
    print("\nEvaluation (train+val set):")
    evaluate_model(y_trainval, final_model.predict(X_trainval), "Train+Val")
    final_test = evaluate_model(y_test, final_model.predict(X_test), "Test")
    
    # Train on ALL data for predictions
    print("\n" + "=" * 70)
    print("PRODUCTION MODEL (all data)")
    print("=" * 70)
    
    production_model = create_model()
    production_model.fit(X, y)
    
    print("\nTraining metrics (all data):")
    evaluate_model(y, production_model.predict(X), "Full Training Set")
    
    # Generate predictions
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)
    
    predictions = production_model.predict(X_eval)
    
    print(f"\nPrediction statistics:")
    print(f"  Count: {len(predictions)}")
    print(f"  Mean:  {predictions.mean():.6f}")
    print(f"  Std:   {predictions.std():.6f}")
    print(f"  Min:   {predictions.min():.6f}")
    print(f"  Max:   {predictions.max():.6f}")
    
    # Save predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, PREDICTIONS_FILE)
    
    output_df = pd.DataFrame({TARGET_COL: predictions})
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Also save to root for submission
    output_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Predictions also saved to: {PREDICTIONS_FILE}")
    
    # Save final metrics
    metrics_df = pd.DataFrame([{
        'model': 'HistGradientBoostingRegressor',
        'id': ID,
        'target': TARGET_COL,
        'test_rmse': final_test['rmse'],
        'test_r2': final_test['r2'],
        'test_mae': final_test['mae'],
        'hyperparameters': str(BEST_PARAMS)
    }])
    metrics_path = os.path.join(OUTPUT_DIR, 'final_model_metrics_91.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Test Performance:")
    print(f"  R²:   {final_test['r2']:.4f}")
    print(f"  RMSE: {final_test['rmse']:.4f}")
    print(f"\nOutput file: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
