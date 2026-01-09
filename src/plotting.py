"""
Plotting functions for visualization.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import PLOT_DIR
from src.logger import get_logger

logger = get_logger("plotting")


def plot_target_distribution(y, title, filename):
    """Plot histogram of target distribution."""
    logger.info(f"Plotting target distribution: {filename}")
    plt.figure()
    plt.hist(y, bins=50)
    plt.title(title)
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=160)
    plt.close()
    logger.info(f"Saved: {filepath}")


def plot_feature_histograms(X_num: pd.DataFrame, top_cols, filename_prefix="hist"):
    """Plot histograms for top features."""
    logger.info(f"Plotting {len(top_cols)} feature histograms")
    for col in top_cols:
        plt.figure()
        plt.hist(X_num[col].dropna().values, bins=50)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        filepath = os.path.join(PLOT_DIR, f"{filename_prefix}_{col}.png")
        plt.savefig(filepath, dpi=160)
        plt.close()
    logger.info(f"Saved feature histograms to: {PLOT_DIR}")


def plot_corr_heatmap(corr: pd.DataFrame, title: str, filename: str):
    """Plot correlation heatmap."""
    logger.info(f"Plotting correlation heatmap: {filename}")
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=200)
    plt.close()
    logger.info(f"Saved: {filepath}")


def plot_r2_for_best(best_name, r2_train, r2_val, r2_test, filename="r2_best_model.png"):
    """Plot R² comparison for best model."""
    logger.info(f"Plotting R² for best model: {best_name}")
    plt.figure(figsize=(6, 4))
    labels = ["Train", "Val", "Test"]
    values = [r2_train, r2_val, r2_test]
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("R²")
    plt.title(f"R² for Best Model: {best_name}")
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=200)
    plt.close()
    logger.info(f"Saved: {filepath}")
