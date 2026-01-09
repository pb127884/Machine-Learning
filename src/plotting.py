# plotting.py - visualization functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import PLOT_DIR
from src.logger import get_logger

log = get_logger("plotting")

def plot_target_distribution(y, title, filename):
    plt.figure()
    plt.hist(y, bins=50)
    plt.title(title)
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=160)
    plt.close()
    log.info(f"saved {path}")

def plot_feature_histograms(X_num, cols, prefix="hist"):
    # plot histogram for each feature
    log.info(f"plotting {len(cols)} feature histograms")
    for col in cols:
        plt.figure()
        plt.hist(X_num[col].dropna().values, bins=50)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_{col}.png"), dpi=160)
        plt.close()

def plot_corr_heatmap(corr, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()

def plot_r2_comparison(name, r2_train, r2_val, r2_test, filename="r2_best_model.png"):
    plt.figure(figsize=(6, 4))
    x = np.arange(3)
    plt.bar(x, [r2_train, r2_val, r2_test])
    plt.xticks(x, ["Train", "Val", "Test"])
    plt.ylabel("R²")
    plt.title(f"R² for {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()
    log.info(f"saved r2 plot")
