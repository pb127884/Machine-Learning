# eda.py - exploratory data analysis functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.config import PLOT_DIR, CORR_TOP_FEATURES
from src.logger import get_logger, LOG_FILE
from src.utils import header
from src.plotting import plot_target_distribution, plot_corr_heatmap

log = get_logger("eda")

def eda_summary(X, y, X_eval):
    header("EDA SUMMARY")
    
    log.info("Dataset:")
    log.info(f"  train samples: {len(X):,}")
    log.info(f"  eval samples: {len(X_eval):,}")
    log.info(f"  features: {X.shape[1]}")
    
    # check for nulls and duplicates
    null_count = X.isnull().sum().sum()
    null_pct = (null_count / (X.shape[0] * X.shape[1])) * 100
    dup_rows = X.duplicated().sum()
    
    log.info(f"  null values: {null_count} ({null_pct:.2f}%)")
    log.info(f"  duplicate rows: {dup_rows}")
    
    log.info("Target stats:")
    log.info(f"  mean: {y.mean():.6f}, median: {np.median(y):.6f}")
    log.info(f"  std: {y.std():.6f}")
    log.info(f"  range: [{y.min():.6f}, {y.max():.6f}]")

def mean_median_std_check(X, y):
    header("MEAN/MEDIAN/STD CHECK")
    
    X_num = X.select_dtypes(include=[np.number])
    log.info(f"numeric columns: {X_num.shape[1]}")
    
    if X_num.shape[1] > 0:
        means = X_num.mean().sort_values(ascending=False)
        stds = X_num.std().sort_values(ascending=False)
        
        log.info("top 10 by mean:")
        log.info(str(means.head(10).round(4)))
        log.info("top 10 by std:")
        log.info(str(stds.head(10).round(4)))
    
    # target stats
    y = np.asarray(y)
    log.info(f"target: mean={y.mean():.4f} median={np.median(y):.4f} std={y.std():.4f}")

def duplication_checks(X):
    header("DUPLICATION CHECK")
    
    dup_rows = X.duplicated().sum()
    log.info(f"duplicate rows: {dup_rows}")
    
    # check for duplicate columns using hash
    col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_cols = col_hash.duplicated().sum()
    
    if dup_cols == 0:
        log.info("no duplicate columns found")
    else:
        log.info(f"found {dup_cols} duplicate columns")

def leakage_overlap_check(X_a, X_b, name_a, name_b, max_rows=3000):
    # check if same rows appear in both sets
    A = X_a.iloc[:max_rows]
    B = X_b.iloc[:max_rows]
    
    ha = pd.util.hash_pandas_object(A, index=False).values
    hb = pd.util.hash_pandas_object(B, index=False).values
    
    overlap = len(set(ha) & set(hb))
    log.info(f"overlap {name_a} vs {name_b}: {overlap}")

def basic_data_distribution(X):
    header("DATA DISTRIBUTION")
    
    log.info(f"shape: {X.shape}")
    
    # check missing values
    missing = X.isna().mean().sort_values(ascending=False)
    log.info("missing % (top 10):")
    log.info(str((missing.head(10) * 100).round(2)))
    
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] > 0:
        variances = X_num.var().sort_values(ascending=False)
        log.info("top 10 by variance:")
        log.info(str(variances.head(10).round(4)))

def target_distribution(y_train, y_val, y_test):
    header("TARGET DISTRIBUTION")
    
    for name, arr in [("train", y_train), ("val", y_val), ("test", y_test)]:
        arr = np.asarray(arr)
        log.info(f"{name}: mean={arr.mean():.4f} std={arr.std():.4f}")
    
    # save plots
    plot_target_distribution(y_train, "Train Target", "target_train_hist.png")
    plot_target_distribution(y_val, "Val Target", "target_val_hist.png")
    plot_target_distribution(y_test, "Test Target", "target_test_hist.png")

def outlier_check(X):
    header("OUTLIER CHECK")
    
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        log.info("no numeric columns")
        return
    
    # IQR method
    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    outlier_rate = ((X_num < lower) | (X_num > upper)).sum() / len(X_num)
    outlier_rate = outlier_rate.sort_values(ascending=False)
    
    log.info("outlier rate (top 10):")
    log.info(str((outlier_rate.head(10) * 100).round(2)))

def correlation_analysis(X, y):
    header("CORRELATION ANALYSIS")
    
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        log.info("no numeric columns")
        return
    
    # correlation with target
    y_s = pd.Series(y, name="target")
    corr = X_num.corrwith(y_s).sort_values(key=lambda s: s.abs(), ascending=False)
    
    log.info("top 20 by |correlation|:")
    log.info(str(corr.head(20).round(6)))
    
    # plot histogram
    plt.figure()
    plt.hist(corr.dropna().values, bins=60)
    plt.title("Feature-Target Correlation Distribution")
    plt.xlabel("correlation")
    plt.ylabel("count")
    plt.savefig(os.path.join(PLOT_DIR, "corr_distribution.png"), dpi=160)
    plt.close()
    
    # heatmap of top features
    top = corr.dropna().head(min(CORR_TOP_FEATURES, X_num.shape[1])).index.tolist()
    corr_mat = X_num[top].corr()
    plot_corr_heatmap(corr_mat, f"Top {len(top)} Features Correlation", f"corr_heatmap.png")
    
    log.info(f"saved correlation plots")
