"""
Exploratory Data Analysis functions.
"""
import numpy as np
import pandas as pd
from src.config import PLOT_DIR, CORR_TOP_FEATURES
from src.logger import get_logger, LOG_FILE
from src.utils import header
from src.plotting import (
    plot_target_distribution, 
    plot_corr_heatmap
)

logger = get_logger("eda")


def eda_summary(X, y, X_eval):
    """Print consolidated EDA summary report."""
    header("EDA SUMMARY REPORT")
    
    logger.info("Dataset Overview:")
    logger.info(f"  Training samples: {len(X):,}")
    logger.info(f"  Evaluation samples: {len(X_eval):,}")
    logger.info(f"  Features: {X.shape[1]}")
    
    logger.info("Data Quality Checks:")
    null_count = X.isnull().sum().sum()
    null_pct = (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
    logger.info(f"  Null values: {null_count} ({null_pct:.2f}%)")
    
    dup_rows = X.duplicated().sum()
    logger.info(f"  Duplicate rows: {dup_rows}")
    
    col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_cols = col_hash.duplicated().sum()
    logger.info(f"  Duplicate columns: {dup_cols}")
    
    logger.info("Target Statistics:")
    logger.info(f"  Mean: {y.mean():.6f}")
    logger.info(f"  Median: {np.median(y):.6f}")
    logger.info(f"  Std: {y.std():.6f}")
    logger.info(f"  Range: [{y.min():.6f}, {y.max():.6f}]")
    
    logger.info("All EDA checks completed!")
    logger.info(f"Log file saved to: {LOG_FILE}")


def mean_median_std_check(X: pd.DataFrame, y: np.ndarray):
    """Perform mean/median/std statistical checks on features and target."""
    header("MEAN / MEDIAN / STD CHECK (X + y)")
    
    X_num = X.select_dtypes(include=[np.number])
    logger.info(f"Numeric columns: {X_num.shape[1]}")
    
    if X_num.shape[1] > 0:
        means = X_num.mean().sort_values(ascending=False)
        medians = X_num.median().sort_values(ascending=False)
        stds = X_num.std().sort_values(ascending=False)
        
        logger.info("Top 10 feature means:")
        logger.info(str(means.head(10).round(6)))
        logger.info("Top 10 feature medians:")
        logger.info(str(medians.head(10).round(6)))
        logger.info("Top 10 feature std:")
        logger.info(str(stds.head(10).round(6)))
    
    y = np.asarray(y)
    logger.info("y stats:")
    logger.info(f"mean={y.mean():.6f} median={np.median(y):.6f} std={y.std():.6f} min={y.min():.6f} max={y.max():.6f}")


def duplication_checks(X: pd.DataFrame):
    """Check for duplicate rows and columns."""
    header("DUPLICATION CHECKS (ROWS + COLUMNS)")
    
    dup_rows = int(X.duplicated().sum())
    logger.info(f"Duplicate rows: {dup_rows}")
    
    col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_col_mask = col_hash.duplicated(keep=False)
    dup_cols = X.columns[dup_col_mask].tolist()
    
    if not dup_cols:
        logger.info("Duplicate columns: none detected")
        return
    
    logger.info(f"Potential duplicate columns found: {len(dup_cols)}")
    groups = {}
    for c in dup_cols:
        groups.setdefault(col_hash[c], []).append(c)
    
    shown = 0
    for _, cols in groups.items():
        if len(cols) > 1:
            logger.info(f"Duplicate column group: {cols}")
            shown += 1
            if shown >= 10:
                logger.info("... (showing first 10 groups)")
                break


def leakage_overlap_check(X_a: pd.DataFrame, X_b: pd.DataFrame, name_a="A", name_b="B", max_rows=3000):
    """Check for data leakage between splits."""
    A = X_a.iloc[:max_rows]
    B = X_b.iloc[:max_rows]
    
    ha = pd.util.hash_pandas_object(A, index=False).values
    hb = pd.util.hash_pandas_object(B, index=False).values
    
    overlap = len(set(ha).intersection(set(hb)))
    logger.info(f"Leak-check overlap {name_a} vs {name_b} (first {max_rows} rows): {overlap}")


def basic_data_distribution(X: pd.DataFrame):
    """Analyze data distribution, missingness, and variation."""
    header("DATA DISTRIBUTION (X) + MISSINGNESS + VARIATION")
    
    logger.info(f"Shape: {X.shape}")
    
    missing = X.isna().mean().sort_values(ascending=False)
    logger.info("Top 15 columns by missing %:")
    logger.info(str((missing.head(15) * 100).round(2)))
    
    X_num = X.select_dtypes(include=[np.number])
    logger.info(f"Numeric columns: {X_num.shape[1]}")
    
    if X_num.shape[1] == 0:
        logger.info("No numeric columns detected.")
        return
    
    variances = X_num.var(numeric_only=True).sort_values(ascending=False)
    stds = X_num.std(numeric_only=True).sort_values(ascending=False)
    
    logger.info("Top 15 features by variance:")
    logger.info(str(variances.head(15).round(6)))
    logger.info("Top 15 features by std:")
    logger.info(str(stds.head(15).round(6)))


def target_distribution(y_train, y_val, y_test):
    """Check and plot target distribution across splits."""
    header("TARGET DISTRIBUTION CHECK (train/val/test)")
    
    def describe_y(name, arr):
        arr = np.asarray(arr)
        logger.info(
            f"{name:<7} mean={np.mean(arr):.6f} std={np.std(arr):.6f} "
            f"min={np.min(arr):.6f} max={np.max(arr):.6f}"
        )
    
    describe_y("train", y_train)
    describe_y("val", y_val)
    describe_y("test", y_test)
    
    plot_target_distribution(y_train, "Target distribution (Train)", "target_train_hist.png")
    plot_target_distribution(y_val, "Target distribution (Val)", "target_val_hist.png")
    plot_target_distribution(y_test, "Target distribution (Test)", "target_test_hist.png")
    logger.info(f"Saved target histograms to: {PLOT_DIR}")


def outlier_check_report_only(X: pd.DataFrame):
    """Report outliers using IQR method (no handling)."""
    header("OUTLIER CHECK (REPORT ONLY - NO HANDLING)")
    
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        logger.info("No numeric columns -> skipping outlier check.")
        return
    
    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    outlier_counts = ((X_num.lt(lower)) | (X_num.gt(upper))).sum().sort_values(ascending=False)
    outlier_rates = (outlier_counts / len(X_num)).sort_values(ascending=False)
    
    logger.info("Top 15 features by IQR-outlier rate:")
    logger.info(str((outlier_rates.head(15) * 100).round(2)))


def correlation_analysis(X: pd.DataFrame, y: np.ndarray):
    """Analyze feature-target and feature-feature correlations."""
    header("CORRELATION (NUMERIC ONLY): feature-target + feature-feature")
    
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        logger.info("No numeric columns -> skipping correlation analysis.")
        return
    
    import matplotlib.pyplot as plt
    
    y_s = pd.Series(y, name="target")
    corr_to_target = X_num.corrwith(y_s).sort_values(key=lambda s: s.abs(), ascending=False)
    
    logger.info("Top 20 features by |corr(feature, target)|:")
    logger.info(str(corr_to_target.head(20).round(6)))
    
    # Plot correlation distribution
    plt.figure()
    plt.hist(corr_to_target.dropna().values, bins=60)
    plt.title("Distribution of corr(feature, target)")
    plt.xlabel("correlation")
    plt.ylabel("count")
    plt.tight_layout()
    import os
    plt.savefig(os.path.join(PLOT_DIR, "corr_feature_target_distribution.png"), dpi=160)
    plt.close()
    
    # Heatmap of top features
    top_feats = corr_to_target.dropna().head(min(CORR_TOP_FEATURES, X_num.shape[1])).index.tolist()
    corr_mat = X_num[top_feats].corr()
    plot_corr_heatmap(
        corr_mat,
        f"Feature-Feature Corr (Top {len(top_feats)} by |corr w/ target|)",
        f"corr_heatmap_top{len(top_feats)}.png",
    )
    
    logger.info(f"Saved correlation plots to: {PLOT_DIR}")
