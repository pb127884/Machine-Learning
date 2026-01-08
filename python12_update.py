"""
python12_update.py - ML Pipeline (Updated Version)

Changes from python12.py:
- Moved EDA after split so we only look at training data
- Fixed stacking to have preprocessing inside each base model
- After retraining on train+val, only report train and test (not val)
- Added leakage audit to check for suspicious features
- Reduced hyperparameter grid to avoid overfitting to validation set
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor


# ----------------------------
# CONFIG
# ----------------------------
SEED = 42

DATA_X_PATH = "dataset_91.csv"
DATA_Y_PATH = "target_91.csv"
DATA_EVAL_PATH = "EVAL_91.csv"
TARGET_COL = "target01"

OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUT_DIR, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def setup_logging():
    """Setup dual logging to console and file."""
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

_original_print = print
def print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)

MAX_FEATURE_HISTS = 20
CORR_TOP_FEATURES = 50


def eda_summary(X_train, y_train, X_eval):
    """Print consolidated EDA summary report - ON TRAIN DATA ONLY."""
    header("EDA SUMMARY REPORT (TRAIN DATA ONLY)")
    
    print("Dataset Overview:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Evaluation samples: {len(X_eval):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    print("\nData Quality Checks (on training data):")
    null_count = X_train.isnull().sum().sum()
    null_pct = (X_train.isnull().sum().sum() / (X_train.shape[0] * X_train.shape[1])) * 100
    print(f"  Null values: {null_count} ({null_pct:.2f}%)")
    
    dup_rows = X_train.duplicated().sum()
    print(f"  Duplicate rows: {dup_rows}")
    
    col_hash = X_train.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_cols = col_hash.duplicated().sum()
    print(f"  Duplicate columns: {dup_cols}")
    
    print("\nTarget Statistics (training only):")
    print(f"  Mean: {y_train.mean():.6f}")
    print(f"  Median: {np.median(y_train):.6f}")
    print(f"  Std: {y_train.std():.6f}")
    print(f"  Range: [{y_train.min():.6f}, {y_train.max():.6f}]")
    
    print("\nAll EDA checks completed!")
    print(f"Log file saved to: {LOG_FILE}")

def header(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------
# PLOTTING
# ----------------------------
def plot_target_distribution(y, title, filename):
    plt.figure()
    plt.hist(y, bins=50)
    plt.title(title)
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=160)
    plt.close()


def plot_feature_histograms(X_num: pd.DataFrame, top_cols, filename_prefix="hist"):
    for col in top_cols:
        plt.figure()
        plt.hist(X_num[col].dropna().values, bins=50)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{filename_prefix}_{col}.png"), dpi=160)
        plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()


def plot_r2_for_best(best_name, r2_train, r2_test, filename="r2_best_model.png"):
    """FIX C: Only plot Train and Test (no Val after retraining)."""
    plt.figure(figsize=(6, 4))
    labels = ["Train", "Test"]
    values = [r2_train, r2_test]
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("R2")
    plt.title(f"R2 for Best Model: {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()
    print(f"[OK] Saved Best Model R2 graph to: {os.path.join(PLOT_DIR, filename)}")


# ----------------------------
# DATA CHECKS (RUN ON TRAIN ONLY)
# ----------------------------
def mean_median_std_check(X: pd.DataFrame, y: np.ndarray, label="TRAIN"):
    """FIX A: Run on train data only."""
    header(f"MEAN / MEDIAN / STD CHECK ({label} DATA ONLY)")

    X_num = X.select_dtypes(include=[np.number])
    print("Numeric columns:", X_num.shape[1])

    if X_num.shape[1] > 0:
        means = X_num.mean().sort_values(ascending=False)
        medians = X_num.median().sort_values(ascending=False)
        stds = X_num.std().sort_values(ascending=False)

        print("\nTop 10 feature means:")
        print(means.head(10).round(6))
        print("\nTop 10 feature medians:")
        print(medians.head(10).round(6))
        print("\nTop 10 feature std:")
        print(stds.head(10).round(6))

    y = np.asarray(y)
    print("\ny stats:")
    print(f"mean={y.mean():.6f} median={np.median(y):.6f} std={y.std():.6f} min={y.min():.6f} max={y.max():.6f}")


def duplication_checks(X: pd.DataFrame, label="TRAIN"):
    """FIX A: Run on train data only."""
    header(f"DUPLICATION CHECKS ({label} DATA ONLY)")
    dup_rows = int(X.duplicated().sum())
    print("Duplicate rows:", dup_rows)

    col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_col_mask = col_hash.duplicated(keep=False)
    dup_cols = X.columns[dup_col_mask].tolist()

    if not dup_cols:
        print("Duplicate columns: none detected")
        return

    print("Potential duplicate columns found:", len(dup_cols))
    groups = {}
    for c in dup_cols:
        groups.setdefault(col_hash[c], []).append(c)

    shown = 0
    for _, cols in groups.items():
        if len(cols) > 1:
            print("Duplicate column group:", cols)
            shown += 1
            if shown >= 10:
                print("... (showing first 10 groups)")
                break


def leakage_overlap_check(X_a: pd.DataFrame, X_b: pd.DataFrame, name_a="A", name_b="B", max_rows=3000):
    """Checks if EXACT same rows appear in both splits (leakage risk)."""
    A = X_a.iloc[:max_rows]
    B = X_b.iloc[:max_rows]

    ha = pd.util.hash_pandas_object(A, index=False).values
    hb = pd.util.hash_pandas_object(B, index=False).values

    overlap = len(set(ha).intersection(set(hb)))
    print(f"Leak-check overlap {name_a} vs {name_b} (first {max_rows} rows): {overlap}")


def basic_data_distribution(X: pd.DataFrame, label="TRAIN"):
    """FIX A: Run on train data only."""
    header(f"DATA DISTRIBUTION ({label} DATA ONLY)")
    print("Shape:", X.shape)

    missing = X.isna().mean().sort_values(ascending=False)
    print("\nTop 15 columns by missing %:")
    print((missing.head(15) * 100).round(2))

    X_num = X.select_dtypes(include=[np.number])
    print("\nNumeric columns:", X_num.shape[1])

    if X_num.shape[1] == 0:
        print("No numeric columns detected.")
        return

    variances = X_num.var(numeric_only=True).sort_values(ascending=False)
    stds = X_num.std(numeric_only=True).sort_values(ascending=False)

    print("\nTop 15 features by variance:")
    print(variances.head(15).round(6))

    print("\nTop 15 features by std:")
    print(stds.head(15).round(6))


def target_distribution(y_train, y_val, y_test):
    header("TARGET DISTRIBUTION CHECK (train/val/test)")

    def describe_y(name, arr):
        arr = np.asarray(arr)
        print(
            f"{name:<7} mean={np.mean(arr):.6f} std={np.std(arr):.6f} "
            f"min={np.min(arr):.6f} max={np.max(arr):.6f}"
        )

    describe_y("train", y_train)
    describe_y("val", y_val)
    describe_y("test", y_test)

    plot_target_distribution(y_train, "Target distribution (Train)", "target_train_hist.png")
    plot_target_distribution(y_val, "Target distribution (Val)", "target_val_hist.png")
    plot_target_distribution(y_test, "Target distribution (Test)", "target_test_hist.png")
    print(f"[OK] Saved target histograms to: {PLOT_DIR}")


def outlier_check_report_only(X: pd.DataFrame, label="TRAIN"):
    """FIX A: Run on train data only."""
    header(f"OUTLIER CHECK ({label} DATA ONLY)")

    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        print("No numeric columns -> skipping outlier check.")
        return

    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_counts = ((X_num.lt(lower)) | (X_num.gt(upper))).sum().sort_values(ascending=False)
    outlier_rates = (outlier_counts / len(X_num)).sort_values(ascending=False)

    print("\nTop 15 features by IQR-outlier rate:")
    print((outlier_rates.head(15) * 100).round(2))


def correlation_analysis(X: pd.DataFrame, y: np.ndarray, label="TRAIN"):
    """FIX A: Run on TRAIN data only - correlation analysis."""
    header(f"CORRELATION ANALYSIS ({label} DATA ONLY)")

    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        print("No numeric columns -> skipping correlation analysis.")
        return

    y_s = pd.Series(y, name="target")
    corr_to_target = X_num.corrwith(y_s).sort_values(key=lambda s: s.abs(), ascending=False)

    print("\nTop 20 features by |corr(feature, target)|:")
    print(corr_to_target.head(20).round(6))

    # Check for suspiciously high correlation (potential leakage)
    high_corr = corr_to_target[corr_to_target.abs() > 0.95]
    if len(high_corr) > 0:
        print("\n[WARNING] Features with |correlation| > 0.95 (potential leakage):")
        print(high_corr)

    plt.figure()
    plt.hist(corr_to_target.dropna().values, bins=60)
    plt.title("Distribution of corr(feature, target) - TRAIN ONLY")
    plt.xlabel("correlation")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "corr_feature_target_distribution.png"), dpi=160)
    plt.close()

    top_feats = corr_to_target.dropna().head(min(CORR_TOP_FEATURES, X_num.shape[1])).index.tolist()
    corr_mat = X_num[top_feats].corr()
    plot_corr_heatmap(
        corr_mat,
        f"Feature-Feature Corr (Top {len(top_feats)} by |corr w/ target|) - TRAIN ONLY",
        f"corr_heatmap_top{len(top_feats)}.png",
    )

    print(f"[OK] Saved correlation plots to: {PLOT_DIR}")


def leakage_audit(X_train, y_train):
    """
    Check for target leakage in features.
    This runs extra checks beyond just correlation.
    """
    header("LEAKAGE AUDIT (checking for suspicious features)")
    
    X_num = X_train.select_dtypes(include=[np.number]).copy()
    y_arr = np.asarray(y_train)
    
    suspicious = []
    
    # Check 1: Very high correlation (> 0.95)
    print("\n1) Checking for |correlation| > 0.95 with target...")
    y_s = pd.Series(y_arr, name="target")
    corr_to_target = X_num.corrwith(y_s)
    high_corr = corr_to_target[corr_to_target.abs() > 0.95]
    if len(high_corr) > 0:
        print(f"   WARNING: {len(high_corr)} features with |corr| > 0.95:")
        for col, c in high_corr.items():
            print(f"      {col}: {c:.4f}")
            suspicious.append((col, "high_corr", c))
    else:
        print("   OK: No features with |corr| > 0.95")
    
    # Check 2: Near-perfect match (feature ~= target)
    print("\n2) Checking for near-exact match with target...")
    for col in X_num.columns:
        match_rate = np.mean(np.isclose(X_num[col].values, y_arr, atol=1e-6))
        if match_rate > 0.9:
            print(f"   WARNING: {col} matches target in {match_rate*100:.1f}% of rows")
            suspicious.append((col, "exact_match", match_rate))
    if not any(s[1] == "exact_match" for s in suspicious):
        print("   OK: No features match target exactly")
    
    # Check 3: Row index baseline (can we predict from row order?)
    print("\n3) Checking if row index predicts target (time/order leakage)...")
    row_idx = np.arange(len(y_arr)).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    lr_idx = LinearRegression()
    lr_idx.fit(row_idx, y_arr)
    idx_r2 = lr_idx.score(row_idx, y_arr)
    if idx_r2 > 0.1:
        print(f"   WARNING: Row index alone gives R2 = {idx_r2:.4f} (may indicate time leakage)")
        suspicious.append(("row_index", "time_leak", idx_r2))
    else:
        print(f"   OK: Row index R2 = {idx_r2:.4f} (no time leakage detected)")
    
    # Check 4: Columns with suspicious names
    print("\n4) Checking for suspicious column names...")
    bad_names = ["target", "label", "y", "outcome", "score", "result"]
    for col in X_num.columns:
        col_lower = col.lower()
        for bad in bad_names:
            if bad in col_lower:
                print(f"   WARNING: Column '{col}' has suspicious name (contains '{bad}')")
                suspicious.append((col, "bad_name", bad))
    if not any(s[1] == "bad_name" for s in suspicious):
        print("   OK: No suspicious column names")
    
    # Summary
    print("\n" + "-" * 50)
    if suspicious:
        print(f"AUDIT RESULT: {len(suspicious)} potential issues found")
        for col, reason, val in suspicious:
            print(f"   - {col}: {reason} ({val})")
    else:
        print("AUDIT RESULT: No leakage detected")
    print("-" * 50)
    
    return suspicious


def permutation_leakage_test(X_train, y_train, X_val, y_val, seed=42):
    """
    Permutation test: shuffle y_train and check if model still performs well.
    If R² stays high after shuffling, there's likely target leakage somewhere.
    Expected: R² should be near 0 or negative with shuffled target.
    """
    header("PERMUTATION LEAKAGE TEST")
    print("Shuffling y_train and fitting a quick model...")
    print("If R² is still high after shuffle, something is leaking target info.")
    
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    # Shuffle y_train
    rng = np.random.RandomState(seed)
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)
    
    # Quick model on shuffled target
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=seed)
    model.fit(X_train, y_train_shuffled)
    
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    r2_train = r2_score(y_train_shuffled, pred_train)
    r2_val = r2_score(y_val, pred_val)  # val target is NOT shuffled
    
    print(f"\nWith SHUFFLED y_train:")
    print(f"  Train R2 (shuffled): {r2_train:.4f}")
    print(f"  Val R2 (real):       {r2_val:.4f}")
    
    # Now fit on real target for comparison
    model_real = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=seed)
    model_real.fit(X_train, y_train)
    pred_val_real = model_real.predict(X_val)
    r2_val_real = r2_score(y_val, pred_val_real)
    
    print(f"\nWith REAL y_train:")
    print(f"  Val R2 (real):       {r2_val_real:.4f}")
    
    print("\n" + "-" * 50)
    if r2_val > 0.3:
        print(f"WARNING: Shuffled model has Val R2 = {r2_val:.4f}")
        print("         This might indicate target leakage!")
    else:
        print(f"OK: Shuffled model has Val R2 = {r2_val:.4f} (expected near 0)")
    print("-" * 50)
    
    return r2_val


def drift_check(X_train, X_val, top_n=20):
    """
    Check for distribution drift between train and validation sets.
    Large differences in mean/std can explain gaps in performance.
    """
    header("TRAIN vs VAL DISTRIBUTION DRIFT CHECK")
    
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_val_num = X_val.select_dtypes(include=[np.number])
    
    # Compute mean and std for each feature
    train_mean = X_train_num.mean()
    val_mean = X_val_num.mean()
    train_std = X_train_num.std()
    val_std = X_val_num.std()
    
    # Compute drift as normalized difference
    mean_diff = (val_mean - train_mean).abs()
    std_ratio = (val_std / train_std.replace(0, np.nan)).fillna(1)
    
    # Normalize by train std
    normalized_drift = (mean_diff / train_std.replace(0, np.nan)).fillna(0)
    
    # Sort by drift
    drift_sorted = normalized_drift.sort_values(ascending=False)
    
    print(f"\nTop {top_n} features by mean drift (normalized by train std):")
    print("-" * 50)
    
    high_drift_count = 0
    for i, (col, drift) in enumerate(drift_sorted.head(top_n).items()):
        status = "WARNING" if drift > 0.2 else "OK"
        if drift > 0.2:
            high_drift_count += 1
        print(f"  {i+1:2d}. {col}: drift = {drift:.4f} [{status}]")
    
    print("\n" + "-" * 50)
    if high_drift_count > 5:
        print(f"WARNING: {high_drift_count} features have drift > 0.2")
        print("         This may explain train-val performance gap.")
    else:
        print(f"OK: Only {high_drift_count} features have drift > 0.2")
    print("-" * 50)
    
    return drift_sorted


# ----------------------------
# STRATIFIED SPLIT
# ----------------------------
def make_strat_bins(y, n_bins=10):
    y = np.asarray(y)
    try:
        b = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        return np.asarray(b, dtype=int)
    except Exception:
        b = pd.qcut(pd.Series(y).rank(method="average"), q=n_bins, labels=False, duplicates="drop")
        return np.asarray(b, dtype=int)


def stratified_70_15_15_split(X, y, seed=42, n_bins=10):
    bins = make_strat_bins(y, n_bins=n_bins)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(X, bins))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y[temp_idx]

    bins_temp = make_strat_bins(y_temp, n_bins=n_bins)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(X_temp, bins_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------
# HYPER TUNING (NO-CV)
# ------------------------------------------------
def hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=42):
    """Hyper-tune HistGradientBoosting using ONLY the Validation set (no CV)."""

    header("HYPER TUNING (NO CV): HistGradientBoostingRegressor using VALIDATION ONLY")

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

    # Reduced grid to avoid overfitting to validation set
    # Original was 864 combos, now ~48 combos
    param_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [5, 7],
        "model__max_leaf_nodes": [31, 63],
        "model__min_samples_leaf": [20],
        "model__l2_regularization": [0.0, 0.1],
        "model__max_iter": [300, 500],
    }

    grid = list(ParameterGrid(param_grid))
    print(f"Total combinations to try: {len(grid)}")

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
            print(f"[{i:>5}/{len(grid)}] best_val_rmse={best_val_rmse:.6f}")

    header("BEST HGB PARAMS (by Validation RMSE) - NO CV")
    print("Best validation RMSE:", round(best_val_rmse, 6))
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tune_df = pd.DataFrame(tuning_rows).sort_values("val_rmse")
    tune_df.to_csv(os.path.join(OUT_DIR, "hgb_tuning_no_cv.csv"), index=False)
    print(f"[OK] Saved tuning table: {os.path.join(OUT_DIR, 'hgb_tuning_no_cv.csv')}")

    return best, best_params, tune_df


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    # ------------------------------------------------
    # 1) LOAD DATA
    # ------------------------------------------------
    header("LOAD DATA")

    X = pd.read_csv(DATA_X_PATH)
    y = pd.read_csv(DATA_Y_PATH)[TARGET_COL].values
    X_eval = pd.read_csv(DATA_EVAL_PATH)

    print("X shape     :", X.shape)
    print("y shape     :", y.shape)
    print("X_eval shape:", X_eval.shape)

    # ------------------------------------------------
    # 2) TRAIN / VAL / TEST SPLIT FIRST (before any EDA involving y)
    # ------------------------------------------------
    header("TRAIN / VAL / TEST SPLIT (70/15/15) - STRATIFIED ON TARGET")
    print("[FIX A] Split data BEFORE any EDA involving target correlation")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_70_15_15_split(
        X, y, seed=SEED, n_bins=10
    )

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # ------------------------------------------------
    # 3) EDA ON TRAIN DATA ONLY (FIX A)
    # ------------------------------------------------
    header("[FIX A] EDA ON TRAIN DATA ONLY (NO PEEKING AT VAL/TEST)")
    
    mean_median_std_check(X_train, y_train, label="TRAIN")
    duplication_checks(X_train, label="TRAIN")
    basic_data_distribution(X_train, label="TRAIN")
    outlier_check_report_only(X_train, label="TRAIN")
    correlation_analysis(X_train, y_train, label="TRAIN")
    leakage_audit(X_train, y_train)  # Extra checks for target leakage

    target_distribution(y_train, y_val, y_test)

    header("DATA LEAKAGE CHECK (OVERLAP BETWEEN SPLITS)")
    leakage_overlap_check(X_train, X_val, "train", "val")
    leakage_overlap_check(X_train, X_test, "train", "test")
    leakage_overlap_check(X_val, X_test, "val", "test")

    # Extra bulletproof checks
    permutation_leakage_test(X_train, y_train, X_val, y_val, seed=SEED) 
    drift_check(X_train, X_val, top_n=15)

    # Feature plotting (top variance numeric)
    header("FEATURE PLOTTING (Top variance numeric features)")
    X_num = X_train.select_dtypes(include=[np.number])
    if X_num.shape[1] > 0:
        variances = X_num.var().sort_values(ascending=False)
        top_cols = variances.head(min(MAX_FEATURE_HISTS, len(variances))).index.tolist()
        plot_feature_histograms(X_num, top_cols, filename_prefix="hist")
        print(f"[OK] Saved feature histograms to: {PLOT_DIR}")
    else:
        print("No numeric features detected -> skipping histograms.")

    # ------------------------------------------------
    # 4) PREPROCESSING PIPELINES
    # ------------------------------------------------
    basic_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])

    scaled_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])

    # ------------------------------------------------
    # 5) MODELS (FIX B: Stacking with preprocessing inside each base estimator)
    # ------------------------------------------------
    header("DEFINE MODELS (WITH FIX B: PREPROCESSING INSIDE BASE ESTIMATORS)")

    ridge = Pipeline([("prep", scaled_prep), ("model", Ridge(alpha=1.0, random_state=SEED))])
    dt = Pipeline([("prep", basic_prep), ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED))])
    rf = Pipeline([("prep", basic_prep), ("model", RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=1))])

    hgb = Pipeline([("prep", basic_prep),
                    ("model", HistGradientBoostingRegressor(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=SEED
                    ))])

    # FIX B: Each base estimator in Stacking has its own preprocessing pipeline
    # This prevents cross-fold leakage inside stacking CV
    stack = StackingRegressor(
        estimators=[
            ("ridge", Pipeline([
                ("prep", clone(scaled_prep)),
                ("model", Ridge(alpha=1.0, random_state=SEED))
            ])),
            ("dt", Pipeline([
                ("prep", clone(basic_prep)),
                ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED))
            ])),
            ("rf", Pipeline([
                ("prep", clone(basic_prep)),
                ("model", RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=1))
            ])),
        ],
        final_estimator=Ridge(alpha=1.0, random_state=SEED),
        passthrough=False,  # Changed to False for cleaner stacking
        cv=5
    )

    models = {
        "Ridge": ridge,
        "DecisionTree": dt,
        "RandomForest": rf,
        "HistGradientBoosting": hgb,
        "Stacking": stack,
    }

    # ------------------------------------------------
    # 6) TRAIN ON TRAIN -> EVALUATE ON VALIDATION (baseline selection)
    # ------------------------------------------------
    header("BASELINE MODEL SELECTION USING VALIDATION SET")

    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

        tr_pred = pipe.predict(X_train)
        va_pred = pipe.predict(X_val)

        tr = evaluate(y_train, tr_pred)
        va = evaluate(y_val, va_pred)

        gap_rmse = va["rmse"] - tr["rmse"]
        results.append((name, va["rmse"], va["mae"], va["r2"], tr["rmse"], tr["r2"], gap_rmse))

        print(
            f"{name:18s} | "
            f"TRAIN RMSE {tr['rmse']:.6f} R2 {tr['r2']:.6f} || "
            f"VAL RMSE {va['rmse']:.6f} R2 {va['r2']:.6f} || "
            f"GAP {gap_rmse:.6f}"
        )

    results_df = pd.DataFrame(
        results,
        columns=["model", "val_rmse", "val_mae", "val_r2", "train_rmse", "train_r2", "gap_rmse"]
    ).sort_values("val_rmse")

    header("BASELINE BEST MODEL (by Validation RMSE)")
    print(results_df.iloc[0])

    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUT_DIR, "baseline_metrics_table.csv"), index=False)
    print(f"[OK] Saved baseline metrics: {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")

    # ------------------------------------------------
    # 7) HYPER TUNING (NO CV) for HistGradientBoosting using VAL ONLY
    # ------------------------------------------------
    tuned_hgb, tuned_params, tune_df = hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=SEED)

    # Evaluate tuned model (fit train only)
    header("TUNED HGB (FIT TRAIN ONLY) -> TRAIN/VAL/TEST RESULTS")
    tuned_hgb.fit(X_train, y_train)

    pred_tr = tuned_hgb.predict(X_train)
    pred_va = tuned_hgb.predict(X_val)
    pred_te = tuned_hgb.predict(X_test)

    m_tr = evaluate(y_train, pred_tr)
    m_va = evaluate(y_val, pred_va)
    m_te = evaluate(y_test, pred_te)

    print("TUNED HGB (train-only fit)")
    print(f"Train RMSE: {m_tr['rmse']:.6f} | Train R2: {m_tr['r2']:.6f}")
    print(f"Val   RMSE: {m_va['rmse']:.6f} | Val   R2: {m_va['r2']:.6f}")
    print(f"Test  RMSE: {m_te['rmse']:.6f} | Test  R2: {m_te['r2']:.6f}")
    print(f"GAP (Val-Train RMSE): {m_va['rmse'] - m_tr['rmse']:.6f}")

    # ------------------------------------------------
    # 8) FINAL TRAIN (TRAIN + VAL) -> TEST ONCE (tuned model)
    # FIX C: Do NOT report Val metrics after retraining on train+val
    # ------------------------------------------------
    header("FINAL EVALUATION ON TEST SET (USED ONCE) - TUNED HGB TRAINED ON (TRAIN+VAL)")
    print("[FIX C] After training on train+val, we only report Train and Test metrics")
    print("[FIX C] Val metrics are NOT valid anymore since model has seen validation data")

    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = np.concatenate([y_train, y_val])

    tuned_hgb.fit(X_train_final, y_train_final)

    pred_train_final = tuned_hgb.predict(X_train_final)
    pred_test = tuned_hgb.predict(X_test)

    train_final_m = evaluate(y_train_final, pred_train_final)
    test_m = evaluate(y_test, pred_test)

    print("\nBEST MODEL: Tuned HistGradientBoostingRegressor (no CV)")
    print(f"Train (train+val) RMSE: {train_final_m['rmse']:.6f} | R2: {train_final_m['r2']:.6f}")
    print(f"Test            RMSE: {test_m['rmse']:.6f} | R2: {test_m['r2']:.6f}")
    print(f"GAP (Test-Train RMSE): {test_m['rmse'] - train_final_m['rmse']:.6f}")

    # FIX C: Only plot Train and Test (no Val)
    plot_r2_for_best("Tuned HistGradientBoosting (no CV)", train_final_m["r2"], test_m["r2"])

    # Save final metrics (without misleading val metrics)
    final_metrics = pd.DataFrame([{
        "model": "TunedHistGradientBoosting_noCV",
        "train_final_rmse": train_final_m["rmse"],
        "train_final_mae": train_final_m["mae"],
        "train_final_r2": train_final_m["r2"],
        "test_rmse": test_m["rmse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2"],
        "best_params": str(tuned_params),
    }])
    final_metrics.to_csv(os.path.join(OUT_DIR, "final_metrics_tuned_hgb_no_cv.csv"), index=False)
    print(f"[OK] Saved final metrics: {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")

    # ------------------------------------------------
    # 9) TRAIN ON FULL DATA -> PREDICT EVAL (OFFICIAL FILE)
    # ------------------------------------------------
    header("TRAIN ON FULL DATA + PREDICT EVAL (OFFICIAL SUBMISSION FILE) - TUNED HGB")

    tuned_hgb.fit(X, y)
    eval_pred = tuned_hgb.predict(X_eval)

    out = pd.DataFrame({TARGET_COL: eval_pred.astype(float)})

    official_path = os.path.join(OUT_DIR, "EVAL_target01_91.csv")
    out.to_csv(official_path, index=False)

    extra_path = os.path.join(OUT_DIR, "EVAL_target01_best.csv")
    out.to_csv(extra_path, index=False)

    print(f"[OK] Saved OFFICIAL: {official_path}")
    print(f"[OK] Saved EXTRA   : {extra_path}")
    print(out.head())

    # Print consolidated EDA summary (on train data)
    eda_summary(X_train, y_train, X_eval)

    header("DONE")
    print(f"All plots saved in: {PLOT_DIR}")
    print(f"Baseline metrics saved in : {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")
    print(f"Tuning table saved in     : {os.path.join(OUT_DIR, 'hgb_tuning_no_cv.csv')}")
    print(f"Run log saved in          : {LOG_FILE}")
    print(f"Final metrics saved in    : {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
