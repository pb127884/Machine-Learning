import os
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
# CONFIG (KEEP PATHS AS YOU GAVE)
# ----------------------------
SEED = 42

DATA_X_PATH = "dataset_91.csv"
DATA_Y_PATH = "target_91.csv"
DATA_EVAL_PATH = "EVAL_91.csv"
TARGET_COL = "target01"

OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MAX_FEATURE_HISTS = 20
CORR_TOP_FEATURES = 50


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


def plot_r2_for_best(best_name, r2_train, r2_val, r2_test, filename="r2_best_model.png"):
    plt.figure(figsize=(6, 4))
    labels = ["Train", "Val", "Test"]
    values = [r2_train, r2_val, r2_test]
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("R²")
    plt.title(f"R² for Best Model: {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()
    print(f"✅ Saved Best Model R² graph to: {os.path.join(PLOT_DIR, filename)}")


# ----------------------------
# REQUIRED: MEAN / MEDIAN / STD CHECKS
# ----------------------------
def mean_median_std_check(X: pd.DataFrame, y: np.ndarray):
    header("MEAN / MEDIAN / STD CHECK (X + y)")

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


# ----------------------------
# DATA LEAKAGE / DUPLICATION CHECKS
# ----------------------------
def duplication_checks(X: pd.DataFrame):
    header("DUPLICATION CHECKS (ROWS + COLUMNS)")
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
    """
    Checks if EXACT same rows appear in both splits (leakage risk).
    Hash-based, fast.
    """
    A = X_a.iloc[:max_rows]
    B = X_b.iloc[:max_rows]

    ha = pd.util.hash_pandas_object(A, index=False).values
    hb = pd.util.hash_pandas_object(B, index=False).values

    overlap = len(set(ha).intersection(set(hb)))
    print(f"Leak-check overlap {name_a} vs {name_b} (first {max_rows} rows): {overlap}")


# ----------------------------
# DATA DISTRIBUTION + VARIATION
# ----------------------------
def basic_data_distribution(X: pd.DataFrame):
    header("DATA DISTRIBUTION (X) + MISSINGNESS + VARIATION")
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
    print(f"✅ Saved target histograms to: {PLOT_DIR}")


def outlier_check_report_only(X: pd.DataFrame):
    header("OUTLIER CHECK (REPORT ONLY - NO HANDLING)")

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


def correlation_analysis(X: pd.DataFrame, y: np.ndarray):
    header("CORRELATION (NUMERIC ONLY): feature-target + feature-feature")

    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        print("No numeric columns -> skipping correlation analysis.")
        return

    y_s = pd.Series(y, name="target")
    corr_to_target = X_num.corrwith(y_s).sort_values(key=lambda s: s.abs(), ascending=False)

    print("\nTop 20 features by |corr(feature, target)|:")
    print(corr_to_target.head(20).round(6))

    plt.figure()
    plt.hist(corr_to_target.dropna().values, bins=60)
    plt.title("Distribution of corr(feature, target)")
    plt.xlabel("correlation")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "corr_feature_target_distribution.png"), dpi=160)
    plt.close()

    top_feats = corr_to_target.dropna().head(min(CORR_TOP_FEATURES, X_num.shape[1])).index.tolist()
    corr_mat = X_num[top_feats].corr()
    plot_corr_heatmap(
        corr_mat,
        f"Feature-Feature Corr (Top {len(top_feats)} by |corr w/ target|)",
        f"corr_heatmap_top{len(top_feats)}.png",
    )

    print(f"✅ Saved correlation plots to: {PLOT_DIR}")


# ----------------------------
# GAP FIX: STRATIFIED SPLIT ON TARGET BINS (still 70/15/15)
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
# NO-CV HYPER TUNING (VALIDATION ONLY)
# ------------------------------------------------
def hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=42):
    """
    Hyper-tune HistGradientBoosting using ONLY the Validation set (no CV).
    Trains on train, evaluates on val, picks best by val RMSE.
    """

    header("HYPER TUNING (NO CV): HistGradientBoostingRegressor using VALIDATION ONLY")

    # preprocessing identical to your basic_prep
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

    # Grid: many combos (be careful: huge grids = huge runtime)
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
    print(f"Total combinations to try: {len(grid)}")

    best = None
    best_params = None
    best_val_rmse = np.inf

    # Optional: save all tuning results
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

    # Save tuning table
    os.makedirs(OUT_DIR, exist_ok=True)
    tune_df = pd.DataFrame(tuning_rows).sort_values("val_rmse")
    tune_df.to_csv(os.path.join(OUT_DIR, "hgb_tuning_no_cv.csv"), index=False)
    print(f"✅ Saved tuning table: {os.path.join(OUT_DIR, 'hgb_tuning_no_cv.csv')}")

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
    # REQUIRED METHOD: mean/median/std checks
    # ------------------------------------------------
    mean_median_std_check(X, y)

    # ------------------------------------------------
    # EDA SECTION (kept)
    # ------------------------------------------------
    duplication_checks(X)
    basic_data_distribution(X)
    outlier_check_report_only(X)          # report only
    correlation_analysis(X, y)            # analysis + plots only (NOT feature selection)

    # ------------------------------------------------
    # 2) TRAIN / VAL / TEST SPLIT (70/15/15) - stratification
    # ------------------------------------------------
    header("TRAIN / VAL / TEST SPLIT (70/15/15) - STRATIFIED ON TARGET (REDUCES GAP)")

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_70_15_15_split(
        X, y, seed=SEED, n_bins=10
    )

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    target_distribution(y_train, y_val, y_test)

    header("DATA LEAKAGE CHECK (OVERLAP BETWEEN SPLITS)")
    leakage_overlap_check(X_train, X_val, "train", "val")
    leakage_overlap_check(X_train, X_test, "train", "test")
    leakage_overlap_check(X_val, X_test, "val", "test")

    # Feature plotting (top variance numeric)
    header("FEATURE PLOTTING (Top variance numeric features)")
    X_num = X_train.select_dtypes(include=[np.number])
    if X_num.shape[1] > 0:
        variances = X_num.var().sort_values(ascending=False)
        top_cols = variances.head(min(MAX_FEATURE_HISTS, len(variances))).index.tolist()
        plot_feature_histograms(X_num, top_cols, filename_prefix="hist")
        print(f"✅ Saved feature histograms to: {PLOT_DIR}")
    else:
        print("No numeric features detected -> skipping histograms.")

    # ------------------------------------------------
    # 3) PREPROCESSING PIPELINES (same)
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
    # 4) MODELS (same + stack)
    # ------------------------------------------------
    header("DEFINE MODELS (BASELINE MODELS)")

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

    stack = Pipeline([
        ("prep", basic_prep),
        ("model", StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=1.0, random_state=SEED)),
                ("dt", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED)),
                ("rf", RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=1)),
            ],
            final_estimator=Ridge(alpha=1.0, random_state=SEED),
            passthrough=True
        ))
    ])

    models = {
        "Ridge": ridge,
        "DecisionTree": dt,
        "RandomForest": rf,
        "HistGradientBoosting": hgb,
        "Stacking": stack,
    }

    # ------------------------------------------------
    # 5) TRAIN ON TRAIN → EVALUATE ON VALIDATION (baseline selection)
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
            f"TRAIN RMSE {tr['rmse']:.6f} R² {tr['r2']:.6f} || "
            f"VAL RMSE {va['rmse']:.6f} R² {va['r2']:.6f} || "
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
    print(f"✅ Saved baseline metrics: {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")

    # ------------------------------------------------
    # 5B) HYPER TUNING (NO CV) for HistGradientBoosting using VAL ONLY
    # ------------------------------------------------
    tuned_hgb, tuned_params, tune_df = hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=SEED)

    # Evaluate tuned model (fit train only)
    header("TUNED HGB (FIT TRAIN ONLY) → TRAIN/VAL/TEST RESULTS")
    tuned_hgb.fit(X_train, y_train)

    pred_tr = tuned_hgb.predict(X_train)
    pred_va = tuned_hgb.predict(X_val)
    pred_te = tuned_hgb.predict(X_test)

    m_tr = evaluate(y_train, pred_tr)
    m_va = evaluate(y_val, pred_va)
    m_te = evaluate(y_test, pred_te)

    print("TUNED HGB (train-only fit)")
    print(f"Train RMSE: {m_tr['rmse']:.6f} | Train R²: {m_tr['r2']:.6f}")
    print(f"Val   RMSE: {m_va['rmse']:.6f} | Val   R²: {m_va['r2']:.6f}")
    print(f"Test  RMSE: {m_te['rmse']:.6f} | Test  R²: {m_te['r2']:.6f}")
    print(f"GAP (Val-Train RMSE): {m_va['rmse'] - m_tr['rmse']:.6f}")

    # ------------------------------------------------
    # 6) FINAL TRAIN (TRAIN + VAL) → TEST ONCE (tuned model)
    # ------------------------------------------------
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

    print("\nBEST MODEL: Tuned HistGradientBoostingRegressor (no CV)")
    print(f"Train RMSE: {train_m['rmse']:.6f} | Train R²: {train_m['r2']:.6f}")
    print(f"Val   RMSE: {val_m['rmse']:.6f} | Val   R²: {val_m['r2']:.6f}")
    print(f"Test  RMSE: {test_m['rmse']:.6f} | Test  R²: {test_m['r2']:.6f}")
    print(f"GAP (Val-Train RMSE): {val_m['rmse'] - train_m['rmse']:.6f}")

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
    print(f"✅ Saved final metrics: {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")

    # ------------------------------------------------
    # 7) TRAIN ON FULL DATA → PREDICT EVAL (OFFICIAL FILE)
    # ------------------------------------------------
    header("TRAIN ON FULL DATA + PREDICT EVAL (OFFICIAL SUBMISSION FILE) - TUNED HGB")

    tuned_hgb.fit(X, y)
    eval_pred = tuned_hgb.predict(X_eval)

    out = pd.DataFrame({TARGET_COL: eval_pred.astype(float)})

    official_path = os.path.join(OUT_DIR, "EVAL_target01_91.csv")
    out.to_csv(official_path, index=False)

    extra_path = os.path.join(OUT_DIR, "EVAL_target01_best.csv")
    out.to_csv(extra_path, index=False)

    print(f"✅ Saved OFFICIAL: {official_path}")
    print(f"✅ Saved EXTRA   : {extra_path}")
    print(out.head())

    header("DONE")
    print(f"All plots saved in: {PLOT_DIR}")
    print(f"Baseline metrics saved in : {os.path.join(OUT_DIR, 'baseline_metrics_table.csv')}")
    print(f"Tuning table saved in     : {os.path.join(OUT_DIR, 'hgb_tuning_no_cv.csv')}")
    print(f"Final metrics saved in    : {os.path.join(OUT_DIR, 'final_metrics_tuned_hgb_no_cv.csv')}")


if __name__ == "__main__":
    main()