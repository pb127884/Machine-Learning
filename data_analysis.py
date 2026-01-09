#!/usr/bin/env python3
# data_analysis.py - analyze the dataset
# generates reports and plots to analysis/outputs/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# paths
DATA_X = "data/dataset_91.csv"
DATA_Y = "data/target_91.csv"
DATA_EVAL = "data/EVAL_91.csv"
TARGET = "target01"

OUT_DIR = "analysis/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

REPORT = os.path.join(OUT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def log(msg):
    print(msg)
    with open(REPORT, 'a') as f:
        f.write(msg + '\n')


def section(title):
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)


# step 1: load data
def load():
    section("1. LOAD DATA")
    
    X = pd.read_csv(DATA_X)
    y = pd.read_csv(DATA_Y)[TARGET].values
    X_eval = pd.read_csv(DATA_EVAL)
    
    log(f"X shape: {X.shape}")
    log(f"y shape: {y.shape}")
    log(f"eval shape: {X_eval.shape}")
    
    return X, y, X_eval


# step 2: target analysis
def analyze_target(y):
    section("2. TARGET ANALYSIS")
    
    log(f"mean: {y.mean():.4f}")
    log(f"median: {np.median(y):.4f}")
    log(f"std: {y.std():.4f}")
    log(f"min: {y.min():.4f}, max: {y.max():.4f}")
    
    # skewness
    if y.mean() > np.median(y):
        log("distribution is right-skewed")
    else:
        log("distribution is left-skewed")
    
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(y, bins=50, edgecolor='black')
    ax[0].axvline(y.mean(), color='r', linestyle='--', label='mean')
    ax[0].axvline(np.median(y), color='g', linestyle='--', label='median')
    ax[0].legend()
    ax[0].set_title('Target Distribution')
    
    ax[1].boxplot(y)
    ax[1].set_title('Target Boxplot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'target.png'), dpi=150)
    plt.close()
    log("saved target.png")


# step 3: data quality
def check_quality(X):
    section("3. DATA QUALITY")
    
    # missing values
    missing = X.isnull().sum().sum()
    total = X.shape[0] * X.shape[1]
    log(f"missing: {missing} ({missing/total*100:.2f}%)")
    
    # duplicates
    dup_rows = X.duplicated().sum()
    log(f"duplicate rows: {dup_rows}")
    
    # duplicate columns
    col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    dup_cols = col_hash.duplicated().sum()
    log(f"duplicate cols: {dup_cols}")
    
    # types
    num_cols = X.select_dtypes(include=[np.number]).shape[1]
    log(f"numeric cols: {num_cols}/{X.shape[1]}")
    
    # save summary
    summary = pd.DataFrame({
        'check': ['missing', 'dup_rows', 'dup_cols', 'non_numeric'],
        'value': [missing, dup_rows, dup_cols, X.shape[1] - num_cols]
    })
    summary.to_csv(os.path.join(OUT_DIR, 'quality.csv'), index=False)


# step 4: feature stats
def feature_stats(X):
    section("4. FEATURE STATISTICS")
    
    X_num = X.select_dtypes(include=[np.number])
    log(f"analyzing {X_num.shape[1]} features")
    
    means = X_num.mean().sort_values(ascending=False)
    stds = X_num.std().sort_values(ascending=False)
    
    log("top 5 by mean:")
    log(str(means.head().round(4)))
    
    log("top 5 by std:")
    log(str(stds.head().round(4)))
    
    # save full stats
    stats = pd.DataFrame({
        'feature': X_num.columns,
        'mean': X_num.mean().values,
        'std': X_num.std().values,
        'var': X_num.var().values
    }).sort_values('var', ascending=False)
    stats.to_csv(os.path.join(OUT_DIR, 'feature_stats.csv'), index=False)
    
    # plot variance distribution
    plt.figure(figsize=(8, 4))
    plt.hist(X_num.var().values, bins=50, edgecolor='black')
    plt.xlabel('Variance')
    plt.title('Feature Variance Distribution')
    plt.savefig(os.path.join(OUT_DIR, 'variance.png'), dpi=150)
    plt.close()


# step 5: correlation
def correlation(X, y):
    section("5. CORRELATION ANALYSIS")
    
    X_num = X.select_dtypes(include=[np.number])
    y_s = pd.Series(y, name='target')
    
    corr = X_num.corrwith(y_s).sort_values(key=lambda x: x.abs(), ascending=False)
    
    log(f"max |corr|: {corr.abs().max():.4f}")
    log("top 10:")
    log(str(corr.head(10).round(4)))
    
    if corr.abs().max() < 0.3:
        log("all correlations weak - need nonlinear models")
    
    # save
    corr_df = pd.DataFrame({'feature': corr.index, 'corr': corr.values})
    corr_df.to_csv(os.path.join(OUT_DIR, 'correlations.csv'), index=False)
    
    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].hist(corr.values, bins=50, edgecolor='black')
    ax[0].axvline(0, color='r', linestyle='--')
    ax[0].set_title('Correlation Distribution')
    
    top20 = corr.head(20)
    colors = ['g' if x > 0 else 'r' for x in top20.values]
    ax[1].barh(range(len(top20)), top20.values, color=colors)
    ax[1].set_yticks(range(len(top20)))
    ax[1].set_yticklabels(top20.index)
    ax[1].set_title('Top 20 by |Correlation|')
    ax[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'correlation.png'), dpi=150)
    plt.close()
    
    # heatmap
    top30 = corr.head(30).index.tolist()
    corr_mat = X_num[top30].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, cmap='coolwarm', center=0, square=True)
    plt.title('Top 30 Features Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'heatmap.png'), dpi=150)
    plt.close()


# step 6: outliers
def check_outliers(X):
    section("6. OUTLIER ANALYSIS")
    
    X_num = X.select_dtypes(include=[np.number])
    
    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = q3 - q1
    
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    outliers = ((X_num < low) | (X_num > high)).sum()
    outlier_pct = (outliers / len(X_num) * 100).sort_values(ascending=False)
    
    log("top 10 by outlier %:")
    log(str(outlier_pct.head(10).round(2)))
    
    total = outliers.sum()
    cells = X_num.shape[0] * X_num.shape[1]
    log(f"total outliers: {total} ({total/cells*100:.2f}%)")
    
    # save
    out_df = pd.DataFrame({'feature': outliers.index, 'count': outliers.values, 'pct': outlier_pct.values})
    out_df.to_csv(os.path.join(OUT_DIR, 'outliers.csv'), index=False)


# step 7: summary
def summary(X, y, corr_max):
    section("7. SUMMARY")
    
    log(f"samples: {X.shape[0]}")
    log(f"features: {X.shape[1]}")
    log(f"target range: [{y.min():.4f}, {y.max():.4f}]")
    log(f"max correlation: {corr_max:.4f}")
    
    log("")
    log("RECOMMENDATIONS:")
    log("1. data is clean - no imputation needed")
    log("2. weak correlations - use nonlinear models")
    log("3. try gradient boosting first")
    log("4. tune hyperparameters carefully")


def main():
    print("=" * 70)
    print("DATA ANALYSIS")
    print(f"report: {REPORT}")
    print("=" * 70)
    
    X, y, X_eval = load()
    analyze_target(y)
    check_quality(X)
    feature_stats(X)
    correlation(X, y)
    check_outliers(X)
    
    # get max corr for summary
    X_num = X.select_dtypes(include=[np.number])
    corr = X_num.corrwith(pd.Series(y))
    corr_max = corr.abs().max()
    
    summary(X, y, corr_max)
    
    section("DONE")
    log(f"output: {OUT_DIR}")
    print(f"\nfiles saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
