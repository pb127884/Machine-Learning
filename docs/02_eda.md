# Step 2: Exploratory Data Analysis (EDA)

## Overview
Comprehensive data quality checks and statistical analysis.

## Source Code
- **Module**: `src/eda.py`
- **Functions**: `mean_median_std_check()`, `duplication_checks()`, `basic_data_distribution()`, `outlier_check_report_only()`, `correlation_analysis()`

## Checks Performed

### 1. Statistical Summary
```python
mean_median_std_check(X, y)
```
- Top 10 features by mean, median, std
- Target statistics (mean, median, std, min, max)

### 2. Duplication Checks
```python
duplication_checks(X)
```
- Duplicate row detection
- Duplicate column detection (hash-based)

### 3. Data Distribution
```python
basic_data_distribution(X)
```
- Shape verification
- Missing value % per column
- Variance and std per feature

### 4. Outlier Detection
```python
outlier_check_report_only(X)
```
- IQR-based outlier rate per feature
- Report only (no removal)

### 5. Correlation Analysis
```python
correlation_analysis(X, y)
```
- Feature-target correlations
- Feature-feature correlations (top features)

## Generated Plots
- `corr_feature_target_distribution.png` - Correlation distribution histogram
- `corr_heatmap_top50.png` - Feature correlation heatmap

## Expected Results

| Check | Result |
|-------|--------|
| Null values | 0 (0.00%) |
| Duplicate rows | 0 |
| Duplicate columns | 0 |
| Top correlation | feat_254 (-0.162) |
