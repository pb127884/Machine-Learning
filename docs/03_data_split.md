# Step 3: Train/Validation/Test Split

## Overview
Stratified data splitting to ensure balanced target distribution across splits.

## Source Code
- **Module**: `src/preprocessing.py`
- **Functions**: `stratified_70_15_15_split()`, `make_strat_bins()`

## Split Ratio
| Set | Percentage | Samples |
|-----|------------|---------|
| Train | 70% | 7,000 |
| Validation | 15% | 1,500 |
| Test | 15% | 1,500 |

## Stratification Method

1. Create bins from continuous target using `pd.qcut()`
2. Use `StratifiedShuffleSplit` for balanced splits
3. Verify no overlap between splits

```python
from src.preprocessing import stratified_70_15_15_split

X_train, X_val, X_test, y_train, y_val, y_test = stratified_70_15_15_split(
    X, y, seed=42, n_bins=10
)
```

## Data Leakage Check

```python
from src.eda import leakage_overlap_check

leakage_overlap_check(X_train, X_val, "train", "val")
leakage_overlap_check(X_train, X_test, "train", "test")
leakage_overlap_check(X_val, X_test, "val", "test")
```

**Expected**: 0 overlap between all splits

## Target Distribution Verification

| Split | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Train | 0.7816 | 0.2366 | 0.3548 | 1.4362 |
| Val | 0.7821 | 0.2376 | 0.3589 | 1.4719 |
| Test | 0.7824 | 0.2382 | 0.3577 | 1.4128 |

## Generated Plots
- `target_train_hist.png`
- `target_val_hist.png`
- `target_test_hist.png`
