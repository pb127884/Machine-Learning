# Step 6: Hyperparameter Tuning

## Overview
Grid search optimization for HistGradientBoostingRegressor using validation set only (no cross-validation).

## Source Code
- **Module**: `src/tuning.py`
- **Function**: `hypertune_hgb_no_cv()`

## Hyperparameter Grid

| Parameter | Values | Count |
|-----------|--------|-------|
| learning_rate | [0.01, 0.03, 0.05, 0.1] | 4 |
| max_depth | [3, 5, 7, None] | 4 |
| max_leaf_nodes | [15, 31, 63] | 3 |
| min_samples_leaf | [10, 20, 50] | 3 |
| l2_regularization | [0.0, 0.1, 1.0] | 3 |
| max_bins | [128, 255] | 2 |
| max_iter | [200, 400, 800] | 3 |

**Total combinations**: 4 × 4 × 3 × 3 × 3 × 2 × 3 = **2,592**

## Tuning Process

```python
from src.tuning import hypertune_hgb_no_cv

best_model, best_params, tune_df = hypertune_hgb_no_cv(
    X_train, y_train, X_val, y_val, seed=42
)
```

## Tuning Progress

| Iteration | Best Val RMSE |
|-----------|---------------|
| 1 | 0.2257 |
| 50 | 0.1500 |
| 100 | 0.1330 |
| 350 | 0.1188 |
| 450 | **0.1142** |
| 2592 | **0.1142** |

## Best Hyperparameters

```python
{
    "learning_rate": 0.03,
    "max_depth": None,
    "max_iter": 800,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_bins": 255
}
```

## Improvement

| Metric | Before Tuning | After Tuning | Improvement |
|--------|---------------|--------------|-------------|
| Val RMSE | 0.1504 | **0.1142** | **24%** |

## Output File
- `outputs/hgb_tuning_no_cv.csv` - All 2592 tuning results
