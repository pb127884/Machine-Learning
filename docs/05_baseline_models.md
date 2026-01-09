# Step 5: Baseline Models

## Overview
Train and compare multiple regression algorithms to select the best baseline.

## Source Code
- **Module**: `src/models.py`
- **Function**: `get_models()`

## Models Compared

| Model | Description | Preprocessing |
|-------|-------------|---------------|
| **Ridge** | L2 regularized linear regression | Scaled |
| **DecisionTree** | max_depth=8, min_samples_leaf=20 | Basic |
| **RandomForest** | 300 trees, sqrt features | Basic |
| **HistGradientBoosting** | Early stopping, 300 iterations | Basic |
| **Stacking** | Ridge + DT + RF → Ridge meta | Basic |

## Code Example

```python
from src.models import get_models

models = get_models()
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
```

## Baseline Results

| Model | Train RMSE | Train R² | Val RMSE | Val R² | GAP |
|-------|------------|----------|----------|--------|-----|
| Ridge | 0.2219 | 0.1199 | 0.2303 | 0.0604 | 0.0083 |
| DecisionTree | 0.2000 | 0.2851 | 0.2441 | -0.0554 | 0.0440 |
| RandomForest | 0.0852 | 0.8703 | 0.2304 | 0.0591 | 0.1452 |
| **HistGradientBoosting** | 0.0841 | 0.8737 | **0.1504** | **0.5991** | 0.0663 |
| Stacking | 0.1635 | 0.5226 | 0.2298 | 0.0641 | 0.0664 |

## Best Baseline Model

**HistGradientBoosting** selected with:
- Lowest Validation RMSE: **0.1504**
- Highest Validation R²: **0.5991**

## Output File
- `outputs/baseline_metrics_table.csv`
