# Step 8: Predictions

## Overview
Generate predictions for the evaluation dataset using the final tuned model.

## Source Code
- **Module**: `main.py`
- **Function**: `generate_predictions()`

## Process

1. Train final model on **full dataset** (train + val + test)
2. Predict on evaluation set (10,000 samples)
3. Save predictions to CSV

```python
# Train on full data
model.fit(X, y)

# Generate predictions
eval_pred = model.predict(X_eval)

# Save
pd.DataFrame({TARGET_COL: eval_pred}).to_csv("outputs/EVAL_target01_91.csv")
```

## Output Files

| File | Description |
|------|-------------|
| `outputs/EVAL_target01_91.csv` | Official submission file |
| `outputs/EVAL_target01_best.csv` | Backup copy |

## Sample Predictions

```csv
target01
0.540740
1.005208
0.958835
0.888208
0.600802
```

## Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Count | 10,000 |
| Mean | ~0.78 |
| Min | ~0.35 |
| Max | ~1.47 |

## Usage

```python
import pandas as pd

predictions = pd.read_csv("outputs/EVAL_target01_91.csv")
print(predictions.head())
```
