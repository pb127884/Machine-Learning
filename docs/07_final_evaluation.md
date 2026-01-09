# Step 7: Final Evaluation

## Overview
Final model evaluation on the held-out test set (used only once).

## Source Code
- **Module**: `main.py`
- **Function**: `final_evaluation()`

## Evaluation Strategy

1. **Train-only fit**: Evaluate tuned model on train/val/test
2. **Train+Val fit**: Retrain on combined train+val, evaluate on test

## Results: Train-Only Fit

| Set | RMSE | R² |
|-----|------|-----|
| Train | 0.0716 | 0.9083 |
| Val | 0.1142 | 0.7691 |
| Test | 0.1139 | 0.7712 |

**GAP (Val-Train RMSE)**: 0.0425

## Results: Final Model (Train+Val Fit)

| Set | RMSE | R² |
|-----|------|-----|
| Train | 0.0839 | 0.8743 |
| Val | 0.0827 | 0.8788 |
| **Test** | **0.1102** | **0.7860** |

**GAP (Val-Train RMSE)**: -0.0012 (excellent!)

## Key Observations

✅ **Low GAP**: Model generalizes well, minimal overfitting  
✅ **Consistent R²**: Test R² (0.786) close to Val R² (0.879)  
✅ **Improvement**: Test RMSE improved from baseline 0.150 to 0.110

## Generated Plots
- `outputs/plots/r2_best_model.png` - R² comparison chart

## Output File
- `outputs/final_metrics_tuned_hgb_no_cv.csv`
