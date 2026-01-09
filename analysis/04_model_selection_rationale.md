# Step 4: Model Selection - My Reasoning

## Models I Tested

I tried 5 different approaches to find the best model.

### My Model Setup Code

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

# Basic preprocessing
basic_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=0.0)),
])

# Models I defined
models = {
    "Ridge": Pipeline([
        ("prep", scaled_prep), 
        ("model", Ridge(alpha=1.0))
    ]),
    
    "DecisionTree": Pipeline([
        ("prep", basic_prep), 
        ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20))
    ]),
    
    "RandomForest": Pipeline([
        ("prep", basic_prep), 
        ("model", RandomForestRegressor(n_estimators=300, max_features="sqrt"))
    ]),
    
    "HistGradientBoosting": Pipeline([
        ("prep", basic_prep),
        ("model", HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            early_stopping=True
        ))
    ]),
    
    "Stacking": Pipeline([
        ("prep", basic_prep),
        ("model", StackingRegressor(
            estimators=[("ridge", Ridge()), ("dt", DecisionTreeRegressor())],
            final_estimator=Ridge()
        ))
    ])
}
```

### Training and Evaluation Code

```python
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    
    tr_pred = pipe.predict(X_train)
    va_pred = pipe.predict(X_val)
    
    tr_rmse = rmse(y_train, tr_pred)
    va_rmse = rmse(y_val, va_pred)
    tr_r2 = r2_score(y_train, tr_pred)
    va_r2 = r2_score(y_val, va_pred)
    gap = va_rmse - tr_rmse
    
    print(f"{name}: Train RMSE {tr_rmse:.4f} | Val RMSE {va_rmse:.4f} | Gap {gap:.4f}")
    results.append((name, va_rmse, va_r2, gap))
```

### My Results

| Model | Train RMSE | Val RMSE | Val R² | GAP |
|-------|------------|----------|--------|-----|
| Ridge | 0.2219 | 0.2303 | 0.060 | 0.008 |
| DecisionTree | 0.2000 | 0.2441 | -0.055 | 0.044 |
| RandomForest | 0.0852 | 0.2304 | 0.059 | **0.145** |
| **HistGradientBoosting** | 0.0841 | **0.1504** | **0.599** | 0.066 |
| Stacking | 0.1635 | 0.2298 | 0.064 | 0.066 |

### My Analysis of Each Model

#### Ridge (Linear) - Why It Failed
- Only 6% R² on validation
- **My reasoning:** Feature analysis showed max correlation is 0.16, so linear combinations can't capture patterns
- I expected this based on my correlation analysis

#### DecisionTree - Why It Failed  
- Negative R² (worse than predicting mean!)
- High GAP means overfitting
- **My reasoning:** Single tree memorizes training data, doesn't generalize

#### RandomForest - Why It Failed
- Similar to Ridge on validation (6% R²)
- Extremely high GAP (0.145) = severe overfitting
- **My reasoning:** Even 300 trees couldn't learn the pattern; bagging alone isn't enough

#### Stacking - Why It Failed
- Combining bad models = still bad
- **My reasoning:** If Ridge and DT are weak, stacking them won't help

#### HistGradientBoosting - Why It Worked!
- **60% R²** - much better than others
- Reasonable GAP (0.066)
- **My reasoning:** Boosting builds trees sequentially, each fixing previous errors

## Other Models I Considered But Didn't Use

| Model | Why I Didn't Use It |
|-------|---------------------|
| **Lasso Regression** | Linear model - would fail like Ridge |
| **ElasticNet** | Linear model - same issue |
| **SVR (Support Vector Regression)** | Slow with 273 features, tuning difficult |
| **KNN Regression** | Slow, doesn't work well in high dimensions |
| **XGBoost** | Similar to HistGradientBoosting, HGB is faster |
| **LightGBM** | Would try if HGB didn't work well |
| **Neural Network** | Overkill for this data size, harder to tune |

## My Decision

I selected **HistGradientBoosting** because:
1. Best validation performance (0.599 R²)
2. Balances train/val performance (not overfitting badly)
3. Built-in early stopping prevents overfitting
4. Fast training with histogram-based splits
5. Good at combining many weak features
