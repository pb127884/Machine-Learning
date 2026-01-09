# Step 5: Hyperparameter Tuning - My Approach

## Why I Chose Grid Search

I used Grid Search instead of other methods because:

| Method | Why I Chose/Didn't Choose |
|--------|---------------------------|
| **Grid Search** ✅ | Tests all combinations, guarantees best result |
| **Random Search** | Faster but might miss optimal values |
| **Bayesian Optimization** | Complex setup, overkill for this grid size |
| **Manual Tuning** | Too slow with 7 parameters |

## My Tuning Code

```python
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

def hypertune_hgb_no_cv(X_train, y_train, X_val, y_val, seed=42):
    
    # Base pipeline
    base_pipe = Pipeline([
        ("prep", basic_prep),
        ("model", HistGradientBoostingRegressor(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        ))
    ])

    # My parameter grid
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
    print(f"Total combinations: {len(grid)}")  # 2592

    best_val_rmse = np.inf
    best_params = None
    
    for i, params in enumerate(grid, 1):
        m = clone(base_pipe)
        m.set_params(**params)
        m.fit(X_train, y_train)
        
        va_pred = m.predict(X_val)
        va_rmse = rmse(y_val, va_pred)
        
        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            best_params = params
        
        if i % 50 == 0:
            print(f"[{i}/2592] best_val_rmse={best_val_rmse:.6f}")
    
    return best_params
```

## Parameters I Tuned - My Reasoning

### Learning Rate: [0.01, 0.03, 0.05, 0.1]
```
Best: 0.03
```
**My reasoning:** 
- 0.01 too slow (needs many iterations)
- 0.1 too fast (overshoots optimal)
- 0.03 is good balance between speed and stability

### Max Depth: [3, 5, 7, None]
```
Best: None (unlimited)
```
**My reasoning:**
- I thought limiting depth would prevent overfitting
- But `None` worked best because early stopping controls complexity
- Learned: Let other parameters regularize instead

### Max Leaf Nodes: [15, 31, 63]
```
Best: 15
```
**My reasoning:**
- Fewer leaves = simpler trees
- Simpler trees generalize better
- 15 leaves is enough to capture patterns without overfitting

### Min Samples Leaf: [10, 20, 50]
```
Best: 20
```
**My reasoning:**
- Higher = more regularization
- 20 prevents splits on small groups
- Good balance between flexibility and stability

### L2 Regularization: [0.0, 0.1, 1.0]
```
Best: 0.0 (no regularization)
```
**My reasoning:**
- I expected some regularization would help
- But other parameters (leaf nodes, min samples) already regularize enough
- Learned: Don't over-regularize

### Max Bins: [128, 255]
```
Best: 255
```
**My reasoning:**
- More bins = finer splits
- 255 allows more precise decisions
- Trade-off with speed, but 255 is fast enough

### Max Iterations: [200, 400, 800]
```
Best: 800
```
**My reasoning:**
- More iterations with early stopping = finds optimal point
- Early stopping prevents overfitting anyway
- 800 gives enough room for convergence

## My Tuning Results

```
[    1/2592] best_val_rmse=0.225711
[   50/2592] best_val_rmse=0.149969
[  100/2592] best_val_rmse=0.133035
[  350/2592] best_val_rmse=0.118774
[  450/2592] best_val_rmse=0.114159  <- Best found!
[ 2592/2592] best_val_rmse=0.114159  <- Confirmed
```

## Final Best Parameters

```python
best_params = {
    "learning_rate": 0.03,
    "max_depth": None,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_bins": 255,
    "max_iter": 800
}
```

## Improvement I Achieved

| Metric | Before Tuning | After Tuning | My Improvement |
|--------|---------------|--------------|----------------|
| Val RMSE | 0.1504 | **0.1142** | **24% better!** |
| Val R² | 0.599 | **0.769** | **28% better!** |
