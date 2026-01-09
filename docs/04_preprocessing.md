# Step 4: Preprocessing

## Overview
Data preprocessing pipelines for handling missing values and feature scaling.

## Source Code
- **Module**: `src/preprocessing.py`
- **Functions**: `get_basic_prep()`, `get_scaled_prep()`

## Preprocessing Pipelines

### Basic Pipeline
```python
from src.preprocessing import get_basic_prep

basic_prep = get_basic_prep()
# Steps: MedianImputer → VarianceThreshold
```

| Step | Purpose |
|------|---------|
| `SimpleImputer(strategy="median")` | Fill missing values with median |
| `VarianceThreshold(threshold=0.0)` | Remove zero-variance features |

### Scaled Pipeline
```python
from src.preprocessing import get_scaled_prep

scaled_prep = get_scaled_prep()
# Steps: MedianImputer → VarianceThreshold → StandardScaler
```

| Step | Purpose |
|------|---------|
| `SimpleImputer(strategy="median")` | Fill missing values |
| `VarianceThreshold(threshold=0.0)` | Remove constant features |
| `StandardScaler()` | Standardize features (mean=0, std=1) |

## Usage in Models

- **Ridge Regression**: Uses `scaled_prep` (requires scaling)
- **Tree-based models**: Use `basic_prep` (scale-invariant)

## Note on Current Dataset

> The dataset has **0% missing values**, so imputation does not modify any data. The imputer is kept for robustness with new datasets.
