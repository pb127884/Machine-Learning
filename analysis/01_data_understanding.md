# Step 1: Data Understanding - My Analysis

## What I Did First

I started by loading the data to understand what I'm working with.

### Code I Used

```python
import pandas as pd
import numpy as np

# Load data
X = pd.read_csv("data/dataset_91.csv")
y = pd.read_csv("data/target_91.csv")["target01"].values
X_eval = pd.read_csv("data/EVAL_91.csv")

# Check shapes
print("X shape     :", X.shape)
print("y shape     :", y.shape)
print("X_eval shape:", X_eval.shape)
```

**Output:**
```
X shape     : (10000, 273)
y shape     : (10000,)
X_eval shape: (10000, 273)
```

### My Observations

1. I have 10,000 training samples with 273 features
2. Same number of evaluation samples (10,000)
3. All features are numeric (no categorical)

### Target Analysis Code

```python
# Check target statistics
print(f"mean={y.mean():.6f}")
print(f"median={np.median(y):.6f}")
print(f"std={y.std():.6f}")
print(f"min={y.min():.6f}")
print(f"max={y.max():.6f}")
```

**Output:**
```
mean=0.781754 median=0.671680 std=0.236976 min=0.354770 max=1.471922
```

### What I Noticed

- Mean (0.78) is greater than Median (0.67) → **right-skewed distribution**
- Values are bounded between 0.35 and 1.47
- Standard deviation is moderate (0.24)

### Methods I Considered But Didn't Use

| Method | Why I Didn't Use It |
|--------|---------------------|
| **Log transformation** | Target is bounded and not extremely skewed |
| **Box-Cox transformation** | Values are already in reasonable range |
| **Normalization** | Not needed since target isn't on extreme scale |

### My Decision

I decided to keep the target as-is because:
1. The skewness is not extreme
2. Tree-based models handle skewed data well
3. Value range is bounded and reasonable
