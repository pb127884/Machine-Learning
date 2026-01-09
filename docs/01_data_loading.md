# Step 1: Data Loading

## Overview
Load training features, targets, and evaluation data from CSV files.

## Source Code
- **Module**: `src/config.py`, `main.py`
- **Function**: `load_data()`

## Data Files

| File | Description | Shape |
|------|-------------|-------|
| `data/dataset_91.csv` | Training features (X) | (10000, 273) |
| `data/target_91.csv` | Target variable (y) | (10000,) |
| `data/EVAL_91.csv` | Evaluation features | (10000, 273) |

## Configuration

```python
# src/config.py
DATA_X_PATH = "data/dataset_91.csv"
DATA_Y_PATH = "data/target_91.csv"
DATA_EVAL_PATH = "data/EVAL_91.csv"
TARGET_COL = "target01"
```

## Expected Output

```
LOAD DATA
==========================================================================================
X shape     : (10000, 273)
y shape     : (10000,)
X_eval shape: (10000, 273)
```

## Code Example

```python
import pandas as pd
from src.config import DATA_X_PATH, DATA_Y_PATH, TARGET_COL

X = pd.read_csv(DATA_X_PATH)
y = pd.read_csv(DATA_Y_PATH)[TARGET_COL].values
```
