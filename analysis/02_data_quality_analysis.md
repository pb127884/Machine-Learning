# Step 2: Data Quality Analysis - My Approach

## How I Checked Data Quality

I performed systematic checks to find any data issues before modeling.

### 1. Missing Values Check

**Code I Used:**
```python
# Check missing values
missing = X.isna().mean().sort_values(ascending=False)
print("Top 15 columns by missing %:")
print((missing.head(15) * 100).round(2))

# Total missing
null_count = X.isnull().sum().sum()
null_pct = (null_count / (X.shape[0] * X.shape[1])) * 100
print(f"Null values: {null_count} ({null_pct:.2f}%)")
```

**Output:**
```
Null values: 0 (0.00%)
```

**My Finding:** No missing values at all - data is complete.

### 2. Duplicate Rows Check

**Code I Used:**
```python
# Check duplicate rows
dup_rows = X.duplicated().sum()
print(f"Duplicate rows: {dup_rows}")
```

**Output:**
```
Duplicate rows: 0
```

### 3. Duplicate Columns Check

**Code I Used:**
```python
# Hash-based duplicate column detection
col_hash = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
dup_col_mask = col_hash.duplicated(keep=False)
dup_cols = X.columns[dup_col_mask].tolist()
print(f"Duplicate columns: {len(dup_cols)}")
```

**Output:**
```
Duplicate columns: none detected
```

### 4. Data Types Check

**Code I Used:**
```python
X_num = X.select_dtypes(include=[np.number])
print(f"Numeric columns: {X_num.shape[1]}")
print(f"Total columns: {X.shape[1]}")
```

**Output:**
```
Numeric columns: 273
Total columns: 273
```

All features are numeric.

## Methods I Considered But Didn't Need

| Method | Why Not Needed |
|--------|----------------|
| **Imputation (mean/median/mode)** | No missing values (0%) |
| **Dropping rows with NaN** | No NaN rows exist |
| **Feature removal for duplicates** | No duplicate columns found |
| **Categorical encoding (One-Hot, Label)** | All features already numeric |
| **Type conversion** | All types already correct |

## My Preprocessing Decision

Since data is clean, I kept preprocessing simple:
```python
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# I kept imputer for robustness (future data may have nulls)
basic_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=0.0)),
])
```
