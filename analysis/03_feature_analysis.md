# Step 3: Feature Analysis - My Investigation

## How I Analyzed 273 Features

With so many features, I needed to understand their characteristics.

### 1. Variance Analysis

**Code I Used:**
```python
X_num = X.select_dtypes(include=[np.number])
variances = X_num.var(numeric_only=True).sort_values(ascending=False)
stds = X_num.std(numeric_only=True).sort_values(ascending=False)

print("Top 15 features by variance:")
print(variances.head(15).round(6))
```

**Output:**
```
feat_54     212.130877
feat_82     210.743873
feat_140    209.931298
...
```

**My Observation:** All features have similar high variance (200-212). No zero-variance features to remove.

### 2. Correlation with Target

**Code I Used:**
```python
# Feature-target correlation
y_s = pd.Series(y, name="target")
corr_to_target = X_num.corrwith(y_s).sort_values(key=lambda s: s.abs(), ascending=False)

print("Top 20 features by |corr(feature, target)|:")
print(corr_to_target.head(20).round(6))
```

**Output:**
```
feat_254   -0.161660
feat_184   -0.159781
feat_50     0.146340
feat_34     0.071623
...
```

**Critical Observation:** Maximum correlation is only **0.16**! This is very weak.

### 3. Correlation Visualization

**Code I Used:**
```python
import matplotlib.pyplot as plt

# Histogram of correlations
plt.figure()
plt.hist(corr_to_target.dropna().values, bins=60)
plt.title("Distribution of corr(feature, target)")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig("outputs/plots/corr_feature_target_distribution.png")
plt.close()

# Heatmap of top features
top_feats = corr_to_target.dropna().head(50).index.tolist()
corr_mat = X_num[top_feats].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_mat.values, aspect="auto")
plt.title("Feature-Feature Correlation Heatmap")
plt.colorbar()
plt.savefig("outputs/plots/corr_heatmap_top50.png")
plt.close()
```

### 4. Outlier Detection

**Code I Used:**
```python
# IQR-based outlier detection
q1 = X_num.quantile(0.25)
q3 = X_num.quantile(0.75)
iqr = (q3 - q1).replace(0, np.nan)

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_counts = ((X_num.lt(lower)) | (X_num.gt(upper))).sum()
outlier_rates = (outlier_counts / len(X_num)).sort_values(ascending=False)

print("Top 15 features by IQR-outlier rate:")
print((outlier_rates.head(15) * 100).round(2))
```

**Output:**
```
feat_181    0.95%
feat_155    0.91%
...
```

Less than 1% outliers per feature - not significant.

## Methods I Considered But Didn't Use

| Method | Why I Didn't Use It |
|--------|---------------------|
| **PCA (dimensionality reduction)** | All features have similar variance; PCA wouldn't help much |
| **Feature selection by correlation threshold** | Max correlation is 0.16, cutting would remove important features |
| **Outlier removal (IQR/Z-score)** | Only <1% outliers, trees handle them well |
| **Log transform features** | Features already have normal-ish distributions |
| **Polynomial features** | 273 features already high-dimensional |
| **Remove multicollinear features (VIF)** | No extreme correlations found |

## My Key Insight

**Low correlations (max 0.16) mean:**
1. No single feature strongly predicts target
2. Linear models will fail (need combinations)
3. Need non-linear models to find complex patterns
4. Tree ensembles can combine weak signals effectively

This insight guided my model selection!
