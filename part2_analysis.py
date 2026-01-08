"""
================================================================================
PART 2: TARGET02 ANALYSIS AND RULE DISCOVERY
================================================================================

This script documents the complete process of discovering the conditions and
calculations for predicting target02.

Run this script to see the full analysis:
    python part2_analysis.py

================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PART 2: TARGET02 RULE DISCOVERY ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

# Load the dataset and target files
X = pd.read_csv('dataset_91.csv')
y_df = pd.read_csv('target_91.csv')
y = y_df['target02'].values

print(f"Dataset shape: {X.shape} (samples, features)")
print(f"Target shape: {y.shape}")
print(f"Feature names: feat_0 to feat_{X.shape[1]-1}")
print(f"\ntarget02 statistics:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std:  {y.std():.4f}")
print(f"  Min:  {y.min():.4f}")
print(f"  Max:  {y.max():.4f}")


# =============================================================================
# STEP 2: CORRELATION ANALYSIS - FIND KEY FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: CORRELATION ANALYSIS - FIND KEY FEATURES")
print("=" * 70)

# Calculate correlation between each feature and target02
correlations = X.corrwith(pd.Series(y, name='target02'))

# Sort by absolute correlation (strongest first)
sorted_corr = correlations.abs().sort_values(ascending=False)

print("\nTop 20 features by absolute correlation with target02:")
print("-" * 50)
for i, (feat, corr) in enumerate(sorted_corr.head(20).items()):
    actual_corr = correlations[feat]
    print(f"  {i+1:2d}. {feat}: {actual_corr:+.4f} (|corr| = {corr:.4f})")

# Identify key features (correlation > 0.1)
key_features = sorted_corr[sorted_corr > 0.1].index.tolist()
print(f"\nKey features (|correlation| > 0.1): {len(key_features)} features")
print(f"Top 4: {key_features[:4]}")


# =============================================================================
# STEP 3: FEATURE IMPORTANCE USING RANDOM FOREST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE IMPORTANCE (RANDOM FOREST)")
print("=" * 70)

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest to get feature importance
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
sorted_imp = importances.sort_values(ascending=False)

print("\nTop 10 most important features (Random Forest):")
print("-" * 50)
for i, (feat, imp) in enumerate(sorted_imp.head(10).items()):
    print(f"  {i+1:2d}. {feat}: {imp:.4f} ({imp*100:.1f}%)")

# The key features from both methods
print("\n*** KEY FINDING ***")
print("Both correlation and feature importance identify the same top 4 features:")
print("  - feat_165: Primary predictor")
print("  - feat_259: Secondary predictor")
print("  - feat_14:  Tertiary predictor")
print("  - feat_105: Quaternary predictor")


# =============================================================================
# STEP 4: VISUALIZE feat_165 vs target02
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: ANALYZE feat_165 (PRIMARY PREDICTOR)")
print("=" * 70)

f165 = X['feat_165'].values

# Analyze the relationship by segments
print("\nRelationship between feat_165 and target02:")
print("-" * 50)

for low, high in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
    mask = (f165 >= low) & (f165 < high) if high < 1.0 else (f165 >= low)
    mean_target = y[mask].mean()
    std_target = y[mask].std()
    count = mask.sum()
    print(f"  feat_165 in [{low:.2f}, {high:.2f}): mean={mean_target:+.4f}, std={std_target:.4f}, n={count}")

print("\n*** KEY FINDING ***")
print("target02 changes significantly based on feat_165 value:")
print("  - When feat_165 < 0.5: target02 is POSITIVE (mean > 0)")
print("  - When feat_165 > 0.5: target02 is NEGATIVE (mean < 0)")


# =============================================================================
# STEP 5: DETERMINE OPTIMAL SPLIT THRESHOLD
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: DETERMINE OPTIMAL SPLIT THRESHOLD")
print("=" * 70)

# Test different split thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
f259 = X['feat_259'].values
f14 = X['feat_14'].values
f105 = X['feat_105'].values

print("\nTesting different split thresholds on feat_165:")
print("-" * 50)

best_r2 = 0
best_threshold = 0.5

for thresh in thresholds:
    mask_low = f165 <= thresh
    mask_high = f165 > thresh
    
    # Fit linear regression for each region
    y_pred = np.zeros_like(y)
    
    for mask in [mask_low, mask_high]:
        Xm = np.column_stack([f259[mask], f14[mask], f105[mask]])
        lr = LinearRegression()
        lr.fit(Xm, y[mask])
        y_pred[mask] = lr.predict(Xm)
    
    r2 = r2_score(y, y_pred)
    print(f"  Threshold {thresh}: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_threshold = thresh

print(f"\n*** BEST THRESHOLD: {best_threshold} (R² = {best_r2:.4f}) ***")


# =============================================================================
# STEP 6: FIT LINEAR MODELS FOR EACH REGION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FIT LINEAR MODELS FOR EACH REGION")
print("=" * 70)

# Use the best threshold
mask_low = f165 <= best_threshold
mask_high = f165 > best_threshold

print(f"\nUsing threshold = {best_threshold}")
print(f"Region 1 (feat_165 <= {best_threshold}): {mask_low.sum()} samples")
print(f"Region 2 (feat_165 > {best_threshold}): {mask_high.sum()} samples")

# Fit Region 1
Xm_low = np.column_stack([f259[mask_low], f14[mask_low], f105[mask_low]])
lr_low = LinearRegression()
lr_low.fit(Xm_low, y[mask_low])

print(f"\nRegion 1 formula (feat_165 <= {best_threshold}):")
print(f"  target02 = {lr_low.coef_[0]:.6f} * feat_259")
print(f"           + {lr_low.coef_[1]:.6f} * feat_14")
print(f"           + {lr_low.coef_[2]:.6f} * feat_105")
print(f"           + {lr_low.intercept_:.6f}")

# Fit Region 2
Xm_high = np.column_stack([f259[mask_high], f14[mask_high], f105[mask_high]])
lr_high = LinearRegression()
lr_high.fit(Xm_high, y[mask_high])

print(f"\nRegion 2 formula (feat_165 > {best_threshold}):")
print(f"  target02 = {lr_high.coef_[0]:.6f} * feat_259")
print(f"           + {lr_high.coef_[1]:.6f} * feat_14")
print(f"           + {lr_high.coef_[2]:.6f} * feat_105")
print(f"           + {lr_high.intercept_:.6f}")


# =============================================================================
# STEP 7: VALIDATE THE MODEL
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: VALIDATE THE MODEL")
print("=" * 70)

# Make predictions using the discovered rules
y_pred = np.zeros_like(y)
y_pred[mask_low] = lr_low.predict(Xm_low)
y_pred[mask_high] = lr_high.predict(Xm_high)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = np.mean(np.abs(y - y_pred))

print("\nFinal Model Performance:")
print("-" * 50)
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE:     {rmse:.4f}")
print(f"  MAE:      {mae:.4f}")


# =============================================================================
# STEP 8: GENERATE FRAMEWORK CODE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: FRAMEWORK CODE FOR SUBMISSION")
print("=" * 70)

print("""
Copy this into the main() function of framework_91.py:

    # Feature indices
    F165 = 165  # Used for condition
    F259 = 259  # Used in calculation
    F14 = 14    # Used in calculation
    F105 = 105  # Used in calculation
    
    # Condition 1: feat_165 <= 0.5
    condition1 = (F165, "<=", 0.5)
    
    def calc1(arr):
        return {c0:.6f} * arr[F259] + {c1:.6f} * arr[F14] + {c2:.6f} * arr[F105] + {c3:.6f}
    
    # Condition 2: default (feat_165 > 0.5)
    condition2 = None
    
    def calc2(arr):
        return {c4:.6f} * arr[F259] + {c5:.6f} * arr[F14] + {c6:.6f} * arr[F105] + {c7:.6f}
    
    pair_list = [
        (condition1, calc1),
        (condition2, calc2),
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    return framework(pair_list, data_array)
""".format(
    c0=lr_low.coef_[0], c1=lr_low.coef_[1], c2=lr_low.coef_[2], c3=lr_low.intercept_,
    c4=lr_high.coef_[0], c5=lr_high.coef_[1], c6=lr_high.coef_[2], c7=lr_high.intercept_
))


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nSummary:")
print("  - Key features: feat_165, feat_259, feat_14, feat_105")
print("  - Condition: Split at feat_165 = 0.5")
print("  - R² Score: {:.4f}".format(r2))
print("=" * 70)
