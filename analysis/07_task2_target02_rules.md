# Task 2: Rule-Based Prediction System for target02

## Overview

This document describes the development of a rule-based prediction system for `target02` that is suitable for deployment on edge devices. The system uses only simple conditional statements and numerical constants - no complex ML models or libraries beyond basic numpy operations.

**ID:** 91  
**Target Variable:** target02  
**Output File:** `framework_91.py`

---

## 1. Problem Understanding

### Objective
Create a lightweight prediction system for `target02` that:
- Uses only simple conditional rules (comparisons: `>`, `>=`, `<`, `<=`, `==`, `!=`)
- Can run on edge devices with limited computational resources
- Avoids complex ML libraries and models
- Maintains reasonable prediction accuracy

### Data Summary
- **Training Data:** `data/dataset_91.csv` - 10,000 samples × 273 features
- **Target Data:** `data/target_91.csv` - Contains `target01` and `target02`
- **Evaluation Data:** `data/EVAL_91.csv` - 10,000 samples × 273 features

### Target Variable Statistics
| Metric | Value |
|--------|-------|
| Count | 10,000 |
| Mean | 0.5157 |
| Std | 0.7282 |
| Min | -0.8790 |
| 25% | -0.1315 |
| 50% | 0.4821 |
| 75% | 1.0911 |
| Max | 2.7512 |

---

## 2. Feature Analysis Methodology

### 2.1 Decision Tree Feature Importance

I trained a `DecisionTreeRegressor` (max_depth=4, min_samples_leaf=100) to identify the most important features:

**Top Features by Decision Tree Importance:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **feat_165** | **81.6%** |
| 2 | feat_259 | 9.1% |
| 3 | feat_14 | 6.5% |
| 4 | feat_105 | 2.9% |

### 2.2 Correlation Analysis (Initial)

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | feat_165 | **-0.6290** |
| 2 | feat_259 | +0.2301 |
| 3 | feat_105 | +0.1575 |
| 4 | feat_14 | +0.0413 |

### Key Finding
The decision tree identified **feat_14** as more important than correlation analysis suggested. This is because feat_14 helps refine predictions within specific feat_165/feat_259 segments.

### 2.3 Feature Selection Rationale

Selected features for the rule system:
1. **feat_165** (index 165) - Primary predictor, importance = 81.6%
2. **feat_259** (index 259) - Secondary predictor, importance = 9.1%
3. **feat_14** (index 14) - Tertiary predictor, importance = 6.5%
4. **feat_105** (index 105) - Quaternary predictor, importance = 2.9%

**Why Decision Tree Feature Selection?**
- Decision trees capture non-linear relationships
- Importance scores reflect actual predictive power
- Tree structure reveals optimal thresholds
- Rules are directly extractable

---

## 3. Decision Tree Structure

### 3.1 Tree Visualization

```
|--- feat_165 <= 0.50
|   |--- feat_259 <= 0.54
|   |   |--- feat_14 <= 0.49
|   |   |   |--- feat_165 <= 0.20 → 0.428
|   |   |   |--- feat_165 > 0.20 → 0.822
|   |   |--- feat_14 > 0.49
|   |   |   |--- feat_259 <= 0.28 → 0.896
|   |   |   |--- feat_259 > 0.28 → 1.158
|   |--- feat_259 > 0.54
|   |   |--- feat_14 <= 0.54
|   |   |   |--- feat_259 <= 0.86 → 1.126
|   |   |   |--- feat_259 > 0.86 → 1.396
|   |   |--- feat_14 > 0.54
|   |   |   |--- feat_165 <= 0.20 → 1.869
|   |   |   |--- feat_165 > 0.20 → 1.347
|--- feat_165 > 0.50
|   |--- feat_165 <= 0.70
|   |   |--- feat_259 <= 0.46
|   |   |   |--- feat_105 <= 0.50 → -0.143
|   |   |   |--- feat_105 > 0.50 → -0.365
|   |   |--- feat_259 > 0.46
|   |   |   |--- feat_105 <= 0.50 → -0.369
|   |   |   |--- feat_105 > 0.50 → -0.584
|   |--- feat_165 > 0.70
|   |   |--- feat_14 <= 0.45
|   |   |   |--- feat_105 <= 0.49 → 0.169
|   |   |   |--- feat_105 > 0.49 → 0.551
|   |   |--- feat_14 > 0.45
|   |   |   |--- feat_105 <= 0.56 → -0.242
|   |   |   |--- feat_105 > 0.56 → 0.145
```

### 3.2 Exact Thresholds

| Feature | Threshold | Usage |
|---------|-----------|-------|
| feat_165 | 0.499972 | Primary split |
| feat_165 | 0.700854 | Secondary split |
| feat_165 | 0.201313 | Sub-split |
| feat_165 | 0.200357 | Sub-split |
| feat_259 | 0.539155 | Left branch split |
| feat_259 | 0.282283 | Sub-split |
| feat_259 | 0.857818 | Sub-split |
| feat_259 | 0.462765 | Middle branch split |
| feat_14 | 0.488796 | Left branch split |
| feat_14 | 0.537704 | Sub-split |
| feat_14 | 0.454465 | Right branch split |
| feat_105 | 0.502041 | Middle branch split |
| feat_105 | 0.495375 | Sub-split |
| feat_105 | 0.488271 | Sub-split |
| feat_105 | 0.557603 | Sub-split |

### 3.3 Leaf Values (16 buckets)

| Leaf | Condition | Value | Samples |
|------|-----------|-------|---------|
| 1 | f165≤0.50, f259≤0.54, f14≤0.49, f165≤0.20 | 0.428199 | 513 |
| 2 | f165≤0.50, f259≤0.54, f14≤0.49, f165>0.20 | 0.821636 | 817 |
| 3 | f165≤0.50, f259≤0.54, f14>0.49, f259≤0.28 | 0.895559 | 709 |
| 4 | f165≤0.50, f259≤0.54, f14>0.49, f259>0.28 | 1.158007 | 641 |
| 5 | f165≤0.50, f259>0.54, f14≤0.54, f259≤0.86 | 1.126337 | 857 |
| 6 | f165≤0.50, f259>0.54, f14≤0.54, f259>0.86 | 1.396311 | 386 |
| 7 | f165≤0.50, f259>0.54, f14>0.54, f165≤0.20 | 1.869131 | 460 |
| 8 | f165≤0.50, f259>0.54, f14>0.54, f165>0.20 | 1.346590 | 639 |
| 9 | 0.50<f165≤0.70, f259≤0.46, f105≤0.50 | -0.143440 | 510 |
| 10 | 0.50<f165≤0.70, f259≤0.46, f105>0.50 | -0.364609 | 442 |
| 11 | 0.50<f165≤0.70, f259>0.46, f105≤0.50 | -0.368569 | 529 |
| 12 | 0.50<f165≤0.70, f259>0.46, f105>0.50 | -0.584090 | 515 |
| 13 | f165>0.70, f14≤0.45, f105≤0.49 | 0.169210 | 638 |
| 14 | f165>0.70, f14≤0.45, f105>0.49 | 0.550530 | 730 |
| 15 | f165>0.70, f14>0.45, f105≤0.56 | -0.241965 | 903 |
| 16 | f165>0.70, f14>0.45, f105>0.56 | 0.145418 | 711 |

---

## 4. Implementation

### 4.1 Code Structure

The `framework_91.py` file implements:

1. **predict_decision_tree()** - Main prediction function with nested if-else logic
2. **cond_eval()** / **multi_cond_eval()** - Legacy condition evaluation (compatibility)
3. **predict_single()** / **predict_batch()** - User-friendly interfaces
4. **evaluate_predictions()** - Calculates RMSE, MAE, R² metrics

### 4.2 Pure Python Implementation

```python
def predict(f165, f259, f14, f105):
    if f165 <= 0.499972:
        if f259 <= 0.539155:
            if f14 <= 0.488796:
                if f165 <= 0.201313:
                    return 0.428199
                else:
                    return 0.821636
            else:
                if f259 <= 0.282283:
                    return 0.895559
                else:
                    return 1.158007
        else:
            if f14 <= 0.537704:
                if f259 <= 0.857818:
                    return 1.126337
                else:
                    return 1.396311
            else:
                if f165 <= 0.200357:
                    return 1.869131
                else:
                    return 1.346590
    else:
        if f165 <= 0.700854:
            if f259 <= 0.462765:
                if f105 <= 0.502041:
                    return -0.143440
                else:
                    return -0.364609
            else:
                if f105 <= 0.495375:
                    return -0.368569
                else:
                    return -0.584090
        else:
            if f14 <= 0.454465:
                if f105 <= 0.488271:
                    return 0.169210
                else:
                    return 0.550530
            else:
                if f105 <= 0.557603:
                    return -0.241965
                else:
                    return 0.145418
```

---

## 5. Evaluation Results

### 5.1 Final Performance

| Metric | Value |
|--------|-------|
| **RMSE** | **0.2567** |
| **MAE** | 0.1980 |
| **R²** | **0.8757** |

### 5.2 Comparison with Previous Approach

| Method | RMSE | R² | Improvement |
|--------|------|----|-------------|
| Baseline (mean) | 0.7282 | 0.0 | - |
| 6-bucket rules (v1) | 0.3541 | 0.7636 | 51.4% |
| **16-bucket DT rules (v2)** | **0.2567** | **0.8757** | **64.7%** |

### 5.3 Why Decision Tree Rules Are Better

1. **More Features**: Uses 4 features (feat_165, feat_259, feat_14, feat_105) vs 2
2. **More Buckets**: 16 leaf nodes vs 6 segments
3. **Optimal Thresholds**: Thresholds are optimized by the algorithm
4. **Captures Interactions**: Tree structure captures feature interactions

---

## 6. Output Files

### 6.1 Generated Files

| File | Description |
|------|-------------|
| `framework_91.py` | Main prediction script (updated) |
| `outputs/EVAL_target02_91.csv` | Predictions for evaluation data |
| `analysis/07_task2_target02_rules.md` | This documentation |

### 6.2 Prediction Statistics (EVAL Data)

| Statistic | Value |
|-----------|-------|
| Mean | 0.5078 |
| Std | 0.6847 |
| Min | -0.5841 |
| Max | 1.8691 |

---

## 7. Usage Instructions

### 7.1 Run Prediction with Validation

```bash
python framework_91.py --validate
```

### 7.2 Run Prediction Only

```bash
python framework_91.py
```

### 7.3 Custom Input/Output Paths

```bash
python framework_91.py --eval_file_path path/to/data.csv --output_path path/to/output.csv
```

---

## 8. Edge Device Considerations

### 8.1 Computational Requirements
- **Operations per prediction:** Maximum 4 comparisons (tree depth = 4)
- **Memory:** 15 threshold values + 16 prediction constants
- **Dependencies:** Only numpy (can be replaced with pure Python)

### 8.2 Execution Time
- Pure if-else logic: O(1) time complexity
- No loops, no function calls (inlinable)
- Suitable for microcontrollers and embedded systems

---

## 9. Conclusions

### What I Did
1. Used correlation analysis to identify initial candidate features
2. Trained Decision Tree (depth=4) to find optimal features and thresholds
3. Identified 4 key features: feat_165, feat_259, feat_14, feat_105
4. Extracted 16 rules from the tree structure
5. Implemented as nested if-else for edge deployment
6. Achieved R² = 0.8757 (vs 0.7636 with simple rules)

### Why Decision Tree Approach Works Better
- Captures non-linear relationships automatically
- Finds optimal split points algorithmically
- Identifies important feature interactions
- 16 buckets provide finer granularity than 6

### Performance Summary
- **RMSE improved**: 0.3541 → 0.2567 (27% reduction)
- **R² improved**: 0.7636 → 0.8757 (15% increase)
- **Still edge-deployable**: Only if-else comparisons

---

*Document updated with Decision Tree analysis*  
*ID: 91*
