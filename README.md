# Machine Learning Regression Pipeline

A comprehensive machine learning pipeline for regression tasks featuring automated EDA, model selection, hyperparameter tuning, and prediction generation.

## 📋 Overview

This project implements an end-to-end machine learning workflow for regression problems. It includes:

- **Exploratory Data Analysis (EDA)**: Automated statistical checks, correlation analysis, and visualization
- **Data Quality Checks**: Duplicate detection, data leakage prevention, and outlier analysis
- **Model Selection**: Comparison of multiple regression algorithms
- **Hyperparameter Tuning**: Grid search optimization using validation set
- **Evaluation**: Comprehensive metrics (RMSE, MAE, R²) with train/val/test splits

## 🗂️ Project Structure

```
Machine-Learning/
├── python12.py              # Main ML pipeline script
├── python.py                # Alternative/previous version
├── dataset_91.csv           # Feature dataset
├── target_91.csv            # Target variable
├── EVAL_91.csv              # Evaluation dataset for predictions
├── outputs/                 # Generated outputs
│   ├── plots/               # Visualization plots
│   ├── baseline_metrics_table.csv
│   ├── hgb_tuning_no_cv.csv
│   ├── final_metrics_tuned_hgb_no_cv.csv
│   ├── EVAL_target01_91.csv # Official predictions
│   └── EVAL_target01_best.csv
└── README.md
```

## 🔧 Requirements

### Dependencies

```
numpy
pandas
matplotlib
scikit-learn
```

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 🚀 Usage

### Running the Pipeline

```bash
python python12.py
```

**Logs are automatically saved to:** `outputs/run_log_YYYYMMDD_HHMMSS.log`

### Running on Server

```bash
# The script automatically creates a timestamped log file
python python12.py

# Or use nohup for background execution on Linux server:
nohup python python12.py > outputs/run_stdout.log 2>&1 &
```

After the run completes, share the log file from `outputs/run_log_*.log` for analysis.

### Input Data

The pipeline expects the following CSV files in the project root:

| File | Description |
|------|-------------|
| `dataset_91.csv` | Feature matrix (X) |
| `target_91.csv` | Target variable with column `target01` |
| `EVAL_91.csv` | Evaluation set for final predictions |

## 📊 Pipeline Stages

### 1. Data Loading & EDA

- **Statistical Checks**: Mean, median, standard deviation analysis
- **Duplication Detection**: Row and column duplicate identification
- **Distribution Analysis**: Missing values, variance, and outlier detection
- **Correlation Analysis**: Feature-target and feature-feature correlations

### 2. Data Splitting

- **Stratified Split**: 70% train / 15% validation / 15% test
- Uses target binning for stratification to ensure balanced distributions
- **Data Leakage Check**: Validates no overlap between splits

### 3. Preprocessing

Two preprocessing pipelines:

| Pipeline | Steps |
|----------|-------|
| Basic | Median Imputer → Variance Threshold |
| Scaled | Median Imputer → Variance Threshold → Standard Scaler |

> **Note**: `SimpleImputer` checks for null values and imputes with median if found. Current dataset has **no null values** (0%), so no imputation occurs. Kept for robustness with new data.

### 4. Models

| Model | Description |
|-------|-------------|
| **Ridge** | Linear regression with L2 regularization |
| **DecisionTree** | Tree-based regressor (max_depth=8) |
| **RandomForest** | Ensemble of 300 trees |
| **HistGradientBoosting** | Gradient boosting with early stopping |
| **Stacking** | Meta-ensemble combining Ridge, DT, and RF |

### 5. Hyperparameter Tuning

Grid search on HistGradientBoosting with parameters:

- Learning rate: [0.01, 0.03, 0.05, 0.1]
- Max depth: [3, 5, 7, None]
- Max leaf nodes: [15, 31, 63]
- Min samples leaf: [10, 20, 50]
- L2 regularization: [0.0, 0.1, 1.0]
- Max bins: [128, 255]
- Max iterations: [200, 400, 800]

### 6. Final Training & Prediction

- Best model trained on (train + validation) data
- Final evaluation on held-out test set
- Predictions generated for evaluation dataset

## 📈 Output Files

### Plots (in `outputs/plots/`)

- `target_train_hist.png`, `target_val_hist.png`, `target_test_hist.png` - Target distributions
- `corr_feature_target_distribution.png` - Correlation distribution
- `corr_heatmap_top*.png` - Feature correlation heatmap
- `r2_best_model.png` - R² comparison chart
- `hist_*.png` - Feature histograms

### Metrics (in `outputs/`)

- `baseline_metrics_table.csv` - All baseline model comparisons
- `hgb_tuning_no_cv.csv` - Complete hyperparameter tuning results
- `final_metrics_tuned_hgb_no_cv.csv` - Final model performance

### Predictions (in `outputs/`)

- `EVAL_target01_91.csv` - Official submission file
- `EVAL_target01_best.csv` - Best model predictions

## ⚙️ Configuration

Key parameters in `python12.py`:

```python
SEED = 42                    # Random seed for reproducibility
TARGET_COL = "target01"      # Target column name
MAX_FEATURE_HISTS = 20       # Number of feature histograms to generate
CORR_TOP_FEATURES = 50       # Top features for correlation heatmap
```

## 📉 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error - primary selection metric |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of Determination |
| **GAP** | Validation RMSE - Train RMSE (overfitting indicator) |

## 🔍 Key Features

- ✅ **Stratified Splitting**: Ensures balanced target distribution across splits
- ✅ **Data Leakage Prevention**: Validates no sample overlap between sets
- ✅ **Automated EDA**: Comprehensive analysis with visualizations
- ✅ **Early Stopping**: Prevents overfitting in gradient boosting
- ✅ **Reproducibility**: Fixed random seed throughout

## 📝 License

This project is for educational and research purposes.

## 👤 Author

Machine Learning Project
