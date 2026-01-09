# Machine Learning Regression Pipeline

A comprehensive, modular ML pipeline for regression tasks with automated EDA, model selection, hyperparameter tuning, and prediction generation.

## � Quick Start

```bash
# Run the pipeline
python main.py

# View logs
cat outputs/logs/run_*.log
```

## 📁 Project Structure

```
Machine-Learning/
├── main.py                   # Entry point
├── src/                      # Source modules
│   ├── config.py             # Configuration
│   ├── logger.py             # Logging setup
│   ├── utils.py              # Utilities
│   ├── eda.py                # EDA functions
│   ├── plotting.py           # Visualizations
│   ├── preprocessing.py      # Data preprocessing
│   ├── models.py             # Model definitions
│   └── tuning.py             # Hyperparameter tuning
├── data/                     # Input data
│   ├── dataset_91.csv
│   ├── target_91.csv
│   └── EVAL_91.csv
├── outputs/                  # Generated outputs
│   ├── logs/                 # Timestamped logs
│   ├── plots/                # Visualizations
│   └── *.csv                 # Results & predictions
├── docs/                     # Step-by-step documentation
└── analysis/                 # Data analysis methodology
```

## 📊 Pipeline Stages

| Step | Description | Documentation |
|------|-------------|---------------|
| 1 | Data Loading | [docs/01_data_loading.md](docs/01_data_loading.md) |
| 2 | EDA & Quality Checks | [docs/02_eda.md](docs/02_eda.md) |
| 3 | Train/Val/Test Split | [docs/03_data_split.md](docs/03_data_split.md) |
| 4 | Preprocessing | [docs/04_preprocessing.md](docs/04_preprocessing.md) |
| 5 | Baseline Models | [docs/05_baseline_models.md](docs/05_baseline_models.md) |
| 6 | Hyperparameter Tuning | [docs/06_hyperparameter_tuning.md](docs/06_hyperparameter_tuning.md) |
| 7 | Final Evaluation | [docs/07_final_evaluation.md](docs/07_final_evaluation.md) |
| 8 | Predictions | [docs/08_predictions.md](docs/08_predictions.md) |

## � Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 📈 Results Summary

| Metric | Baseline | After Tuning | Improvement |
|--------|----------|--------------|-------------|
| Val RMSE | 0.150 | 0.114 | **24%** |
| Test R² | 0.599 | 0.786 | **31%** |

## 📝 Configuration

Edit `src/config.py` to modify:

```python
SEED = 42                    # Random seed
TARGET_COL = "target01"      # Target column
MAX_FEATURE_HISTS = 20       # Histograms to generate
CORR_TOP_FEATURES = 50       # Top features for correlation
```

## � Output Files

| File | Description |
|------|-------------|
| `baseline_metrics_table.csv` | All baseline model results |
| `hgb_tuning_no_cv.csv` | Hyperparameter tuning results |
| `final_metrics_tuned_hgb_no_cv.csv` | Final model metrics |
| `EVAL_target01_91.csv` | Official predictions |

## 📊 Logging

All runs are logged to `outputs/logs/run_YYYYMMDD_HHMMSS.log` with:
- Timestamps
- Module names
- Log levels (INFO, WARNING, ERROR)

## 👤 Author

Machine Learning Project
