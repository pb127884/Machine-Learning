"""
Configuration constants and paths for the ML pipeline.
"""
import os

# Random seed for reproducibility
SEED = 42

# Data paths
DATA_X_PATH = "data/dataset_91.csv"
DATA_Y_PATH = "data/target_91.csv"
DATA_EVAL_PATH = "data/EVAL_91.csv"
TARGET_COL = "target01"

# Output directories
OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
LOG_DIR = os.path.join(OUT_DIR, "logs")

# EDA parameters
MAX_FEATURE_HISTS = 20
CORR_TOP_FEATURES = 50

# Ensure directories exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
