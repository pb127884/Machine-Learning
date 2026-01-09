# config.py - project settings
import os

SEED = 42

# data files
DATA_X_PATH = "data/dataset_91.csv"
DATA_Y_PATH = "data/target_91.csv"
DATA_EVAL_PATH = "data/EVAL_91.csv"
TARGET_COL = "target01"

# output folders
OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
LOG_DIR = os.path.join(OUT_DIR, "logs")

# analysis params
MAX_FEATURE_HISTS = 20
CORR_TOP_FEATURES = 50

# create folders if needed
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
