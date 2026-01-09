# logger.py - handles logging to console and file
import os
import sys
import logging
from datetime import datetime
from src.config import LOG_DIR

# log file with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def setup_logging(level=logging.INFO):
    # setup both console and file logging
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []
    
    # console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # file output
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def get_logger(name):
    return logging.getLogger(name)

# init logging when module loads
_root = setup_logging()
