"""
Model definitions for the ML pipeline.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    HistGradientBoostingRegressor,
    StackingRegressor
)
from src.config import SEED
from src.preprocessing import get_basic_prep, get_scaled_prep
from src.logger import get_logger

logger = get_logger("models")


def get_models():
    """
    Get dictionary of all baseline models.
    
    Returns:
        dict: Model name -> Pipeline
    """
    logger.info("Initializing baseline models")
    
    basic_prep = get_basic_prep()
    scaled_prep = get_scaled_prep()
    
    ridge = Pipeline([
        ("prep", scaled_prep), 
        ("model", Ridge(alpha=1.0, random_state=SEED))
    ])
    
    dt = Pipeline([
        ("prep", get_basic_prep()), 
        ("model", DecisionTreeRegressor(
            max_depth=8, 
            min_samples_leaf=20, 
            random_state=SEED
        ))
    ])
    
    rf = Pipeline([
        ("prep", get_basic_prep()), 
        ("model", RandomForestRegressor(
            n_estimators=300, 
            max_features="sqrt", 
            random_state=SEED, 
            n_jobs=1
        ))
    ])
    
    hgb = Pipeline([
        ("prep", get_basic_prep()),
        ("model", HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=SEED
        ))
    ])
    
    stack = Pipeline([
        ("prep", get_basic_prep()),
        ("model", StackingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=1.0, random_state=SEED)),
                ("dt", DecisionTreeRegressor(
                    max_depth=8, 
                    min_samples_leaf=20, 
                    random_state=SEED
                )),
                ("rf", RandomForestRegressor(
                    n_estimators=200, 
                    max_features="sqrt", 
                    random_state=SEED, 
                    n_jobs=1
                )),
            ],
            final_estimator=Ridge(alpha=1.0, random_state=SEED),
            passthrough=True
        ))
    ])
    
    models = {
        "Ridge": ridge,
        "DecisionTree": dt,
        "RandomForest": rf,
        "HistGradientBoosting": hgb,
        "Stacking": stack,
    }
    
    logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
    return models
