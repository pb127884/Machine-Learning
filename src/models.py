# models.py - model definitions
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from src.config import SEED
from src.preprocessing import get_basic_pipeline, get_scaled_pipeline
from src.logger import get_logger

log = get_logger("models")

def get_all_models():
    log.info("setting up models")
    
    basic = get_basic_pipeline()
    scaled = get_scaled_pipeline()
    
    models = {
        # linear model - needs scaling
        "Ridge": Pipeline([
            ("prep", scaled), 
            ("model", Ridge(alpha=1.0, random_state=SEED))
        ]),
        
        # decision tree
        "DecisionTree": Pipeline([
            ("prep", get_basic_pipeline()), 
            ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED))
        ]),
        
        # random forest
        "RandomForest": Pipeline([
            ("prep", get_basic_pipeline()), 
            ("model", RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=SEED, n_jobs=1))
        ]),
        
        # gradient boosting - my main model
        "HistGradientBoosting": Pipeline([
            ("prep", get_basic_pipeline()),
            ("model", HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=SEED
            ))
        ]),
        
        # stacking ensemble
        "Stacking": Pipeline([
            ("prep", get_basic_pipeline()),
            ("model", StackingRegressor(
                estimators=[
                    ("ridge", Ridge(alpha=1.0, random_state=SEED)),
                    ("dt", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED)),
                    ("rf", RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=1)),
                ],
                final_estimator=Ridge(alpha=1.0, random_state=SEED),
                passthrough=True
            ))
        ])
    }
    
    log.info(f"defined {len(models)} models")
    return models
