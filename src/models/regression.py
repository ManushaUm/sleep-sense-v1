from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_random_forest_pipeline(params=None) -> Pipeline:
    """Build a Pipeline with SimpleImputer, StandardScaler, and RandomForestRegressor."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**params))
    ])

def get_xgboost_pipeline(params=None) -> Pipeline:
    """Build a Pipeline with SimpleImputer, StandardScaler, and XGBRegressor."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(**params))
    ])

def get_hyperparameter_grids() -> dict:
    """Return hyperparameter grids for hyperparameter optimization."""
    return {
        'rf': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [4, 6, 8],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },
        'xgb': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 4, 5],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.7, 0.8, 0.9]
        }
    }
