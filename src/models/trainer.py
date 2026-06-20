import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.evaluation.metrics import evaluate_metrics
from src.models.baseline import UserMeanBaseline
from src.models.regression import get_random_forest_pipeline, get_xgboost_pipeline
from src.models.anomaly import UserIsolationForest
from src.models.anomaly_ae import UserAutoencoderAnomalyDetector

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"

def load_data(parquet_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed merged dataset and split into features and targets."""
    if parquet_path is None:
        parquet_path = Path(__file__).resolve().parents[2] / "data" / "preprocessed" / "merged" / "dataset.parquet"
        
    df = pd.read_parquet(parquet_path)
    
    # Target
    y = df['sleep_score']
    
    # Feature columns (everything except user_id, date, sleep_score)
    exclude_cols = ['user_id', 'date', 'sleep_score']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    user_ids = df['user_id']
    
    return df, X, y, user_ids

def run_louo_cv(model_name: str, X: pd.DataFrame, y: pd.Series, user_ids: pd.Series) -> tuple[np.ndarray, dict]:
    """
    Run Leave-One-User-Out (LOUO) cross-validation for a given model.
    """
    unique_users = user_ids.unique()
    y_pred = np.zeros(len(y))
    
    # Re-insert user_id temporarily for splitting and baseline
    X_full = X.copy()
    X_full['user_id'] = user_ids
    
    for uid in unique_users:
        train_idx = user_ids != uid
        val_idx = user_ids == uid
        
        X_train = X_full[train_idx]
        y_train = y[train_idx]
        
        X_val = X_full[val_idx]
        
        # Fit model
        if model_name == 'baseline':
            model = UserMeanBaseline()
            model.fit(X_train, y_train)
        elif model_name == 'rf':
            model = get_random_forest_pipeline()
            model.fit(X_train.drop(columns=['user_id']), y_train)
        elif model_name == 'xgb':
            model = get_xgboost_pipeline()
            model.fit(X_train.drop(columns=['user_id']), y_train)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        # Predict
        if model_name == 'baseline':
            preds = model.predict(X_val)
        else:
            preds = model.predict(X_val.drop(columns=['user_id']))
            
        y_pred[val_idx] = preds
        
    # Evaluate overall metrics
    metrics = evaluate_metrics(y, y_pred)
    return y_pred, metrics

def train_and_register_best_model():
    """
    Load dataset, train baseline, RF, and XGBoost using LOUO-CV, log experiments,
    and fit the final models on the ENTIRE dataset to save in registry.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    
    df, X, y, user_ids = load_data()
    
    print("Running Leave-One-User-Out (LOUO) CV for baseline...")
    _, baseline_metrics = run_louo_cv('baseline', X, y, user_ids)
    print("Baseline metrics:", baseline_metrics)
    
    print("Running LOUO CV for Random Forest...")
    _, rf_metrics = run_louo_cv('rf', X, y, user_ids)
    print("Random Forest metrics:", rf_metrics)
    
    print("Running LOUO CV for XGBoost...")
    _, xgb_metrics = run_louo_cv('xgb', X, y, user_ids)
    print("XGBoost metrics:", xgb_metrics)
    
    # Log results to experiments.csv
    exp_path = REGISTRY_DIR / "experiments.csv"
    
    records = []
    if exp_path.exists():
        try:
            records = pd.read_csv(exp_path).to_dict(orient='records')
        except Exception:
            pass
            
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for name, met in [('baseline', baseline_metrics), ('rf', rf_metrics), ('xgb', xgb_metrics)]:
        row = {
            'timestamp': timestamp_str,
            'model': name,
            'mae': met['mae'],
            'rmse': met['rmse'],
            'binary_accuracy': met['binary_accuracy'],
            'ordinal_accuracy': met['ordinal_accuracy'],
            'exact_accuracy': met['exact_accuracy']
        }
        records.append(row)
        
    pd.DataFrame(records).to_csv(exp_path, index=False)
    print(f"Logged experiment results to {exp_path}")
    
    # Fit the best model (XGBoost) on the entire dataset and save it
    print("Training final XGBoost model on entire dataset...")
    final_xgb_pipeline = get_xgboost_pipeline()
    final_xgb_pipeline.fit(X, y)
    
    xgb_path = REGISTRY_DIR / "xgboost_model.pkl"
    joblib.dump(final_xgb_pipeline, xgb_path)
    print(f"Saved final XGBoost model pipeline to {xgb_path}")
    
    # Fit and save final Random Forest model pipeline
    print("Training final Random Forest model on entire dataset...")
    final_rf_pipeline = get_random_forest_pipeline()
    final_rf_pipeline.fit(X, y)
    
    rf_path = REGISTRY_DIR / "rf_model.pkl"
    joblib.dump(final_rf_pipeline, rf_path)
    print(f"Saved final Random Forest model pipeline to {rf_path}")
    
    # Extract and save the scaler from the pipeline
    scaler = final_xgb_pipeline.named_steps['scaler']
    scaler_path = REGISTRY_DIR / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")
    
    # Fit and save final Isolation Forest anomaly detection model
    print("Training final Isolation Forest anomaly detection model per user...")
    final_iso_forest = UserIsolationForest()
    # Re-insert user_id and date for grouping in fit
    df_for_anomaly = X.copy()
    df_for_anomaly['user_id'] = user_ids
    final_iso_forest.fit(df_for_anomaly)
    
    iso_path = REGISTRY_DIR / "isoforest_model.pkl"
    joblib.dump(final_iso_forest, iso_path)
    print(f"Saved final Isolation Forest model to {iso_path}")
    
    # Fit and save final PyTorch Autoencoder anomaly detector
    print("Training final PyTorch Autoencoder anomaly detection model per user...")
    final_ae_detector = UserAutoencoderAnomalyDetector()
    final_ae_detector.fit(df_for_anomaly)
    
    ae_path = REGISTRY_DIR / "autoencoder_model.pkl"
    joblib.dump(final_ae_detector, ae_path)
    print(f"Saved final PyTorch Autoencoder model to {ae_path}")

if __name__ == '__main__':
    train_and_register_best_model()
