import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"

def load_xgb_pipeline():
    """Load the trained XGBoost pipeline from model registry."""
    path = REGISTRY_DIR / "xgboost_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run trainer first.")
    return joblib.load(path)

def explain_dataset(X: pd.DataFrame, save_importance: bool = True) -> tuple:
    """
    Explain all predictions in X using SHAP TreeExplainer.
    
    Returns:
        explainer, shap_values (SHAP Explainer objects), and X_preprocessed (numpy array)
    """
    pipeline = load_xgb_pipeline()
    
    # 1. Preprocess features using pipeline's imputer and scaler steps
    X_imputed = pipeline.named_steps['imputer'].transform(X)
    X_scaled = pipeline.named_steps['scaler'].transform(X_imputed)
    X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2. Extract final model
    model = pipeline.named_steps['model']
    
    # 3. Create explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_preprocessed)
    
    if save_importance:
        # Calculate global mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance_dict = dict(zip(X.columns, [float(x) for x in mean_abs_shap]))
        # Sort by importance descending
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Save to registry
        out_path = REGISTRY_DIR / "feature_importance.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(importance_dict, f, indent=4)
        print(f"Saved global feature importance to {out_path}")
        
    return explainer, shap_values, X_preprocessed

def get_top_3_shap_contributors(df_row: pd.DataFrame) -> list:
    """
    Find top 3 SHAP contributors (features driving the prediction away from baseline)
    for a single daily feature vector.
    
    df_row: DataFrame with 1 row containing features.
    
    Returns:
        list of dicts: [{'feature': '...', 'shap_value': float, 'feature_value': float}]
    """
    pipeline = load_xgb_pipeline()
    X_imputed = pipeline.named_steps['imputer'].transform(df_row)
    X_scaled = pipeline.named_steps['scaler'].transform(X_imputed)
    X_preprocessed = pd.DataFrame(X_scaled, columns=df_row.columns)
    
    model = pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_preprocessed)[0] # first row
    
    contributors = []
    for col, val, shap_val in zip(df_row.columns, df_row.iloc[0], shap_vals):
        contributors.append({
            'feature': col,
            'shap_value': float(shap_val),
            'feature_value': float(val)
        })
        
    # Sort by absolute SHAP values descending
    contributors = sorted(contributors, key=lambda x: abs(x['shap_value']), reverse=True)
    return contributors[:3]

def plot_shap_summary(shap_values, X_preprocessed, save_path: str = None):
    """Generate and save SHAP summary plot."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_preprocessed, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved SHAP summary plot to {save_path}")
    plt.close()
