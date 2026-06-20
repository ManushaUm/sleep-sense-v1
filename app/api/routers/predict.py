import os
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pathlib import Path

from app.api import schemas
from src.db.database import get_db
from src.db import crud
from src.data.preprocessor import score_to_label
from src.evaluation.explainability import get_top_3_shap_contributors
from src.advice.generator import generate_advice

router = APIRouter()

REGISTRY_DIR = Path(__file__).resolve().parents[3] / "models" / "registry"
FEATURES_DIR = Path(__file__).resolve().parents[3] / "data" / "preprocessed" / "features"

# Load models safely (None if not found, endpoints will raise 500 error if missing)
xgb_path = REGISTRY_DIR / "xgboost_model.pkl"
xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None

iso_path = REGISTRY_DIR / "isoforest_model.pkl"
iso_model = joblib.load(iso_path) if iso_path.exists() else None

@router.post("/predict", response_model=schemas.PredictionResponse, tags=["Predict"])
def predict_anonymous(features: schemas.DailyFeaturesInput):
    """
    Predict sleep quality score and label anonymously from a raw features input vector.
    """
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not loaded in model registry.")
        
    # Convert input to DataFrame (1 row) and align column order
    input_dict = features.dict()
    df_row = pd.DataFrame([input_dict])
    if hasattr(xgb_model, 'feature_names_in_'):
        df_row = df_row[list(xgb_model.feature_names_in_)]
    
    # Predict continuous sleep score
    pred_score = float(xgb_model.predict(df_row)[0])
    pred_score = float(np.clip(pred_score, 0.0, 3.0))
    pred_label = score_to_label(pred_score)
    
    # Get top 3 SHAP contributors
    try:
        top_features = get_top_3_shap_contributors(df_row)
    except Exception:
        top_features = []
        
    # Generate sleep advice
    try:
        advice_list = generate_advice(top_features)
    except Exception:
        advice_list = [
            "Maintain a consistent sleep schedule.",
            "Limit screen time 1 hour before bed.",
            "Try light daytime walking or physical activity."
        ]
        
    return schemas.PredictionResponse(
        predicted_score=pred_score,
        predicted_label=pred_label,
        anomaly_flag=0, # Anonymous predict defaults to normal anomaly status
        top_features=top_features,
        advice=advice_list
    )

@router.post("/predict/{user_id}", response_model=schemas.PredictionResponse, tags=["Predict"])
def predict_user(user_id: str, date: str = Query(..., description="Date YYYY-MM-DD"), db: Session = Depends(get_db)):
    """
    Predict sleep quality score, label, and anomaly status for a stored user and date.
    Looks up daily features in database or preprocessed CSV cache, computes prediction,
    saves it to predictions database table, and returns the response.
    """
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not loaded in model registry.")
        
    # 1. Look up daily features
    db_feat = crud.get_daily_features(db, user_id, date)
    import json
    
    if db_feat:
        features_dict = json.loads(db_feat.features_json)
    else:
        # Try local CSV cache
        csv_path = FEATURES_DIR / f"{user_id}_features.csv"
        if not csv_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Daily features not found for user {user_id} in database or cache."
            )
        try:
            df_user = pd.read_csv(csv_path)
            # Find matching date
            df_day = df_user[df_user['date'] == date]
            if df_day.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No features found for user {user_id} on date {date} in cache."
                )
            # Convert row to dictionary (ignoring non-feature target columns)
            exclude_cols = ['user_id', 'date', 'sleep_score']
            features_dict = {c: float(df_day.iloc[0][c]) for c in df_day.columns if c not in exclude_cols}
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error reading user features cache: {str(e)}")
            
    # Convert features to 1-row DataFrame and align column order
    validated_features = schemas.DailyFeaturesInput(**features_dict)
    df_row = pd.DataFrame([validated_features.dict()])
    if hasattr(xgb_model, 'feature_names_in_'):
        df_row = df_row[list(xgb_model.feature_names_in_)]
    
    # 2. Predict sleep score and label
    pred_score = float(xgb_model.predict(df_row)[0])
    pred_score = float(np.clip(pred_score, 0.0, 3.0))
    pred_label = score_to_label(pred_score)
    
    # 3. Anomaly detection using Isolation Forest
    anomaly_flag = 0
    if iso_model is not None:
        try:
            df_anomaly = df_row.copy()
            df_anomaly['user_id'] = user_id
            df_anomaly['date'] = date
            anomaly_flag = int(iso_model.predict_anomaly_flag(df_anomaly)[0])
        except Exception:
            pass
            
    # 4. Get top 3 SHAP contributors
    try:
        top_features = get_top_3_shap_contributors(df_row)
    except Exception:
        top_features = []
        
    # 5. Generate sleep advice
    try:
        advice_list = generate_advice(top_features)
    except Exception:
        advice_list = [
            "Maintain a consistent sleep schedule.",
            "Limit screen time 1 hour before bed.",
            "Try light daytime walking or physical activity."
        ]
        
    # 6. Save prediction to DB
    try:
        crud.create_or_update_user(db, user_id=user_id)
        crud.create_or_update_prediction(
            db=db,
            user_id=user_id,
            date=date,
            predicted_score=pred_score,
            predicted_label=pred_label,
            anomaly_flag=anomaly_flag,
            top_features=top_features,
            advice_list=advice_list
        )
    except Exception:
        pass
        
    return schemas.PredictionResponse(
        user_id=user_id,
        date=date,
        predicted_score=pred_score,
        predicted_label=pred_label,
        anomaly_flag=anomaly_flag,
        top_features=top_features,
        advice=advice_list
    )
