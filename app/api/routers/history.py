import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db import crud
from app.api import schemas

router = APIRouter()

@router.get("/history/{user_id}", response_model=list[schemas.PredictionHistoryItem], tags=["History"])
def get_history(user_id: str, limit: int = Query(30, description="Max number of historical records to return"), db: Session = Depends(get_db)):
    """
    Get prediction history (predicted score, label, SHAP drivers, advice, actual rating)
    for a specific user.
    """
    records = crud.get_user_predictions_history(db, user_id, limit=limit)
    response = []
    for r in records:
        top_feats = json.loads(r.top_features_json) if r.top_features_json else []
        advice = json.loads(r.advice_json) if r.advice_json else []
        
        response.append(schemas.PredictionHistoryItem(
            date=r.date,
            predicted_score=r.predicted_score,
            predicted_label=r.predicted_label,
            anomaly_flag=r.anomaly_flag,
            top_features=top_feats,
            advice=advice,
            actual_rating=r.actual_rating,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ))
    return response

@router.get("/anomalies/{user_id}", response_model=list[schemas.PredictionHistoryItem], tags=["History"])
def get_anomalies(user_id: str, db: Session = Depends(get_db)):
    """
    Get all daily predictions that were flagged as behaviorally anomalous for a specific user.
    """
    records = crud.get_user_anomalies(db, user_id)
    response = []
    for r in records:
        top_feats = json.loads(r.top_features_json) if r.top_features_json else []
        advice = json.loads(r.advice_json) if r.advice_json else []
        
        response.append(schemas.PredictionHistoryItem(
            date=r.date,
            predicted_score=r.predicted_score,
            predicted_label=r.predicted_label,
            anomaly_flag=r.anomaly_flag,
            top_features=top_feats,
            advice=advice,
            actual_rating=r.actual_rating,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ))
    return response
