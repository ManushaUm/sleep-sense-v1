import json
from sqlalchemy.orm import Session
from src.db import models

def get_user(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def get_all_users(db: Session):
    return db.query(models.User).all()

def create_or_update_user(db: Session, user_id: str, psqi_pre_score: float = None, psqi_post_score: float = None, personality_dict: dict = None, hashed_password: str = None, profile_picture_url: str = None):
    db_user = get_user(db, user_id)
    p_json = json.dumps(personality_dict) if personality_dict else None
    
    if db_user:
        if psqi_pre_score is not None:
            db_user.psqi_pre_score = psqi_pre_score
        if psqi_post_score is not None:
            db_user.psqi_post_score = psqi_post_score
        if p_json is not None:
            db_user.personality_json = p_json
        if hashed_password is not None:
            db_user.hashed_password = hashed_password
        if profile_picture_url is not None:
            db_user.profile_picture_url = profile_picture_url
    else:
        db_user = models.User(
            user_id=user_id,
            hashed_password=hashed_password,
            psqi_pre_score=psqi_pre_score,
            psqi_post_score=psqi_post_score,
            personality_json=p_json,
            profile_picture_url=profile_picture_url
        )
        db.add(db_user)
        
    db.commit()
    db.refresh(db_user)
    return db_user

def get_daily_features(db: Session, user_id: str, date: str):
    return db.query(models.DailyFeatures).filter(
        models.DailyFeatures.user_id == user_id,
        models.DailyFeatures.date == date
    ).first()

def create_or_update_daily_features(db: Session, user_id: str, date: str, features_dict: dict):
    db_feat = get_daily_features(db, user_id, date)
    
    if db_feat:
        try:
            existing = json.loads(db_feat.features_json)
        except Exception:
            existing = {}
        # Merge: update existing features with newly provided non-None keys
        existing.update({k: v for k, v in features_dict.items() if v is not None})
        f_json = json.dumps(existing)
        db_feat.features_json = f_json
    else:
        # Keep only non-None values for initial creation to let Pydantic defaults handle rest
        f_json = json.dumps({k: v for k, v in features_dict.items() if v is not None})
        db_feat = models.DailyFeatures(
            user_id=user_id,
            date=date,
            features_json=f_json
        )
        db.add(db_feat)
        
    db.commit()
    db.refresh(db_feat)
    return db_feat

def get_prediction(db: Session, user_id: str, date: str):
    return db.query(models.Prediction).filter(
        models.Prediction.user_id == user_id,
        models.Prediction.date == date
    ).first()

def create_or_update_prediction(db: Session, user_id: str, date: str,
                                predicted_score: float, predicted_label: str,
                                anomaly_flag: int = 0, top_features: list = None,
                                advice_list: list = None, actual_rating: str = None):
    db_pred = get_prediction(db, user_id, date)
    top_json = json.dumps(top_features) if top_features else None
    adv_json = json.dumps(advice_list) if advice_list else None
    
    if db_pred:
        db_pred.predicted_score = predicted_score
        db_pred.predicted_label = predicted_label
        db_pred.anomaly_flag = anomaly_flag
        if top_json is not None:
            db_pred.top_features_json = top_json
        if adv_json is not None:
            db_pred.advice_json = adv_json
        if actual_rating is not None:
            db_pred.actual_rating = actual_rating
    else:
        db_pred = models.Prediction(
            user_id=user_id,
            date=date,
            predicted_score=predicted_score,
            predicted_label=predicted_label,
            anomaly_flag=anomaly_flag,
            top_features_json=top_json,
            advice_json=adv_json,
            actual_rating=actual_rating
        )
        db.add(db_pred)
        
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_user_predictions_history(db: Session, user_id: str, limit: int = 30):
    return db.query(models.Prediction).filter(
        models.Prediction.user_id == user_id
    ).order_by(models.Prediction.date.desc()).limit(limit).all()

def get_user_anomalies(db: Session, user_id: str):
    return db.query(models.Prediction).filter(
        models.Prediction.user_id == user_id,
        models.Prediction.anomaly_flag == 1
    ).order_by(models.Prediction.date.desc()).all()
