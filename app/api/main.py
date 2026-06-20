import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.db.database import engine, Base, get_db
from src.db import crud, models
from src.data.feature_store import build_dataset
from app.api.routers import health, predict, advice, history
from app.api import schemas
from src.data.preprocessor import score_to_label

# Create database tables at startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SleepSense API",
    description="Passive Daytime Behavior Sleep Quality Predictor API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(advice.router)
app.include_router(history.router)

@app.get("/users", response_model=list[schemas.UserResponse], tags=["Users"])
def list_users(db: Session = Depends(get_db)):
    """List all ingested users stored in the SQLite database."""
    users = crud.get_all_users(db)
    response = []
    for u in users:
        p_dict = json.loads(u.personality_json) if u.personality_json else None
        response.append(schemas.UserResponse(
            user_id=u.user_id,
            psqi_pre_score=u.psqi_pre_score,
            psqi_post_score=u.psqi_post_score,
            personality=p_dict
        ))
    return response

@app.post("/users/{user_id}/ingest", tags=["Users"])
def ingest_user_data(user_id: str, db: Session = Depends(get_db)):
    """
    Run ETL pipelines to build and load feature vectors for a specific user into the database.
    """
    from src.data.loader import get_all_users
    all_users = get_all_users()
    if user_id not in all_users:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in raw data files.")
        
    try:
        # Build features for this user
        df_user = build_dataset([user_id])
        if df_user.empty:
            raise HTTPException(status_code=400, detail=f"No valid sleep EMA ratings or features found for user {user_id}.")
            
        # 1. Create or update user
        psqi_val = float(df_user.iloc[0]['psqi_pre_score']) if 'psqi_pre_score' in df_user.columns else None
        
        # Extract personality
        pers_dict = {
            'extraversion': float(df_user.iloc[0]['personality_extraversion']),
            'agreeableness': float(df_user.iloc[0]['personality_agreeableness']),
            'conscientiousness': float(df_user.iloc[0]['personality_conscientiousness']),
            'neuroticism': float(df_user.iloc[0]['personality_neuroticism']),
            'openness': float(df_user.iloc[0]['personality_openness'])
        } if 'personality_extraversion' in df_user.columns else None
        
        crud.create_or_update_user(
            db=db,
            user_id=user_id,
            psqi_pre_score=psqi_val,
            personality_dict=pers_dict
        )
        
        # 2. Load all feature rows into daily_features table
        exclude_cols = ['user_id', 'date', 'sleep_score']
        feature_cols = [c for c in df_user.columns if c not in exclude_cols]
        
        for _, row in df_user.iterrows():
            date_str = row['date']
            feats_dict = {c: float(row[c]) for c in feature_cols}
            crud.create_or_update_daily_features(
                db=db,
                user_id=user_id,
                date=date_str,
                features_dict=feats_dict
            )
            
            # Post-hoc target rating alignment in DB predictions
            target_rating = score_to_label(float(row['sleep_score'])) if 'sleep_score' in df_user.columns else None
            if target_rating:
                db_pred = crud.get_prediction(db, user_id, date_str)
                if db_pred:
                    db_pred.actual_rating = target_rating
                    db.commit()
                    
        return {
            "status": "success",
            "message": f"Successfully ingested {len(df_user)} days of data for user {user_id} into database."
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"ETL Ingestion failed: {str(e)}")
