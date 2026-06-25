import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db import crud, models
from app.api import security

router = APIRouter()

@router.get("/advice/{user_id}", response_model=list[str], tags=["Advice"])
def get_user_advice(user_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(security.get_current_user)):
    """
    Get the most recent sleep advice suggestions for a specific user.
    """
    if current_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's data.")
    # Fetch the latest prediction record for this user
    history = crud.get_user_predictions_history(db, user_id, limit=1)
    if history:
        pred = history[0]
        if pred.advice_json:
            try:
                return json.loads(pred.advice_json)
            except Exception:
                pass
                
    # Default fallback advice if no predictions logged yet
    return [
        "Try to charge your phone outside the bedroom to avoid late-night screen time.",
        "A 20-minute outdoor walk tomorrow will help stabilize your circadian rhythm.",
        "Schedule tomorrow's study sessions earlier in the day to prevent bedtime stress."
    ]
