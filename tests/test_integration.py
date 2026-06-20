import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_all_users, load_ema_sleep
from src.features.activity_features import extract_activity_features
from src.advice.generator import generate_advice
from app.api.main import app

client = TestClient(app)

def test_data_loader():
    """Verify that user list and sleep data can be loaded from raw files."""
    users = get_all_users()
    assert len(users) > 0
    assert "u00" in users
    
    # Load u00 sleep EMAs
    df_sleep = load_ema_sleep("u00")
    assert not df_sleep.empty
    assert "date" in df_sleep.columns
    assert "rate_score" in df_sleep.columns

def test_activity_extractor():
    """Verify that the activity feature extractor returns a valid dict schema."""
    feats = extract_activity_features("u00", "2013-03-25")
    assert isinstance(feats, dict)
    expected_keys = [
        "stationary_ratio", "walking_minutes", "running_minutes", 
        "exercise_detected", "peak_activity_hour", "activity_bout_count"
    ]
    for key in expected_keys:
        assert key in feats

def test_model_inference():
    """Verify that the registered XGBoost model can perform inference on standard dummy features."""
    import joblib
    from pathlib import Path
    
    xgb_path = Path(__file__).resolve().parents[1] / "models" / "registry" / "xgboost_model.pkl"
    assert xgb_path.exists(), "Register your model first by running trainer.py"
    
    model = joblib.load(xgb_path)
    
    # Create a dummy features dataframe matching the expected schema and align columns
    from app.api.schemas import DailyFeaturesInput
    dummy_input = DailyFeaturesInput().dict()
    df_row = pd.DataFrame([dummy_input])
    if hasattr(model, 'feature_names_in_'):
        df_row = df_row[list(model.feature_names_in_)]
    
    pred = float(model.predict(df_row)[0])
    assert 0.0 <= pred <= 3.0

def test_advice_generator():
    """Verify that advice generator outputs exactly 3 recommendations."""
    dummy_shap = [
        {'feature': 'unlock_count_late_night', 'shap_value': -0.15, 'feature_value': 4.0},
        {'feature': 'walking_minutes', 'shap_value': 0.12, 'feature_value': 45.0},
        {'feature': 'stress_level', 'shap_value': -0.05, 'feature_value': 4.0}
    ]
    advice = generate_advice(dummy_shap)
    assert len(advice) == 3
    for s in advice:
        assert isinstance(s, str)
        assert len(s) > 0

def test_api_health():
    """Test API /health endpoint."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}

def test_api_users():
    """Test API /users listing endpoint."""
    r = client.get("/users")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_api_predict_anonymous():
    """Test API /predict endpoint with a valid anonymous feature body."""
    from app.api.schemas import DailyFeaturesInput
    body = DailyFeaturesInput().dict()
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    res = r.json()
    assert "predicted_score" in res
    assert "predicted_label" in res
    assert "advice" in res
    assert len(res["advice"]) == 3
