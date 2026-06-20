import pandas as pd
import numpy as np
from src.data.loader import load_activity
from src.data.preprocessor import add_date_column, add_hour_column

def extract_activity_features(user_id: str, date: str) -> dict:
    """Extract accelerometer-based physical activity features for a given user and date."""
    default_features = {
        "stationary_ratio": 1.0,
        "walking_minutes": 0.0,
        "running_minutes": 0.0,
        "exercise_detected": 0,
        "peak_activity_hour": np.nan,
        "activity_bout_count": 0
    }
    
    try:
        df = load_activity(user_id)
    except Exception:
        return default_features
        
    if df.empty:
        return default_features
        
    df = add_date_column(df)
    df_day = df[df['date'] == date]
    
    if df_day.empty:
        return default_features
        
    total_samples = len(df_day)
    stationary_samples = len(df_day[df_day['activity_inference'] == 0])
    walking_samples = len(df_day[df_day['activity_inference'] == 1])
    running_samples = len(df_day[df_day['activity_inference'] == 2])
    
    stationary_ratio = float(stationary_samples / total_samples)
    walking_minutes = float(walking_samples / total_samples * 1440.0)
    running_minutes = float(running_samples / total_samples * 1440.0)
    
    # Exercise detected if running minutes > 5 or walking minutes > 30
    exercise_detected = 1 if (running_minutes > 5.0 or walking_minutes > 30.0) else 0
    
    # Calculate peak activity hour
    df_day = df_day.copy()
    df_day = add_hour_column(df_day)
    df_day['hour_int'] = df_day['hour'].astype(int)
    
    active_df = df_day[df_day['activity_inference'].isin([1, 2])]
    if not active_df.empty:
        peak_hour = active_df.groupby('hour_int').size().idxmax()
        peak_activity_hour = float(peak_hour)
    else:
        peak_activity_hour = np.nan
        
    # Calculate activity bout count (transitions to active state)
    df_day = df_day.sort_values(by='timestamp').reset_index(drop=True)
    df_day['active'] = df_day['activity_inference'].isin([1, 2])
    bout_count = int((df_day['active'] & ~df_day['active'].shift(1).fillna(False)).sum())
    
    return {
        "stationary_ratio": stationary_ratio,
        "walking_minutes": walking_minutes,
        "running_minutes": running_minutes,
        "exercise_detected": exercise_detected,
        "peak_activity_hour": peak_activity_hour,
        "activity_bout_count": bout_count
    }
