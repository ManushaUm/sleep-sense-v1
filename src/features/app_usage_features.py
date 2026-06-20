import pandas as pd
import numpy as np
from src.data.loader import load_app_usage
from src.data.preprocessor import add_date_column, add_hour_column

def get_app_category(package: str) -> str:
    """Classify an app package name into a category."""
    package = str(package).lower()
    if any(x in package for x in ['facebook', 'twitter', 'instagram', 'whatsapp', 'snapchat', 'talk', 'gm', 'messenger', 'viber', 'line', 'skype', 'social']):
        return 'social'
    elif any(x in package for x in ['youtube', 'netflix', 'spotify', 'pandora', 'music', 'gallery', 'video', 'player', 'game', 'play.games', 'entertainment']):
        return 'entertainment'
    elif any(x in package for x in ['dropbox', 'adobe', 'canvas', 'blackboard', 'dictionary', 'evernote', 'calculator', 'docs', 'sheets', 'slides', 'office', 'pdf', 'reader', 'study']):
        return 'study'
    elif any(x in package for x in ['chrome', 'browser', 'firefox', 'opera', 'safari']):
        return 'browser'
    else:
        return 'system'

def extract_app_usage_features(user_id: str, date: str) -> dict:
    """Extract app usage features for a given user and date."""
    default_features = {
        "app_social_min": 0.0,
        "app_entertainment_evening_min": 0.0,
        "app_late_night_min": 0.0,
        "last_active_app_hour": np.nan,
        "app_diversity_count": 0,
        "app_study_sessions": 0.0
    }
    
    try:
        df = load_app_usage(user_id)
    except Exception:
        return default_features
        
    if df.empty:
        return default_features
        
    df = add_date_column(df)
    df_day = df[df['date'] == date]
    
    if df_day.empty:
        return default_features
        
    df_day = df_day.copy()
    df_day = add_hour_column(df_day)
    df_day['category'] = df_day['package_name'].apply(get_app_category)
    
    # App diversity is the number of unique packages accessed
    app_diversity_count = int(df_day['package_name'].nunique())
    
    # Group by timestamp to analyze unique 20-minute intervals
    df_unique = df_day.groupby('timestamp').first().reset_index()
    
    # 1 sample represents 20 minutes
    SAMPLE_INTERVAL_MIN = 20.0
    
    app_social_min = float((df_unique['category'] == 'social').sum() * SAMPLE_INTERVAL_MIN)
    app_entertainment_evening_min = float(
        ((df_unique['category'] == 'entertainment') & (df_unique['hour'] >= 20.0)).sum() * SAMPLE_INTERVAL_MIN
    )
    app_late_night_min = float((df_unique['hour'] >= 22.0).sum() * SAMPLE_INTERVAL_MIN)
    
    last_active_app_hour = df_unique['hour'].max()
    app_study_sessions = float((df_unique['category'] == 'study').sum() * SAMPLE_INTERVAL_MIN)
    
    return {
        "app_social_min": app_social_min,
        "app_entertainment_evening_min": app_entertainment_evening_min,
        "app_late_night_min": app_late_night_min,
        "last_active_app_hour": float(last_active_app_hour) if not pd.isna(last_active_app_hour) else np.nan,
        "app_diversity_count": app_diversity_count,
        "app_study_sessions": app_study_sessions
    }
