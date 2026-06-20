import pandas as pd
import numpy as np
from src.data.loader import RAW_DIR

# Local cache for parsed phonelock data
_phonelock_cache = {}

def extract_phonelock_features(user_id: str, date: str) -> dict:
    """Extract phone lock/unlock features for a given user and date."""
    default_features = {
        "unlock_count_late_night": 0,
        "unlock_count_evening": 0,
        "unlock_count_daytime": 0,
        "first_unlock_hour": np.nan,
        "last_unlock_hour": np.nan,
        "avg_session_duration_min": 0.0,
        "screen_sessions_count": 0
    }
    
    if user_id not in _phonelock_cache:
        p = RAW_DIR / "sensing" / "phonelock" / f"phonelock_{user_id}.csv"
        if not p.exists():
            _phonelock_cache[user_id] = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(p)
                df.columns = [c.strip() for c in df.columns]
                if df.empty:
                    _phonelock_cache[user_id] = pd.DataFrame()
                else:
                    df['start'] = pd.to_numeric(df['start'], errors='coerce')
                    df['end'] = pd.to_numeric(df['end'], errors='coerce')
                    df = df.dropna()
                    if df.empty:
                        _phonelock_cache[user_id] = pd.DataFrame()
                    else:
                        dt_series = pd.to_datetime(df['start'], unit='s', utc=True)
                        df['date'] = dt_series.dt.strftime('%Y-%m-%d')
                        df['start_hour'] = dt_series.dt.hour + dt_series.dt.minute / 60.0 + dt_series.dt.second / 3600.0
                        _phonelock_cache[user_id] = df
            except Exception:
                _phonelock_cache[user_id] = pd.DataFrame()

    df = _phonelock_cache[user_id]
    if df.empty:
        return default_features
        
    df_day = df[df['date'] == date]
    if df_day.empty:
        return default_features
        
    df_day = df_day.copy()
    
    # Calculate features
    late_night = df_day[df_day['start_hour'] >= 22.0]
    evening = df_day[(df_day['start_hour'] >= 20.0) & (df_day['start_hour'] < 22.0)]
    daytime = df_day[(df_day['start_hour'] >= 6.0) & (df_day['start_hour'] < 20.0)]
    
    unlock_count_late_night = len(late_night)
    unlock_count_evening = len(evening)
    unlock_count_daytime = len(daytime)
    
    first_unlock_hour = df_day['start_hour'].min()
    last_unlock_hour = df_day['start_hour'].max()
    
    durations = (df_day['end'] - df_day['start']) / 60.0
    avg_session_duration_min = float(durations.mean()) if not durations.empty else 0.0
    screen_sessions_count = len(df_day)
    
    return {
        "unlock_count_late_night": unlock_count_late_night,
        "unlock_count_evening": unlock_count_evening,
        "unlock_count_daytime": unlock_count_daytime,
        "first_unlock_hour": float(first_unlock_hour) if not pd.isna(first_unlock_hour) else np.nan,
        "last_unlock_hour": float(last_unlock_hour) if not pd.isna(last_unlock_hour) else np.nan,
        "avg_session_duration_min": avg_session_duration_min,
        "screen_sessions_count": screen_sessions_count
    }
