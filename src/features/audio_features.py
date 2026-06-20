import pandas as pd
import numpy as np
from src.data.loader import load_audio, RAW_DIR
from src.data.preprocessor import add_date_column

# Local cache for conversation data
_conversation_cache = {}

def extract_audio_features(user_id: str, date: str) -> dict:
    """Extract audio and conversation features for a given user and date."""
    default_features = {
        "silence_ratio": 1.0,
        "conversation_ratio": 0.0,
        "social_audio_evening": 0.0
    }
    
    # 1. Silence ratio from audio inference
    try:
        df_audio = load_audio(user_id)
    except Exception:
        df_audio = pd.DataFrame()
        
    silence_ratio = 1.0
    if not df_audio.empty:
        df_audio = add_date_column(df_audio)
        df_audio_day = df_audio[df_audio['date'] == date]
        if not df_audio_day.empty:
            silence_samples = len(df_audio_day[df_audio_day['audio_inference'] == 0])
            silence_ratio = float(silence_samples / len(df_audio_day))
            
    # 2. Conversation ratio and social audio evening from conversation logs
    if user_id not in _conversation_cache:
        p_conv = RAW_DIR / "sensing" / "conversation" / f"conversation_{user_id}.csv"
        if not p_conv.exists():
            _conversation_cache[user_id] = pd.DataFrame()
        else:
            try:
                df_conv = pd.read_csv(p_conv)
                df_conv.columns = [c.strip() for c in df_conv.columns]
                df_conv['start_timestamp'] = pd.to_numeric(df_conv['start_timestamp'], errors='coerce')
                df_conv['end_timestamp'] = pd.to_numeric(df_conv['end_timestamp'], errors='coerce')
                df_conv = df_conv.dropna()
                
                if df_conv.empty:
                    _conversation_cache[user_id] = pd.DataFrame()
                else:
                    dt_series = pd.to_datetime(df_conv['start_timestamp'], unit='s', utc=True)
                    df_conv['date'] = dt_series.dt.strftime('%Y-%m-%d')
                    df_conv['start_hour'] = dt_series.dt.hour + dt_series.dt.minute / 60.0 + dt_series.dt.second / 3600.0
                    df_conv['duration'] = df_conv['end_timestamp'] - df_conv['start_timestamp']
                    _conversation_cache[user_id] = df_conv
            except Exception:
                _conversation_cache[user_id] = pd.DataFrame()
                
    df_conv = _conversation_cache[user_id]
    conversation_ratio = 0.0
    social_audio_evening = 0.0
    
    if not df_conv.empty:
        df_day = df_conv[df_conv['date'] == date]
        if not df_day.empty:
            total_conv_sec = df_day['duration'].sum()
            # Ratio of the 24h day (86400 seconds)
            conversation_ratio = float(total_conv_sec / 86400.0)
            
            # Conversation starting after 19:00 (7 PM)
            evening_conv = df_day[df_day['start_hour'] >= 19.0]
            evening_conv_sec = evening_conv['duration'].sum()
            # Convert to minutes
            social_audio_evening = float(evening_conv_sec / 60.0)
            
    return {
        "silence_ratio": silence_ratio,
        "conversation_ratio": conversation_ratio,
        "social_audio_evening": social_audio_evening
    }
