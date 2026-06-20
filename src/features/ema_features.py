import pandas as pd
import numpy as np
import json
from src.data.loader import RAW_DIR, load_ema_stress, load_ema_exercise

# Memory cache for parsed EMA json structures
# Structure: {user_id: {date: {feature_name: [values]}}}
_ema_json_cache = {}

def _init_ema_cache(user_id: str):
    """Load and parse EMA JSON files once for the given user, grouping by date."""
    if user_id in _ema_json_cache:
        return
        
    user_cache = {}
    
    # Helper to append values to date-keyed cache
    def add_val(date_str, key, val):
        if date_str not in user_cache:
            user_cache[date_str] = {}
        if key not in user_cache[date_str]:
            user_cache[date_str][key] = []
        user_cache[date_str][key].append(val)
        
    from src.data.preprocessor import ts_to_date_str
    
    # 1. Process Mood
    p_mood = RAW_DIR / "EMA" / "response" / "Mood" / f"Mood_{user_id}.json"
    if p_mood.exists():
        try:
            with open(p_mood, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for rec in data:
                    ts = rec.get("resp_time")
                    if ts:
                        d_str = ts_to_date_str(int(ts))
                        happyornot = rec.get("happyornot")
                        happy_val = rec.get("happy")
                        if happyornot in ['2', 2]:
                            add_val(d_str, 'happy', 0.0)
                        elif happy_val is not None:
                            try:
                                add_val(d_str, 'happy', float(happy_val))
                            except Exception:
                                pass
        except Exception:
            pass
            
    # 2. Process Mood 2
    p_mood2 = RAW_DIR / "EMA" / "response" / "Mood 2" / f"Mood 2_{user_id}.json"
    if p_mood2.exists():
        try:
            with open(p_mood2, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for rec in data:
                    ts = rec.get("resp_time")
                    if ts:
                        d_str = ts_to_date_str(int(ts))
                        how_val = rec.get("how")
                        if how_val in ['1', 1]:
                            add_val(d_str, 'happy', 3.0)
                        elif how_val in ['3', 3]:
                            add_val(d_str, 'tired', 1.0)
                        
                        if how_val is not None and how_val not in ['3', 3]:
                            add_val(d_str, 'tired', 0.0)
        except Exception:
            pass
            
    # 3. Process Social
    p_social = RAW_DIR / "EMA" / "response" / "Social" / f"Social_{user_id}.json"
    if p_social.exists():
        try:
            with open(p_social, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for rec in data:
                    ts = rec.get("resp_time")
                    if ts:
                        d_str = ts_to_date_str(int(ts))
                        num_val = rec.get("number")
                        if num_val is not None:
                            try:
                                add_val(d_str, 'social', float(num_val))
                            except Exception:
                                pass
        except Exception:
            pass
            
    # 4. Process Class
    p_class = RAW_DIR / "EMA" / "response" / "Class" / f"Class_{user_id}.json"
    if p_class.exists():
        try:
            with open(p_class, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for rec in data:
                    ts = rec.get("resp_time")
                    if ts:
                        d_str = ts_to_date_str(int(ts))
                        hours_val = rec.get("hours")
                        if hours_val is not None:
                            try:
                                hrs = max(0.0, float(hours_val) - 1.0)
                                add_val(d_str, 'study_hours', hrs)
                            except Exception:
                                pass
        except Exception:
            pass
            
    _ema_json_cache[user_id] = user_cache


def extract_ema_features(user_id: str, date: str) -> dict:
    """Extract daily self-report EMA features for a given user and date."""
    
    # 1. Stress level from load_ema_stress
    try:
        df_stress = load_ema_stress(user_id)
        df_stress_day = df_stress[df_stress['date'] == date]
        stress_level = float(df_stress_day['stress_level'].mean()) if not df_stress_day.empty else np.nan
    except Exception:
        stress_level = np.nan
        
    # 2. Exercise self report from load_ema_exercise
    try:
        df_ex = load_ema_exercise(user_id)
        df_ex_day = df_ex[df_ex['date'] == date]
        exercise_self_report = int(df_ex_day['exercise_bool'].max()) if not df_ex_day.empty else 0
    except Exception:
        exercise_self_report = 0
        
    # Initialize cache for JSON files
    _init_ema_cache(user_id)
    user_cache = _ema_json_cache[user_id]
    
    mood_happy = np.nan
    mood_tired = 0.0
    social_contacts = np.nan
    study_hours_today = 0.0
    
    if date in user_cache:
        day_data = user_cache[date]
        if 'happy' in day_data:
            mood_happy = float(np.mean(day_data['happy']))
        if 'tired' in day_data:
            mood_tired = float(np.mean(day_data['tired']))
        if 'social' in day_data:
            social_contacts = float(np.mean(day_data['social']))
        if 'study_hours' in day_data:
            study_hours_today = float(np.sum(day_data['study_hours']))
            
    return {
        "stress_level": stress_level,
        "mood_happy": mood_happy,
        "mood_tired": mood_tired,
        "social_contacts": social_contacts,
        "exercise_self_report": exercise_self_report,
        "study_hours_today": study_hours_today
    }
