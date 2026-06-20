"""
loader.py — Raw data readers for the StudentLife dataset.

Each function reads one modality for one user and returns a clean DataFrame.
All timestamps are Unix epoch (seconds). Date alignment happens in preprocessor.py.

Usage:
    from src.data.loader import load_ema_sleep, load_activity
    df = load_ema_sleep("u00")
"""

from pathlib import Path
import pandas as pd
import json
import re
import functools

# ── Path config ────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Import study window and converters from preprocessor to align data window
STUDY_START_TS = 1362096000   # 2013-03-01 00:00:00 UTC
STUDY_END_TS   = 1372636800   # 2013-07-01 00:00:00 UTC

EMA_RATE_MAP = {
    "Very good": 3,
    "Fairly good": 2,
    "Fairly bad": 1,
    "Very bad": 0,
}

# ── Caching Layer ──────────────────────────────────────────────────────────────
_csv_cache = {}
_json_cache = {}
_processed_df_cache = {}

def clear_loader_cache():
    """Clear the cached dataframes and json structures to free memory."""
    global _csv_cache, _json_cache, _processed_df_cache
    _csv_cache.clear()
    _json_cache.clear()
    _processed_df_cache.clear()

def cache_processed_df(func):
    """Decorator to cache the processed DataFrame results of a loader function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__, args, tuple(sorted(kwargs.items())))
        if key not in _processed_df_cache:
            df = func(*args, **kwargs)
            _processed_df_cache[key] = df
        return _processed_df_cache[key].copy()
    return wrapper

def load_csv_cached(p, **kwargs):
    """Load a CSV file with caching based on path and arguments."""
    kw_tuple = tuple(sorted(kwargs.items()))
    key = (str(p), kw_tuple)
    if key not in _csv_cache:
        _csv_cache[key] = pd.read_csv(p, **kwargs)
    return _csv_cache[key].copy()

def load_json_cached(p):
    """Load a JSON file with caching based on path."""
    key = str(p)
    if key not in _json_cache:
        with open(p, "r", encoding="utf-8") as f:
            _json_cache[key] = json.load(f)
    return _json_cache[key]


def get_all_users() -> list[str]:
    """Return list of user IDs that have activity data."""
    activity_dir = RAW_DIR / "sensing" / "activity"
    return sorted([
        f.stem.replace("activity_", "")
        for f in activity_dir.glob("activity_u*.csv")
    ])


# ── EMA targets ────────────────────────────────────────────────────────────────

@cache_processed_df
def load_ema_sleep(user_id: str) -> pd.DataFrame:
    """
    Load daily EMA sleep responses for a user.

    Returns DataFrame with columns:
        timestamp (int), date (str YYYY-MM-DD),
        hours (float), rate (str), rate_score (int 0–3)
    """
    p = RAW_DIR / "EMA" / "response" / "Sleep" / f"Sleep_{user_id}.json"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "date", "hours", "rate", "rate_score"])
    
    rate_score_map = {
        '0': 0, '1': 1, '2': 2, '3': 3,
        0: 0, 1: 1, 2: 2, 3: 3
    }
    rate_label_map = {
        '0': 'Very bad', '1': 'Fairly bad', '2': 'Fairly good', '3': 'Very good',
        0: 'Very bad', 1: 'Fairly bad', 2: 'Fairly good', 3: 'Very good'
    }
    
    records = []
    try:
        data = load_json_cached(p)
        for rec in data:
            if "rate" not in rec:
                continue
            rate_val = rec.get("rate")
            if rate_val is None:
                continue
            rate_str_key = str(rate_val).strip()
            
            ts = rec.get("resp_time")
            try:
                ts_val = int(ts)
            except Exception:
                continue
                
            try:
                hours = float(rec.get("hour"))
            except Exception:
                hours = None
                
            if rate_str_key in ['Very good', 'Fairly good', 'Fairly bad', 'Very bad']:
                rate_label = rate_str_key
                rate_score = {'Very bad': 0, 'Fairly bad': 1, 'Fairly good': 2, 'Very good': 3}[rate_str_key]
            else:
                rate_score = rate_score_map.get(rate_val, rate_score_map.get(rate_str_key, None))
                rate_label = rate_label_map.get(rate_val, rate_label_map.get(rate_str_key, 'Unknown'))
                
            # Date string conversion
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d")
            
            records.append({
                "timestamp": ts_val,
                "date": date_str,
                "hours": hours,
                "rate": rate_label,
                "rate_score": rate_score
            })
    except Exception:
        pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[(df["timestamp"] >= STUDY_START_TS) & (df["timestamp"] <= STUDY_END_TS)]
    else:
        df = pd.DataFrame(columns=["timestamp", "date", "hours", "rate", "rate_score"])
    return df


@cache_processed_df
def load_ema_stress(user_id: str) -> pd.DataFrame:
    """
    Load daily EMA stress responses for a user.

    Returns DataFrame with columns:
        timestamp (int), date (str), stress_level (int 1–5)
    """
    p = RAW_DIR / "EMA" / "response" / "Stress" / f"Stress_{user_id}.json"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "date", "stress_level"])
        
    records = []
    try:
        data = load_json_cached(p)
        for rec in data:
            if "level" not in rec:
                continue
            ts = rec.get("resp_time")
            try:
                ts_val = int(ts)
            except Exception:
                continue
            
            try:
                level = int(rec.get("level"))
            except Exception:
                continue
                
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d")
            
            records.append({
                "timestamp": ts_val,
                "date": date_str,
                "stress_level": level
            })
    except Exception:
        pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[(df["timestamp"] >= STUDY_START_TS) & (df["timestamp"] <= STUDY_END_TS)]
    else:
        df = pd.DataFrame(columns=["timestamp", "date", "stress_level"])
    return df


@cache_processed_df
def load_ema_exercise(user_id: str) -> pd.DataFrame:
    """
    Load daily EMA exercise responses for a user.

    Returns DataFrame with columns:
        timestamp (int), date (str), exercise_bool (int), duration_category (str)
    """
    p = RAW_DIR / "EMA" / "response" / "Exercise" / f"Exercise_{user_id}.json"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "date", "exercise_bool", "duration_category"])
        
    duration_map = {
        '1': 'None', '2': '<30 mins', '3': '30-60 mins', '4': '60-90 mins', '5': '>90mins',
        1: 'None', 2: '<30 mins', 3: '30-60 mins', 4: '60-90 mins', 5: '>90mins'
    }
    
    records = []
    try:
        data = load_json_cached(p)
        for rec in data:
            ts = rec.get("resp_time")
            try:
                ts_val = int(ts)
            except Exception:
                continue
                
            have_val = rec.get("have")
            if have_val in ['1', 1]:
                exercise_bool = 1
            elif have_val in ['2', 2]:
                exercise_bool = 0
            else:
                exercise_bool = 0
                
            dur_val = rec.get("exercise")
            duration_cat = duration_map.get(dur_val, 'None')
            
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d")
            
            records.append({
                "timestamp": ts_val,
                "date": date_str,
                "exercise_bool": exercise_bool,
                "duration_category": duration_cat
            })
    except Exception:
        pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[(df["timestamp"] >= STUDY_START_TS) & (df["timestamp"] <= STUDY_END_TS)]
    else:
        df = pd.DataFrame(columns=["timestamp", "date", "exercise_bool", "duration_category"])
    return df


@cache_processed_df
def load_ema_activity(user_id: str) -> pd.DataFrame:
    """
    Load daily EMA activity/mood responses for a user.

    Returns DataFrame with columns:
        timestamp (int), date (str), working (int), relaxing (int), ...
    """
    p = RAW_DIR / "EMA" / "response" / "Activity" / f"Activity_{user_id}.json"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "date", "working", "relaxing", "other_working", "other_relaxing"])
        
    records = []
    try:
        data = load_json_cached(p)
        for rec in data:
            ts = rec.get("resp_time")
            try:
                ts_val = int(ts)
            except Exception:
                continue
                
            def to_int(val):
                try:
                    return int(val)
                except Exception:
                    return None
                    
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d")
            
            records.append({
                "timestamp": ts_val,
                "date": date_str,
                "working": to_int(rec.get("working")),
                "relaxing": to_int(rec.get("relaxing")),
                "other_working": to_int(rec.get("other_working")),
                "other_relaxing": to_int(rec.get("other_relaxing"))
            })
    except Exception:
        pass
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[(df["timestamp"] >= STUDY_START_TS) & (df["timestamp"] <= STUDY_END_TS)]
    else:
        df = pd.DataFrame(columns=["timestamp", "date", "working", "relaxing", "other_working", "other_relaxing"])
    return df


# ── Sensing modalities ─────────────────────────────────────────────────────────

@cache_processed_df
def load_activity(user_id: str) -> pd.DataFrame:
    """
    Load accelerometer activity inference data.

    Returns DataFrame with columns:
        timestamp (int), activity_inference (int)
        [0=stationary, 1=walking, 2=running, 3=unknown]
    """
    p = RAW_DIR / "sensing" / "activity" / f"activity_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "activity_inference"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={'activity inference': 'activity_inference'})
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['activity_inference'] = pd.to_numeric(df['activity_inference'], errors='coerce')
    df = df.dropna()
    df = df[(df['timestamp'] >= STUDY_START_TS) & (df['timestamp'] <= STUDY_END_TS)]
    df['timestamp'] = df['timestamp'].astype(int)
    df['activity_inference'] = df['activity_inference'].astype(int)
    from src.data.preprocessor import add_date_column, add_hour_column
    df = add_date_column(df)
    df = add_hour_column(df)
    return df[['timestamp', 'activity_inference', 'date', 'hour']].reset_index(drop=True)


@cache_processed_df
def load_phonelock(user_id: str) -> pd.DataFrame:
    """
    Load phone lock/unlock events.

    Returns DataFrame with columns:
        timestamp (int), event (int)  [0=lock, 1=unlock]
    """
    p = RAW_DIR / "sensing" / "phonelock" / f"phonelock_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "event"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna()
    
    df_unlock = pd.DataFrame({'timestamp': df['start'], 'event': 1})
    df_lock = pd.DataFrame({'timestamp': df['end'], 'event': 0})
    
    df_events = pd.concat([df_unlock, df_lock], ignore_index=True)
    df_events = df_events.sort_values(by='timestamp').reset_index(drop=True)
    df_events['timestamp'] = df_events['timestamp'].astype(int)
    df_events['event'] = df_events['event'].astype(int)
    
    df_events = df_events[(df_events['timestamp'] >= STUDY_START_TS) & (df_events['timestamp'] <= STUDY_END_TS)]
    return df_events.reset_index(drop=True)


@cache_processed_df
def load_app_usage(user_id: str) -> pd.DataFrame:
    """
    Load running app (foreground app) data.

    Returns DataFrame with columns:
        timestamp (int), package_name (str), class_name (str)
    """
    p = RAW_DIR / "app_usage" / f"running_app_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "package_name", "class_name"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    
    df = df.rename(columns={
        'RUNNING_TASKS_topActivity_mPackage': 'package_name',
        'RUNNING_TASKS_topActivity_mClass': 'class_name'
    })
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Filter study window
    df = df[(df['timestamp'] >= STUDY_START_TS) & (df['timestamp'] <= STUDY_END_TS)]
    df['timestamp'] = df['timestamp'].astype(int)
    from src.data.preprocessor import add_date_column, add_hour_column
    df = add_date_column(df)
    df = add_hour_column(df)
    return df[['timestamp', 'package_name', 'class_name', 'date', 'hour']].reset_index(drop=True)


@cache_processed_df
def load_audio(user_id: str) -> pd.DataFrame:
    """
    Load audio inference data.

    Returns DataFrame with columns:
        timestamp (int), audio_inference (int)
        [0=silence, 1=voice, 2=noise, 3=unknown]
    """
    p = RAW_DIR / "sensing" / "audio" / f"audio_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "audio_inference"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={'audio inference': 'audio_inference'})
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['audio_inference'] = pd.to_numeric(df['audio_inference'], errors='coerce')
    df = df.dropna()
    df = df[(df['timestamp'] >= STUDY_START_TS) & (df['timestamp'] <= STUDY_END_TS)]
    df['timestamp'] = df['timestamp'].astype(int)
    df['audio_inference'] = df['audio_inference'].astype(int)
    from src.data.preprocessor import add_date_column, add_hour_column
    df = add_date_column(df)
    df = add_hour_column(df)
    return df[['timestamp', 'audio_inference', 'date', 'hour']].reset_index(drop=True)


@cache_processed_df
def load_conversation(user_id: str) -> pd.DataFrame:
    """
    Load conversation detection data.

    Returns DataFrame with columns:
        timestamp (int), inferred_conversation (int) [0=no, 1=yes]
    """
    p = RAW_DIR / "sensing" / "conversation" / f"conversation_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "inferred_conversation"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    df['start_timestamp'] = pd.to_numeric(df['start_timestamp'], errors='coerce')
    df['end_timestamp'] = pd.to_numeric(df['end_timestamp'], errors='coerce')
    df = df.dropna()
    
    df_start = pd.DataFrame({'timestamp': df['start_timestamp'], 'inferred_conversation': 1})
    df_end = pd.DataFrame({'timestamp': df['end_timestamp'], 'inferred_conversation': 0})
    
    df_events = pd.concat([df_start, df_end], ignore_index=True)
    df_events = df_events.sort_values(by='timestamp').reset_index(drop=True)
    df_events['timestamp'] = df_events['timestamp'].astype(int)
    df_events['inferred_conversation'] = df_events['inferred_conversation'].astype(int)
    
    df_events = df_events[(df_events['timestamp'] >= STUDY_START_TS) & (df_events['timestamp'] <= STUDY_END_TS)]
    return df_events.reset_index(drop=True)


@cache_processed_df
def load_gps(user_id: str) -> pd.DataFrame:
    """
    Load GPS location data.

    Returns DataFrame with columns:
        timestamp (int), double_latitude (float), double_longitude (float),
        double_speed (float), double_accuracy (float)
    """
    p = RAW_DIR / "sensing" / "gps" / f"gps_{user_id}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "double_latitude", "double_longitude", "double_speed", "double_accuracy"])
        
    df = load_csv_cached(p, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'time': 'timestamp',
        'latitude': 'double_latitude',
        'longitude': 'double_longitude',
        'speed': 'double_speed',
        'accuracy': 'double_accuracy'
    })
    
    df = df[['timestamp', 'double_latitude', 'double_longitude', 'double_speed', 'double_accuracy']]
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['double_latitude'] = pd.to_numeric(df['double_latitude'], errors='coerce')
    df['double_longitude'] = pd.to_numeric(df['double_longitude'], errors='coerce')
    df['double_speed'] = pd.to_numeric(df['double_speed'], errors='coerce')
    df['double_accuracy'] = pd.to_numeric(df['double_accuracy'], errors='coerce')
    df = df.dropna()
    df = df[(df['timestamp'] >= STUDY_START_TS) & (df['timestamp'] <= STUDY_END_TS)]
    df['timestamp'] = df['timestamp'].astype(int)
    from src.data.preprocessor import add_date_column, add_hour_column
    df = add_date_column(df)
    df = add_hour_column(df)
    return df.reset_index(drop=True)


# ── Surveys ────────────────────────────────────────────────────────────────────

@cache_processed_df
def load_psqi() -> pd.DataFrame:
    """
    Load Pittsburgh Sleep Quality Index survey data for all users.

    Returns DataFrame with columns:
        uid (str), type (str pre/post), overall_quality (str),
        psqi_score (float — computed from components)
    """
    p = RAW_DIR / "survey" / "psqi.csv"
    if not p.exists():
        return pd.DataFrame(columns=["uid", "type", "overall_quality", "psqi_score"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    
    overall_quality_col = 'During the past month, how would you rate your sleep quality overall?'
    q2_col = 'During the past month, how long (in minutes) has it usually taken you to fall asleep each night?'
    q5a_col = 'a. Cannot get to sleep within 30 minutes'
    q4_col = 'During the past month, how many hours of actual sleep did you get at night? (This may be different than the number of hours you spent in bed.)'
    
    q1_col = 'During the past month, what time have you usually gone to bed at night?'
    q3_col = 'When have you usually gotten up in the morning?'
    
    dist_cols = [
        'b. Wake up in the middle of the night or early morning',
        'c. Have to get up to use the bathroom',
        'd. Cannot breathe comfortably',
        'e. Cough or snore loudly',
        'f. Feel too cold',
        'g. Feel too hot',
        'h. Have bad dreams',
        'i. Have pain',
        'j. Other reason(s)'
    ]
    
    q6_col = 'During the past month, how often have you taken medicine (prescribed or over the counter) to help you sleep?'
    q7_col = 'During the past month, how often have you had trouble staying awake while driving, eating meals, or engaging in social activity?'
    q8_col = 'During the past month, how much of a problem has it been for you to keep up enthusiasm to get things done?'
    
    scores = []
    for idx, row in df.iterrows():
        # Component 1
        q9_val = str(row.get(overall_quality_col, '')).strip()
        comp1 = {'Very good': 0, 'Fairly good': 1, 'Fairly bad': 2, 'Very bad': 3}.get(q9_val, 1)
        
        # Component 2
        q2_str = str(row.get(q2_col, ''))
        q2_match = re.search(r'(\d+)', q2_str)
        q2_min = int(q2_match.group(1)) if q2_match else 15
        
        q2_score = 0
        if q2_min <= 15: q2_score = 0
        elif q2_min <= 30: q2_score = 1
        elif q2_min <= 60: q2_score = 2
        else: q2_score = 3
        
        q5a_val = str(row.get(q5a_col, '')).strip()
        q5a_score = {
            'Not during the past month': 0,
            'Less than once week': 1, 'Less than once a week': 1,
            'Once or a twice week': 2, 'Once or twice a week': 2,
            'Three or a more times week': 3, 'Three or more times a week': 3
        }.get(q5a_val, 0)
        
        comp2_sum = q2_score + q5a_score
        if comp2_sum == 0: comp2 = 0
        elif comp2_sum <= 2: comp2 = 1
        elif comp2_sum <= 4: comp2 = 2
        else: comp2 = 3
        
        # Component 3
        q4_str = str(row.get(q4_col, ''))
        q4_match = re.search(r'(\d+)', q4_str)
        q4_hours = float(q4_match.group(1)) if q4_match else 7.0
        if q4_str == '10-Sep':
            q4_hours = 9.5
        elif q4_str == '8-Sep':
            q4_hours = 8.0
            
        if q4_hours > 7: comp3 = 0
        elif q4_hours >= 6: comp3 = 1
        elif q4_hours >= 5: comp3 = 2
        else: comp3 = 3
        
        # Component 4
        def parse_time(time_str):
            if pd.isna(time_str):
                return None
            ts_str = str(time_str).strip().upper()
            is_pm = 'PM' in ts_str
            is_am = 'AM' in ts_str
            t_match = re.search(r'(\d+)(?::(\d+))?', ts_str)
            if not t_match:
                return None
            h = int(t_match.group(1))
            m = int(t_match.group(2)) if t_match.group(2) else 0
            if is_pm and h < 12: h += 12
            elif is_am and h == 12: h = 0
            return h + m / 60.0
            
        bt = parse_time(row.get(q1_col))
        wt = parse_time(row.get(q3_col))
        if bt is not None and wt is not None:
            hours_in_bed = (wt - bt) % 24
            if hours_in_bed == 0: hours_in_bed = 8.0
        else:
            hours_in_bed = 8.0
            
        efficiency = (q4_hours / hours_in_bed) * 100
        if efficiency > 85: comp4 = 0
        elif efficiency >= 75: comp4 = 1
        elif efficiency >= 65: comp4 = 2
        else: comp4 = 3
        
        # Component 5
        comp5_sum = 0
        freq_map = {
            'Not during the past month': 0,
            'Less than once week': 1, 'Less than once a week': 1,
            'Once or a twice week': 2, 'Once or twice a week': 2,
            'Three or a more times week': 3, 'Three or more times a week': 3
        }
        for d_col in dist_cols:
            val = str(row.get(d_col, '')).strip()
            comp5_sum += freq_map.get(val, 0)
            
        if comp5_sum == 0: comp5 = 0
        elif comp5_sum <= 9: comp5 = 1
        elif comp5_sum <= 18: comp5 = 2
        else: comp5 = 3
        
        # Component 6
        q6_val = str(row.get(q6_col, '')).strip()
        comp6 = freq_map.get(q6_val, 0)
        
        # Component 7
        q7_val = str(row.get(q7_col, '')).strip()
        q7_score = freq_map.get(q7_val, 0)
        
        q8_val = str(row.get(q8_col, '')).strip()
        q8_score = {
            'No problem at all': 0,
            'Only a very slight problem': 1,
            'Somewhat of a problem': 2,
            'A very big problem': 3
        }.get(q8_val, 0)
        
        comp7_sum = q7_score + q8_score
        if comp7_sum == 0: comp7 = 0
        elif comp7_sum <= 2: comp7 = 1
        elif comp7_sum <= 4: comp7 = 2
        else: comp7 = 3
        
        psqi_score = float(comp1 + comp2 + comp3 + comp4 + comp5 + comp6 + comp7)
        scores.append(psqi_score)
        
    df['psqi_score'] = scores
    df = df.rename(columns={'uid': 'uid', 'type': 'type', overall_quality_col: 'overall_quality'})
    return df[['uid', 'type', 'overall_quality', 'psqi_score']].reset_index(drop=True)


@cache_processed_df
def load_big_five() -> pd.DataFrame:
    """
    Load Big Five personality survey data for all users.

    Returns DataFrame with columns:
        uid (str), openness, conscientiousness, extraversion,
        agreeableness, neuroticism (all float)
    """
    p = RAW_DIR / "survey" / "BigFive.csv"
    if not p.exists():
        return pd.DataFrame(columns=["uid", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
        
    df = load_csv_cached(p)
    df.columns = [c.strip() for c in df.columns]
    
    cols = df.columns
    col_map = {}
    for i in range(1, 45):
        for col in cols:
            if re.search(r'-\s*' + str(i) + r'\b', col):
                col_map[i] = col
                break
                
    val_map = {
        'Disagree Strongly': 1, 'Disagree strongly': 1,
        'Disagree a little': 2,
        'Neither agree nor disagree': 3,
        'Agree a little': 4,
        'Agree strongly': 5
    }
    
    e_items = [(1, False), (6, True), (11, False), (16, False), (21, True), (26, False), (31, True), (36, False)]
    a_items = [(2, True), (7, False), (12, True), (17, False), (22, False), (27, True), (32, False), (37, True), (42, False)]
    c_items = [(3, False), (8, True), (13, False), (18, True), (23, True), (28, False), (33, False), (38, False), (43, True)]
    n_items = [(4, False), (9, True), (14, False), (19, False), (24, True), (29, False), (34, True), (39, False)]
    o_items = [(5, False), (10, False), (15, False), (20, False), (25, False), (30, False), (35, True), (40, False), (41, True), (44, False)]
    
    def get_score(row, items):
        vals = []
        for item_num, is_reverse in items:
            col_name = col_map.get(item_num)
            if not col_name:
                continue
            raw_val = row.get(col_name)
            numeric_val = val_map.get(raw_val, 3)
            if is_reverse:
                numeric_val = 6 - numeric_val
            vals.append(numeric_val)
        return float(sum(vals) / len(vals)) if vals else 3.0
        
    records = []
    for idx, row in df.iterrows():
        uid = row.get('uid')
        if not uid:
            continue
        extraversion = get_score(row, e_items)
        agreeableness = get_score(row, a_items)
        conscientiousness = get_score(row, c_items)
        neuroticism = get_score(row, n_items)
        openness = get_score(row, o_items)
        
        records.append({
            'uid': uid,
            'extraversion': extraversion,
            'agreeableness': agreeableness,
            'conscientiousness': conscientiousness,
            'neuroticism': neuroticism,
            'openness': openness
        })
        
    return pd.DataFrame(records)
