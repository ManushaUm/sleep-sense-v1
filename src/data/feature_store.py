"""
feature_store.py — Builds and caches daily feature vectors per user.

Main entry points:
    build_daily_features(user_id, date) → dict
    build_dataset(user_ids, start_date, end_date) → pd.DataFrame
    save_preprocessed() → writes to data/preprocessed/
"""

from pathlib import Path
import pandas as pd
import numpy as np

PREPROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocessed"

from src.features.notes_nlp import NotesNLPExtractor
nlp_extractor = NotesNLPExtractor()
# Fit Word2Vec on simulated notes corpus
nlp_extractor.fit_word2vec()


def build_daily_features(user_id: str, date: str) -> dict:
    """
    Compute all ~30 features for a given (user_id, date) pair.

    Args:
        user_id: e.g. 'u00'
        date: 'YYYY-MM-DD'

    Returns:
        dict of feature_name → value (float or int)
    """
    from src.features.phonelock_features import extract_phonelock_features
    from src.features.activity_features import extract_activity_features
    from src.features.app_usage_features import extract_app_usage_features
    from src.features.audio_features import extract_audio_features
    from src.features.gps_features import extract_gps_features
    from src.features.ema_features import extract_ema_features
    
    feats = {}
    
    # 1. Phonelock features
    feats.update(extract_phonelock_features(user_id, date))
    
    # 2. Activity features
    feats.update(extract_activity_features(user_id, date))
    
    # 3. App usage features
    feats.update(extract_app_usage_features(user_id, date))
    
    # 4. Audio features
    feats.update(extract_audio_features(user_id, date))
    
    # 5. GPS features
    feats.update(extract_gps_features(user_id, date))
    
    # 6. EMA features
    feats.update(extract_ema_features(user_id, date))
    
    # 7. NLP features (Word2Vec similarities)
    feats.update(nlp_extractor.extract_nlp_features(user_id, date))
    
    # Context features: day_of_week, is_weekend
    dt = pd.to_datetime(date)
    feats["day_of_week"] = int(dt.dayofweek)
    feats["is_weekend"] = 1 if dt.dayofweek >= 5 else 0
    
    return feats


def build_dataset(user_ids: list[str] | None = None,
                  start_date: str | None = None,
                  end_date: str | None = None) -> pd.DataFrame:
    """
    Build the full feature matrix + target labels for the given users and date range.

    Returns:
        DataFrame with columns: user_id, date, <all features>, sleep_score (target)
    """
    from src.data.loader import get_all_users, load_ema_sleep, load_psqi, load_big_five, clear_loader_cache
    
    # Clear cache before starting
    clear_loader_cache()
    
    if user_ids is None:
        user_ids = get_all_users()
        
    # Load surveys as static user features
    try:
        psqi_df = load_psqi()
        psqi_pre = psqi_df[psqi_df['type'] == 'pre'].drop_duplicates(subset=['uid']).copy()
        psqi_pre = psqi_pre.rename(columns={'psqi_score': 'psqi_pre_score'})
        psqi_dict = dict(zip(psqi_pre['uid'], psqi_pre['psqi_pre_score']))
    except Exception:
        psqi_dict = {}
        
    try:
        bf_df = load_big_five()
        bf_df = bf_df.drop_duplicates(subset=['uid'])
        bf_dict = bf_df.set_index('uid').to_dict(orient='index')
    except Exception:
        bf_dict = {}
        
    global_median_psqi = float(pd.Series(list(psqi_dict.values())).median()) if psqi_dict else 5.0
    
    rows = []
    for uid in user_ids:
        print(f"[{uid}] Building features...", flush=True)
        # Load targets (Sleep EMAs)
        try:
            sleep_df = load_ema_sleep(uid)
        except Exception:
            clear_loader_cache()
            continue
            
        if sleep_df.empty:
            clear_loader_cache()
            continue
            
        sleep_df = sleep_df.dropna(subset=['rate_score'])
        if sleep_df.empty:
            clear_loader_cache()
            continue
            
        # Filter date range if provided
        if start_date:
            sleep_df = sleep_df[sleep_df['date'] >= start_date]
        if end_date:
            sleep_df = sleep_df[sleep_df['date'] <= end_date]
            
        for _, row in sleep_df.iterrows():
            target_date = row['date']
            target_score = row['rate_score']
            
            # Build daily features
            try:
                feats = build_daily_features(uid, target_date)
            except Exception:
                continue
                
            row_dict = {
                "user_id": uid,
                "date": target_date,
            }
            row_dict.update(feats)
            
            # Merge surveys
            row_dict["psqi_pre_score"] = float(psqi_dict.get(uid, global_median_psqi))
            
            user_bf = bf_dict.get(uid, {})
            row_dict["personality_extraversion"] = float(user_bf.get("extraversion", 3.0))
            row_dict["personality_agreeableness"] = float(user_bf.get("agreeableness", 3.0))
            row_dict["personality_conscientiousness"] = float(user_bf.get("conscientiousness", 3.0))
            row_dict["personality_neuroticism"] = float(user_bf.get("neuroticism", 3.0))
            row_dict["personality_openness"] = float(user_bf.get("openness", 3.0))
            
            # Target
            row_dict["sleep_score"] = int(target_score)
            
            rows.append(row_dict)
            
        # Clear cache for the current user to free memory
        clear_loader_cache()
        
    clear_loader_cache()
    return pd.DataFrame(rows)


def save_preprocessed(df: pd.DataFrame) -> None:
    """
    Save the dataset to data/preprocessed/merged/dataset.parquet
    and per-user feature CSVs to data/preprocessed/features/.
    """
    if df.empty:
        return
        
    # Ensure preprocessed directories exist
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PREPROCESSED_DIR / "features").mkdir(parents=True, exist_ok=True)
    (PREPROCESSED_DIR / "targets").mkdir(parents=True, exist_ok=True)
    (PREPROCESSED_DIR / "merged").mkdir(parents=True, exist_ok=True)
    
    # 1. Export merged dataset
    parquet_path = PREPROCESSED_DIR / "merged" / "dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # 2. Export per-user features and targets
    feature_cols = [c for c in df.columns if c not in ['sleep_score']]
    
    for uid in df['user_id'].unique():
        user_df = df[df['user_id'] == uid]
        
        # Save features
        feat_path = PREPROCESSED_DIR / "features" / f"{uid}_features.csv"
        user_df[feature_cols].to_csv(feat_path, index=False)
        
        # Save targets
        target_path = PREPROCESSED_DIR / "targets" / f"{uid}_targets.csv"
        user_df[['date', 'sleep_score']].to_csv(target_path, index=False)


def load_preprocessed() -> pd.DataFrame:
    """Load the cached merged dataset from data/preprocessed/merged/dataset.parquet."""
    parquet_path = PREPROCESSED_DIR / "merged" / "dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {parquet_path}. "
            "Run build_dataset() and save_preprocessed() first."
        )
    return pd.read_parquet(parquet_path)
