import pandas as pd
import numpy as np
from src.data.loader import load_gps
from src.data.preprocessor import add_date_column

def extract_gps_features(user_id: str, date: str) -> dict:
    """Extract GPS-based mobility features for a given user and date."""
    default_features = {
        "location_entropy": 0.0,
        "mobility_radius": 0.0,
        "unique_locations_count": 0
    }
    
    try:
        df = load_gps(user_id)
    except Exception:
        return default_features
        
    if df.empty:
        return default_features
        
    df = add_date_column(df)
    df_day = df[df['date'] == date]
    
    if df_day.empty:
        return default_features
        
    # Round coordinates to 3 decimal places (approx. 110m resolution)
    df_day = df_day.copy()
    df_day['lat_round'] = df_day['double_latitude'].round(3)
    df_day['lng_round'] = df_day['double_longitude'].round(3)
    df_day['loc_str'] = df_day['lat_round'].astype(str) + "," + df_day['lng_round'].astype(str)
    
    unique_locations_count = int(df_day['loc_str'].nunique())
    
    # Calculate Location Entropy
    counts = df_day['loc_str'].value_counts()
    probs = counts / counts.sum()
    entropy = -float((probs * np.log(probs + 1e-9)).sum())
    
    # Calculate Mobility Radius (in km)
    if len(df_day) >= 2:
        lat_var = df_day['double_latitude'].var()
        lng_var = df_day['double_longitude'].var()
        if pd.isna(lat_var) or pd.isna(lng_var):
            mobility_radius = 0.0
        else:
            mobility_radius = float(np.sqrt(lat_var + lng_var) * 111.3) # 1 deg ~ 111.3 km
    else:
        mobility_radius = 0.0
        
    return {
        "location_entropy": entropy,
        "mobility_radius": mobility_radius,
        "unique_locations_count": unique_locations_count
    }
