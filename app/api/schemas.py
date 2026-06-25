from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UserCreate(BaseModel):
    user_id: str = Field(..., description="Unique user identifier, e.g. u00")
    psqi_pre_score: Optional[float] = Field(None, description="Pre-study PSQI score")
    psqi_post_score: Optional[float] = Field(None, description="Post-study PSQI score")
    personality: Optional[Dict[str, float]] = Field(None, description="Big Five personality scores dict")

class CalendarEventInput(BaseModel):
    summary: str = Field(..., description="Calendar event title")
    start_time: str = Field(..., description="ISO 8601 start timestamp")
    end_time: str = Field(..., description="ISO 8601 end timestamp")

class UserResponse(BaseModel):
    user_id: str
    psqi_pre_score: Optional[float] = None
    psqi_post_score: Optional[float] = None
    personality: Optional[Dict[str, float]] = None
    profile_picture_url: Optional[str] = None

    class Config:
        from_attributes = True


class DailyFeaturesInput(BaseModel):
    # Lock / unlock
    unlock_count_late_night: float = 0.0
    unlock_count_evening: float = 0.0
    unlock_count_daytime: float = 0.0
    first_unlock_hour: Optional[float] = None
    last_unlock_hour: Optional[float] = None
    avg_session_duration_min: float = 0.0
    screen_sessions_count: float = 0.0
    
    # Activity
    stationary_ratio: float = 1.0
    walking_minutes: float = 0.0
    running_minutes: float = 0.0
    exercise_detected: int = 0
    peak_activity_hour: Optional[float] = None
    activity_bout_count: float = 0.0
    
    # App usage
    app_social_min: float = 0.0
    app_entertainment_evening_min: float = 0.0
    app_late_night_min: float = 0.0
    last_active_app_hour: Optional[float] = None
    app_diversity_count: float = 0.0
    app_study_sessions: float = 0.0
    
    # Audio
    silence_ratio: float = 1.0
    conversation_ratio: float = 0.0
    social_audio_evening: float = 0.0
    
    # GPS
    location_entropy: float = 0.0
    mobility_radius: float = 0.0
    unique_locations_count: float = 0.0
    
    # EMA
    stress_level: Optional[float] = None
    mood_happy: Optional[float] = None
    mood_tired: float = 0.0
    social_contacts: Optional[float] = None
    exercise_self_report: int = 0
    study_hours_today: float = 0.0
    
    # Context
    day_of_week: int = 0
    is_weekend: int = 0
    psqi_pre_score: float = 5.0
    
    # Personality
    personality_extraversion: float = 3.0
    personality_agreeableness: float = 3.0
    personality_conscientiousness: float = 3.0
    personality_neuroticism: float = 3.0
    personality_openness: float = 3.0
    
    # NLP Similarity
    nlp_caffeine_similarity: float = 0.0
    nlp_screen_similarity: float = 0.0
    nlp_stress_similarity: float = 0.0
    
    # Calendar Events
    calendar_events: Optional[List[CalendarEventInput]] = None


class PredictionResponse(BaseModel):
    user_id: Optional[str] = None
    date: Optional[str] = None
    predicted_score: float = Field(..., description="Continuous regression score in range 0.0 - 3.0")
    predicted_label: str = Field(..., description="Sleep quality label (Very bad - Very good)")
    anomaly_flag: int = Field(0, description="1 if behavior is anomalous, 0 if normal")
    top_features: List[Dict[str, Any]] = Field([], description="Top 3 SHAP drivers of prediction")
    advice: List[str] = Field([], description="List of 3 sleep recommendation strings")

class PredictionHistoryItem(BaseModel):
    date: str
    predicted_score: float
    predicted_label: str
    anomaly_flag: int
    top_features: List[Dict[str, Any]] = []
    advice: List[str] = []
    actual_rating: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True

class UserRegister(BaseModel):
    user_id: str = Field(..., description="Unique user identifier, e.g. u00")
    password: str = Field(..., description="User password")
    psqi_pre_score: Optional[float] = Field(None, description="Pre-study PSQI score")
    psqi_post_score: Optional[float] = Field(None, description="Post-study PSQI score")
    personality: Optional[Dict[str, float]] = Field(None, description="Big Five personality scores dict")

class UserLogin(BaseModel):
    user_id: str = Field(..., description="Unique user identifier, e.g. u00")
    password: str = Field(..., description="User password")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    profile_picture_url: Optional[str] = None

class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., description="Google ID Token JWT")


