from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from src.db.database import Base

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    psqi_pre_score = Column(Float, nullable=True)
    psqi_post_score = Column(Float, nullable=True)
    personality_json = Column(String, nullable=True) # JSON string of BigFive traits

class DailyFeatures(Base):
    __tablename__ = "daily_features"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(String, nullable=False) # 'YYYY-MM-DD'
    features_json = Column(String, nullable=False) # JSON string of all features
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uix_user_date_features"),
    )

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    date = Column(String, nullable=False) # 'YYYY-MM-DD'
    predicted_score = Column(Float, nullable=False)
    predicted_label = Column(String, nullable=False)
    anomaly_flag = Column(Integer, default=0) # 0 = normal, 1 = anomalous
    top_features_json = Column(String, nullable=True) # JSON of top 3 SHAP drivers
    advice_json = Column(String, nullable=True) # JSON of advice list
    actual_rating = Column(String, nullable=True) # Ground truth if available
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uix_user_date_predictions"),
    )
