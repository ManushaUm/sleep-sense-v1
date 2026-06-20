import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class UserIsolationForest:
    """
    Fits a separate Isolation Forest model per user on their daily behavioral features
    to detect personalized day-to-day anomalies.
    """
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models_ = {}
        self.imputers_ = {}
        self.scalers_ = {}
        self.global_imputer = SimpleImputer(strategy='median')
        self.global_scaler = StandardScaler()
        self.global_model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X: pd.DataFrame, y=None) -> 'UserIsolationForest':
        # Align features (exclude user_id and date if present)
        exclude_cols = ['user_id', 'date']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        # Fit a global fallback model first
        X_feat = X[feature_cols].copy()
        X_imp_global = self.global_imputer.fit_transform(X_feat)
        X_scale_global = self.global_scaler.fit_transform(X_imp_global)
        self.global_model.fit(X_scale_global)
        
        # Group by user and fit personalized models
        unique_users = X['user_id'].unique() if 'user_id' in X.columns else []
        for uid in unique_users:
            user_df = X[X['user_id'] == uid][feature_cols].copy()
            if len(user_df) >= 5: # Need enough historical days to fit IsolationForest
                imp = SimpleImputer(strategy='median')
                scale = StandardScaler()
                
                user_imp = imp.fit_transform(user_df)
                user_scale = scale.fit_transform(user_imp)
                
                model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
                model.fit(user_scale)
                
                self.models_[uid] = model
                self.imputers_[uid] = imp
                self.scalers_[uid] = scale
                
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores (lower scores = more anomalous).
        Outputs are in range [-1.0, 1.0].
        """
        exclude_cols = ['user_id', 'date']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        scores = []
        for _, row in X.iterrows():
            uid = row.get('user_id')
            row_feat = pd.DataFrame([row[feature_cols]])
            
            if uid in self.models_:
                imp = self.imputers_[uid]
                scale = self.scalers_[uid]
                model = self.models_[uid]
                
                row_imp = imp.transform(row_feat)
                row_scale = scale.transform(row_imp)
                score = model.score_samples(row_scale)[0]
            else:
                # Fallback to global model
                row_imp = self.global_imputer.transform(row_feat)
                row_scale = self.global_scaler.transform(row_imp)
                score = self.global_model.score_samples(row_scale)[0]
                
            scores.append(float(score))
            
        return np.array(scores)

    def predict_anomaly_flag(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary anomaly flags (1 = anomalous, 0 = normal).
        Flags samples with score below the threshold (sklearn returns -1 for anomaly).
        """
        exclude_cols = ['user_id', 'date']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        flags = []
        for _, row in X.iterrows():
            uid = row.get('user_id')
            row_feat = pd.DataFrame([row[feature_cols]])
            
            if uid in self.models_:
                imp = self.imputers_[uid]
                scale = self.scalers_[uid]
                model = self.models_[uid]
                
                row_imp = imp.transform(row_feat)
                row_scale = scale.transform(row_imp)
                # sklearn IsolationForest predict returns -1 for anomaly, 1 for normal
                pred = model.predict(row_scale)[0]
                flag = 1 if pred == -1 else 0
            else:
                row_imp = self.global_imputer.transform(row_feat)
                row_scale = self.global_scaler.transform(row_imp)
                pred = self.global_model.predict(row_scale)[0]
                flag = 1 if pred == -1 else 0
                
            flags.append(flag)
            
        return np.array(flags)
