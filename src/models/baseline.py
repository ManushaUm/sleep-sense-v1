import numpy as np
import pandas as pd

class UserMeanBaseline:
    """
    A baseline regressor that predicts sleep quality based on the user's
    historical average sleep rating. If the user is new/unseen during inference,
    it falls back to the global mean of the training set.
    """
    def __init__(self):
        self.user_means_ = {}
        self.global_mean_ = 2.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'UserMeanBaseline':
        # Align target y with features X
        temp_df = X.copy()
        temp_df['target'] = y
        
        # Calculate mean for each user
        self.user_means_ = temp_df.groupby('user_id')['target'].mean().to_dict()
        self.global_mean_ = float(y.mean()) if len(y) > 0 else 2.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            uid = row.get('user_id')
            preds.append(self.user_means_.get(uid, self.global_mean_))
        return np.array(preds)
