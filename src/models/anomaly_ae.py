import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class AutoencoderModel(nn.Module):
    """
    Standard PyTorch feedforward autoencoder for tabular data reconstruction.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        # Encoder: compresses input features to a latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        # Decoder: reconstructs the input from the latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class UserAutoencoderAnomalyDetector:
    """
    Fits a PyTorch Autoencoder per user on their daily behavioral features
    to detect day-to-day anomalies based on reconstruction loss.
    """
    def __init__(self, contamination: float = 0.1, epochs: int = 50, batch_size: int = 16, lr: float = 0.01):
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.models_ = {}
        self.imputers_ = {}
        self.scalers_ = {}
        self.thresholds_ = {}
        
        # Fallbacks
        self.global_imputer = SimpleImputer(strategy='median')
        self.global_scaler = StandardScaler()
        self.global_model = None
        self.global_threshold = 0.0

    def _train_model(self, X_scaled: np.ndarray, input_dim: int) -> tuple[AutoencoderModel, float]:
        """Train PyTorch autoencoder and return trained model and reconstruction loss threshold."""
        model = AutoencoderModel(input_dim=input_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        dataset = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset.size()[0])
            for i in range(0, dataset.size()[0], self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x = dataset[indices]
                
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
        # Compute reconstruction error on train set to set threshold
        model.eval()
        with torch.no_grad():
            reconstructed = model(dataset)
            # Row-wise MSE
            mse_errors = torch.mean((reconstructed - dataset) ** 2, dim=1).numpy()
            
        # Set threshold at the (100 - contamination * 100) percentile
        threshold = float(np.percentile(mse_errors, 100 * (1 - self.contamination)))
        return model, threshold

    def fit(self, X: pd.DataFrame, y=None) -> 'UserAutoencoderAnomalyDetector':
        exclude_cols = ['user_id', 'date', 'sleep_score']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        # Fit global fallback model
        X_feat = X[feature_cols].copy()
        X_imp_global = self.global_imputer.fit_transform(X_feat)
        X_scale_global = self.global_scaler.fit_transform(X_imp_global)
        
        input_dim = len(feature_cols)
        self.global_model, self.global_threshold = self._train_model(X_scale_global, input_dim)
        
        # Fit per-user personalized autoencoders
        if 'user_id' in X.columns:
            unique_users = X['user_id'].unique()
            for uid in unique_users:
                user_df = X[X['user_id'] == uid][feature_cols].copy()
                if len(user_df) >= 10:  # Need enough days to train neural net
                    user_imp = self.global_imputer.transform(user_df)
                    user_scale = self.global_scaler.transform(user_imp)
                    
                    model, thresh = self._train_model(user_scale, input_dim)
                    self.models_[uid] = model
                    self.thresholds_[uid] = thresh
                    
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute reconstruction MSE error per daily feature vector.
        Higher reconstruction MSE = more anomalous behavior.
        """
        exclude_cols = ['user_id', 'date', 'sleep_score']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        scores = []
        for _, row in X.iterrows():
            uid = row.get('user_id')
            row_feat = pd.DataFrame([row[feature_cols]])
            
            if uid in self.models_:
                model = self.models_[uid]
            else:
                model = self.global_model
                
            row_imp = self.global_imputer.transform(row_feat)
            row_scale = self.global_scaler.transform(row_imp)
            
            model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(row_scale, dtype=torch.float32)
                reconstructed = model(tensor_x)
                mse = float(torch.mean((reconstructed - tensor_x) ** 2).item())
            scores.append(mse)
            
        return np.array(scores)

    def predict_anomaly_flag(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary anomaly flags (1 = anomalous, 0 = normal).
        Flags samples where reconstruction MSE exceeds the user-specific threshold.
        """
        exclude_cols = ['user_id', 'date', 'sleep_score']
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        flags = []
        for _, row in X.iterrows():
            uid = row.get('user_id')
            row_feat = pd.DataFrame([row[feature_cols]])
            
            if uid in self.models_:
                model = self.models_[uid]
                thresh = self.thresholds_[uid]
            else:
                model = self.global_model
                thresh = self.global_threshold
                
            row_imp = self.global_imputer.transform(row_feat)
            row_scale = self.global_scaler.transform(row_imp)
            
            model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(row_scale, dtype=torch.float32)
                reconstructed = model(tensor_x)
                mse = float(torch.mean((reconstructed - tensor_x) ** 2).item())
                
            flag = 1 if mse > thresh else 0
            flags.append(flag)
            
        return np.array(flags)
