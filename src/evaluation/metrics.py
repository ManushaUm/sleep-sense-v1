import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression and custom classification metrics for SleepSense model evaluation.
    
    Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - Binary Accuracy: Good sleep (score >= 1.5) vs Bad sleep (score < 1.5)
        - Ordinal Accuracy: Prediction is within 1 label of the true label (abs diff <= 1)
        - Exact Accuracy: Rounded prediction exactly equals the true label (abs diff == 0)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 1. Binary Accuracy
    # Mapping sleep score to binary label: Good (>= 1.5) vs Bad (< 1.5)
    # y_true is integer [0, 1, 2, 3] in raw data, so >= 1.5 translates to >= 2 (2 or 3)
    y_true_binary = (y_true >= 1.5).astype(int)
    y_pred_binary = (y_pred >= 1.5).astype(int)
    binary_acc = np.mean(y_true_binary == y_pred_binary)
    
    # 2. Ordinal Accuracy (within 1 label difference after rounding)
    # Round predictions to nearest integer: [0, 1, 2, 3] clipped
    y_pred_rounded = np.clip(np.round(y_pred), 0, 3).astype(int)
    y_true_rounded = np.clip(np.round(y_true), 0, 3).astype(int)
    ordinal_acc = np.mean(np.abs(y_true_rounded - y_pred_rounded) <= 1)
    
    # Simple integer exact match accuracy
    exact_acc = np.mean(y_true_rounded == y_pred_rounded)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "binary_accuracy": float(binary_acc),
        "ordinal_accuracy": float(ordinal_acc),
        "exact_accuracy": float(exact_acc)
    }
