import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_pose_metrics(y_true, y_pred, target_cols=None):
    """
    Computes MAE, RMSE, and R2 for regression outputs.
    """
    if target_cols is None:
        target_cols = ["x", "y", "z", "yaw"]
        
    results = {}
    mae_list, rmse_list, r2_list = [], [], []
    
    for i, dim in enumerate(target_cols):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        results[f"MAE_{dim}"] = float(mae)
        results[f"RMSE_{dim}"] = float(rmse)
        results[f"R2_{dim}"] = float(r2)
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
    results["avg_MAE"] = float(np.mean(mae_list))
    results["avg_RMSE"] = float(np.mean(rmse_list))
    results["avg_R2"] = float(np.mean(r2_list))
    return results
