"""
Evaluator Module for Prophet Demand Forecasting.
Computes time-series accuracy metrics: MAE, RMSE, MAPE, WMAPE, Bias, R2.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute comprehensive time-series forecasting evaluation metrics.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    
    # Avoid division by zero in MAPE
    non_zero_mask = y_t != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
    else:
        mape = 0.0
        
    # WMAPE (Weighted Absolute Percentage Error) - Standard in Supply Chain
    total_actual = float(np.sum(y_t))
    wmape = float(np.sum(np.abs(y_t - y_p)) / total_actual * 100.0) if total_actual > 0 else 0.0
    
    # Mean Forecast Bias (Over / Under forecasting)
    bias = float(np.mean(y_p - y_t))
    
    # R-squared
    r2 = float(r2_score(y_t, y_p))
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_percent": round(mape, 2),
        "WMAPE_percent": round(wmape, 2),
        "Bias": round(bias, 2),
        "R2_Score": round(r2, 4),
        "Total_Actual_Demand": int(np.sum(y_t)),
        "Total_Forecasted_Demand": int(np.sum(y_p))
    }
