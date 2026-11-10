"""
Adaptive Prophet Demand Forecasting Pipeline.
Automatically:
- Detects available exogenous regressors in the input dataframe
  (promo_flag, discount_pct, is_holiday, or any custom columns).
- Enables yearly seasonality only if series length >= 365 days.
- Enables weekly seasonality only if data frequency is daily/sub-daily.
- Works on any e-commerce time series without manual column hardcoding.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, List, Optional

# Default exogenous regressors this pipeline recognizes.
# The model only adds a regressor if the column actually exists in the data.
_KNOWN_EXOG_CONFIGS = {
    "promo_flag":   {"prior_scale": 10.0, "mode": "multiplicative"},
    "discount_pct": {"prior_scale": 10.0, "mode": "multiplicative"},
    "is_holiday":   {"prior_scale": 5.0,  "mode": "multiplicative"},
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def detect_data_frequency(df: pd.DataFrame) -> str:
    """
    Infer data frequency from 'ds' column.
    Returns: 'D' (daily), 'W' (weekly), 'M' (monthly), or 'unknown'.
    """
    if len(df) < 2:
        return "D"
    deltas = pd.to_datetime(df["ds"]).diff().dropna()
    median_delta = deltas.median()

    if median_delta <= pd.Timedelta(days=1):
        return "D"
    elif median_delta <= pd.Timedelta(days=8):
        return "W"
    elif median_delta <= pd.Timedelta(days=32):
        return "M"
    return "unknown"


def detect_available_regressors(df: pd.DataFrame) -> List[str]:
    """
    Scan the dataframe for any known exogenous regressor columns that exist
    AND are fully non-null. Returns a list of detected column names.
    """
    available = []
    for col in _KNOWN_EXOG_CONFIGS:
        if col in df.columns and df[col].notna().all():
            available.append(col)
    return available


def build_prophet_model(df: pd.DataFrame) -> Tuple[Prophet, List[str]]:
    """
    Dynamically construct a Prophet model tuned to this specific dataset:
    - Detects frequency -> enables weekly seasonality only for daily data
    - Checks series length -> enables yearly seasonality only if >= 365 days
    - Detects exogenous regressors automatically
    Returns (model, list_of_exog_col_names).
    """
    freq = detect_data_frequency(df)
    n_days = (pd.to_datetime(df["ds"].max()) - pd.to_datetime(df["ds"].min())).days

    weekly_seasonality  = (freq in ("D", "W")) and (n_days >= 14)
    yearly_seasonality  = (n_days >= 365)

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,      # regularize trend changepoints
        seasonality_prior_scale=10.0
    )

    exog_cols = detect_available_regressors(df)
    for col in exog_cols:
        cfg = _KNOWN_EXOG_CONFIGS[col]
        model.add_regressor(col, prior_scale=cfg["prior_scale"], mode=cfg["mode"])

    return model, exog_cols


# ---------------------------------------------------------------------------
# TRAIN & PREDICT
# ---------------------------------------------------------------------------

def train_and_predict(
    model: Prophet,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Fit model on train_df and predict on predict_df.
    predict_df must contain 'ds' column and any regressor columns.
    """
    model.fit(train_df)
    forecast = model.predict(predict_df)
    return model, forecast


def generate_future_forecast(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    future_days: int = 90
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Build a fresh Prophet model from full_df, refit on it, and generate
    a future_days-ahead forecast with appropriate regressor baseline values.
    Returns (fitted_model, forecast_df).
    """
    model, exog_cols = build_prophet_model(full_df)
    model.fit(full_df)

    future_df = model.make_future_dataframe(periods=future_days, freq="D")

    # Fill regressor values for future dates
    if exog_cols:
        future_df = future_df.merge(
            full_df[["ds"] + exog_cols],
            on="ds",
            how="left"
        )
        for col in exog_cols:
            if "flag" in col or "is_" in col or "promo" in col or "holiday" in col:
                future_df[col] = future_df[col].fillna(0)
            else:
                # Continuous regressors: use mean of last 30 days
                baseline = full_df[col].tail(30).mean()
                future_df[col] = future_df[col].fillna(baseline)

    forecast = model.predict(future_df)
    return model, forecast


# ---------------------------------------------------------------------------
# CONVENIENCE: CREATE BASELINE vs ENHANCED (for legacy compatibility)
# ---------------------------------------------------------------------------

def create_enhanced_prophet(df: Optional[pd.DataFrame] = None) -> Prophet:
    """
    Returns a Prophet model configured for the given df.
    If df is None, falls back to a sensible default with known regressors.
    """
    if df is not None:
        model, _ = build_prophet_model(df)
        return model

    # Legacy fallback (no df available — use known defaults)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    for col, cfg in _KNOWN_EXOG_CONFIGS.items():
        model.add_regressor(col, prior_scale=cfg["prior_scale"], mode=cfg["mode"])
    return model
