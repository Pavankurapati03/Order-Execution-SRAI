"""
Adaptive Auto-SARIMAX Forecasting Pipeline.
Uses pmdarima for automatic order selection (p, d, q)(P, D, Q)_m
and statsmodels SARIMAX for exogenous regressor support.
Fully adaptive: works on any e-commerce time series dataset
without manual hyperparameter tuning.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DATA INSPECTION UTILITIES
# ---------------------------------------------------------------------------

def detect_seasonality_period(df: pd.DataFrame) -> int:
    """
    Automatically infer seasonal period m from the date column 'ds'.
    - Daily frequency -> m = 7  (weekly seasonality)
    - Weekly frequency -> m = 52
    - Monthly frequency -> m = 12
    Falls back to m = 7 if undetermined.
    """
    if len(df) < 2:
        return 7

    df_sorted = df.sort_values("ds").reset_index(drop=True)
    deltas = df_sorted["ds"].diff().dropna()
    median_delta = deltas.median()

    if median_delta <= pd.Timedelta(days=1):
        return 7    # Daily data -> weekly seasonality
    elif median_delta <= pd.Timedelta(days=8):
        return 52   # Weekly data -> yearly seasonality
    elif median_delta <= pd.Timedelta(days=32):
        return 12   # Monthly data -> yearly seasonality
    else:
        return 4    # Quarterly / other


def extract_exog_cols(df: pd.DataFrame,
                      reserved: List[str] = ("ds", "y")) -> List[str]:
    """
    Automatically detect numeric exogenous regressor columns in df,
    excluding reserved columns.
    Returns list of column names (may be empty).
    """
    return [
        col for col in df.columns
        if col not in reserved
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().all()
    ]


def get_exog_array(df: pd.DataFrame, exog_cols: List[str]) -> Optional[np.ndarray]:
    """Return a 2D numpy array of exog features, or None if no exog cols."""
    if not exog_cols:
        return None
    return df[exog_cols].values.astype(float)


# ---------------------------------------------------------------------------
# AUTO-ARIMA ORDER DETECTION
# ---------------------------------------------------------------------------

def auto_select_sarima_order(
    train_series: np.ndarray,
    m: int,
    exog: Optional[np.ndarray] = None
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Use pmdarima stepwise Auto-ARIMA to find optimal SARIMA order.
    Returns (order, seasonal_order) tuples for statsmodels SARIMAX.
    """
    import pmdarima as pm

    print(f"   [Auto-ARIMA] Searching optimal order (m={m}, exog_features={exog.shape[1] if exog is not None else 0})...")

    # Limit max_p, max_q to keep it fast on large datasets
    auto_model = pm.auto_arima(
        train_series,
        X=exog,
        m=m,
        seasonal=True,
        stepwise=True,
        information_criterion="aic",
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        max_d=2, max_D=1,
        start_p=1, start_q=1,
        start_P=0, start_Q=0,
        error_action="ignore",
        suppress_warnings=True,
        n_jobs=1
    )

    order = auto_model.order                # (p, d, q)
    seasonal_order = auto_model.seasonal_order  # (P, D, Q, m)
    print(f"   [Auto-ARIMA] Best order: SARIMA{order}x{seasonal_order}")
    return order, seasonal_order


# ---------------------------------------------------------------------------
# SARIMAX TRAIN + PREDICT
# ---------------------------------------------------------------------------

def train_sarimax(
    train_df: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    exog_cols: List[str]
) -> Any:
    """
    Fit a SARIMAX model on training data.
    Returns fitted statsmodels SARIMAXResults object.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train_y = train_df["y"].values.astype(float)
    train_exog = get_exog_array(train_df, exog_cols)

    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(
        disp=False,
        maxiter=200,
        method="lbfgs"
    )
    return result


def predict_sarimax(
    result: Any,
    test_df: pd.DataFrame,
    exog_cols: List[str]
) -> np.ndarray:
    """
    Generate in-sample / out-of-sample predictions for test set.
    Returns numpy array of predictions (clipped to >= 0).
    """
    n_test = len(test_df)
    test_exog = get_exog_array(test_df, exog_cols)

    forecast_obj = result.forecast(steps=n_test, exog=test_exog)
    preds = np.array(forecast_obj).clip(0, None)
    return preds


def forecast_future_sarimax(
    result: Any,
    future_days: int,
    last_df: pd.DataFrame,
    exog_cols: List[str]
) -> pd.DataFrame:
    """
    Generate future multi-step forecasts beyond the training window.
    For exogenous variables, uses the mean of the last 30 days as the
    assumed baseline for future periods (no-promo baseline).
    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
    """
    last_date = last_df["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    # Build future exog: use rolling mean of last 30 days as neutral baseline
    future_exog = None
    if exog_cols:
        baseline = last_df[exog_cols].tail(30).mean().values
        future_exog = np.tile(baseline, (future_days, 1)).astype(float)
        # Set promo-like binary features to 0 for conservative baseline
        for i, col in enumerate(exog_cols):
            if "flag" in col or "is_" in col or "promo" in col:
                future_exog[:, i] = 0.0

    forecast_result = result.get_forecast(steps=future_days, exog=future_exog)
    summary = forecast_result.summary_frame(alpha=0.05)

    yhat = np.array(summary["mean"]).clip(0, None)
    yhat_lower = np.array(summary["mean_ci_lower"]).clip(0, None)
    yhat_upper = np.array(summary["mean_ci_upper"]).clip(0, None)

    return pd.DataFrame({
        "ds": future_dates,
        "yhat": yhat,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper
    })


# ---------------------------------------------------------------------------
# MASTER CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def run_sarimax_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
    future_days: int = 90
) -> Tuple[Any, np.ndarray, Optional[pd.DataFrame], List[str]]:
    """
    Master function: auto-detects frequency, exog cols, fits best SARIMAX,
    returns (fitted_result, test_predictions, future_forecast_df, exog_cols).

    Parameters:
        train_df  : DataFrame with columns ['ds', 'y', ...optional exog...]
        test_df   : DataFrame with same columns for holdout evaluation
        full_df   : Full dataset; if provided, a retrained model generates
                    future_days-ahead forecast
        future_days: number of days to forecast into the future

    Returns:
        result       : fitted statsmodels SARIMAXResults
        test_preds   : np.ndarray of predictions on test_df
        future_df    : DataFrame of future forecast (or None if full_df not given)
        exog_cols    : list of exog column names used
    """
    # 1. Detect seasonal period
    m = detect_seasonality_period(train_df)

    # 2. Detect exogenous features
    exog_cols = extract_exog_cols(train_df)
    if exog_cols:
        print(f"   [SARIMAX] Exogenous regressors detected: {exog_cols}")
    else:
        print("   [SARIMAX] No exogenous regressors detected — running plain SARIMA.")

    # 3. Auto-select best SARIMA order using pmdarima on train series
    train_exog_arr = get_exog_array(train_df, exog_cols)
    order, seasonal_order = auto_select_sarima_order(
        train_df["y"].values.astype(float),
        m=m,
        exog=train_exog_arr
    )

    # 4. Fit SARIMAX on train data
    print("   [SARIMAX] Fitting final SARIMAX model on training data...")
    result = train_sarimax(train_df, order, seasonal_order, exog_cols)
    print(f"   [SARIMAX] Model fitted. AIC={result.aic:.2f}")

    # 5. Predict on test holdout
    test_preds = predict_sarimax(result, test_df, exog_cols)
    print(f"   [SARIMAX] Test predictions generated ({len(test_preds)} steps).")

    # 6. Optionally refit on full data and forecast future
    future_df = None
    if full_df is not None and future_days > 0:
        print("   [SARIMAX] Refitting on full dataset for future forecast...")
        full_result = train_sarimax(full_df, order, seasonal_order, exog_cols)
        future_df = forecast_future_sarimax(full_result, future_days, full_df, exog_cols)
        print(f"   [SARIMAX] Future {future_days}-day forecast generated.")

    return result, test_preds, future_df, exog_cols
