"""
Data Loader Module for Demand Forecasting.
Handles loading raw CSV, SKU filtering, daily aggregation, and Prophet/LSTM/SARIMAX format preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_raw_data(csv_path: str = "fmcg_sales_3years_1M_rows.csv") -> pd.DataFrame:
    """Load raw FMCG sales CSV dataset."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_prophet_df(
    df: pd.DataFrame, 
    sku_id: str = "SKU0001", 
    store_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter dataset for a target SKU (and optional store_id) and aggregate into
    daily time-series format (ds, y) with exogenous regressor features.
    Compatible with Prophet, SARIMAX, and LSTM pipelines.
    """
    filtered_df = df[df['sku_id'] == sku_id].copy()
    if store_id:
        filtered_df = filtered_df[filtered_df['store_id'] == store_id]
        
    daily_df = filtered_df.groupby('date').agg(
        y=('units_sold', 'sum'),
        promo_flag=('promo_flag', 'max'),
        discount_pct=('discount_pct', 'mean'),
        is_holiday=('is_holiday', 'max'),
        temperature=('temperature', 'mean'),
        rain_mm=('rain_mm', 'mean'),
        stock_out_flag=('stock_out_flag', 'max')
    ).reset_index().rename(columns={'date': 'ds'})
    
    daily_df = daily_df.sort_values('ds').reset_index(drop=True)
    return daily_df

def split_train_test(
    df: pd.DataFrame, 
    test_start_date: str = "2023-07-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train and holdout test sets based on cut-off date.
    Default split: Train (2021-01-01 to 2023-06-30), Test (2023-07-01 to 2023-12-31).
    """
    cutoff = pd.to_datetime(test_start_date)
    train_df = df[df['ds'] < cutoff].copy()
    test_df = df[df['ds'] >= cutoff].copy()
    return train_df, test_df
