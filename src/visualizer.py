"""
Visualizer Module for Prophet Demand Forecasting.
Generates publication-quality charts for holdout evaluation, component decomposition,
and 90-day future demand projections.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from prophet import Prophet

# Configure modern, elegant plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_actual_vs_forecast(
    test_df: pd.DataFrame, 
    forecast_df: pd.DataFrame, 
    title: str = "Prophet Holdout Evaluation (Actual vs Predicted)",
    save_path: str = "artifacts/plots/actual_vs_forecast.png"
):
    """Plot actual holdout test demand vs Prophet predictions with confidence intervals."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Merge test_df with forecast_df on ds
    merged = pd.merge(test_df[['ds', 'y']], forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    
    ax.plot(merged['ds'], merged['y'], label='Actual Demand (y)', color='#1f77b4', lw=2, marker='o', ms=3, alpha=0.85)
    ax.plot(merged['ds'], merged['yhat'], label='Prophet Forecast (yhat)', color='#ff7f0e', lw=2.5, linestyle='--')
    ax.fill_between(
        merged['ds'], 
        merged['yhat_lower'], 
        merged['yhat_upper'], 
        color='#ff7f0e', 
        alpha=0.2, 
        label='95% Confidence Interval'
    )
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Daily Units Sold", labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def plot_components(
    model: Prophet, 
    forecast_df: pd.DataFrame,
    save_path: str = "artifacts/plots/components_decomposition.png"
):
    """Plot Prophet model component breakdown (Trend, Weekly, Yearly, Extra Regressors)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = model.plot_components(forecast_df, figsize=(10, 8))
    fig.suptitle("Prophet Component Decomposition (Trend & Seasonality)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {save_path}")

def plot_future_forecast(
    historical_df: pd.DataFrame, 
    forecast_df: pd.DataFrame,
    future_days: int = 90,
    save_path: str = "artifacts/plots/future_90day_forecast.png"
):
    """Plot complete 3-year historical demand alongside 90-day future demand forecast into 2024."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    # Historical actuals
    ax.plot(historical_df['ds'], historical_df['y'], label='Historical Actual Sales (2021-2023)', color='#2ca02c', lw=1, alpha=0.7)
    
    # Full fitted & future forecast
    future_start = historical_df['ds'].max()
    future_mask = forecast_df['ds'] > future_start
    
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Fitted / Projected Demand', color='#d62728', lw=1.8)
    ax.plot(forecast_df.loc[future_mask, 'ds'], forecast_df.loc[future_mask, 'yhat'], label=f'Future {future_days}-Day Forecast (2024)', color='#9467bd', lw=2.5, linestyle='-')
    ax.fill_between(
        forecast_df.loc[future_mask, 'ds'], 
        forecast_df.loc[future_mask, 'yhat_lower'], 
        forecast_df.loc[future_mask, 'yhat_upper'], 
        color='#9467bd', 
        alpha=0.3, 
        label='95% Forecast Uncertainty Bound'
    )
    
    ax.axvline(x=future_start, color='#7f7f7f', linestyle=':', lw=1.5, label='Forecast Horizon Start')
    ax.set_title(f"3-Year Historical Sales & Future {future_days}-Day Demand Projection (Forecast-Based Planning)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Daily Units Sold", labelpad=8)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
