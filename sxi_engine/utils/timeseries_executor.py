# agent2/utils/timeseries_executor.py
"""
Time Series Executor - Handles LSTM/ARIMA/Prophet forecasting
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent2.utils.path_utils import resolve_media_path
from agent2.utils.plotly_export import save_plot
# ML Libraries
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# This style is more robust in newer TensorFlow versions

from agent2.utils.json_utils import make_json_safe


class TimeSeriesExecutor:
    """
    Executor for time series forecasting tasks.
    Supports LSTM, ARIMA, and Prophet models.
    """
    
    def __init__(self, request, dataframe_path, target_info, hyperparameters=None):
        """
        Args:
            request: Django request object
            dataframe_path: Path to CSV/Excel file
            target_info: Dict with target configuration
            hyperparameters: Dict with model hyperparameters (optional)
        """
        self.request = request
        self.dataframe_path = dataframe_path
        self.target_info = target_info
        self.hyperparameters = hyperparameters or {}
        self.buyerid = request.session.get('buyerid', request.user.id)
        
        # Output directory
        self.output_dir = Path(settings.MEDIA_ROOT) / 'files' / 'chatbot' / str(self.buyerid)
        self.output_dir.mkdir(parents=True, exist_ok=True)        
        self.model_type = self.hyperparameters.get('model_type', 'LSTM')
    
    def execute(self):
        """
        Main execution method.
        
        Returns:
            dict: Results with forecast, metrics, and plots
        """
        try:
            # Load and prepare data
            df = self.load_dataframe()
            df_prepared = self.prepare_timeseries_data(df)
            
            # Detect frequency
            frequency = self.detect_frequency(df_prepared)
            
            # Split data
            train_size = int(len(df_prepared) * 0.8)
            train_data = df_prepared[:train_size]
            test_data = df_prepared[train_size:]
            
            # Build and train model
            if self.model_type == 'LSTM':
                model, scaler, X_test, y_test = self.build_and_train_lstm(train_data, test_data)
                predictions = self.predict_lstm(model, X_test, scaler)
            else:
                # Placeholder for other models
                raise NotImplementedError(f"{self.model_type} not yet implemented")
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, predictions)
            
            # Generate forecast
            forecast_horizon = int(self.target_info.get('Forecast Horizon', 6))
            forecast = self.generate_forecast(model, train_data, scaler, forecast_horizon)
            
            # Detect patterns
            trend = self.detect_trend(df_prepared)
            seasonality = self.detect_seasonality(df_prepared)
            
            # Generate plots
            plots = self.generate_plots(
                df_prepared, 
                test_data.index, 
                predictions, 
                forecast,
                forecast_horizon
            )
            
            # Format results
            return {
                'success': True,
                'forecast_values': forecast.tolist(),
                'forecast_horizon': forecast_horizon,
                'detected_frequency': frequency,
                'algorithm_used': self.model_type,
                'model_params': self.hyperparameters,
                'validation_metrics': metrics,
                'trend_direction': trend,
                'seasonality_detected': seasonality,
                'is_stationary': self.check_stationarity(df_prepared),
                'plots': plots,
                'raw_data_shape': df_prepared.shape
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"TimeSeriesExecutor Error: {error_trace}")
            
            return {
                'error': True,
                'message': str(e),
                'trace': error_trace
            }
    
    def load_dataframe(self):
        """Load dataset from MEDIA URL or filesystem path"""


        # ✅ Step 1: take path from object
        path = self.dataframe_path

        # ✅ Step 2: normalize MEDIA_URL → MEDIA_ROOT
        path = resolve_media_path(path)
        path = os.path.normpath(path)
        print(f"[TimeSeriesExecutor] Loading dataframe from: {path}")
        # ✅ Step 3: load file
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TimeSeriesExecutor: dataset not found → {path}"
            )

        # 5️⃣ Load file
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)

        elif path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)

        else:
            raise ValueError(f"Unsupported file format: {path}")
        
    def prepare_timeseries_data(self, df):
        """Prepare time series data"""
        # Get column names
        time_col = self.target_info.get('Time Column', 'date')
        target_col = self.target_info.get('Target Outcome', 'sales')
        
        # Convert time column to datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Set time as index
        df = df.set_index(time_col)
        
        # Sort by time
        df = df.sort_index()
        
        # Extract target column as series
        series = df[target_col]
        
        # Handle missing values
        series = series.fillna(method='ffill').fillna(method='bfill')
        
        return series.to_frame()
    
    def detect_frequency(self, df):
        """Detect time series frequency"""
        try:
            inferred = pd.infer_freq(df.index)
            if inferred:
                return inferred
        except:
            pass
        
        # Manual detection based on median difference
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(days=1):
            return "D"  # Daily
        elif median_diff <= pd.Timedelta(days=7):
            return "W"  # Weekly
        elif median_diff <= pd.Timedelta(days=31):
            return "M"  # Monthly
        else:
            return "Y"  # Yearly
    
    def build_and_train_lstm(self, train_data, test_data):
        """Build and train LSTM model"""
        # if not KERAS:
        #     raise ImportError("TensorFlow/Keras required for LSTM")
        
        # Get hyperparameters
        lstm_layers = self.hyperparameters.get('lstm_layers', 2)
        lstm_units = self.hyperparameters.get('lstm_units', 128)
        epochs = self.hyperparameters.get('epochs', 50)
        dropout = self.hyperparameters.get('dropout', 0.2)
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        lookback = self.hyperparameters.get('lookback', 12)
        
        # Scale data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, lookback)
        X_test, y_test = self.create_sequences(test_scaled, lookback)
        
        # Build model
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(lstm_units, return_sequences=(lstm_layers > 1), 
                      input_shape=(lookback, 1)))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, lstm_layers):
            return_seq = (i < lstm_layers - 1)
            model.add(LSTM(lstm_units, return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Train
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return model, scaler, X_test, y_test
    
    def create_sequences(self, data, lookback):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def predict_lstm(self, model, X_test, scaler):
        """Make predictions with LSTM"""
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions_scaled = model.predict(X_test_reshaped, verbose=0)
        predictions = scaler.inverse_transform(predictions_scaled)
        return predictions.flatten()
    
    def generate_forecast(self, model, train_data, scaler, horizon):
        """Generate future forecast"""
        lookback = self.hyperparameters.get('lookback', 12)
        
        # Get last sequence
        train_scaled = scaler.transform(train_data)
        last_sequence = train_scaled[-lookback:]
        
        forecast = []
        current_sequence = last_sequence.copy()
        
        for _ in range(horizon):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape((1, lookback, 1))
            
            # Predict next value
            next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            forecast.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform
        forecast_array = np.array(forecast).reshape(-1, 1)
        forecast_original = scaler.inverse_transform(forecast_array)
        
        return forecast_original.flatten()
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2)
        }
    
    def detect_trend(self, df):
        """Detect overall trend"""
        values = df.iloc[:, 0].values
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "upward"
        elif slope < -0.01:
            return "downward"
        else:
            return "stable"
    
    def detect_seasonality(self, df):
        """Detect seasonality"""
        # Simple seasonality detection
        # In production, use statsmodels seasonal_decompose
        values = df.iloc[:, 0].values
        
        if len(values) < 24:
            return False
        
        # Check for repeating patterns
        autocorr = np.correlate(values - values.mean(), values - values.mean(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Look for significant peaks
        peaks = (autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:])
        return bool(np.any(peaks & (autocorr[1:-1] > 0.3)))
    
    def check_stationarity(self, df):
        """Check if series is stationary"""
        # Simplified check - in production use ADF test
        values = df.iloc[:, 0].values
        
        # Check if mean and std are stable
        mid = len(values) // 2
        mean_first = np.mean(values[:mid])
        mean_second = np.mean(values[mid:])
        
        std_first = np.std(values[:mid])
        std_second = np.std(values[mid:])
        
        mean_stable = abs(mean_first - mean_second) < (0.1 * mean_first)
        std_stable = abs(std_first - std_second) < (0.1 * std_first)
        
        return mean_stable and std_stable
    
    def generate_plots(self, df, test_index, predictions, forecast, horizon):
        """Generate visualization plots"""
        target_col = self.target_info.get('Target Outcome', 'sales')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Historical Data & Forecast', 'Prediction Accuracy'),
            vertical_spacing=0.15
        )
        
        # Plot 1: Full timeline with forecast
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add predictions on test set
        fig.add_trace(
            go.Scatter(
                x=test_index,
                y=predictions,
                mode='lines',
                name='Predictions',
                line=dict(color='orange', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add forecast
        last_date = df.index[-1]
        freq = self.detect_frequency(df)
        forecast_dates = pd.date_range(start=last_date, periods=horizon+1, freq=freq)[1:]
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Plot 2: Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=df.loc[test_index].iloc[:, 0],
                y=predictions,
                mode='markers',
                name='Test Set',
                marker=dict(color='purple', size=8)
            ),
            row=2, col=1
        )
        
        # Add perfect prediction line
        min_val = min(df.loc[test_index].iloc[:, 0].min(), predictions.min())
        max_val = max(df.loc[test_index].iloc[:, 0].max(), predictions.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Actual", row=2, col=1)
        fig.update_yaxes(title_text=target_col, row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"{self.model_type} Time Series Forecast"
        )
        
        # Save plot
        plot_path = self.output_dir / f"timeseries_forecast_{self.buyerid}.html"
        plot_result = save_plot(
            fig,
            str(plot_path),
            title=f"{self.model_type} Time Series Forecast",
        )
        
        return {
            'forecast_plot': plot_result["html_path"].replace("\\", "/"),
            'forecast_plot_png': plot_result["image_path"].replace("\\", "/"),
            'forecast_plot_export_message': plot_result["message"],
            'interactive': True
        }



class SARIMAExecutor:
    def __init__(self, freq="D"):
        self.freq = freq

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

    def detect_seasonality(self, series, max_lag=365, threshold=0.3, min_period=2):
        series = series.dropna()

        if len(series) < min_period * 2:
            return False, None

        max_lag = min(max_lag, len(series) // 2)

        acf_vals = acf(series, nlags=max_lag, fft=True)
        acf_vals[0] = 0

        candidate_lags = np.where(acf_vals > threshold)[0]
        candidate_lags = candidate_lags[candidate_lags >= min_period]

        if len(candidate_lags) == 0:
            return False, None

        m = candidate_lags[np.argmax(acf_vals[candidate_lags])]
        return True, int(m)

    def train(self, df, date_col, target_col):
        ts = df[[date_col, target_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.sort_values(date_col)
        ts.set_index(date_col, inplace=True)
        ts = ts.asfreq(self.freq)

        ts[target_col] = ts[target_col].interpolate()

        series = ts[target_col]

        seasonal, m = self.detect_seasonality(series)

        auto_model = pm.auto_arima(
            series,
            seasonal=seasonal,
            m=m if seasonal else 1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        return model, ts, order, seasonal_order, seasonal, m

    def forecast(self, model, steps, last_date):
        forecast = model.get_forecast(steps=steps)
        conf_int = forecast.conf_int()

        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=self.freq),
            periods=steps,
            freq=self.freq
        )

        return pd.DataFrame({
            "date": forecast_index,
            "forecast": forecast.predicted_mean.values,
            "lower_ci": conf_int.iloc[:, 0].values,
            "upper_ci": conf_int.iloc[:, 1].values
        })

    def evaluate(self, model, train_ts, test_ts, target_col):
        steps = len(test_ts)

        forecast = model.get_forecast(steps=steps)
        preds = forecast.predicted_mean.values

        y_true = test_ts[target_col].values

        metrics = {
            "RMSE": self.rmse(y_true, preds),
            "MAPE": self.mape(y_true, preds),
            "R2": r2_score(y_true, preds)
        }

        return metrics, y_true, preds

    def sarima_plots(self, model, ts, forecast_df, target_col):
        # ------------------ 1. Actual vs Fitted
        plt.figure()
        plt.plot(ts.index, ts[target_col], label="Actual")
        plt.plot(ts.index, model.fittedvalues, label="Fitted")
        plt.legend()
        plt.title("Actual vs Fitted")
        plt.show()

        # ------------------ 2. Residuals
        residuals = model.resid

        plt.figure()
        plt.plot(residuals)
        plt.title("Residuals")
        plt.show()

        # ------------------ 3. Residual Distribution
        plt.figure()
        residuals.plot(kind="kde")
        plt.title("Residual Density")
        plt.show()

        # ------------------ 4. ACF
        plot_acf(residuals.dropna(), lags=30)
        plt.show()

        # ------------------ 5. PACF
        plot_pacf(residuals.dropna(), lags=30)
        plt.show()

        # ------------------ 6. Forecast Plot
        plt.figure()
        plt.plot(ts.index, ts[target_col], label="History")
        plt.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast")
        plt.fill_between(
            forecast_df["date"],
            forecast_df["lower_ci"],
            forecast_df["upper_ci"],
            alpha=0.3
        )
        plt.legend()
        plt.title("SARIMA Forecast")
        plt.show()

    def seasonality_explain(self, seasonal, m):
        return {
            "seasonality_detected": seasonal,
            "seasonal_period": m,
            "interpretation": (
                f"Recurring pattern detected every {m} time steps"
                if seasonal else
                "No statistically significant repeating pattern detected"
            )
        }

    def model_structure(self, order, seasonal_order):
        p, d, q = order
        P, D, Q, m = seasonal_order

        return {
            "non_seasonal_order": {
                "AR_terms": p,
                "Differencing": d,
                "MA_terms": q
            },
            "seasonal_order": {
                "Seasonal_AR": P,
                "Seasonal_Differencing": D,
                "Seasonal_MA": Q,
                "Seasonal_Period": m
            }
        }

    def fit_quality(self, ts, model, target_col):
        fitted = model.fittedvalues
        actual = ts[target_col]
        error = actual - fitted

        return {
            "mean_fit_error": round(error.mean(), 3),
            "fit_volatility": round(error.std(), 3)
        }

    def residual_diagnostics(self, model, max_lag=30):
        residuals = model.resid.dropna()
        acf_vals = acf(residuals, nlags=max_lag)

        significant_lags = np.where(np.abs(acf_vals) > 0.2)[0]
        significant_lags = significant_lags[significant_lags > 0]

        return {
            "residual_mean": round(residuals.mean(), 4),
            "residual_std": round(residuals.std(), 4),
            "autocorrelation_leakage": len(significant_lags) > 0
        }

    def uncertainty_insights(self, forecast_df):
        forecast_df["ci_width"] = (
            forecast_df["upper_ci"] - forecast_df["lower_ci"]
        )

        return {
            "average_confidence_width": round(forecast_df["ci_width"].mean(), 3),
            "max_confidence_width": round(forecast_df["ci_width"].max(), 3)
        }

    def prediction_behavior(self, test_ts, preds, target_col):
        errors = test_ts[target_col].values - preds

        return {
            "mean_forecast_error": round(errors.mean(), 3),
            "error_volatility": round(errors.std(), 3)
        }

    def llm_payload(
        self,
        model,
        ts,
        forecast_df,
        test_ts,
        preds,
        target_col,
        order,
        seasonal_order,
        seasonal,
        m,
        metrics
    ):
        return {
            "model": "SARIMA",
            "metrics": {
                "RMSE": round(metrics["RMSE"], 3),
                "MAPE_percent": round(metrics["MAPE"], 2),
                "R2_score": round(metrics["R2"], 3)
            },
            "seasonality_analysis": self.seasonality_explain(seasonal, m),
            "model_structure": self.model_structure(order, seasonal_order),
            "fit_diagnostics": self.fit_quality(ts, model, target_col),
            "residual_diagnostics": self.residual_diagnostics(model),
            "uncertainty_analysis": self.uncertainty_insights(forecast_df),
            "forecast_behavior": self.prediction_behavior(test_ts, preds, target_col)
        }


class ProphetExecutor:
    def __init__(
        self,
        request,
        freq="D",
        seasonalities="auto",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    ):
        self.freq = freq
        self.seasonalities = seasonalities
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
        self.prophet_df = None
        self.request = request
        self.buyerid = request.session.get('buyerid', request.user.id)

        self.output_dir = Path(settings.MEDIA_ROOT) / 'files' / 'chatbot' / str(self.buyerid)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def train(self, df, date_col, target_col):
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df[date_col] = pd.to_datetime(prophet_df[date_col])

        prophet_df = prophet_df.rename(
            columns={date_col: "ds", target_col: "y"}
        ).sort_values("ds")

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )

        # Auto seasonality
        if self.seasonalities == "auto":
            model.add_seasonality(name="weekly", period=7, fourier_order=3)
            model.add_seasonality(name="yearly", period=365.25, fourier_order=10)

        elif isinstance(self.seasonalities, dict):
            for name, params in self.seasonalities.items():
                model.add_seasonality(
                    name=name,
                    period=params["period"],
                    fourier_order=params["fourier_order"]
                )

        model.fit(prophet_df)

        self.model = model
        self.prophet_df = prophet_df

        return model, prophet_df


    def forecast(self, periods):
        future = self.model.make_future_dataframe(periods=periods, freq=self.freq)
        forecast = self.model.predict(future)
        return forecast


    # def plots(self, forecast, actual_df=None):
    #     fig1 = self.model.plot(forecast)
    #     plt.title("Prophet Forecast")
    #     plt.show()

    #     fig2 = self.model.plot_components(forecast)
    #     plt.show()

    #     if actual_df is not None:
    #         merged = actual_df.merge(
    #             forecast[["ds", "yhat"]],
    #             on="ds",
    #             how="left"
    #         )

    #         plt.figure()
    #         plt.plot(merged["ds"], merged["y"], label="Actual")
    #         plt.plot(merged["ds"], merged["yhat"], label="Predicted")
    #         plt.legend()
    #         plt.title("Actual vs Predicted")
    #         plt.show()

    #     plt.figure()
    #     plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    #     plt.fill_between(
    #         forecast["ds"],
    #         forecast["yhat_lower"],
    #         forecast["yhat_upper"],
    #         alpha=0.3
    #     )
    #     plt.legend()
    #     plt.title("Forecast Uncertainty")
    #     plt.show()

    def plots(self, forecast, actual_df=None):
        from pathlib import Path
        # from plotly import plot_plotly, plot_components_plotly
        from prophet.plot import plot_plotly, plot_components_plotly
        import plotly.graph_objects as go

        self.output_dir = Path(settings.MEDIA_ROOT) / 'files' / 'chatbot' / str(self.buyerid)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # fig1 = plot_plotly(self.model, forecast)
        # fig1.update_layout(title="Prophet Forecast")
        import plotly.graph_objects as go

        last_hist_date = self.model.history['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_hist_date]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.model.history['ds'],
            y=self.model.history['y'],
            mode="lines",
            name="History"
        ))
        fig1.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode="lines",
            name="Forecast"
        ))

        # # Upper CI
        # fig.add_trace(go.Scatter(
        #     x=future_forecast['ds'],
        #     y=future_forecast['yhat_upper'],
        #     mode="lines",
        #     line=dict(width=0),
        #     showlegend=False
        # ))

        # # Lower CI + Fill
        # fig.add_trace(go.Scatter(
        #     x=future_forecast['ds'],
        #     y=future_forecast['yhat_lower'],
        #     mode="lines",
        #     fill="tonexty",
        #     name="Confidence Interval",
        #     line=dict(width=0),
        #     opacity=0.3
        # ))

        fig1.update_layout(
            title="Prophet Forecast",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )

        #fig1.show()

        forecast_path = self.output_dir / f"prophet_forecast_{self.buyerid}.html"
        forecast_result = save_plot(fig1, str(forecast_path), title="Prophet Forecast")
        saved_paths["forecast_plot"] = forecast_result["html_path"]
        saved_paths["forecast_plot_png"] = forecast_result["image_path"]
        saved_paths["forecast_plot_export_message"] = forecast_result["message"]
        
        # ----------------------------------
        # 2. Components plot (Plotly)
        # ----------------------------------
        fig2 = plot_components_plotly(self.model, forecast)
        components_path = self.output_dir / f"prophet_components_{self.buyerid}.html"
        components_result = save_plot(fig2, str(components_path), title="Prophet Components")
        saved_paths["components_plot"] = components_result["html_path"]
        saved_paths["components_plot_png"] = components_result["image_path"]
        saved_paths["components_plot_export_message"] = components_result["message"]

        # ----------------------------------
        # 3. Actual vs Predicted (Plotly)
        # ----------------------------------
        if actual_df is not None:
            merged = actual_df.merge(
                forecast[["ds", "yhat"]],
                on="ds",
                how="left"
            )

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], name="Actual"))
            fig3.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], name="Predicted"))
            fig3.update_layout(title="Actual vs Predicted")

            avp_path = self.output_dir / f"actual_vs_predicted_{self.buyerid}.html"
            avp_result = save_plot(fig3, str(avp_path), title="Actual vs Predicted")
            saved_paths["actual_vs_predicted_plot"] = avp_result["html_path"]
            saved_paths["actual_vs_predicted_plot_png"] = avp_result["image_path"]
            saved_paths["actual_vs_predicted_plot_export_message"] = avp_result["message"]

        # ----------------------------------
        # 4. Forecast Uncertainty (Plotly)
        # ----------------------------------
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
        fig4.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            fill=None,
            mode="lines",
            line_color="lightgrey",
            showlegend=False
        ))
        fig4.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill="tonexty",
            mode="lines",
            line_color="lightgrey",
            name="Confidence Interval"
        ))

        fig4.update_layout(title="Forecast Uncertainty")

        uncertainty_path = self.output_dir / f"forecast_uncertainty_{self.buyerid}.html"
        uncertainty_result = save_plot(fig4, str(uncertainty_path), title="Forecast Uncertainty")
        saved_paths["uncertainty_plot"] = uncertainty_result["html_path"]
        saved_paths["uncertainty_plot_png"] = uncertainty_result["image_path"]
        saved_paths["uncertainty_plot_export_message"] = uncertainty_result["message"]

        return saved_paths


    def evaluate(self, forecast_df):
        eval_df = self.prophet_df.merge(
            forecast_df[["ds", "yhat"]],
            on="ds",
            how="inner"
        )

        y_true = eval_df["y"].values
        y_pred = eval_df["yhat"].values

        metrics = {
            "RMSE": self.rmse(y_true, y_pred),
            "MAPE": self.mape(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

        plt.figure(figsize=(10, 5))
        plt.plot(eval_df["ds"], y_true, label="Actual")
        plt.plot(eval_df["ds"], y_pred, label="Predicted")
        plt.legend()
        plt.title("Prophet: Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()

        return metrics


    @staticmethod
    def extract_trend_insights(forecast):
        trend_start = forecast["trend"].iloc[0]
        trend_end = forecast["trend"].iloc[-1]
        trend_change_pct = ((trend_end - trend_start) / abs(trend_start)) * 100

        return {
            "trend_start_value": round(trend_start, 2),
            "trend_end_value": round(trend_end, 2),
            "trend_direction": "increasing" if trend_change_pct > 0 else "decreasing",
            "trend_change_percent": round(trend_change_pct, 2)
        }

    @staticmethod
    def extract_seasonality_strength(forecast):
        seasonality_info = {}

        if "weekly" in forecast.columns:
            seasonality_info["weekly_strength"] = round(
                forecast["weekly"].abs().mean(), 3
            )

        if "yearly" in forecast.columns:
            seasonality_info["yearly_strength"] = round(
                forecast["yearly"].abs().mean(), 3
            )

        return seasonality_info

    @staticmethod
    def extract_uncertainty_insights(forecast):
        forecast = forecast.copy()
        forecast["uncertainty_width"] = (
            forecast["yhat_upper"] - forecast["yhat_lower"]
        )

        return {
            "avg_uncertainty_width": round(
                forecast["uncertainty_width"].mean(), 3
            ),
            "max_uncertainty_width": round(
                forecast["uncertainty_width"].max(), 3
            ),
            "uncertainty_trend": "increasing"
            if forecast["uncertainty_width"].iloc[-1] >
               forecast["uncertainty_width"].iloc[0]
            else "stable/decreasing"
        }

    @staticmethod
    def extract_prediction_deviation(actual_df, forecast):
        merged = actual_df.merge(
            forecast[["ds", "yhat"]],
            on="ds",
            how="inner"
        )

        merged["error"] = merged["y"] - merged["yhat"]

        return {
            "mean_error": round(merged["error"].mean(), 3),
            "max_overprediction": round(merged["error"].min(), 3),
            "max_underprediction": round(merged["error"].max(), 3),
            "error_volatility": round(merged["error"].std(), 3)
        }


    def llm_payload(self, actual_df, forecast):
        return {
            "model": "Prophet",
            "metrics": self.evaluate(forecast),
            "trend_analysis": self.extract_trend_insights(forecast),
            "seasonality_analysis": self.extract_seasonality_strength(forecast),
            "uncertainty_analysis": self.extract_uncertainty_insights(forecast),
            "prediction_behavior": self.extract_prediction_deviation(actual_df, forecast)
        }



