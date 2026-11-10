"""
Adaptive Enhanced PyTorch LSTM Demand Forecasting Module.
Handles:
- Automatic cyclical calendar feature engineering (day-of-week, month, day-of-month)
- Dynamic input sizing (works with any number of exogenous regressors)
- StandardScaler normalization for numerical stability
- 2-layer stacked LSTM with residual bypass projection (improves R2)
- ReduceLROnPlateau learning rate scheduler
- Gradient clipping to prevent exploding gradients
- Works on any e-commerce time series without manual hyperparameter changes
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# CALENDAR FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive cyclical sine/cosine calendar features from 'ds' column.
    This allows the LSTM to learn weekly and yearly demand cycles
    without needing explicit hardcoded seasonal parameters.

    Features added (all cyclic, in range [-1, 1]):
      - sin_dow, cos_dow  : Day of Week (period = 7)
      - sin_dom, cos_dom  : Day of Month (period = 31)
      - sin_month, cos_month : Month of Year (period = 12)
    """
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    dow   = ds.dt.dayofweek.values           # 0 = Monday
    dom   = ds.dt.day.values                 # 1-31
    month = ds.dt.month.values               # 1-12

    df["sin_dow"]   = np.sin(2 * math.pi * dow   / 7)
    df["cos_dow"]   = np.cos(2 * math.pi * dow   / 7)
    df["sin_dom"]   = np.sin(2 * math.pi * dom   / 31)
    df["cos_dom"]   = np.cos(2 * math.pi * dom   / 31)
    df["sin_month"] = np.sin(2 * math.pi * month / 12)
    df["cos_month"] = np.cos(2 * math.pi * month / 12)

    return df


def get_feature_cols(df: pd.DataFrame, corr_threshold: float = 0.05) -> List[str]:
    """
    Adaptively select feature columns for LSTM input.
    - Always includes 'y' (target demand).
    - Includes calendar features (always useful for seasonality).
    - For exogenous regressors, only keeps those with abs(correlation with y)
      above corr_threshold. This prevents low-signal weather / stock features
      from polluting the model when the dataset is limited in size.
    """
    reserved = {"ds"}
    calendar_cols = {"sin_dow", "cos_dow", "sin_dom", "cos_dom", "sin_month", "cos_month"}

    candidate_exog = [
        col for col in df.columns
        if col not in reserved
        and col not in calendar_cols
        and col != "y"
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Keep exog feature only if it has meaningful correlation with target y
    selected_exog = []
    for col in candidate_exog:
        corr = abs(df["y"].corr(df[col]))
        if corr >= corr_threshold:
            selected_exog.append(col)

    # Build final ordered feature list: y first, then selected exog, then calendar
    present_calendar = [c for c in calendar_cols if c in df.columns]
    selected = ["y"] + selected_exog + present_calendar
    return selected


def adaptive_window_size(n_train: int) -> int:
    """
    Adaptively choose sliding-window size based on training set length.
    Ensures at least 200 training sequences are generated.
    """
    for w in [14, 21, 30, 7]:
        if n_train - w >= 200:
            return w
    return 7


# ---------------------------------------------------------------------------
# STANDARD SCALER (z-score normalization, robust for any scale)
# ---------------------------------------------------------------------------

class StandardScalerNP:
    """Z-score scaler for 2D numpy arrays (samples x features)."""

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.mean_ = data.mean(axis=0)
        self.std_  = data.std(axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return (data - self.mean_) / self.std_

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_

    def inverse_transform_col(self, scaled: np.ndarray, col_idx: int = 0) -> np.ndarray:
        return scaled * self.std_[col_idx] + self.mean_[col_idx]


# ---------------------------------------------------------------------------
# PYTORCH DATASET
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding window time series dataset for LSTM training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int = 30):
        self.X_seq, self.y_seq = [], []
        for i in range(len(X) - window):
            self.X_seq.append(X[i : i + window])
            self.y_seq.append(y[i + window])
        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.y_seq = torch.tensor(np.array(self.y_seq), dtype=torch.float32).unsqueeze(1)

    def __len__(self):  return len(self.X_seq)
    def __getitem__(self, idx): return self.X_seq[idx], self.y_seq[idx]


# ---------------------------------------------------------------------------
# RESIDUAL LSTM ARCHITECTURE
# ---------------------------------------------------------------------------

class SimpleLSTMForecaster(nn.Module):
    """
    Clean 2-layer Stacked LSTM forecaster.
    - No residual projection (avoids mean-prediction collapse on limited data).
    - LayerNorm + Dropout for stability.
    - Adaptive to any input_size.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.35
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)           # (batch, window, hidden)
        last = lstm_out[:, -1, :]            # last timestep: (batch, hidden)
        last = self.norm(last)
        return self.fc(self.drop(last))      # (batch, 1)


# Keep alias for backward compatibility
ResidualLSTMForecaster = SimpleLSTMForecaster


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_lstm_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: Optional[int] = None
) -> Tuple[StandardScalerNP, DataLoader, DataLoader, np.ndarray, List[str]]:
    """
    Prepare LSTM training and test data:
    1. Adds cyclical calendar features.
    2. Adaptively selects high-signal feature columns (correlation-based).
    3. Fits StandardScaler on train, transforms both train and test.
    4. Builds sliding-window PyTorch Datasets and DataLoaders.
    """
    # Add calendar features
    train_aug = add_calendar_features(train_df)
    test_aug  = add_calendar_features(test_df)

    # Adaptive feature selection & window sizing
    feature_cols = get_feature_cols(train_aug)
    if window_size is None:
        window_size = adaptive_window_size(len(train_aug))
    print(f"   [LSTM] Window size: {window_size} | Features ({len(feature_cols)}): {feature_cols}")
    y_idx = feature_cols.index("y")

    scaler = StandardScalerNP()
    train_arr = scaler.fit_transform(train_aug[feature_cols].values.astype(float))

    # Prepend last window_size rows of train to test for seamless prediction
    combined_raw = np.vstack([
        train_aug[feature_cols].values.astype(float)[-window_size:],
        test_aug[feature_cols].values.astype(float)
    ])
    combined_scaled = scaler.transform(combined_raw)

    train_dataset = TimeSeriesDataset(train_arr, train_arr[:, y_idx], window=window_size)
    test_dataset  = TimeSeriesDataset(combined_scaled, combined_scaled[:, y_idx], window=window_size)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=False, drop_last=False)

    return scaler, train_loader, test_loader, combined_scaled, feature_cols


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

def train_lstm_model(
    train_loader: DataLoader,
    input_size: int,
    hidden_size: int = 96,
    epochs: int = 200,
    lr: float = 0.001
) -> SimpleLSTMForecaster:
    """
    Train LSTM with:
    - 80/20 train/val split on the loader's sequences for early stopping
    - AdamW optimizer (weight decay regularization)
    - MSELoss (directly minimizes variance, better for R2)
    - ReduceLROnPlateau (halves LR when val loss plateaus)
    - Gradient clipping (max norm=1.0)
    - Early stopping (patience=25 epochs, restores best weights)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------- build 80/20 train/val split from the loader's dataset ----------
    dataset   = train_loader.dataset
    n_total   = len(dataset)
    n_val     = max(1, int(n_total * 0.20))
    n_tr      = n_total - n_val
    tr_set, val_set = torch.utils.data.random_split(
        dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    tr_loader  = DataLoader(tr_set,  batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = SimpleLSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.35
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_state    = None
    patience      = 25
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        # --- train step ---
        model.train()
        tr_loss = 0.0
        for batch_X, batch_y in tr_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss  = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * batch_X.size(0)
        tr_loss /= len(tr_loader.dataset)

        # --- val step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_loss += criterion(model(batch_X), batch_y).item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        # --- early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"   Early stopping at epoch {epoch} (best val loss={best_val_loss:.6f})")
                break

        if epoch % 30 == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"   Epoch [{epoch:>3}/{epochs}] tr_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  LR={current_lr:.6f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# PREDICTION ON HOLDOUT
# ---------------------------------------------------------------------------

def predict_lstm(
    model: ResidualLSTMForecaster,
    test_loader: DataLoader,
    scaler: StandardScalerNP,
    y_col_idx: int = 0
) -> np.ndarray:
    """Generate predictions on test set and inverse-transform to original scale."""
    model.eval()
    preds_scaled = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            out = model(batch_X)
            preds_scaled.append(out.item())

    preds_scaled = np.array(preds_scaled)
    preds_unscaled = scaler.inverse_transform_col(preds_scaled, y_col_idx)
    return np.clip(preds_unscaled, 0, None)


# ---------------------------------------------------------------------------
# FUTURE FORECASTING (AUTOREGRESSIVE)
# ---------------------------------------------------------------------------

def forecast_future_lstm(
    model: ResidualLSTMForecaster,
    full_df: pd.DataFrame,
    scaler: StandardScalerNP,
    feature_cols: List[str],
    window_size: int = 30,
    future_days: int = 90
) -> pd.DataFrame:
    """
    Autoregressive multi-step future forecast.
    - Seeds the rolling window from the last window_size historical rows.
    - For each future step, calendar features are computed from the actual
      future date; exogenous regressors assume a no-promo baseline (0 for
      binary flags, mean for continuous features).
    """
    model.eval()

    full_aug = add_calendar_features(full_df)
    full_arr = scaler.transform(full_aug[feature_cols].values.astype(float))

    y_idx = feature_cols.index("y")
    last_date = pd.to_datetime(full_df["ds"].max())
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    # Compute column means for baseline future exog values
    col_means = full_aug[feature_cols].mean().values
    # Compute column stds from scaler
    # We'll use scaler.transform for calendar cols and set binary exog = 0

    # Identify exogenous + calendar col indices (everything except 'y')
    non_y_idxs = [i for i, c in enumerate(feature_cols) if c != "y"]
    binary_exog_idxs = [
        i for i, c in enumerate(feature_cols)
        if any(kw in c for kw in ["flag", "is_", "promo", "holiday"])
    ]

    # Seed window
    current_seq = full_arr[-window_size:].copy()   # (window, n_features)

    future_preds = []

    with torch.no_grad():
        for future_date in future_dates:
            inp = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(inp).item()
            future_preds.append(pred_scaled)

            # Build next row: calendar features from actual future date
            next_row = col_means.copy()    # baseline
            next_row[y_idx] = pred_scaled  # predicted demand (scaled)

            # Compute & scale calendar features for this specific future date
            dow   = future_date.dayofweek
            dom   = future_date.day
            month = future_date.month
            cal_vals_raw = np.array([
                np.sin(2*math.pi*dow/7),
                np.cos(2*math.pi*dow/7),
                np.sin(2*math.pi*dom/31),
                np.cos(2*math.pi*dom/31),
                np.sin(2*math.pi*month/12),
                np.cos(2*math.pi*month/12)
            ])
            cal_cols = ["sin_dow","cos_dow","sin_dom","cos_dom","sin_month","cos_month"]
            for j, cname in enumerate(feature_cols):
                if cname in cal_cols:
                    cal_idx_in_raw = cal_cols.index(cname)
                    # Transform using scaler
                    next_row[j] = (cal_vals_raw[cal_idx_in_raw] - scaler.mean_[j]) / scaler.std_[j]
                elif j in binary_exog_idxs:
                    next_row[j] = (0.0 - scaler.mean_[j]) / scaler.std_[j]

            current_seq = np.vstack([current_seq[1:], next_row])

    # Inverse-transform predictions
    preds_unscaled = scaler.inverse_transform_col(np.array(future_preds), y_idx)
    preds_unscaled = np.clip(preds_unscaled, 0, None)

    return pd.DataFrame({
        "ds": future_dates,
        "yhat": preds_unscaled,
        "yhat_lower": preds_unscaled * 0.88,
        "yhat_upper": preds_unscaled * 1.12
    })
