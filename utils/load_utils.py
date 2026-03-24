from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import holidays

DateTimeLike = Union[str, datetime, pd.Timestamp]

SUPPORTED_HOUSEHOLD_ID = 1
FORECAST_HORIZON = 96  # 96 x 15 min = 24h


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def _resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)

    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]

    found = next((candidate for candidate in candidates if candidate.exists()), None)
    if found is not None:
        return found

    raise FileNotFoundError(f"Could not find file: {path_like}")


# ------------------------------------------------------------
# Time parsing
# ------------------------------------------------------------

def _parse_forecast_time(forecast_time: DateTimeLike) -> pd.Timestamp:
    ts = pd.Timestamp(forecast_time)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


# ------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------

def load_feature_engineered_dataset(
    csv_path: Union[str, Path] = "data/shifted-date-residential1_feature_engineered_full.csv",
) -> pd.DataFrame:

    csv_path = _resolve_path(csv_path)

    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    return df


# ------------------------------------------------------------
# Feature builder (critical)
# ------------------------------------------------------------

def _build_feature_row_from_dataset(
    df_all: pd.DataFrame,
    forecast_time: pd.Timestamp,
    feature_cols: list,
) -> pd.DataFrame:

    df_all = df_all.copy()
    df_all.index = pd.to_datetime(df_all.index, utc=True)
    df_all = df_all.sort_index()

    if "load" not in df_all.columns:
        raise ValueError("Dataset must contain 'load' column")

    if forecast_time not in df_all.index:
        raise ValueError(f"forecast_time {forecast_time} not found in dataset index")

    # --------------------------------------------------------
    # Historical load (ONLY before forecast_time)
    # --------------------------------------------------------
    df_hist = df_all.loc[df_all.index < forecast_time, ["load"]].copy()

    if len(df_hist) < 192:
        raise ValueError("Not enough history for lag_192")

    # FIX: enforce continuous 15-min index
    df_hist = df_hist.asfreq("15min")
    df_hist["load"] = df_hist["load"].ffill()

    # --------------------------------------------------------
    # Row at forecast time (for weather)
    # --------------------------------------------------------
    row_at_forecast = df_all.loc[[forecast_time]].copy()

    de_holidays = holidays.Germany()

    feature_row = {
        "hour": forecast_time.hour,
        "day_of_week": forecast_time.dayofweek,
        "month": forecast_time.month,
        "is_weekend": int(forecast_time.dayofweek >= 5),
        "is_holiday": int(
            forecast_time.tz_localize(None).normalize() in de_holidays
        ),
        "lag_1": df_hist["load"].iloc[-1],
        "lag_2": df_hist["load"].iloc[-2],
        "lag_3": df_hist["load"].iloc[-3],
        "lag_96": df_hist["load"].iloc[-96],
        "lag_192": df_hist["load"].iloc[-192],
        "delta_1": df_hist["load"].iloc[-1] - df_hist["load"].iloc[-2],
        "delta_2": df_hist["load"].iloc[-2] - df_hist["load"].iloc[-3],
        "rolling_mean_1h": df_hist["load"].iloc[-4:].mean(),
        "rolling_mean_24h": df_hist["load"].iloc[-96:].mean(),
        "rolling_std_1h": df_hist["load"].iloc[-4:].std(),
    }

    # --------------------------------------------------------
    # Weather features (validated)
    # --------------------------------------------------------
    for col in row_at_forecast.columns:
        if col in feature_cols:
            val = row_at_forecast.iloc[0][col]

            if pd.isna(val):
                raise ValueError(f"Weather feature {col} is NaN at forecast_time")

            feature_row[col] = val

    X_latest = pd.DataFrame([feature_row])

    # enforce exact feature order
    missing_cols = [col for col in feature_cols if col not in X_latest.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    return X_latest[feature_cols]


# ------------------------------------------------------------
# Main forecast function
# ------------------------------------------------------------

def get_daily_load_forecast(
    forecast_time: DateTimeLike,
    household_id: int = 1,
    model_path: Union[str, Path] = "models/load_forecast_model.pkl",
    feature_dataset_path: Union[str, Path] = "data/shifted-date-residential1_feature_engineered_full.csv",
) -> pd.DataFrame:
    """
    Forecast next 24h using:
    - historical load (from dataset)
    - weather (already stored in dataset)

    Train-once → predict-many design
    """

    if household_id != SUPPORTED_HOUSEHOLD_ID:
        raise ValueError(
            f"Model only supports household_id={SUPPORTED_HOUSEHOLD_ID}"
        )

    forecast_time = _parse_forecast_time(forecast_time)

    # --------------------------------------------------------
    # Load model (with feature_cols)
    # --------------------------------------------------------
    model_path = _resolve_path(model_path)
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
    else:
        raise ValueError("Model must be saved as dict with model + feature_cols")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    df_all = load_feature_engineered_dataset(feature_dataset_path)

    if forecast_time > df_all.index.max():
        raise ValueError("forecast_time beyond dataset range")

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------
    X_latest = _build_feature_row_from_dataset(
        df_all=df_all,
        forecast_time=forecast_time,
        feature_cols=feature_cols,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    y_pred = model.predict(X_latest).flatten()

    future_times = pd.date_range(
        start=forecast_time,
        periods=FORECAST_HORIZON,
        freq="15min",
        tz="UTC",
    )

    return pd.DataFrame({
        "time": future_times,
        "predicted_kwh": np.round(y_pred, 4),
        "predicted_kw": np.round(y_pred * 4, 4),
        "household": f"residential{household_id}",
        "source": "historical_simulation_forecast",
    })


__all__ = [
    "load_feature_engineered_dataset",
    "get_daily_load_forecast",
]