from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import holidays

DateLike = Union[str, date, datetime]

# This model is currently trained for household 1 only.
SUPPORTED_HOUSEHOLD_ID = 1

FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "sunshine_duration",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_96",
    "lag_192",
    "rolling_mean_1h",
    "rolling_mean_24h",
]


def _parse_target_date(target_date: Optional[DateLike]) -> date:
    if target_date is None:
        return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).date()
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    if isinstance(target_date, str):
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    raise TypeError("target_date must be None, date, datetime, or YYYY-MM-DD string")


def _resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    found = next((c for c in candidates if c.exists()), None)
    if found:
        return found

    raise FileNotFoundError(f"Could not find file: {path_like}")


def load_baseline_household_history(
    csv_path: Union[str, Path] = "data/household_data_15min_singleindex.csv",
    household_id: int = 1,
) -> pd.DataFrame:
    csv_path = _resolve_path(csv_path)

    df = pd.read_csv(csv_path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.set_index("utc_timestamp").sort_index()

    grid_import_col = f"DE_KN_residential{household_id}_grid_import"
    pv_col = f"DE_KN_residential{household_id}_pv"
    grid_export_col = f"DE_KN_residential{household_id}_grid_export"

    if grid_import_col not in df.columns:
        raise ValueError(f"Missing required column: {grid_import_col}")

    house_df = df[[grid_import_col]].copy()
    house_df[pv_col] = df[pv_col] if pv_col in df.columns else 0.0
    house_df[grid_export_col] = df[grid_export_col] if grid_export_col in df.columns else 0.0

    house_diff = house_df.diff()

    house_diff["load"] = (
        house_diff[grid_import_col]
        + house_diff[pv_col]
        - house_diff[grid_export_col]
    ).clip(lower=0)

    return house_diff[["load"]].dropna().copy()


def load_weather_features(
    weather_path: Union[str, Path] = "data/weather_full_15min.csv",
) -> pd.DataFrame:
    weather_path = _resolve_path(weather_path)

    weather_df = pd.read_csv(weather_path)
    weather_df["utc_timestamp"] = pd.to_datetime(weather_df["utc_timestamp"], utc=True)
    weather_df = weather_df.sort_values("utc_timestamp").reset_index(drop=True)

    if "source" in weather_df.columns:
        weather_df = weather_df.drop(columns=["source"])

    return weather_df


def _build_feature_row(
    load_history_df: pd.DataFrame,
    weather_row_df: pd.DataFrame,
) -> pd.DataFrame:
    load_history_df = load_history_df.copy()
    load_history_df.index = pd.to_datetime(load_history_df.index, utc=True)
    load_history_df = load_history_df.sort_index()

    if "load" not in load_history_df.columns:
        raise ValueError("load_history_df must contain a 'load' column")

    if len(load_history_df) < 192:
        raise ValueError("Not enough load history to compute lag_192")

    weather_row_df = weather_row_df.copy()
    weather_row_df["utc_timestamp"] = pd.to_datetime(weather_row_df["utc_timestamp"], utc=True)
    weather_row_df = weather_row_df.set_index("utc_timestamp").sort_index()

    if len(weather_row_df) != 1:
        raise ValueError("weather_row_df must contain exactly one row")

    forecast_time = weather_row_df.index[0]
    weather_row = weather_row_df.iloc[0].to_dict()

    de_holidays = holidays.Germany()

    feature_row = {
        "hour": forecast_time.hour,
        "day_of_week": forecast_time.dayofweek,
        "month": forecast_time.month,
        "is_weekend": int(forecast_time.dayofweek >= 5),
        "is_holiday": int(forecast_time.tz_localize(None).normalize() in de_holidays),
        "lag_1": load_history_df["load"].iloc[-1],
        "lag_2": load_history_df["load"].iloc[-2],
        "lag_3": load_history_df["load"].iloc[-3],
        "lag_96": load_history_df["load"].iloc[-96],
        "lag_192": load_history_df["load"].iloc[-192],
        "rolling_mean_1h": load_history_df["load"].iloc[-4:].mean(),
        "rolling_mean_24h": load_history_df["load"].iloc[-96:].mean(),
    }

    for col, val in weather_row.items():
        feature_row[col] = val

    X_latest = pd.DataFrame([feature_row])

    missing_cols = [col for col in FEATURE_COLS if col not in X_latest.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    return X_latest[FEATURE_COLS]


def get_daily_load_forecast(
    target_date: Optional[DateLike] = None,
    household_id: int = 1,
    model_path: Union[str, Path] = "models/load_forecast_model.pkl",
    household_csv_path: Union[str, Path] = "data/household_data_15min_singleindex.csv",
    weather_csv_path: Union[str, Path] = "data/weather_full_15min.csv",
) -> pd.DataFrame:
    """
    Weather-aware prototype day-ahead household load forecast.
    Returns 96 rows at 15-minute UTC resolution.
    """

    if household_id != SUPPORTED_HOUSEHOLD_ID:
        raise ValueError(
            f"Current saved model is trained only for household_id={SUPPORTED_HOUSEHOLD_ID}"
        )

    target_day = _parse_target_date(target_date)
    model_path = _resolve_path(model_path)

    model = joblib.load(model_path)

    load_history_df = load_baseline_household_history(
        csv_path=household_csv_path,
        household_id=household_id,
    )

    weather_df = load_weather_features(weather_path=weather_csv_path)

    target_day_ts = pd.Timestamp(target_day, tz="UTC")

    weather_day_df = weather_df[
        weather_df["utc_timestamp"].dt.normalize() == target_day_ts.normalize()
    ].copy()

    if weather_day_df.empty:
        raise ValueError(f"No weather rows found for target_date={target_day}")

    # Use the first 15-minute weather row of the target day as forecast origin
    weather_row_df = weather_day_df.iloc[[0]].copy()
    forecast_time = weather_row_df["utc_timestamp"].iloc[0]

    # Keep only load history available before the forecast origin
    load_history_df = load_history_df[
        load_history_df.index < forecast_time
    ].copy()

    if load_history_df.empty:
        raise ValueError("No historical load data available before forecast_time")

    X_latest = _build_feature_row(
        load_history_df=load_history_df,
        weather_row_df=weather_row_df,
    )

    y_pred = model.predict(X_latest).flatten()

    future_times = pd.date_range(
        start=forecast_time,
        periods=96,
        freq="15min",
        tz="UTC",
    )

    return pd.DataFrame({
        "time": future_times,
        "predicted_kwh": np.round(y_pred, 4),
        "predicted_kw": np.round(y_pred * 4, 4),
        "household": f"residential{household_id}",
        "source": "weather_aware_load_model",
    })


__all__ = [
    "FEATURE_COLS",
    "get_daily_load_forecast",
    "load_baseline_household_history",
    "load_weather_features",
]