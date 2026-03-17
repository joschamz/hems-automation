from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import holidays

DateLike = Union[str, date, datetime]


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


def get_daily_load_forecast(
    target_date: Optional[DateLike] = None,
    household_id: int = 1,
    model_path: Union[str, Path] = "models/load_forecast_model.pkl",
    csv_path: Union[str, Path] = "data/household_data_15min_singleindex.csv",
) -> pd.DataFrame:
    """
    Prototype day-ahead household load forecast using a pre-trained baseline model.
    Returns 96 rows at 15-minute UTC resolution.
    """
    target_day = _parse_target_date(target_date)
    model_path = _resolve_path(model_path)

    model = joblib.load(model_path)
    df_house = load_baseline_household_history(csv_path=csv_path, household_id=household_id)

    df_inf = df_house.copy()
    df_inf["hour"] = df_inf.index.hour
    df_inf["day_of_week"] = df_inf.index.dayofweek
    df_inf["month"] = df_inf.index.month
    df_inf["is_weekend"] = (df_inf["day_of_week"] >= 5).astype(int)
    de_holidays = holidays.Germany()
    df_inf["is_holiday"] = df_inf.index.tz_localize(None).normalize().map(
    lambda d: int(d in de_holidays)
    )

    df_inf["lag_1"] = df_inf["load"].shift(1)
    df_inf["lag_4"] = df_inf["load"].shift(4)
    df_inf["lag_96"] = df_inf["load"].shift(96)
    df_inf["rolling_mean_1h"] = df_inf["load"].rolling(4).mean()
    df_inf["rolling_mean_24h"] = df_inf["load"].rolling(96).mean()

    df_inf = df_inf.dropna().copy()

    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "lag_1",
        "lag_4",
        "lag_96",
        "rolling_mean_1h",
        "rolling_mean_24h",
    ]

    X_latest = df_inf[feature_cols].iloc[[-1]]
    y_pred = model.predict(X_latest).flatten()

    future_times = pd.date_range(
        start=pd.Timestamp(target_day.isoformat(), tz="UTC"),
        periods=96,
        freq="15min",
    )

    return pd.DataFrame({
        "time": future_times,
        "predicted_kwh": np.round(y_pred, 4),
        "predicted_kw": np.round(y_pred * 4, 4),
        "household": f"residential{household_id}",
        "source": "baseline_load_model",
    })


__all__ = ["get_daily_load_forecast", "load_baseline_household_history"]