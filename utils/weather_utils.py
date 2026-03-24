from pathlib import Path
from typing import Union, Optional
import json

import numpy as np
import pandas as pd
import requests

PathLike = Union[str, Path]

ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 60

COMMON_HOURLY_WEATHER_VARS = [
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
]


def _resolve_path(path_like: PathLike) -> Path:
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


def _resolve_output_path(path_like: PathLike) -> Path:
    path = Path(path_like)

    if path.is_absolute():
        return path

    return Path(__file__).resolve().parents[1] / path


def load_weather_config(config_path: PathLike = "config.json") -> dict:
    config_path = _resolve_path(config_path)

    with config_path.open("r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    required_keys = {"lat", "lon"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required weather config keys: {sorted(missing_keys)}")

    return config


def _normalize_calendar(household_calendar: pd.DataFrame) -> pd.DataFrame:
    if "utc_timestamp" not in household_calendar.columns:
        raise ValueError("household_calendar must contain 'utc_timestamp'")

    calendar_df = household_calendar.copy()
    calendar_df["utc_timestamp"] = pd.to_datetime(
        calendar_df["utc_timestamp"], utc=True, errors="coerce"
    )
    calendar_df = (
        calendar_df
        .dropna(subset=["utc_timestamp"])
        .sort_values("utc_timestamp")
        .drop_duplicates(subset=["utc_timestamp"])
        .reset_index(drop=True)
    )

    if calendar_df.empty:
        raise ValueError("household_calendar is empty")

    return calendar_df


def _build_api_params(
    config: dict,
    start_date: str,
    end_date: str,
) -> dict:
    params = {
        "latitude": config["lat"],
        "longitude": config["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(COMMON_HOURLY_WEATHER_VARS),
        "timezone": "UTC",
    }

    if (
        "global_tilted_irradiance" in COMMON_HOURLY_WEATHER_VARS
        and {"tilt", "azimuth"}.issubset(config)
    ):
        params["tilt"] = config["tilt"]
        params["azimuth"] = config["azimuth"]

    return params


def _parse_hourly_payload(payload: dict, source_label: str) -> pd.DataFrame:
    if isinstance(payload, dict) and payload.get("error"):
        reason = payload.get("reason", "Unknown API error")
        raise ValueError(f"Open-Meteo API error: {reason}")

    hourly_payload = payload.get("hourly", {})
    if "time" not in hourly_payload:
        raise ValueError(
            f"Missing 'time' in hourly payload. Available keys: {list(hourly_payload.keys())}"
        )

    chunk_df = pd.DataFrame(hourly_payload)
    chunk_df["time"] = pd.to_datetime(chunk_df["time"], utc=True, errors="coerce")
    chunk_df = chunk_df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for weather_col in COMMON_HOURLY_WEATHER_VARS:
        if weather_col not in chunk_df.columns:
            chunk_df[weather_col] = np.nan

    chunk_df = chunk_df[["time", *COMMON_HOURLY_WEATHER_VARS]].copy()
    chunk_df["source"] = source_label

    return chunk_df


def fetch_raw_historical_weather(
    household_calendar: pd.DataFrame,
    config_path: PathLike = "config.json",
    save_raw_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    Fetch raw hourly historical weather data from the Open-Meteo archive API
    for the timestamp range covered by household_calendar.
    """
    calendar_df = _normalize_calendar(household_calendar)
    config = load_weather_config(config_path)

    historical_start_date = calendar_df["utc_timestamp"].min().date()
    historical_end_date = calendar_df["utc_timestamp"].max().date()

    month_periods = pd.period_range(
        start=historical_start_date,
        end=historical_end_date,
        freq="M",
    )

    chunk_frames = []

    for period in month_periods:
        chunk_start = max(historical_start_date, period.start_time.date())
        chunk_end = min(historical_end_date, period.end_time.date())

        params = _build_api_params(
            config=config,
            start_date=chunk_start.isoformat(),
            end_date=chunk_end.isoformat(),
        )

        response = requests.get(
            ARCHIVE_ENDPOINT,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        payload = response.json()
        chunk_df = _parse_hourly_payload(
            payload=payload,
            source_label="open_meteo_archive",
        )
        chunk_frames.append(chunk_df)

    if not chunk_frames:
        raise ValueError("No historical weather chunks were fetched.")

    raw_weather_df = (
        pd.concat(chunk_frames, ignore_index=True)
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="first")
        .reset_index(drop=True)
    )

    if save_raw_path is not None:
        save_raw_path = _resolve_output_path(save_raw_path)
        save_raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_weather_df.to_csv(save_raw_path, index=False)

    return raw_weather_df


def fetch_raw_forecast_weather(
    household_calendar: pd.DataFrame,
    config_path: PathLike = "config.json",
) -> pd.DataFrame:
    """
    Fetch raw hourly forecast weather data from the Open-Meteo forecast API
    for the timestamp range covered by household_calendar.
    """
    calendar_df = _normalize_calendar(household_calendar)
    config = load_weather_config(config_path)

    forecast_start_date = calendar_df["utc_timestamp"].min().date()
    forecast_end_date = calendar_df["utc_timestamp"].max().date()

    params = _build_api_params(
        config=config,
        start_date=forecast_start_date.isoformat(),
        end_date=forecast_end_date.isoformat(),
    )

    response = requests.get(
        FORECAST_ENDPOINT,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    raw_weather_df = _parse_hourly_payload(
        payload=payload,
        source_label="open_meteo_forecast",
    )

    return raw_weather_df


def build_full_weather_15min(
    raw_weather_df: pd.DataFrame,
    household_calendar: pd.DataFrame,
    save_full_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    Convert raw hourly weather data into a clean 15-minute UTC weather table
    aligned to the household calendar.
    """
    if "utc_timestamp" not in household_calendar.columns:
        raise ValueError("household_calendar must contain 'utc_timestamp'")

    if "time" not in raw_weather_df.columns:
        raise ValueError("raw_weather_df must contain 'time'")

    calendar_df = household_calendar.copy()
    calendar_df["utc_timestamp"] = pd.to_datetime(
        calendar_df["utc_timestamp"], utc=True, errors="coerce"
    )
    calendar_df = calendar_df.dropna(subset=["utc_timestamp"])
    calendar_df = (
        calendar_df
        .sort_values("utc_timestamp")
        .drop_duplicates(subset=["utc_timestamp"])
        .reset_index(drop=True)
    )

    if calendar_df.empty:
        raise ValueError("household_calendar is empty after timestamp parsing")

    working_df = raw_weather_df.copy()
    working_df["time"] = pd.to_datetime(working_df["time"], utc=True, errors="coerce")
    working_df = working_df.dropna(subset=["time"]).sort_values("time")
    working_df = working_df.drop_duplicates(subset=["time"], keep="first")
    working_df = working_df.set_index("time")

    weather_cols = [col for col in COMMON_HOURLY_WEATHER_VARS if col in working_df.columns]
    if not weather_cols:
        raise ValueError("No expected weather columns found in raw_weather_df")

    for weather_col in weather_cols:
        working_df[weather_col] = pd.to_numeric(working_df[weather_col], errors="coerce")

    calendar_index = pd.DatetimeIndex(calendar_df["utc_timestamp"]).tz_convert("UTC")
    calendar_index = pd.DatetimeIndex(calendar_index.sort_values().unique())

    combined_index = working_df.index.union(calendar_index)

    full_weather_indexed = working_df[weather_cols].reindex(combined_index).sort_index()
    full_weather_indexed = full_weather_indexed.interpolate(method="time", limit_direction="both")
    full_weather_indexed = full_weather_indexed.ffill().bfill()

    full_weather_df = full_weather_indexed.reindex(calendar_index).reset_index()
    full_weather_df = full_weather_df.rename(columns={"index": "utc_timestamp"})
    full_weather_df["source"] = "open_meteo_interpolated"

    if save_full_path is not None:
        save_full_path = _resolve_output_path(save_full_path)
        save_full_path.parent.mkdir(parents=True, exist_ok=True)
        full_weather_df.to_csv(save_full_path, index=False)

    return full_weather_df


def add_derived_weather_features(full_weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived weather features.
    Use this only if the training pipeline also uses these columns.
    """
    df = full_weather_df.copy()

    df["heating_degree_18c"] = np.maximum(18 - df["temperature_2m"], 0)
    df["cooling_degree_22c"] = np.maximum(df["temperature_2m"] - 22, 0)
    df["is_raining"] = (df["rain"].fillna(0) > 0).astype(int)
    df["is_snowing"] = (df["snowfall"].fillna(0) > 0).astype(int)
    df["is_dark"] = (df["shortwave_radiation"].fillna(0) < 10).astype(int)

    wind_dir_rad = np.deg2rad(df["wind_direction_10m"].fillna(0))
    df["wind_u_10m"] = df["wind_speed_10m"].fillna(0) * np.cos(wind_dir_rad)
    df["wind_v_10m"] = df["wind_speed_10m"].fillna(0) * np.sin(wind_dir_rad)

    return df


def build_weather_for_household_calendar(
    household_timestamps: Union[pd.Index, pd.Series, pd.DataFrame],
    config_path: PathLike = "config.json",
    save_raw_path: Optional[PathLike] = None,
    save_full_path: Optional[PathLike] = None,
    include_derived: bool = False,
) -> pd.DataFrame:
    """
    Build weather data aligned to the household calendar.

    Logic:
    - past timestamps  -> archive API
    - future timestamps -> forecast API
    """
    if isinstance(household_timestamps, pd.DataFrame):
        if "utc_timestamp" not in household_timestamps.columns:
            raise ValueError("DataFrame input must contain 'utc_timestamp'")
        household_calendar = household_timestamps[["utc_timestamp"]].copy()
    elif isinstance(household_timestamps, (pd.Index, pd.Series)):
        household_calendar = pd.DataFrame({
            "utc_timestamp": pd.to_datetime(household_timestamps, utc=True, errors="coerce")
        })
    else:
        raise TypeError(
            "household_timestamps must be a DatetimeIndex, Series, or DataFrame with utc_timestamp"
        )

    household_calendar["utc_timestamp"] = pd.to_datetime(
        household_calendar["utc_timestamp"],
        utc=True,
        errors="coerce",
    )
    household_calendar = household_calendar.dropna(subset=["utc_timestamp"])
    household_calendar = (
        household_calendar
        .drop_duplicates(subset=["utc_timestamp"])
        .sort_values("utc_timestamp")
        .reset_index(drop=True)
    )

    if household_calendar.empty:
        raise ValueError("household_timestamps produced an empty calendar")

    now_utc = pd.Timestamp.now(tz="UTC")

    historical_calendar = household_calendar[
        household_calendar["utc_timestamp"] < now_utc
    ].copy()

    forecast_calendar = household_calendar[
        household_calendar["utc_timestamp"] >= now_utc
    ].copy()

    raw_frames = []

    if not historical_calendar.empty:
        historical_raw_df = fetch_raw_historical_weather(
            household_calendar=historical_calendar,
            config_path=config_path,
            save_raw_path=None,
        )
        raw_frames.append(historical_raw_df)

    if not forecast_calendar.empty:
        forecast_raw_df = fetch_raw_forecast_weather(
            household_calendar=forecast_calendar,
            config_path=config_path,
        )
        raw_frames.append(forecast_raw_df)

    if not raw_frames:
        raise ValueError("No weather data could be fetched for the requested calendar")

    raw_weather_df = (
        pd.concat(raw_frames, ignore_index=True)
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="first")
        .reset_index(drop=True)
    )

    if save_raw_path is not None:
        save_raw_path = _resolve_output_path(save_raw_path)
        save_raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_weather_df.to_csv(save_raw_path, index=False)

    full_weather_df = build_full_weather_15min(
        raw_weather_df=raw_weather_df,
        household_calendar=household_calendar,
        save_full_path=save_full_path,
    )

    if include_derived:
        return add_derived_weather_features(full_weather_df)

    return full_weather_df


__all__ = [
    "COMMON_HOURLY_WEATHER_VARS",
    "fetch_raw_historical_weather",
    "fetch_raw_forecast_weather",
    "build_full_weather_15min",
    "add_derived_weather_features",
    "build_weather_for_household_calendar",
    "load_weather_config",
]