import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import requests

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
REQUIRED_CONFIG_KEYS = ("lat", "lon", "kwp", "tilt", "azimuth", "yield_factor")
TEMP_COEFFICIENT = -0.004


DateLike = Union[str, date, datetime]


def load_solar_config(config_path: Union[str, Path] = "user_config.json") -> Dict[str, float]:
    resolved_path = _resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    parsed_config: Dict[str, float] = {}
    for key in REQUIRED_CONFIG_KEYS:
        try:
            parsed_config[key] = float(config[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Config value for '{key}' must be numeric") from error

    return parsed_config


def get_daily_solar_kwh(
    target_date: Optional[DateLike] = None,
    mode: str = "auto",
    config_path: Union[str, Path] = "user_config.json",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Return one day (24h) of 15-minute predicted solar output in kWh.

    Columns:
    - time  (timezone-aware UTC)
    - predicted_kwh  (energy for each 15-minute interval)
    - predicted_kw
    - temperature_2m
    - source
    """
    config = load_solar_config(config_path)
    day = _parse_target_date(target_date)
    use_forecast = _resolve_mode(day, mode) == "forecast"

    try:
        return _fetch_day(day, config, use_forecast)
    except Exception as error:
        if not allow_fallback:
            raise
        return _fallback_clear_sky(day, config, error)


def _resolve_config_path(config_path: Union[str, Path]) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found at {path}")

    candidates = [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]
    found = next((c for c in candidates if c.exists()), None)
    if found:
        return found

    raise FileNotFoundError(
        f"Config file '{config_path}' not found. Checked: {', '.join(str(c) for c in candidates)}"
    )


def _parse_target_date(target_date: Optional[DateLike]) -> date:
    if target_date is None:
        return date.today() + timedelta(days=1)
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    if isinstance(target_date, str):
        try:
            return datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError("target_date must use YYYY-MM-DD format") from error
    raise TypeError("target_date must be None, date, datetime, or YYYY-MM-DD string")


def _resolve_mode(day: date, mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"auto", "forecast", "historical"}:
        raise ValueError("mode must be one of: auto, forecast, historical")
    today = date.today()
    if normalized == "auto":
        return "historical" if day < today else "forecast"
    if normalized == "forecast" and day < today:
        raise ValueError("Forecast mode cannot be used for past dates")
    if normalized == "historical" and day >= today:
        raise ValueError("Historical mode can only be used for past dates")
    return normalized


def _request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(f"Open-Meteo API error: {payload.get('reason', 'Unknown API error')}")
    return payload


def _fetch_day(day: date, config: Dict[str, float], use_forecast: bool) -> pd.DataFrame:
    """Fetch one day of solar irradiance from Open-Meteo (forecast or archive API)."""
    common_params = {
        "latitude": config["lat"],
        "longitude": config["lon"],
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "tilt": config["tilt"],
        "azimuth": config["azimuth"],
        "timezone": "UTC",
    }
    irradiance_vars = "global_tilted_irradiance,direct_radiation,diffuse_radiation,temperature_2m"

    if use_forecast:
        payload = _request_json(FORECAST_ENDPOINT, {**common_params, "minutely_15": irradiance_vars})
        if "minutely_15" not in payload:
            raise ValueError("Forecast response does not contain minutely_15 data")
        return _to_output_frame(pd.DataFrame(payload["minutely_15"]), day, config, source="forecast_api")
    else:
        payload = _request_json(ARCHIVE_ENDPOINT, {**common_params, "hourly": irradiance_vars})
        if "hourly" not in payload:
            raise ValueError("Historical response does not contain hourly data")
        return _to_output_frame(pd.DataFrame(payload["hourly"]), day, config, source="historical_api")


def _to_output_frame(frame: pd.DataFrame, day: date, config: Dict[str, float], source: str) -> pd.DataFrame:
    if "time" not in frame:
        raise ValueError("API response is missing the 'time' field")

    if "global_tilted_irradiance" not in frame:
        if {"direct_radiation", "diffuse_radiation"}.issubset(frame.columns):
            frame["global_tilted_irradiance"] = (
                pd.to_numeric(frame["direct_radiation"], errors="coerce")
                + pd.to_numeric(frame["diffuse_radiation"], errors="coerce")
            )
        else:
            raise ValueError("No usable irradiance variables in API response")

    normalized = _normalize_to_15_minute(frame, day)

    irradiance = normalized["global_tilted_irradiance"].clip(lower=0)
    base_kw = (irradiance / 1000.0) * config["kwp"] * config["yield_factor"]

    temperature = normalized["temperature_2m"]
    temp_factor = np.where(
        temperature > 25.0,
        1.0 + (temperature - 25.0) * TEMP_COEFFICIENT,
        1.0,
    )

    predicted_kw = np.maximum(0.0, base_kw * temp_factor)

    return pd.DataFrame({
        "time": normalized["time"],
        "predicted_kwh": np.round(predicted_kw * 0.25, 4),
        "predicted_kw": np.round(predicted_kw, 4),
        "temperature_2m": np.round(temperature, 2),
        "source": source,
    })


def _normalize_to_15_minute(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["time"]).sort_values("time")

    keep_columns = ["global_tilted_irradiance", "temperature_2m"]
    for column in keep_columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.set_index("time")
    target_index = pd.date_range(start=pd.Timestamp(day.isoformat(), tz="UTC"), periods=96, freq="15min")

    # Union the raw index (e.g. 24 hourly points) with the 96-slot target so that
    # interpolate("time") can fill across the full range before we reindex to 15-min.
    combined_index = frame.index.union(target_index)

    normalized = frame[keep_columns].reindex(combined_index).sort_index()
    normalized = normalized.interpolate(method="time", limit_direction="both").ffill().bfill()
    normalized = normalized.reindex(target_index)
    normalized.index.name = "time"

    return normalized.reset_index()


def _fallback_clear_sky(day: date, config: Dict[str, float], error: Exception) -> pd.DataFrame:
    """Estimate solar output using clear-sky geometry when the API is unavailable."""
    day_of_year = day.timetuple().tm_yday
    declination = 23.45 * np.sin(np.radians((360.0 / 365.0) * (day_of_year - 81)))

    times = pd.date_range(start=pd.Timestamp(day.isoformat(), tz="UTC"), periods=96, freq="15min")
    hour_angles = np.radians(15.0 * (times.hour + times.minute / 60.0 - 12.0))

    lat_rad = np.radians(config["lat"])
    dec_rad = np.radians(declination)
    cos_zenith = np.clip(
        np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angles),
        -1.0, 1.0,
    )
    zenith_rad = np.arccos(cos_zenith)

    cos_incidence = (
        np.cos(zenith_rad) * np.cos(np.radians(config["tilt"]))
        + np.sin(zenith_rad) * np.sin(np.radians(config["tilt"]))
        * np.cos(hour_angles - np.radians(config["azimuth"]))
    )
    irradiance = np.where(np.degrees(zenith_rad) > 90.0, 0.0, 1000.0 * np.maximum(0.0, cos_incidence))
    predicted_kw = config["kwp"] * config["yield_factor"] * (irradiance / 1000.0)

    return pd.DataFrame({
        "time": times,
        "predicted_kwh": np.round(predicted_kw * 0.25, 4),
        "predicted_kw": np.round(predicted_kw, 4),
        "temperature_2m": np.nan,
        "source": "fallback_model",
        "note": f"Fallback used because API request failed: {error}",
    })


__all__ = ["get_daily_solar_kwh", "load_solar_config"]
