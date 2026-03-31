"""
prices_utils.py — Rolling 48-hour electricity price utility with historical fallback.

Public API
----------
get_daily_prices(
    secrets_dir="secrets",  # directory containing entsoe_api_key.txt
    allow_fallback=True,     # when True, use historical 48h data to fill gaps
) -> pd.DataFrame            # 192 rows × 15-min UTC intervals starting from now

Output columns
--------------
- time           (datetime64, UTC, timezone-aware)
- price_eur_mwh  (float, never NaN)
- price_cent_kwh (float, never NaN)  # price_eur_mwh / 10
- source         (str)  "forecast" | "forecast_shifted" | "historical_shifted" | "fallback_unavailable"

Window semantics
----------------
The returned frame always represents a strict rolling UTC 48-hour window
starting from the current time rounded down to the nearest 15-minute slot.

Fallback mechanism (fill priority per slot)
-------------------------------------------
1. forecast          — direct API value for this slot
2. forecast_shifted  — slot filled from day 1 of the forecast (h0-h24 shifted to h24-h48);
                       only applied to day 2 gaps (day 1 is always covered by historical)
3. historical_shifted — aligned value from the 48h prior historical window
4. fallback_unavailable — no data available from any source
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COUNTRY_CODE = "DE_LU"   # Germany-Luxembourg bidding zone
DEFAULT_TZ = "Europe/Berlin"
PREFERRED_RESOLUTIONS = ("15T", "60T")
HORIZON_HOURS = 48
INTERVAL_MINUTES = 15
INTERVALS_48H = (HORIZON_HOURS * 60) // INTERVAL_MINUTES
INTERVALS_24H = (24 * 60) // INTERVAL_MINUTES
HISTORICAL_LOOKBACK_HOURS = 48  # Fetch prior 48h for fallback coverage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_48h_prices_forecast(
    secrets_dir: Union[str, Path] = "secrets",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Return a rolling UTC 48-hour window of 15-minute day-ahead prices.
    
    Fetches all forecast and historical data, then fills any NaN forecast values
    with aligned historical data, ensuring the output contains no NaN prices.

    Parameters
    ----------
    secrets_dir : str | Path
        Directory that contains ``entsoe_api_key.txt``.
    allow_fallback : bool
        When ``True`` (default), use historical data to fill gaps and return
        complete data. When ``False``, raise if any NaN would result.

    Returns
    -------
    pd.DataFrame
        192 rows, one per 15-minute interval in UTC.
        First row is ``now.floor('15min')`` and last row is ``first + 47h45m``.
        Columns: ``time``, ``price_eur_mwh``, ``price_cent_kwh``, ``source``, ``note``.
        - source: "forecast" (direct API), "forecast_shifted" (day 1 repeated into day 2),
                  "historical_shifted" (48h prior), or "fallback_unavailable".
        - note: Explanation for non-forecast entries.
    """
    start_utc = pd.Timestamp.now(tz="UTC").floor(f"{INTERVAL_MINUTES}min")
    end_utc = start_utc + pd.Timedelta(hours=HORIZON_HOURS)
    api_key = _load_api_key(secrets_dir)
    return _fetch_and_fill_window(
        start_utc,
        end_utc,
        api_key,
        DEFAULT_COUNTRY_CODE,
        DEFAULT_TZ,
        allow_fallback=allow_fallback,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_api_key(secrets_dir: Union[str, Path] = "secrets") -> str:
    """Load the ENTSOE API key from *secrets_dir*/entsoe_api_key.txt."""
    key_file_rel = Path(secrets_dir) / "entsoe_api_key.txt"

    if key_file_rel.is_absolute():
        if key_file_rel.exists():
            return key_file_rel.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"API key file not found at {key_file_rel}")

    candidates = [
        Path.cwd() / key_file_rel,
        Path(__file__).resolve().parents[1] / key_file_rel,
    ]
    found = next((c for c in candidates if c.exists()), None)
    if not found:
        raise FileNotFoundError(
            f"API key file not found. Checked: {', '.join(str(c) for c in candidates)}"
        )
    return found.read_text(encoding="utf-8").strip()


def _fetch_and_fill_window(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    api_key: str,
    country_code: str = DEFAULT_COUNTRY_CODE,
    tz: str = DEFAULT_TZ,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch forecast and historical data, then fill missing forecast values.
    
    Three-step process:
    1. Fetch ALL forecast prices (now to now+48h)
    2. Fetch ALL historical prices (now-48h to now)
    3. Fill NaN slots in priority order:
       a. forecast_shifted — day 1 values (h0-h24) shifted into day 2 (h24-h48) gaps
       b. historical_shifted — aligned 48h-prior historical values
    
    Returns 48h forecast table with source labels and no NaN prices.
    """
    client = EntsoePandasClient(api_key=api_key)
    target_index = _window_index(start_utc)

    # STEP 1: Fetch ALL forecast values (now to now+48h)
    try:
        raw_forecast = _query_prices_with_resolution_fallback(
            client,
            country_code,
            start_utc,
            end_utc,
        )
        # Remove duplicate timestamps (can occur during DST transitions)
        raw_forecast = raw_forecast.sort_index()
        raw_forecast = raw_forecast[~raw_forecast.index.duplicated(keep="first")]
        forecast_normalized = _normalize_to_15_minute(raw_forecast, target_index, tz)
    except Exception:
        if not allow_fallback:
            raise
        forecast_normalized = pd.Series(np.nan, index=target_index, dtype=float)

    # STEP 2: Fetch ALL historical values (now-48h to now)
    hist_start_utc = start_utc - pd.Timedelta(hours=HISTORICAL_LOOKBACK_HOURS)
    hist_target_index = _window_index(hist_start_utc)
    try:
        raw_historical = _query_prices_with_resolution_fallback(
            client,
            country_code,
            hist_start_utc,
            start_utc,
        )
        # Remove duplicate timestamps (can occur during DST transitions)
        raw_historical = raw_historical.sort_index()
        raw_historical = raw_historical[~raw_historical.index.duplicated(keep="first")]
        historical_normalized = _normalize_to_15_minute(raw_historical, hist_target_index, tz)
    except Exception:
        if not allow_fallback:
            raise
        historical_normalized = pd.Series(np.nan, index=hist_target_index, dtype=float)
    
    # STEP 3: Fill NaN slots in priority order
    forecast_mask = forecast_normalized.notna()  # Track original direct forecast values

    # 3a. forecast_shifted: shift day 1 (h0-h24) forward into day 2 (h24-h48) gaps.
    #     No backward shift: missing day 1 slots are always covered by historical data.
    forecast_day1_for_day2 = forecast_normalized.shift(INTERVALS_24H)
    after_forecast_fill = forecast_normalized.where(
        forecast_normalized.notna(), forecast_day1_for_day2
    )
    forecast_shifted_mask = (~forecast_mask) & after_forecast_fill.notna()

    # 3b. historical_shifted: fill remaining NaNs with aligned 48h-prior values.
    #     Use .values for positional alignment since indices are in different time ranges.
    resolved = after_forecast_fill.where(
        after_forecast_fill.notna(), historical_normalized.values
    )

    # Determine source labels
    source = np.where(
        forecast_mask,
        "forecast",
        np.where(
            forecast_shifted_mask,
            "forecast_shifted",
            np.where(
                resolved.notna(),
                "historical_shifted",
                "fallback_unavailable",
            ),
        ),
    )

    # Build result DataFrame for 48h forecast window
    result = pd.DataFrame({
        "time": target_index,
        "price_eur_mwh": np.round(resolved.values, 4),
        "price_cent_kwh": np.round(resolved.values / 10.0, 4),
        "source": source,
    })

    # Add explanatory notes
    result["note"] = np.where(
        result["source"] == "forecast_shifted",
        "Filled from day 1 of the forecast window (hours 0-24 shifted to hours 24-48).",
        np.where(
            result["source"] == "historical_shifted",
            "Filled from historical data (48 hours prior).",
            np.where(
                result["source"] == "fallback_unavailable",
                "No forecast or historical data available.",
                np.nan,
            ),
        ),
    )
    result.loc[result["note"].isna(), "note"] = np.nan
    
    return result


def _normalize_to_15_minute(
    raw_series: pd.Series,
    target_index: pd.DatetimeIndex,
    tz: str = DEFAULT_TZ,
) -> pd.Series:
    """
    Reindex a price Series (hourly or 15-min) to the exact rolling 48h
    target (192 slots) without cross-gap forward filling.

    Hourly API points are expanded to quarter-hour slots within the same hour.
    """
    series = raw_series.copy()
    if series.index.tz is None:
        series.index = series.index.tz_localize(tz)
    series.index = series.index.tz_convert("UTC")

    expanded = _expand_series_to_quarter_hour(series)
    normalized = expanded.reindex(target_index)
    if normalized.dropna().empty:
        raise NoMatchingDataError("No usable values after normalization to 15-minute grid")

    return normalized


def _expand_series_to_quarter_hour(series: pd.Series) -> pd.Series:
    """
    Expand hourly ENTSOE series into quarter-hour points for the same hour.

    If the source is already 15-minute data, values are preserved as-is.
    """
    if len(series) <= 1:
        return series

    diffs = series.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median_step = float(diffs.median()) if not diffs.empty else INTERVAL_MINUTES

    if median_step <= INTERVAL_MINUTES:
        return series

    frames = []
    for offset in (0, 15, 30, 45):
        shifted = series.copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=offset)
        frames.append(shifted)

    expanded = pd.concat(frames).sort_index()
    expanded = expanded[~expanded.index.duplicated(keep="first")]
    return expanded


def _query_prices_with_resolution_fallback(
    client: EntsoePandasClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Query day-ahead prices in a way that works across entsoe-py versions.

    - entsoe-py 0.6.x: resolution defaults to 60T; for many recent dates
      only 15T exists, so we try 15T first and then 60T.
    - entsoe-py 0.7.x: resolution argument is deprecated; if removed in a
      future release, we fall back to calling without it.
    """
    last_error: Optional[Exception] = None

    for resolution in PREFERRED_RESOLUTIONS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return client.query_day_ahead_prices(
                    country_code,
                    start=start,
                    end=end,
                    resolution=resolution,
                )
        except NoMatchingDataError as error:
            last_error = error
            continue
        except TypeError:
            # Future-proof: if resolution is removed from the API, use default call.
            break

    try:
        return client.query_day_ahead_prices(
            country_code,
            start=start,
            end=end,
        )
    except NoMatchingDataError as error:
        if last_error is not None:
            raise last_error from error
        raise


# ---------------------------------------------------------------------------
# Time index helper
# ---------------------------------------------------------------------------

def _window_index(start_utc: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(
        start=start_utc,
        periods=INTERVALS_48H,
        freq=f"{INTERVAL_MINUTES}min",
    )


# ---------------------------------------------------------------------------

get_daily_prices = get_48h_prices_forecast

__all__ = ["get_48h_prices_forecast", "get_daily_prices"]
