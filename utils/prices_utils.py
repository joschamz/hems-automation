"""
prices_utils.py — Day-ahead electricity price utility, modelled after solar_utils.py.

Public API
----------
get_daily_prices(
    target_date=None,       # None → tomorrow; str "YYYY-MM-DD"; date; datetime
    mode="auto",            # "auto" | "forecast" | "historical"
    secrets_dir="secrets",  # directory containing entsoe_api_key.txt
    allow_fallback=True,    # when True, return NaN frame instead of raising on API error
) -> pd.DataFrame           # 96 rows × 15-min UTC intervals for the requested day

Output columns
--------------
- time           (datetime64, UTC, timezone-aware)   — matches solar_utils.py's `time` column
- price_eur_mwh  (float)                             — NaN when source != "entsoe_api"
- price_cent_kwh (float)                             — price_eur_mwh / 10
- source         (str)  "entsoe_api" | "not_published" | "fallback_unavailable"

Historical prices are supported — ENTSOE stores day-ahead prices indefinitely.
The same `query_day_ahead_prices` endpoint handles both past and future dates.

Resolution note
---------------
The returned frame always represents a strict UTC calendar day
(00:00:00–23:45:00 UTC, 96 rows). Internally, ENTSOE data is market-time
based, but this utility normalizes and slices to the requested UTC day.

Resolution depends on market/date and ENTSOE data publication. With SDAC
quarter-MTU go-live, many recent dates (including DE_LU) are available in
15-minute granularity, while older dates can be hourly. This utility is
version-compatible with entsoe-py 0.6.x and 0.7.x by trying 15-minute data
first and falling back to hourly when needed.
"""

import warnings
from datetime import date, datetime
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
# Compatibility for mixed ENTSOE resolutions and entsoe-py versions:
# - 0.6.x defaults to 60T unless explicitly passed and can miss post go-live data
# - 0.7.x deprecates resolution and auto-forces the correct SDAC resolution
PREFERRED_RESOLUTIONS = ("15T", "60T")

DateLike = Union[str, date, datetime]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_daily_prices(
    target_date: Optional[DateLike] = None,
    mode: str = "auto",
    secrets_dir: Union[str, Path] = "secrets",
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Return one UTC calendar day (24 h) of 15-minute day-ahead electricity prices.

    Parameters
    ----------
    target_date : None | str "YYYY-MM-DD" | date | datetime
        UTC day to fetch. ``None`` defaults to *tomorrow in UTC*.
    mode : {"auto", "forecast", "historical"}
        ``"auto"`` picks the correct mode by comparing *target_date* to today.
        ``"forecast"`` is valid only for today or future dates.
        ``"historical"`` is valid only for past dates.
    secrets_dir : str | Path
        Directory that contains ``entsoe_api_key.txt``.
        Resolved with the same two-candidate logic as solar_utils.py's config path.
    allow_fallback : bool
        When ``True`` (default), return a 96-row NaN frame instead of raising
        if the API call fails or the data has not been published yet.

    Returns
    -------
    pd.DataFrame
        96 rows, one per 15-minute interval in UTC for the requested UTC day.
        First row is always ``00:00:00+00:00``, last row ``23:45:00+00:00``.
        Columns: ``time``, ``price_eur_mwh``, ``price_cent_kwh``, ``source``.
        A ``note`` column is appended when the source is not ``"entsoe_api"``.
    """
    day = _parse_target_date(target_date)
    is_forecast = _resolve_mode(day, mode) == "forecast"
    api_key = _load_api_key(secrets_dir)

    try:
        return _fetch_day(day, api_key, DEFAULT_COUNTRY_CODE, DEFAULT_TZ)
    except NoMatchingDataError as exc:
        # "not_published" only makes sense for future dates whose D+1 auction
        # has not run yet.  For historical dates NoMatchingDataError signals a
        # genuine data gap — surface the real reason via fallback_frame.
        if is_forecast:
            warnings.warn(
                f"Day-ahead prices for {day} have not been published yet. "
                "ENTSOE typically publishes D+1 prices around 13:00 CET.",
                UserWarning,
                stacklevel=2,
            )
            return _not_published_frame(day)
        if not allow_fallback:
            raise
        return _fallback_frame(day, exc)
    except Exception as error:
        if not allow_fallback:
            raise
        return _fallback_frame(day, error)


# ---------------------------------------------------------------------------
# Internal helpers — path / config resolution (mirrored from solar_utils.py)
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


def _parse_target_date(target_date: Optional[DateLike]) -> date:
    if target_date is None:
        return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).date()
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
    today = pd.Timestamp.now(tz="UTC").date()
    if normalized == "auto":
        return "historical" if day < today else "forecast"
    if normalized == "forecast" and day < today:
        raise ValueError("Forecast mode cannot be used for past dates")
    if normalized == "historical" and day >= today:
        raise ValueError("Historical mode can only be used for past dates")
    return normalized


# ---------------------------------------------------------------------------
# Internal helpers — data fetching & normalisation
# ---------------------------------------------------------------------------

def _fetch_day(
    day: date,
    api_key: str,
    country_code: str = DEFAULT_COUNTRY_CODE,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """
    Fetch one UTC calendar day of day-ahead prices and return a 96-row UTC
    DataFrame. Works for both past (historical) and future (forecast) dates
    since ENTSOE uses the same endpoint for both.
    """
    client = EntsoePandasClient(api_key=api_key)

    # Query a strict UTC day window and then normalize to 96 UTC 15-min slots.
    start_utc = pd.Timestamp(day.isoformat(), tz="UTC")
    end_utc = start_utc + pd.Timedelta(days=1)

    raw_prices = _query_prices_with_resolution_fallback(
        client,
        country_code,
        start_utc,
        end_utc,
    )

    # Keep one value per timestamp before timezone conversion.
    raw_prices = raw_prices.sort_index()
    raw_prices = raw_prices[~raw_prices.index.duplicated(keep="first")]

    # ENTSOE responses are usually timezone-aware. If not, assume market timezone.
    if raw_prices.index.tz is None:
        raw_prices.index = raw_prices.index.tz_localize(tz)

    # Normalize to UTC and slice strictly to the requested UTC day window.
    raw_utc = raw_prices.tz_convert("UTC")
    day_prices = raw_utc[(raw_utc.index >= start_utc) & (raw_utc.index < end_utc)]

    if day_prices.empty:
        raise NoMatchingDataError(
            f"query_day_ahead_prices returned no data for UTC day {day} after filtering to {start_utc} – {end_utc}"
        )

    normalized = _normalize_to_15_minute(day_prices, start_utc, tz)

    return pd.DataFrame({
        "time": normalized.index,
        "price_eur_mwh": np.round(normalized.values, 4),
        "price_cent_kwh": np.round(normalized.values / 10.0, 4),
        "source": "entsoe_api",
    })


def _normalize_to_15_minute(
    raw_series: pd.Series,
    start_utc: pd.Timestamp,
    tz: str = DEFAULT_TZ,
) -> pd.Series:
    """
    Reindex a price Series (hourly or already 15-min) to exactly one UTC-day
    target (96 slots) using forward-fill — a published price is valid for the
    whole hour (or quarter) it covers.
    """
    series = raw_series.copy()
    # Ensure UTC
    if series.index.tz is None:
        series.index = series.index.tz_localize(tz)
    series.index = series.index.tz_convert("UTC")

    target_index = pd.date_range(
        start=start_utc,
        periods=96,
        freq="15min",
    )

    # Keep observed points and target slots on one timeline so forward-fill can
    # propagate each published value across the intervals it is valid for.
    combined_index = series.index.union(target_index)
    filled = series.reindex(combined_index).sort_index().ffill().bfill()

    return filled.reindex(target_index)


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
# Fallback frames (no API data available)
# ---------------------------------------------------------------------------

def _utc_index(day: date) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(day.isoformat(), tz="UTC"),
        periods=96,
        freq="15min",
    )


def _nan_price_frame(day: date, source: str, note: str) -> pd.DataFrame:
    """Build a 96-row UTC NaN-price frame used by fallback paths."""
    return pd.DataFrame({
        "time": _utc_index(day),
        "price_eur_mwh": np.nan,
        "price_cent_kwh": np.nan,
        "source": source,
        "note": note,
    })


def _not_published_frame(day: date) -> pd.DataFrame:
    """
    Return a 96-row NaN frame for the requested UTC day when day-ahead prices
    are not published yet. ENTSOE typically publishes D+1 prices around 13:00 CET.
    """
    return _nan_price_frame(
        day,
        source="not_published",
        note="Day-ahead prices for this day have not been published yet.",
    )


def _fallback_frame(day: date, error: Exception) -> pd.DataFrame:
    """Return a 96-row NaN frame when the ENTSOE API failed or returned no data."""
    return _nan_price_frame(
        day,
        source="fallback_unavailable",
        note=f"API call failed: {type(error).__name__}: {error}",
    )


# ---------------------------------------------------------------------------

__all__ = ["get_daily_prices"]
