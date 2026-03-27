"""
Aggregated Input Table Builder

This module extracts and merges solar generation, energy prices, and household load forecasts
from multiple sources into a unified dispatch input table at UTC resolution.

The horizon length and interval are loaded from system_config.json:
- optimization_horizon_hours
- interval_minutes

The module implements:
- Smart caching: reuses data from the same day within the configured throttle window
- Fallback strategies: fills missing values from latest available day profiles
- Validation: enforces configured horizon row count and continuous timestamp spacing
- Persistence: saves results and refresh state for downstream dispatch workflows
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.load_utils import get_daily_load_forecast
from utils.prices_utils import get_daily_prices
from utils.solar_utils import get_daily_solar_kwh

# =============================================================================
# CONSTANTS
# =============================================================================

SYSTEM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "system_config.json"
_SYSTEM_CONFIG = json.loads(SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))

INTERVAL_MINUTES = int(_SYSTEM_CONFIG["interval_minutes"])
HOURS_PER_HORIZON = int(_SYSTEM_CONFIG["optimization_horizon_hours"])

if INTERVAL_MINUTES <= 0:
    raise ValueError("interval_minutes must be > 0 in system_config.json")
if (60 % INTERVAL_MINUTES) != 0:
    raise ValueError("interval_minutes must divide 60 in system_config.json")
if HOURS_PER_HORIZON <= 0:
    raise ValueError("optimization_horizon_hours must be > 0 in system_config.json")

INTERVALS_PER_HOUR = 60 // INTERVAL_MINUTES
INTERVALS_PER_DAY = (24 * 60) // INTERVAL_MINUTES
EXPECTED_ROWS_HORIZON = HOURS_PER_HORIZON * INTERVALS_PER_HOUR
DAYS_PER_HORIZON = math.ceil(HOURS_PER_HORIZON / 24)

# =============================================================================
# OUTPUT SCHEMA (EXACT)
# =============================================================================

OUTPUT_COLUMNS = [
    "utc_timestamp",              # datetime64[ns, UTC]  - configured interval UTC timestamps
    "pv_generation_kwh",          # float64 - Solar generation in kWh per interval
    "source_solar",               # object (str) - Source label (API, fallback, etc.)
    "energy_price_buy_cent_kwh",  # float64 - Grid buy price in cent/kWh
    "source_price",               # object (str) - Source label for price
    "household_load_kwh",         # float64 - Household load in kWh per interval
    "source_load",                # object (str) - Source label for load forecast
    "energy_price_sell_cent_kwh", # float64 - Grid sell price in cent/kWh (from system config)
    "source_sell_price",          # object (str) - Source label for sell price
]

# =============================================================================
# PRIVATE HELPERS - Date & Time Utilities
# =============================================================================


def _next_day(day_obj) -> Any:
    """Return the next calendar day at midnight UTC."""
    return (pd.Timestamp(day_obj, tz="UTC") + pd.Timedelta(days=1)).date()


def _day_index(day_obj) -> pd.DatetimeIndex:
    """Generate one day of timestamps at configured interval in UTC."""
    return pd.date_range(
        start=pd.Timestamp(day_obj, tz="UTC"),
        periods=INTERVALS_PER_DAY,
        freq=f"{INTERVAL_MINUTES}min",
    )


# =============================================================================
# PRIVATE HELPERS - Solar Data Fetch
# =============================================================================


def _fetch_horizon_solar(start_day) -> pd.DataFrame:
    """
    Fetch horizon solar data by concatenating configured number of days.

    Returns:
        DataFrame with columns: [time, ...]
    """
    days = [
        (pd.Timestamp(start_day, tz="UTC") + pd.Timedelta(days=offset)).date()
        for offset in range(DAYS_PER_HORIZON)
    ]
    frames = [get_daily_solar_kwh(day) for day in days]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# PRIVATE HELPERS - Price Data Fetch & Fallback
# =============================================================================


def _load_cached_latest_price_profile(
    output_path: Path,
) -> tuple[Optional[pd.DataFrame], Optional[Any]]:
    """
    Extract the latest available daily price profile from cached output CSV.

    Returns:
        (profile_df, day) where profile_df has columns [price_eur_mwh, price_cent_kwh]
        or (None, None) if cache is invalid or unavailable.
    """
    if not output_path.exists():
        return None, None

    try:
        cached = pd.read_csv(output_path)
    except Exception:
        return None, None

    required_cols = {"utc_timestamp", "energy_price_buy_cent_kwh"}
    if not required_cols.issubset(cached.columns):
        return None, None

    cached["utc_timestamp"] = pd.to_datetime(cached["utc_timestamp"], utc=True, errors="coerce")
    cached["energy_price_buy_cent_kwh"] = pd.to_numeric(
        cached["energy_price_buy_cent_kwh"], errors="coerce"
    )
    cached = (
        cached.dropna(subset=["utc_timestamp", "energy_price_buy_cent_kwh"])
        .sort_values("utc_timestamp")
    )

    if cached.empty:
        return None, None

    cached["day"] = cached["utc_timestamp"].dt.floor("D")
    latest_day = cached["day"].max()
    latest_day_df = (
        cached[cached["day"] == latest_day].sort_values("utc_timestamp").reset_index(drop=True)
    )

    if len(latest_day_df) < INTERVALS_PER_DAY:
        return None, None

    latest_day_df = latest_day_df.iloc[:INTERVALS_PER_DAY].copy()
    profile = pd.DataFrame(
        {
            "price_cent_kwh": latest_day_df["energy_price_buy_cent_kwh"].values,
        }
    )
    profile["price_eur_mwh"] = profile["price_cent_kwh"] * 10.0
    profile = profile[["price_eur_mwh", "price_cent_kwh"]].ffill().bfill()

    if profile["price_cent_kwh"].isna().all():
        return None, None

    return profile.reset_index(drop=True), latest_day.date()


def _find_latest_price_profile(
    start_day, output_path: Path, max_lookback_days: int = 3
) -> tuple[pd.DataFrame, Any]:
    """
    Find the latest available daily price profile (cached or from API).

    Search order:
    1. Cached profile from output_path (if same day as last build)
    2. API calls for up to max_lookback_days prior

    Returns:
        (profile_df, source_label) where profile_df has columns [price_eur_mwh, price_cent_kwh]
        and source_label indicates origin ("YYYY-MM-DD", etc.)

    Raises:
        ValueError: if no valid price profile is available from cache or API sources.
    """
    cached_profile, cached_day = _load_cached_latest_price_profile(output_path)
    if cached_profile is not None:
        return cached_profile, cached_day

    last_error: Optional[Exception] = None
    for offset in range(max_lookback_days + 1):
        candidate_day = (pd.Timestamp(start_day, tz="UTC") - pd.Timedelta(days=offset)).date()

        try:
            candidate = get_daily_prices(candidate_day).copy()
        except Exception as exc:
            last_error = exc
            continue

        candidate["time"] = pd.to_datetime(candidate["time"], utc=True, errors="coerce")
        candidate["price_cent_kwh"] = pd.to_numeric(candidate["price_cent_kwh"], errors="coerce")
        candidate["price_eur_mwh"] = pd.to_numeric(candidate["price_eur_mwh"], errors="coerce")
        candidate = candidate.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        if len(candidate) == INTERVALS_PER_DAY and candidate["price_cent_kwh"].notna().any():
            profile = (
                candidate[["price_eur_mwh", "price_cent_kwh"]]
                .ffill()
                .bfill()
                .reset_index(drop=True)
            )
            return profile, candidate_day

    if last_error is not None:
        raise ValueError(
            "No valid price profile available in cache or price sources for the requested horizon. "
            f"Last source error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    raise ValueError(
        "No valid price profile available in cache or price sources for the requested horizon."
    )


def _build_price_day_with_fallback(
    target_day, fallback_profile: pd.DataFrame, fallback_day: Any
) -> pd.DataFrame:
    """
    Build a single day of price data, filling missing values from fallback profile.

    Returns:
        DataFrame with columns [time, price_eur_mwh, price_cent_kwh, source]
    """
    try:
        day_prices = get_daily_prices(target_day).copy()
    except Exception:
        day_prices = pd.DataFrame({"time": _day_index(target_day)})

    for col in ["time", "price_eur_mwh", "price_cent_kwh", "source"]:
        if col not in day_prices.columns:
            day_prices[col] = np.nan

    day_prices["time"] = pd.to_datetime(day_prices["time"], utc=True, errors="coerce")
    day_prices["price_eur_mwh"] = pd.to_numeric(day_prices["price_eur_mwh"], errors="coerce")
    day_prices["price_cent_kwh"] = pd.to_numeric(day_prices["price_cent_kwh"], errors="coerce")

    base = pd.DataFrame({"time": _day_index(target_day)})
    day_prices = (
        base.merge(
            day_prices[["time", "price_eur_mwh", "price_cent_kwh", "source"]],
            on="time",
            how="left",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    missing_mask = day_prices["price_cent_kwh"].isna()
    if missing_mask.any():
        day_prices.loc[missing_mask, "price_eur_mwh"] = fallback_profile.loc[
            missing_mask, "price_eur_mwh"
        ].values
        day_prices.loc[missing_mask, "price_cent_kwh"] = fallback_profile.loc[
            missing_mask, "price_cent_kwh"
        ].values
        day_prices.loc[missing_mask, "source"] = f"fallback_latest_day_profile_{fallback_day}"

    day_prices["source"] = day_prices["source"].fillna(
        f"fallback_latest_day_profile_{fallback_day}"
    )
    return day_prices


def _fetch_horizon_prices(start_day, output_path: Path) -> pd.DataFrame:
    """
    Fetch horizon price data by concatenating configured number of days with fallback handling.

    Returns:
        DataFrame with columns [time, price_eur_mwh, price_cent_kwh, source]
    """
    fallback_profile, fallback_day = _find_latest_price_profile(start_day, output_path)

    days = [
        (pd.Timestamp(start_day, tz="UTC") + pd.Timedelta(days=offset)).date()
        for offset in range(DAYS_PER_HORIZON)
    ]
    frames = [
        _build_price_day_with_fallback(
            day, fallback_profile=fallback_profile, fallback_day=fallback_day
        )
        for day in days
    ]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# PRIVATE HELPERS - Load Data Fetch & Fallback
# =============================================================================


def _load_cached_latest_load_profile(
    output_path: Path,
) -> tuple[Optional[pd.DataFrame], Optional[Any], Optional[str]]:
    """
    Extract the latest available daily load profile from cached output CSV.

    Returns:
        (profile_df, day, household_name) where profile_df has columns [predicted_kwh, predicted_kw]
        or (None, None, None) if cache is invalid or unavailable.
    """
    if not output_path.exists():
        return None, None, None

    try:
        cached = pd.read_csv(output_path)
    except Exception:
        return None, None, None

    required_cols = {"utc_timestamp", "household_load_kwh"}
    if not required_cols.issubset(cached.columns):
        return None, None, None

    cached["utc_timestamp"] = pd.to_datetime(cached["utc_timestamp"], utc=True, errors="coerce")
    cached["household_load_kwh"] = pd.to_numeric(cached["household_load_kwh"], errors="coerce")
    cached = (
        cached.dropna(subset=["utc_timestamp", "household_load_kwh"])
        .sort_values("utc_timestamp")
    )

    if cached.empty:
        return None, None, None

    cached["day"] = cached["utc_timestamp"].dt.floor("D")
    latest_day = cached["day"].max()
    latest_day_df = (
        cached[cached["day"] == latest_day].sort_values("utc_timestamp").reset_index(drop=True)
    )

    if len(latest_day_df) < INTERVALS_PER_DAY:
        return None, None, None

    latest_day_df = latest_day_df.iloc[:INTERVALS_PER_DAY].copy()
    profile = pd.DataFrame(
        {
            "predicted_kwh": latest_day_df["household_load_kwh"].values,
        }
    )
    profile["predicted_kw"] = profile["predicted_kwh"] * 4.0
    profile = profile[["predicted_kwh", "predicted_kw"]].ffill().bfill()

    if profile["predicted_kwh"].isna().all():
        return None, None, None

    return profile.reset_index(drop=True), latest_day.date(), None


def _find_latest_load_profile(
    start_day, output_path: Path, max_lookback_days: int = 7
) -> tuple[pd.DataFrame, Any, Optional[str]]:
    """
    Find the latest available daily load profile (cached or from API).

    Search order:
    1. Cached profile from output_path (if available)
    2. API calls for up to max_lookback_days prior

    Returns:
        (profile_df, source_label, household_name) where profile_df has columns
        [predicted_kwh, predicted_kw]

    Raises:
        ValueError: if no valid load profile is available from cache or forecast sources.
    """
    cached_profile, cached_day, cached_household = _load_cached_latest_load_profile(output_path)
    if cached_profile is not None:
        return cached_profile, cached_day, cached_household

    last_error: Optional[Exception] = None
    for offset in range(max_lookback_days + 1):
        candidate_day = (pd.Timestamp(start_day, tz="UTC") - pd.Timedelta(days=offset)).date()
        try:
            candidate = get_daily_load_forecast(candidate_day).copy()
        except Exception as exc:
            last_error = exc
            continue

        if "predicted_kwh" not in candidate.columns:
            continue

        candidate["time"] = pd.to_datetime(candidate["time"], utc=True, errors="coerce")
        candidate["predicted_kwh"] = pd.to_numeric(candidate["predicted_kwh"], errors="coerce")
        candidate = candidate.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        if len(candidate) == INTERVALS_PER_DAY and candidate["predicted_kwh"].notna().any():
            profile = pd.DataFrame(
                {
                    "predicted_kwh": candidate["predicted_kwh"].values,
                    "predicted_kw": pd.to_numeric(
                        candidate.get("predicted_kw", candidate["predicted_kwh"] * 4.0),
                        errors="coerce",
                    ),
                }
            ).ffill().bfill()

            return profile.reset_index(drop=True), candidate_day, None

    if last_error is not None:
        raise ValueError(
            "No valid load profile available in cache or forecast sources for the requested horizon. "
            f"Last source error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    raise ValueError(
        "No valid load profile available in cache or forecast sources for the requested horizon."
    )


def _build_load_day_with_fallback(
    target_day, fallback_profile: pd.DataFrame, fallback_day: Any
) -> pd.DataFrame:
    """
    Build a single day of load data, filling missing values from fallback profile.

    Returns:
        DataFrame with columns [time, predicted_kwh, predicted_kw, source]
    """
    try:
        day_load = get_daily_load_forecast(target_day).copy()
    except Exception:
        day_load = pd.DataFrame({"time": _day_index(target_day)})

    for col in ["time", "predicted_kwh", "predicted_kw", "source"]:
        if col not in day_load.columns:
            day_load[col] = np.nan

    day_load["time"] = pd.to_datetime(day_load["time"], utc=True, errors="coerce")
    day_load["predicted_kwh"] = pd.to_numeric(day_load["predicted_kwh"], errors="coerce")
    day_load["predicted_kw"] = pd.to_numeric(day_load["predicted_kw"], errors="coerce")

    base = pd.DataFrame({"time": _day_index(target_day)})
    day_load = (
        base.merge(
            day_load[["time", "predicted_kwh", "predicted_kw", "source"]],
            on="time",
            how="left",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    missing_mask = day_load["predicted_kwh"].isna()
    if missing_mask.any():
        day_load.loc[missing_mask, "predicted_kwh"] = fallback_profile.loc[
            missing_mask, "predicted_kwh"
        ].values
        day_load.loc[missing_mask, "predicted_kw"] = fallback_profile.loc[
            missing_mask, "predicted_kw"
        ].values
        day_load.loc[missing_mask, "source"] = f"fallback_latest_day_profile_{fallback_day}"

    day_load["predicted_kw"] = day_load["predicted_kw"].fillna(day_load["predicted_kwh"] * 4.0)
    day_load["source"] = day_load["source"].fillna(
        f"fallback_latest_day_profile_{fallback_day}"
    )

    return day_load


def _fetch_horizon_load(start_day, output_path: Path) -> pd.DataFrame:
    """
    Fetch horizon load data by concatenating configured number of days with fallback handling.

    Returns:
        DataFrame with columns [time, predicted_kwh, predicted_kw, source]
    """
    fallback_profile, fallback_day, _ = _find_latest_load_profile(start_day, output_path)

    days = [
        (pd.Timestamp(start_day, tz="UTC") + pd.Timedelta(days=offset)).date()
        for offset in range(DAYS_PER_HORIZON)
    ]
    frames = [
        _build_load_day_with_fallback(
            day, fallback_profile=fallback_profile, fallback_day=fallback_day
        )
        for day in days
    ]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# PRIVATE HELPERS - Merge & Validation
# =============================================================================


def _build_horizon_df(start_day, output_path: Path) -> pd.DataFrame:
    """
    Merge horizon solar, price, and load data into a single DataFrame.

    Returns:
        Merged DataFrame with columns from solar, prices, and load.

    Raises:
        ValueError: if result row count ≠ 192.
    """
    solar_df = _fetch_horizon_solar(start_day)
    prices_df = _fetch_horizon_prices(start_day, output_path)
    load_df = _fetch_horizon_load(start_day, output_path)

    merged = pd.merge(solar_df, prices_df, on="time", how="inner")
    merged = pd.merge(merged, load_df, on="time", how="inner")

    merged["time"] = pd.to_datetime(merged["time"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    merged = merged.iloc[:EXPECTED_ROWS_HORIZON].copy()

    if len(merged) != EXPECTED_ROWS_HORIZON:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_HORIZON} rows ({HOURS_PER_HORIZON}h at {INTERVAL_MINUTES}-min), got {len(merged)}"
        )

    return merged


def _validate_continuous_timestamps(df: pd.DataFrame) -> None:
    """
    Verify the DataFrame has configured horizon rows with continuous UTC timestamps.

    Raises:
        ValueError: if shape or timestamp spacing is invalid.
    """
    if len(df) != EXPECTED_ROWS_HORIZON:
        raise ValueError(
            f"Expected {EXPECTED_ROWS_HORIZON} rows ({HOURS_PER_HORIZON}h at {INTERVAL_MINUTES}-min), got {len(df)}"
        )

    step_ok = df["time"].diff().dropna().eq(pd.Timedelta(minutes=INTERVAL_MINUTES)).all()
    if not step_ok:
        raise ValueError(
            f"Time index is not a continuous {INTERVAL_MINUTES}-minute UTC series."
        )


# =============================================================================
# PRIVATE HELPERS - Cache & State Management
# =============================================================================


def _should_refresh(
    start_day,
    now_utc: pd.Timestamp,
    output_path: Path,
    update_interval: pd.Timedelta,
) -> tuple[bool, str]:
    """
    Decide whether to refresh the aggregated table or reuse cached version.

    Returns:
        (should_refresh: bool, reason: str)

    Refresh occurs if:
    - output_path does not exist
    """
    if not output_path.exists():
        return True, "aggregated_table.csv not found"

    return True, "always refresh (no state file caching)"


def _reuse_cached_table(output_path: Path, start_day) -> tuple[Optional[pd.DataFrame], bool]:
    """
    Attempt to load and validate cached aggregated table.

    Returns:
        (df, success: bool) - df is the loaded DataFrame, or None if load failed.
    """
    try:
        cached_df = pd.read_csv(output_path)
    except Exception:
        return None, False

    # Required columns (household is no longer included)
    cached_required_cols = {
        "utc_timestamp",
        "pv_generation_kwh",
        "source_solar",
        "energy_price_buy_cent_kwh",
        "source_price",
        "household_load_kwh",
        "source_load",
    }

    if not cached_required_cols.issubset(cached_df.columns):
        return None, False

    # Rename to internal format (predicted_kwh_x, predicted_kwh_y, etc.)
    df = cached_df.rename(
        columns={
            "utc_timestamp": "time",
            "pv_generation_kwh": "predicted_kwh_x",
            "source_solar": "source_x",
            "energy_price_buy_cent_kwh": "price_cent_kwh",
            "source_price": "source_y",
            "household_load_kwh": "predicted_kwh_y",
            "source_load": "source",
        }
    )[["time", "predicted_kwh_x", "source_x", "price_cent_kwh", "source_y", "predicted_kwh_y", "source"]]

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    return df, True


# =============================================================================
# PUBLIC API - Load System Config
# =============================================================================


def _load_system_config(
    config_path: Path = SYSTEM_CONFIG_PATH,
) -> float:
    """
    Load default sell price value from system_config.json.

    Returns:
        default_sell_price_cent_kwh

    Raises:
        FileNotFoundError: if config_path does not exist
        KeyError: if default_sell_price_cent_kwh is missing from config
    """
    if not config_path.exists():
        raise FileNotFoundError(f"system_config.json not found at {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    if "default_sell_price_cent_kwh" not in config:
        raise KeyError("default_sell_price_cent_kwh is required in system_config.json")

    return float(config["default_sell_price_cent_kwh"])


# =============================================================================
# PUBLIC API - Main Orchestrator
# =============================================================================


def build_aggregated_table(
    start_day: "str | Any",
    output_path: Path = Path(__file__).resolve().parents[1] / "data/runtime/aggregated_table.csv",
    update_interval: pd.Timedelta = pd.Timedelta(minutes=15),
    defaults: Optional[dict[str, float]] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Build or reuse a config-driven aggregated input table for battery dispatch.

    **Inputs:**
        - start_day (str or date): Start date for the configured horizon window (e.g., "2026-03-20" or date object).
            UTC timezone is assumed.
    - output_path (Path): Where to write/read the cached aggregated table CSV.
        - update_interval (pd.Timedelta): Minimum time between refreshes for same day.
            Default: 15 minutes. Prevents excessive API calls.
        - defaults (dict, optional): Override sell price loaded from system_config.json.
            Keys: "sell_price_cent_kwh".
    - save_output (bool): Whether to persist output to disk. Default True.

    **Processing Steps:**
    1. Load system_config.json for default sell price.
    2. Apply any overrides from defaults parameter.
    3. Determine if cached version can be reused (same day, within update_interval).
    4. If reuse possible: validate and return cached table.
    5. If refresh needed: fetch solar, prices, load from APIs → merge → validate.
    6. Rename columns to final schema and add sell price from config.
    7. (Optional) Save to CSV and sync state file.

    **Output Schema (configured horizon rows × 9 columns):**
    ```
    utc_timestamp              datetime64[ns, UTC] - configured interval UTC timestamps
    pv_generation_kwh         float64              - Solar generation (kWh)
    source_solar              object (str)         - Data source label
    energy_price_buy_cent_kwh float64              - Grid buy price (cent/kWh)
    source_price              object (str)         - Price source label
    household_load_kwh        float64              - Household load (kWh)
    source_load               object (str)         - Load forecast source label
    energy_price_sell_cent_kwh float64             - Grid sell price (cent/kWh) [from config]
    source_sell_price         object (str)         - Sell price source (always "config_default")
    ```

    **Return Value:**
    pandas.DataFrame with the shape (192, 9) and schema above.

    **Example:**
    ```python
    from utils.aggregated_utils import build_aggregated_table
    from pathlib import Path
    import pandas as pd

    # Simple one-liner usage
    df = build_aggregated_table("2026-03-20")

    # With custom paths and defaults
    df = build_aggregated_table(
        "2026-03-20",
        output_path=Path("data/custom_aggregated.csv"),
        update_interval=pd.Timedelta(minutes=30),
        defaults={"sell_price_cent_kwh": 5.0},
        save_output=True,
    )
    print(df.shape)  # (192, 9)
    print(df.columns.tolist())  # exact output schema
    ```

    **Raises:**
    ValueError: if no price profile is available, if no load profile is available,
    if final table shape is not 192 rows, or timestamps are not continuous.
    """
    # Load sell-price default from system_config.json
    sell_price_cfg = _load_system_config()

    # Apply any parameter overrides
    default_sell_price_cent_kwh = sell_price_cfg

    if defaults:
        default_sell_price_cent_kwh = defaults.get(
            "sell_price_cent_kwh", default_sell_price_cent_kwh
        )

    # Normalize start_day to date object
    start_day = pd.Timestamp(start_day, tz="UTC").date()
    now_utc = pd.Timestamp.now(tz="UTC")

    # Step 1: Check refresh policy
    should_refresh, refresh_reason = _should_refresh(
        start_day, now_utc, output_path, update_interval
    )

    if should_refresh:
        # Step 2a: Build fresh horizon table
        df = _build_horizon_df(start_day, output_path)
        refresh_occurred = True
        print(f"Refreshed horizon table for {start_day} ({refresh_reason}).")
    else:
        # Step 2b: Attempt to reuse cached table
        df, cache_valid = _reuse_cached_table(output_path, start_day)
        if not cache_valid:
            # Cache invalid → rebuild
            df = _build_horizon_df(start_day, output_path)
            refresh_occurred = True
            print("Cached table schema is incompatible; rebuilt a fresh horizon table.")
        else:
            refresh_occurred = False
            print(f"Skipped refresh ({refresh_reason}). Reusing cached horizon table from {output_path}.")

    # Step 3: Validate timestamps
    _validate_continuous_timestamps(df)

    # Step 4: Rename to final output schema
    df_final = df[
        ["time", "predicted_kwh_x", "source_x", "price_cent_kwh", "source_y", "predicted_kwh_y", "source"]
    ].copy()
    df_final.columns = [
        "utc_timestamp",
        "pv_generation_kwh",
        "source_solar",
        "energy_price_buy_cent_kwh",
        "source_price",
        "household_load_kwh",
        "source_load",
    ]

    # Step 5: Add sell price column from config
    df_final["energy_price_sell_cent_kwh"] = default_sell_price_cent_kwh
    df_final["source_sell_price"] = "config_default"

    # Reorder columns to match OUTPUT_COLUMNS
    df_final = df_final[OUTPUT_COLUMNS]

    # Step 6: Persist if refresh occurred
    if save_output and refresh_occurred:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)

        print(f"Saved refreshed horizon table to: {output_path}")
    else:
        print("No file update: using cached or skipped persistence.")
        print(f"Table location: {output_path}")

    return df_final

# Example usage in main.py:
# df = build_aggregated_table(start_day='2026-03-20')