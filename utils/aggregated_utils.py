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
from utils.prices_utils import get_48h_prices_forecast
from utils.solar_utils import get_daily_solar_kwh

# =============================================================================
# CONSTANTS
# =============================================================================

SYSTEM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "system_config.json"
_SYSTEM_CONFIG = json.loads(SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))
history_path = Path(__file__).resolve().parents[1] / "data/history/history_aggregated_table.csv"

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

# Refactored for rolling 48h horizon aligned to last completed interval

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.load_utils import get_daily_load_forecast
from utils.prices_utils import get_daily_prices
from utils.solar_utils import get_daily_solar_kwh

SYSTEM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "system_config.json"
_SYSTEM_CONFIG = json.loads(SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))

INTERVAL_MINUTES = int(_SYSTEM_CONFIG["interval_minutes"])
HOURS_PER_HORIZON = int(_SYSTEM_CONFIG["optimization_horizon_hours"])

INTERVALS_PER_HOUR = 60 // INTERVAL_MINUTES
EXPECTED_ROWS_HORIZON = HOURS_PER_HORIZON * INTERVALS_PER_HOUR

OUTPUT_COLUMNS = [
    "utc_timestamp",
    "pv_generation_kwh",
    "source_solar",
    "energy_price_buy_cent_kwh",
    "source_price",
    "household_load_kwh",
    "source_load",
    "energy_price_sell_cent_kwh",
    "source_sell_price",
]

def _fetch_horizon_load(start_time, end_time, output_path: Path):
    from utils.load_utils import (
        get_daily_load_forecast,
        load_feature_engineered_dataset,
    )

    df_all = load_feature_engineered_dataset("data/load_training_dataset.csv")
    max_available_time = df_all.index.max()

    forecast_time = min(start_time, max_available_time)

    df = get_daily_load_forecast(forecast_time)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    full_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{INTERVAL_MINUTES}min",
        tz="UTC",
    )

    df = (
        df.set_index("time")
        .reindex(full_index)
        .rename_axis("time")
        .reset_index()
    )

    return df


#def _day_index(day_obj) -> pd.DatetimeIndex:
#    return pd.date_range(
#        start=pd.Timestamp(day_obj, tz="UTC"),
#        periods=(24 * 60) // INTERVAL_MINUTES,
#        freq=f"{INTERVAL_MINUTES}min",
#    )


def _fetch_days_range(start_time, end_time):
    required_days = math.ceil((end_time - start_time).total_seconds() / 86400) + 1

    return [
        (start_time + pd.Timedelta(days=i)).date()
        for i in range(required_days)
    ]


def _fetch_horizon_solar(start_time, end_time):
    days = _fetch_days_range(start_time, end_time)
    frames = [get_daily_solar_kwh(day) for day in days]
    return pd.concat(frames, ignore_index=True)


# def _fetch_horizon_prices(start_time, end_time, output_path: Path):
#     df = get_48h_prices_forecast()

#     df["time"] = pd.to_datetime(df["time"], utc=True)

#     # slice to rolling window
#     df = df[
#         (df["time"] >= start_time) &
#         (df["time"] <= end_time)
#     ].copy()

#     return df


def _fetch_horizon_prices(start_time, end_time, output_path: Path):
    df = get_48h_prices_forecast()

    return df

def _build_horizon_df(start_time, end_time, output_path: Path) -> pd.DataFrame:
    solar_df = _fetch_horizon_solar(start_time, end_time)
    prices_df = _fetch_horizon_prices(start_time, end_time, output_path)
    load_df = _fetch_horizon_load(start_time, end_time, output_path)

    merged = pd.merge(solar_df, prices_df, on="time", how="inner")
    merged = pd.merge(merged, load_df, on="time", how="inner")

    merged["time"] = pd.to_datetime(merged["time"], utc=True)
    merged = merged.sort_values("time")

    # rolling window filter
    merged = merged[
        (merged["time"] >= start_time) &
        (merged["time"] <= end_time)
    ].copy()

    merged = merged.iloc[:EXPECTED_ROWS_HORIZON]

    if len(merged) != EXPECTED_ROWS_HORIZON:
        raise ValueError(f"Expected {EXPECTED_ROWS_HORIZON} rows, got {len(merged)}")

    return merged.reset_index(drop=True)


def _validate_continuous_timestamps(df: pd.DataFrame) -> None:
    step_ok = df["time"].diff().dropna().eq(pd.Timedelta(minutes=INTERVAL_MINUTES)).all()
    if not step_ok:
        raise ValueError("Time index not continuous")


def _load_system_config(config_path: Path = SYSTEM_CONFIG_PATH) -> float:
    with open(config_path) as f:
        config = json.load(f)
    return float(config["default_sell_price_cent_kwh"])


def build_aggregated_table(
    output_path: Path = Path(__file__).resolve().parents[1] / "data/runtime/aggregated_table.csv",
    update_interval: pd.Timedelta = pd.Timedelta(minutes=15),
    defaults: Optional[dict[str, float]] = None,
    save_output: bool = True,
) -> pd.DataFrame:

    sell_price_cfg = _load_system_config()
    default_sell_price_cent_kwh = sell_price_cfg

    if defaults:
        default_sell_price_cent_kwh = defaults.get(
            "sell_price_cent_kwh", default_sell_price_cent_kwh
        )

    now_utc = pd.Timestamp.now(tz="UTC")

    # align to last completed interval
    start_time = now_utc.floor(f"{INTERVAL_MINUTES}min")

    end_time = start_time + pd.Timedelta(hours=HOURS_PER_HORIZON) - pd.Timedelta(minutes=INTERVAL_MINUTES)

    df = _build_horizon_df(start_time, end_time, output_path)

    _validate_continuous_timestamps(df)

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

    df_final["energy_price_sell_cent_kwh"] = default_sell_price_cent_kwh
    df_final["source_sell_price"] = "config_default"

    df_final = df_final[OUTPUT_COLUMNS]

    if save_output:
        # --- runtime snapshot (overwrite) ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)

        # --- history append (ONLY NEW ROWS) ---
        history_path.parent.mkdir(parents=True, exist_ok=True)

        if history_path.exists():
            existing = pd.read_csv(history_path, parse_dates=["utc_timestamp"])

            # ensure same dtype
            df_final["utc_timestamp"] = pd.to_datetime(df_final["utc_timestamp"], utc=True)

            existing_times = set(existing["utc_timestamp"])

            # 🔥 filter only new timestamps
            new_rows = df_final[~df_final["utc_timestamp"].isin(existing_times)]

            if not new_rows.empty:
                new_rows.to_csv(
                    history_path,
                    mode="a",
                    header=False,
                    index=False,
                )
                print(f"Appended {len(new_rows)} new rows to history.")
            else:
                print("No new timestamps to append.")

        else:
            # first run → write full file
            df_final.to_csv(history_path, index=False)
            print("Created history file with initial data.")

    return df_final


# Example usage in main.py:
# df = build_aggregated_table(start_day='2026-03-20')