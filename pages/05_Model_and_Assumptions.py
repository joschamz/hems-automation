import json
from pathlib import Path
from datetime import timedelta, date
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import linprog

from utils import (
    get_daily_solar_kwh,
    get_daily_prices,
    get_daily_load_forecast,
)
from utils.load_utils import load_feature_engineered_dataset


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Model & Assumptions",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
        max-width: 1500px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #111827 42%, #1e293b 100%);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 42px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 980px;
    }

    .pill {
        display: inline-block;
        padding: 0.46rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-top: 0.45rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .glass-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
        backdrop-filter: blur(6px);
        height: 100%;
    }

    .kpi-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        height: 100%;
    }

    .kpi-label {
        color: #475569;
        font-size: 0.92rem;
        margin-bottom: 0.3rem;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.72rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-sub {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 760;
        color: #0f172a;
        margin-bottom: 0.6rem;
    }

    .muted {
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.7;
    }

    .ok {
        color: #16a34a;
        font-weight: 700;
    }

    .warn {
        color: #d97706;
        font-weight: 700;
    }

    .bad {
        color: #dc2626;
        font-weight: 700;
    }

    .insight-box {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        height: 100%;
    }

    .insight-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 760;
        margin-bottom: 0.35rem;
    }

    .insight-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .note-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 1.2rem 1.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 0.3rem 0.35rem 0.1rem 0.35rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS / PATHS
# =========================================================
DEFAULT_FEATURE_DATASET_PATH = "data/input/shifted-date-residential1_feature_engineered_full.csv"
DEFAULT_LOAD_MODEL_PATH = "models/load_forecast_model.pkl"
CONFIG_PATH = Path("user_config.json")


# =========================================================
# HELPERS
# =========================================================
def load_user_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "lat", "lon", "kwp", "tilt", "azimuth", "yield_factor",
        "battery_capacity_kwh", "initial_soc_kwh", "min_reserve_kwh",
        "max_charge_kw", "max_discharge_kw",
        "charge_efficiency", "discharge_efficiency",
        "sell_price_cent_kwh", "allow_grid_charging",
        "grid_charge_price_threshold", "cycle_penalty",
        "enforce_solar_first_in_lp", "terminal_soc_value",
        "min_end_soc_kwh",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing configuration keys in user_config.json: {missing}")

    return config


def write_temp_solar_config(lat, lon, kwp, tilt, azimuth, yield_factor) -> str:
    config = {
        "lat": float(lat),
        "lon": float(lon),
        "kwp": float(kwp),
        "tilt": float(tilt),
        "azimuth": float(azimuth),
        "yield_factor": float(yield_factor),
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, tmp, indent=2)
    tmp.close()
    return tmp.name


@st.cache_data(show_spinner=False)
def get_last_available_load_timestamp(feature_dataset_path: str) -> pd.Timestamp:
    df_all = load_feature_engineered_dataset(feature_dataset_path)
    return df_all.index.max()


def remap_load_forecast_to_target_day(load_raw: pd.DataFrame, planning_date: date) -> pd.DataFrame:
    out = load_raw.copy()
    target_start = pd.Timestamp(planning_date).tz_localize("UTC")
    out["time"] = pd.date_range(
        start=target_start,
        periods=len(out),
        freq="15min",
        tz="UTC"
    )
    return out


def build_dispatch_input_table(
    solar_raw: pd.DataFrame,
    price_raw: pd.DataFrame,
    load_raw: pd.DataFrame,
    default_sell_price_cent_kwh: float,
    dynamic_reserve_kwh: float | None = None,
) -> pd.DataFrame:
    solar = solar_raw.copy()
    solar["time"] = pd.to_datetime(solar["time"], utc=True)
    solar = solar.rename(columns={
        "time": "utc_timestamp",
        "predicted_kwh": "pv_generation_kwh",
        "predicted_kw": "pv_generation_kw",
    })
    solar = solar[["utc_timestamp", "pv_generation_kwh", "pv_generation_kw", "source"]].copy()

    price = price_raw.copy()
    price["time"] = pd.to_datetime(price["time"], utc=True)
    price = price.rename(columns={
        "time": "utc_timestamp",
        "price_cent_kwh": "energy_price_buy_cent_kwh",
        "price_eur_mwh": "energy_price_buy_eur_mwh",
    })
    price = price[["utc_timestamp", "energy_price_buy_cent_kwh", "energy_price_buy_eur_mwh", "source"]].copy()
    price["energy_price_sell_cent_kwh"] = float(default_sell_price_cent_kwh)

    load = load_raw.copy()
    load["time"] = pd.to_datetime(load["time"], utc=True)
    load = load.rename(columns={
        "time": "utc_timestamp",
        "predicted_kwh": "household_load_kwh",
        "predicted_kw": "household_load_kw",
    })
    keep_cols = ["utc_timestamp", "household_load_kwh", "household_load_kw"]
    if "household" in load.columns:
        keep_cols.append("household")
    if "source" in load.columns:
        keep_cols.append("source")
    load = load[keep_cols].copy()

    df = (
        solar.merge(price, on="utc_timestamp", how="inner", suffixes=("", "_price"))
             .merge(load, on="utc_timestamp", how="inner", suffixes=("", "_load"))
             .sort_values("utc_timestamp")
             .reset_index(drop=True)
    )

    if dynamic_reserve_kwh is not None:
        df["soc_min_dynamic_kwh"] = float(dynamic_reserve_kwh)

    return df


def prepare_forecast_input(input_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    required_cols = [
        "utc_timestamp",
        "pv_generation_kwh",
        "energy_price_buy_cent_kwh",
        "household_load_kwh",
    ]
    missing_required = [col for col in required_cols if col not in input_df.columns]
    if missing_required:
        raise ValueError(f"Missing required input columns: {missing_required}")

    df = input_df.copy()
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    if df["utc_timestamp"].isna().any():
        raise ValueError("Invalid utc_timestamp values found.")
    if df["utc_timestamp"].duplicated().any():
        raise ValueError("Duplicate utc_timestamp rows found.")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    numeric_cols = [
        "pv_generation_kwh",
        "energy_price_buy_cent_kwh",
        "household_load_kwh",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN after conversion.")

    if "energy_price_sell_cent_kwh" not in df.columns:
        df["energy_price_sell_cent_kwh"] = float(params["default_sell_price_cent_kwh"])
    else:
        df["energy_price_sell_cent_kwh"] = pd.to_numeric(df["energy_price_sell_cent_kwh"], errors="coerce")
        df["energy_price_sell_cent_kwh"] = df["energy_price_sell_cent_kwh"].fillna(float(params["default_sell_price_cent_kwh"]))

    if "soc_min_dynamic_kwh" not in df.columns:
        df["soc_min_dynamic_kwh"] = float(params["soc_min_kwh"])
    else:
        df["soc_min_dynamic_kwh"] = pd.to_numeric(df["soc_min_dynamic_kwh"], errors="coerce")
        df["soc_min_dynamic_kwh"] = df["soc_min_dynamic_kwh"].fillna(float(params["soc_min_kwh"]))

    df["soc_min_dynamic_kwh"] = df["soc_min_dynamic_kwh"].clip(
        lower=float(params["soc_min_kwh"]),
        upper=float(params["soc_max_kwh"]),
    )

    expected_step = pd.Timedelta(minutes=int(params["interval_minutes"]))
    step_ok = df["utc_timestamp"].diff().dropna().eq(expected_step).all()
    if not step_ok:
        raise ValueError("Input timestamps are not continuous at 15-minute resolution.")

    if len(df) != 96:
        raise ValueError(f"Expected 96 rows for 24h horizon, got {len(df)}.")

    return df


def _finalize_dispatch_output(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    cycle_penalty = float(params.get("cycle_penalty_cent_per_kwh", 0.0))
    df["soc_percent"] = (df["soc_kwh"] / params["battery_capacity_kwh"]) * 100.0
    pv_used = df["pv_to_load_kwh"] + df["pv_to_battery_kwh"] + df["export_to_grid_kwh"]
    df["curtailed_pv_kwh"] = np.maximum(0.0, df["pv_generation_kwh"] - pv_used)
    df["total_import_kwh"] = df["grid_to_load_kwh"] + df["grid_to_battery_kwh"]
    df["total_export_kwh"] = df["export_to_grid_kwh"]
    df["interval_cost_cent"] = (
        df["energy_price_buy_cent_kwh"] * df["total_import_kwh"]
        - df["energy_price_sell_cent_kwh"] * df["total_export_kwh"]
        + cycle_penalty * (
            df["pv_to_battery_kwh"] + df["grid_to_battery_kwh"] + df["battery_to_load_kwh"]
        )
    )
    df["cumulative_cost_cent"] = df["interval_cost_cent"].cumsum()
    return df


def run_rule_dispatch(forecast_df: pd.DataFrame, params: dict, initial_soc_kwh: float) -> pd.DataFrame:
    dt_hours = params["interval_minutes"] / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours
    eta_c = params["charge_efficiency"]
    eta_d = params["discharge_efficiency"]
    soc_min = params["soc_min_kwh"]
    soc_max = params["soc_max_kwh"]
    allow_grid_charging = bool(params.get("allow_grid_charging", False))
    grid_charge_threshold = params.get("grid_charge_price_threshold_cent_kwh")

    soc = float(initial_soc_kwh)
    rows = []

    for row in forecast_df.itertuples(index=False):
        timestamp = row.utc_timestamp
        load_remaining = float(row.household_load_kwh)
        pv_remaining = float(row.pv_generation_kwh)
        buy_price = float(row.energy_price_buy_cent_kwh)
        sell_price = float(row.energy_price_sell_cent_kwh)
        interval_soc_min = float(np.clip(getattr(row, "soc_min_dynamic_kwh", soc_min), soc_min, soc_max))

        pv_to_load_kwh = min(load_remaining, pv_remaining)
        load_remaining -= pv_to_load_kwh
        pv_remaining -= pv_to_load_kwh

        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        pv_to_battery_kwh = min(pv_remaining, charge_limit_kwh, charge_headroom_kwh)
        soc += pv_to_battery_kwh * eta_c
        pv_remaining -= pv_to_battery_kwh

        available_discharge_kwh = max(0.0, (soc - interval_soc_min) * eta_d)
        battery_to_load_kwh = min(load_remaining, discharge_limit_kwh, available_discharge_kwh)
        soc -= battery_to_load_kwh / eta_d
        load_remaining -= battery_to_load_kwh

        grid_to_load_kwh = load_remaining

        charge_limit_left_kwh = max(0.0, charge_limit_kwh - pv_to_battery_kwh)
        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        should_grid_charge = (
            allow_grid_charging
            and grid_charge_threshold is not None
            and buy_price <= float(grid_charge_threshold)
        )
        grid_to_battery_kwh = min(charge_limit_left_kwh, charge_headroom_kwh) if should_grid_charge else 0.0
        soc += grid_to_battery_kwh * eta_c

        export_to_grid_kwh = pv_remaining
        soc = float(np.clip(soc, interval_soc_min, soc_max))

        rows.append({
            "utc_timestamp": timestamp,
            "pv_generation_kwh": row.pv_generation_kwh,
            "household_load_kwh": row.household_load_kwh,
            "energy_price_buy_cent_kwh": buy_price,
            "energy_price_sell_cent_kwh": sell_price,
            "grid_to_load_kwh": grid_to_load_kwh,
            "pv_to_load_kwh": pv_to_load_kwh,
            "pv_to_battery_kwh": pv_to_battery_kwh,
            "battery_to_load_kwh": battery_to_load_kwh,
            "grid_to_battery_kwh": grid_to_battery_kwh,
            "export_to_grid_kwh": export_to_grid_kwh,
            "soc_kwh": soc,
            "soc_min_dynamic_kwh": interval_soc_min,
            "method": "rule_based",
        })

    return _finalize_dispatch_output(pd.DataFrame(rows), params)


def run_lp_dispatch(forecast_df: pd.DataFrame, params: dict, initial_soc_kwh: float) -> pd.DataFrame:
    n = len(forecast_df)
    dt_hours = params["interval_minutes"] / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours

    eta_c = float(params["charge_efficiency"])
    eta_d = float(params["discharge_efficiency"])
    soc_min = float(params["soc_min_kwh"])
    soc_max = float(params["soc_max_kwh"])

    if not (soc_min <= initial_soc_kwh <= soc_max):
        raise ValueError(f"Initial SoC {initial_soc_kwh} must be within [{soc_min}, {soc_max}].")

    load = forecast_df["household_load_kwh"].to_numpy(dtype=float)
    pv = forecast_df["pv_generation_kwh"].to_numpy(dtype=float)
    buy = forecast_df["energy_price_buy_cent_kwh"].to_numpy(dtype=float)
    sell = forecast_df["energy_price_sell_cent_kwh"].to_numpy(dtype=float)
    soc_min_dynamic = forecast_df["soc_min_dynamic_kwh"].to_numpy(dtype=float)

    cycle_penalty = float(params.get("cycle_penalty_cent_per_kwh", 0.0))
    enforce_solar_first = bool(params.get("enforce_solar_first_in_lp", True))
    terminal_soc_value = float(params.get("terminal_soc_value_cent_kwh", 0.0))
    min_end_soc_kwh = params.get("min_end_soc_kwh")

    idx_gl = np.arange(0, n)
    idx_pl = np.arange(n, 2 * n)
    idx_pb = np.arange(2 * n, 3 * n)
    idx_bl = np.arange(3 * n, 4 * n)
    idx_gb = np.arange(4 * n, 5 * n)
    idx_ex = np.arange(5 * n, 6 * n)
    idx_soc = np.arange(6 * n, 7 * n + 1)
    num_vars = 7 * n + 1

    c = np.zeros(num_vars, dtype=float)
    c[idx_gl] = buy
    c[idx_gb] = buy
    c[idx_ex] = -sell
    c[idx_pb] += cycle_penalty
    c[idx_gb] += cycle_penalty
    c[idx_bl] += cycle_penalty
    c[idx_soc[-1]] -= terminal_soc_value

    A_eq_rows = []
    b_eq = []

    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_gl[t]] = 1.0
        row[idx_pl[t]] = 1.0
        row[idx_bl[t]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(load[t])

        row = np.zeros(num_vars, dtype=float)
        row[idx_soc[t + 1]] = 1.0
        row[idx_soc[t]] = -1.0
        row[idx_pb[t]] = -eta_c
        row[idx_gb[t]] = -eta_c
        row[idx_bl[t]] = 1.0 / eta_d
        A_eq_rows.append(row)
        b_eq.append(0.0)

    A_ub_rows = []
    b_ub = []

    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_pl[t]] = 1.0
        row[idx_pb[t]] = 1.0
        row[idx_ex[t]] = 1.0
        A_ub_rows.append(row)
        b_ub.append(pv[t])

        row = np.zeros(num_vars, dtype=float)
        row[idx_pb[t]] = 1.0
        row[idx_gb[t]] = 1.0
        A_ub_rows.append(row)
        b_ub.append(charge_limit_kwh)

    if enforce_solar_first:
        for t in range(n):
            row = np.zeros(num_vars, dtype=float)
            row[idx_gl[t]] = 1.0
            row[idx_bl[t]] = 1.0
            A_ub_rows.append(row)
            b_ub.append(max(0.0, load[t] - pv[t]))

    bounds = [(0.0, None)] * num_vars
    for idx in idx_pb:
        bounds[idx] = (0.0, charge_limit_kwh)
    for idx in idx_bl:
        bounds[idx] = (0.0, discharge_limit_kwh)
    for idx in idx_gb:
        if bool(params.get("allow_grid_charging", True)):
            bounds[idx] = (0.0, charge_limit_kwh)
        else:
            bounds[idx] = (0.0, 0.0)

    bounds[idx_soc[0]] = (float(initial_soc_kwh), float(initial_soc_kwh))
    for t in range(n):
        bounds[idx_soc[t + 1]] = (float(soc_min_dynamic[t]), soc_max)

    if min_end_soc_kwh is not None:
        min_end_soc = float(min_end_soc_kwh)
        min_end_soc = max(soc_min, min(min_end_soc, soc_max))
        lb, ub = bounds[idx_soc[-1]]
        bounds[idx_soc[-1]] = (max(lb, min_end_soc), ub)

    result = linprog(
        c=c,
        A_ub=np.array(A_ub_rows, dtype=float) if A_ub_rows else None,
        b_ub=np.array(b_ub, dtype=float) if b_ub else None,
        A_eq=np.array(A_eq_rows, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP optimization failed: {result.message}")

    x = result.x
    output_df = forecast_df.copy()
    output_df["grid_to_load_kwh"] = x[idx_gl]
    output_df["pv_to_load_kwh"] = x[idx_pl]
    output_df["pv_to_battery_kwh"] = x[idx_pb]
    output_df["battery_to_load_kwh"] = x[idx_bl]
    output_df["grid_to_battery_kwh"] = x[idx_gb]
    output_df["export_to_grid_kwh"] = x[idx_ex]
    output_df["soc_kwh"] = x[idx_soc[1:]]
    output_df["method"] = "optimizer_lp"

    return _finalize_dispatch_output(output_df, params)


def run_balance_checks(dispatch_df: pd.DataFrame, params: dict, tolerance: float = 1e-6) -> pd.Series:
    load_balance_error = (
        dispatch_df["pv_to_load_kwh"]
        + dispatch_df["battery_to_load_kwh"]
        + dispatch_df["grid_to_load_kwh"]
        - dispatch_df["household_load_kwh"]
    ).abs().max()

    pv_balance_error = (
        dispatch_df["pv_to_load_kwh"]
        + dispatch_df["pv_to_battery_kwh"]
        + dispatch_df["export_to_grid_kwh"]
        + dispatch_df.get("curtailed_pv_kwh", 0.0)
        - dispatch_df["pv_generation_kwh"]
    ).abs().max()

    soc_ok = dispatch_df["soc_kwh"].between(
        params["soc_min_kwh"] - tolerance,
        params["soc_max_kwh"] + tolerance,
    ).all()

    return pd.Series({
        "rows": len(dispatch_df),
        "max_load_balance_error_kwh": float(load_balance_error),
        "max_pv_balance_error_kwh": float(pv_balance_error),
        "soc_within_bounds": bool(soc_ok),
    })


def build_validation_scorecard(rule_checks: pd.Series, opt_checks: pd.Series) -> dict:
    max_balance_error = max(
        float(rule_checks["max_load_balance_error_kwh"]),
        float(rule_checks["max_pv_balance_error_kwh"]),
        float(opt_checks["max_load_balance_error_kwh"]),
        float(opt_checks["max_pv_balance_error_kwh"]),
    )

    all_soc_ok = bool(rule_checks["soc_within_bounds"]) and bool(opt_checks["soc_within_bounds"])

    integrity_status = "Validated" if max_balance_error <= 1e-6 and all_soc_ok else "Check Required"

    return {
        "integrity_status": integrity_status,
        "max_balance_error_kwh": max_balance_error,
        "soc_ok": all_soc_ok,
    }


def build_parameter_groups(
    planning_date,
    latitude,
    longitude,
    capacity_kwp,
    tilt,
    azimuth,
    yield_factor,
    battery_capacity_kwh,
    initial_soc_kwh,
    min_reserve_kwh,
    soc_max_kwh,
    max_charge_kw,
    max_discharge_kw,
    charge_efficiency,
    discharge_efficiency,
    sell_price_cent_kwh,
    allow_grid_charging,
    grid_charge_price_threshold,
    cycle_penalty,
    enforce_solar_first_in_lp,
    terminal_soc_value,
    min_end_soc_value,
    feature_dataset_path,
    load_model_path,
    last_load_ts,
):
    return {
        "Site & PV System": [
            ("Planning Date", str(planning_date)),
            ("Latitude", latitude),
            ("Longitude", longitude),
            ("PV Capacity (kWp)", capacity_kwp),
            ("Panel Tilt (°)", tilt),
            ("Panel Azimuth (°)", azimuth),
            ("Yield Factor", yield_factor),
        ],
        "Battery & Control Limits": [
            ("Battery Capacity (kWh)", battery_capacity_kwh),
            ("Initial SoC (kWh)", initial_soc_kwh),
            ("Minimum Reserve (kWh)", min_reserve_kwh),
            ("Maximum Effective SoC (kWh)", soc_max_kwh),
            ("Max Charge Power (kW)", max_charge_kw),
            ("Max Discharge Power (kW)", max_discharge_kw),
            ("Charge Efficiency", charge_efficiency),
            ("Discharge Efficiency", discharge_efficiency),
        ],
        "Economic & Optimization Settings": [
            ("Default Sell Price (cent/kWh)", sell_price_cent_kwh),
            ("Allow Grid Charging", allow_grid_charging),
            ("Grid Charge Threshold (cent/kWh)", grid_charge_price_threshold),
            ("Cycle Penalty (cent/kWh)", cycle_penalty),
            ("Enforce Solar First in LP", enforce_solar_first_in_lp),
            ("Terminal SoC Value (cent/kWh)", terminal_soc_value),
            ("Minimum End SoC (kWh)", min_end_soc_value),
        ],
        "Model Artifacts": [
            ("Feature Dataset Path", feature_dataset_path),
            ("Load Model Path", load_model_path),
            ("Load Baseline Timestamp", str(last_load_ts)),
        ],
    }


def build_validation_bar_chart(rule_checks: pd.Series, opt_checks: pd.Series) -> go.Figure:
    labels = [
        "Rule Load Error",
        "Rule PV Error",
        "Opt Load Error",
        "Opt PV Error",
    ]
    values = [
        float(rule_checks["max_load_balance_error_kwh"]),
        float(rule_checks["max_pv_balance_error_kwh"]),
        float(opt_checks["max_load_balance_error_kwh"]),
        float(opt_checks["max_pv_balance_error_kwh"]),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.8f}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Error: %{y:.10f} kWh<extra></extra>",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        title=dict(text="Energy Balance Validation Errors", x=0.02, xanchor="left", font=dict(size=20)),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Absolute Error (kWh)", showgrid=True, gridcolor="rgba(148,163,184,0.20)", zeroline=False),
    )
    return fig


def build_assumption_cards() -> list[tuple[str, str]]:
    return [
        (
            "Solar Forecast Assumption",
            "Solar generation is estimated through the project solar utility using online weather-driven forecast data. Forecast quality therefore depends on upstream weather uncertainty."
        ),
        (
            "Electricity Price Assumption",
            "Electricity buy prices are produced from the project price utility using available market data. Published prices and fallback paths can affect economic realism for the target day."
        ),
        (
            "Load Forecast Assumption",
            "Household demand is generated from the trained baseline load forecasting model and mapped onto the selected planning date. It is not yet personalized with live user smart-meter feedback."
        ),
        (
            "Optimization Assumption",
            "The LP optimizer acts on a deterministic day-ahead forecast. It does not yet optimize under uncertainty, intraday corrections, battery degradation physics, or user comfort constraints."
        ),
    ]


# =========================================================
# CONFIG
# =========================================================
try:
    user_config = load_user_config()
except Exception as exc:
    st.error(f"Failed to load user configuration: {exc}")
    st.stop()

latitude = float(user_config["lat"])
longitude = float(user_config["lon"])
capacity_kwp = float(user_config["kwp"])
tilt = float(user_config["tilt"])
azimuth = float(user_config["azimuth"])
yield_factor = float(user_config["yield_factor"])

feature_dataset_path = user_config.get("feature_dataset_path", DEFAULT_FEATURE_DATASET_PATH)
load_model_path = user_config.get("load_model_path", DEFAULT_LOAD_MODEL_PATH)

battery_capacity_kwh = float(user_config["battery_capacity_kwh"])
initial_soc_kwh = float(user_config["initial_soc_kwh"])
min_reserve_kwh = float(user_config["min_reserve_kwh"])
max_charge_kw = float(user_config["max_charge_kw"])
max_discharge_kw = float(user_config["max_discharge_kw"])
charge_efficiency = float(user_config["charge_efficiency"])
discharge_efficiency = float(user_config["discharge_efficiency"])
sell_price_cent_kwh = float(user_config["sell_price_cent_kwh"])
allow_grid_charging = bool(user_config["allow_grid_charging"])
grid_charge_price_threshold = float(user_config["grid_charge_price_threshold"])
cycle_penalty = float(user_config["cycle_penalty"])
enforce_solar_first_in_lp = bool(user_config["enforce_solar_first_in_lp"])
terminal_soc_value = float(user_config["terminal_soc_value"])
min_end_soc_kwh = float(user_config["min_end_soc_kwh"])


# =========================================================
# SIDEBAR
# =========================================================
tomorrow_default = pd.Timestamp.now(tz="UTC").date() + timedelta(days=1)

with st.sidebar:
    planning_date = st.date_input("Planning Date", value=tomorrow_default)
    st.caption("This page documents the model setup, operational assumptions, and validation status behind the HEMS outputs.")


# =========================================================
# PARAMS
# =========================================================
soc_max_kwh = battery_capacity_kwh * 0.90
initial_soc_kwh = min(max(initial_soc_kwh, min_reserve_kwh), soc_max_kwh)
min_end_soc_value = min_end_soc_kwh if min_end_soc_kwh > 0 else None

PARAMS = {
    "interval_minutes": 15,
    "battery_capacity_kwh": float(battery_capacity_kwh),
    "soc_min_kwh": float(min_reserve_kwh),
    "soc_max_kwh": float(soc_max_kwh),
    "max_charge_kw": float(max_charge_kw),
    "max_discharge_kw": float(max_discharge_kw),
    "charge_efficiency": float(charge_efficiency),
    "discharge_efficiency": float(discharge_efficiency),
    "default_sell_price_cent_kwh": float(sell_price_cent_kwh),
    "allow_grid_charging": bool(allow_grid_charging),
    "grid_charge_price_threshold_cent_kwh": float(grid_charge_price_threshold),
    "cycle_penalty_cent_per_kwh": float(cycle_penalty),
    "enforce_solar_first_in_lp": bool(enforce_solar_first_in_lp),
    "terminal_soc_value_cent_kwh": float(terminal_soc_value),
    "min_end_soc_kwh": min_end_soc_value,
}


# =========================================================
# PIPELINE
# =========================================================
try:
    planning_date = pd.Timestamp(planning_date).date()

    solar_config_path = write_temp_solar_config(
        lat=latitude,
        lon=longitude,
        kwp=capacity_kwp,
        tilt=tilt,
        azimuth=azimuth,
        yield_factor=yield_factor,
    )

    solar_raw = get_daily_solar_kwh(
        target_date=planning_date,
        mode="forecast" if planning_date >= pd.Timestamp.now().date() else "historical",
        config_path=solar_config_path,
        allow_fallback=True,
    )

    price_mode = "forecast" if planning_date >= pd.Timestamp.now(tz="UTC").date() else "historical"
    price_raw = get_daily_prices(
        target_date=planning_date,
        mode=price_mode,
        secrets_dir="secrets",
        allow_fallback=True,
    )

    last_load_ts = get_last_available_load_timestamp(feature_dataset_path)
    load_raw = get_daily_load_forecast(
        forecast_time=last_load_ts,
        household_id=1,
        model_path=load_model_path,
        feature_dataset_path=feature_dataset_path,
    )
    load_raw = remap_load_forecast_to_target_day(load_raw, planning_date)

    dispatch_input_df = build_dispatch_input_table(
        solar_raw=solar_raw,
        price_raw=price_raw,
        load_raw=load_raw,
        default_sell_price_cent_kwh=sell_price_cent_kwh,
        dynamic_reserve_kwh=min_reserve_kwh,
    )

    prepared_forecast_df = prepare_forecast_input(dispatch_input_df, PARAMS)

    rule_dispatch_df = run_rule_dispatch(prepared_forecast_df, PARAMS, initial_soc_kwh)
    optimized_dispatch_df = run_lp_dispatch(prepared_forecast_df, PARAMS, initial_soc_kwh)

    rule_checks = run_balance_checks(rule_dispatch_df, PARAMS)
    opt_checks = run_balance_checks(optimized_dispatch_df, PARAMS)

    validation_summary = build_validation_scorecard(rule_checks, opt_checks)

    parameter_groups = build_parameter_groups(
        planning_date=planning_date,
        latitude=latitude,
        longitude=longitude,
        capacity_kwp=capacity_kwp,
        tilt=tilt,
        azimuth=azimuth,
        yield_factor=yield_factor,
        battery_capacity_kwh=battery_capacity_kwh,
        initial_soc_kwh=initial_soc_kwh,
        min_reserve_kwh=min_reserve_kwh,
        soc_max_kwh=soc_max_kwh,
        max_charge_kw=max_charge_kw,
        max_discharge_kw=max_discharge_kw,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        sell_price_cent_kwh=sell_price_cent_kwh,
        allow_grid_charging=allow_grid_charging,
        grid_charge_price_threshold=grid_charge_price_threshold,
        cycle_penalty=cycle_penalty,
        enforce_solar_first_in_lp=enforce_solar_first_in_lp,
        terminal_soc_value=terminal_soc_value,
        min_end_soc_value=min_end_soc_value,
        feature_dataset_path=feature_dataset_path,
        load_model_path=load_model_path,
        last_load_ts=last_load_ts,
    )

except Exception as exc:
    st.error(f"Model and assumptions pipeline failed: {exc}")
    st.stop()


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">Model & Assumptions Center</div>
    <div class="hero-subtitle">
        This page documents the simulation setup, optimization parameters, data dependencies, and validation checks behind the HEMS outputs.
        It is the model governance layer of the project: the place where assumptions are made explicit and result credibility is tested.
    </div>
    <div style="margin-top:0.65rem;">
        <span class="pill">System Configuration</span>
        <span class="pill">Validation Checks</span>
        <span class="pill">Operational Assumptions</span>
        <span class="pill">Model Transparency</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# KPI ROW
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Validation Status</div>
    <div class="kpi-value">{validation_summary['integrity_status']}</div>
    <div class="kpi-sub">Combined balance and constraint check status</div>
</div>
""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Maximum Balance Error</div>
    <div class="kpi-value">{validation_summary['max_balance_error_kwh']:.8f}</div>
    <div class="kpi-sub">Worst observed load or PV balance deviation (kWh)</div>
</div>
""", unsafe_allow_html=True)

with k3:
    soc_label = "Within Bounds" if validation_summary["soc_ok"] else "Out of Bounds"
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Battery Constraint Status</div>
    <div class="kpi-value">{soc_label}</div>
    <div class="kpi-sub">Rule-based and optimized schedules checked</div>
</div>
""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Planning Horizon</div>
    <div class="kpi-value">96 Intervals</div>
    <div class="kpi-sub">24 hours at 15-minute resolution</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")


# =========================================================
# VALIDATION SUMMARY
# =========================================================
status_css = "ok" if validation_summary["integrity_status"] == "Validated" else "warn"

st.markdown(
    f"""
<div class="glass-card">
    <div class="section-title">Validation Summary</div>
    <div class="{status_css}" style="font-size:1.18rem; margin-bottom:0.5rem;">{validation_summary['integrity_status']}</div>
    <div class="muted">
        The purpose of this layer is to verify that the HEMS outputs are not only visually convincing, but also numerically consistent.
        Load balance, PV balance, and battery state constraints are checked for both the rule-based baseline and the optimized schedule.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")


# =========================================================
# PARAMETER TABLES
# =========================================================
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("System Configuration")
    for section_name, rows in parameter_groups.items():
        st.markdown(
            f"""
<div class="glass-card" style="margin-bottom:1rem;">
    <div class="section-title">{section_name}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        section_df = pd.DataFrame(rows, columns=["Parameter", "Value"])
        st.dataframe(section_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Validation Diagnostics")

    validation_df = pd.DataFrame({
        "Check": [
            "Rule-Based Load Balance Error (kWh)",
            "Rule-Based PV Balance Error (kWh)",
            "Rule-Based SoC Bounds",
            "Optimized Load Balance Error (kWh)",
            "Optimized PV Balance Error (kWh)",
            "Optimized SoC Bounds",
        ],
        "Value": [
            round(rule_checks["max_load_balance_error_kwh"], 8),
            round(rule_checks["max_pv_balance_error_kwh"], 8),
            bool(rule_checks["soc_within_bounds"]),
            round(opt_checks["max_load_balance_error_kwh"], 8),
            round(opt_checks["max_pv_balance_error_kwh"], 8),
            bool(opt_checks["soc_within_bounds"]),
        ],
    })
    st.dataframe(validation_df, use_container_width=True, hide_index=True)

    st.markdown("")
    val_fig = build_validation_bar_chart(rule_checks, opt_checks)
    st.plotly_chart(val_fig, use_container_width=True)

st.markdown("")


# =========================================================
# ASSUMPTIONS
# =========================================================
st.markdown(
    """
<div class="note-card">
    <div class="section-title">Why This Page Matters</div>
    <div class="muted">
        Most dashboards show results but hide assumptions. That is weak engineering. This page exists to expose the assumptions,
        modeling simplifications, and validation evidence that determine whether the outputs should actually be trusted.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

cards = build_assumption_cards()
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

for col, (title, text) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(
            f"""
<div class="insight-box">
    <div class="insight-title">{title}</div>
    <div class="insight-text">{text}</div>
</div>
""",
            unsafe_allow_html=True,
        )