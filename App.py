import json
import tempfile
from pathlib import Path
import base64
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

from utils import (
    get_daily_solar_kwh,
    get_daily_prices,
    get_daily_load_forecast,
)
from utils.load_utils import load_feature_engineered_dataset
from textwrap import dedent


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HEMS Control Center",
    page_icon="⚡",
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
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 16px 38px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        line-height: 1.05;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }

    .hero-pill {
        display: block;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        color: #0f172a;
        font-size: 0.84rem;
        font-weight: 700;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 0.4rem;}
        text-align: center;

    .glass-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        height: 100%;
        backdrop-filter: blur(6px);
   }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
        color: #0f172a;
    }

    .muted {
        color: #475569;
        font-size: 0.92rem;
    }

    .ok {
        color: #22c55e;
        font-weight: 700;
    }

    .warn {
        color: #f59e0b;
        font-weight: 700;
    }

    .bad {
        color: #ef4444;
        font-weight: 700;
    }

    .footer-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.08);
        color: #0f172a;
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }

    .brand-logo {
        width: 82px;
        height: 82px;
        object-fit: contain;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        background: rgba(255,255,255,0.02);
    }

    .hero-cover {
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
    }

    .hero-cover img {
        width: 100%;
        display: block;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(15,23,42,0.08);
        padding: 0.85rem 1rem;
        border-radius: 18px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
DEFAULT_FEATURE_DATASET_PATH = "data/input/shifted-date-residential1_feature_engineered_full.csv"
DEFAULT_LOAD_MODEL_PATH = "models/load_forecast_model.pkl"

ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "logo.png"
COVER_PATH = ASSETS_DIR / "cover.png"


def image_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode()


logo_b64 = image_to_base64(LOGO_PATH)
cover_b64 = image_to_base64(COVER_PATH)


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

    for col in ["pv_generation_kwh", "household_load_kwh"]:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' must be non-negative.")

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

        # 1) PV to load
        pv_to_load_kwh = min(load_remaining, pv_remaining)
        load_remaining -= pv_to_load_kwh
        pv_remaining -= pv_to_load_kwh

        # 2) PV to battery
        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        pv_to_battery_kwh = min(pv_remaining, charge_limit_kwh, charge_headroom_kwh)
        soc += pv_to_battery_kwh * eta_c
        pv_remaining -= pv_to_battery_kwh

        # 3) Battery to load
        available_discharge_kwh = max(0.0, (soc - interval_soc_min) * eta_d)
        battery_to_load_kwh = min(load_remaining, discharge_limit_kwh, available_discharge_kwh)
        soc -= battery_to_load_kwh / eta_d
        load_remaining -= battery_to_load_kwh

        # 4) Grid to load
        grid_to_load_kwh = load_remaining

        # 5) Optional grid charging
        charge_limit_left_kwh = max(0.0, charge_limit_kwh - pv_to_battery_kwh)
        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        should_grid_charge = (
            allow_grid_charging
            and grid_charge_threshold is not None
            and buy_price <= float(grid_charge_threshold)
        )
        grid_to_battery_kwh = min(charge_limit_left_kwh, charge_headroom_kwh) if should_grid_charge else 0.0
        soc += grid_to_battery_kwh * eta_c

        # 6) Export remaining PV
        export_to_grid_kwh = pv_remaining

        soc = float(np.clip(soc, interval_soc_min, soc_max))

        flow_names = [
            "pv_to_load",
            "pv_to_battery",
            "battery_to_load",
            "grid_to_load",
            "grid_to_battery",
            "export_to_grid",
        ]
        flow_vals = [
            pv_to_load_kwh,
            pv_to_battery_kwh,
            battery_to_load_kwh,
            grid_to_load_kwh,
            grid_to_battery_kwh,
            export_to_grid_kwh,
        ]
        decision_rule = " | ".join(n for n, v in zip(flow_names, flow_vals) if v > 1e-9) or "idle"

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
            "decision_rule": decision_rule,
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
    output_df["decision_rule"] = "optimizer_lp"
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


def summarize_dispatch(df: pd.DataFrame) -> dict:
    return {
        "cost_cent": float(df["interval_cost_cent"].sum()),
        "cost_eur": float(df["interval_cost_cent"].sum() / 100.0),
        "grid_import_kwh": float(df["total_import_kwh"].sum()),
        "grid_export_kwh": float(df["total_export_kwh"].sum()),
        "final_soc_kwh": float(df["soc_kwh"].iloc[-1]),
        "final_soc_percent": float(df["soc_percent"].iloc[-1]),
        "pv_to_load_kwh": float(df["pv_to_load_kwh"].sum()),
        "pv_to_battery_kwh": float(df["pv_to_battery_kwh"].sum()),
        "battery_to_load_kwh": float(df["battery_to_load_kwh"].sum()),
        "grid_to_battery_kwh": float(df["grid_to_battery_kwh"].sum()),
        "export_to_grid_kwh": float(df["export_to_grid_kwh"].sum()),
    }


def build_recommendation_text(opt_summary: dict, rule_summary: dict) -> list[str]:
    saving = rule_summary["cost_eur"] - opt_summary["cost_eur"]
    recommendations = []

    if opt_summary["pv_to_battery_kwh"] > 0.1:
        recommendations.append("Store daytime solar surplus in the battery before considering export.")
    if opt_summary["battery_to_load_kwh"] > 0.1:
        recommendations.append("Use the battery to cover demand during higher-price intervals and reduce grid purchases.")
    if opt_summary["grid_to_battery_kwh"] > 0.1:
        recommendations.append("Charge the battery selectively from the grid during low-price windows.")
    if opt_summary["grid_export_kwh"] > 0.1:
        recommendations.append("Export residual PV only after local demand and battery charging opportunities are satisfied.")

    if saving > 0.01:
        recommendations.append(f"The optimized schedule outperforms the rule-based baseline by approximately €{saving:.2f}.")
    else:
        recommendations.append("The optimized schedule is close to the rule-based baseline under the current price and generation profile.")

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================
tomorrow_default = pd.Timestamp.now(tz="UTC").date() + timedelta(days=1)

with st.sidebar:
    st.title("⚙️ Planning Controls")

    st.markdown("### Planning Horizon")
    planning_date = st.date_input("Planning Date", value=tomorrow_default)

    st.markdown("### Location")
    latitude = st.number_input("Latitude", value=47.659216, format="%.6f")
    longitude = st.number_input("Longitude", value=9.175072, format="%.6f")

    st.markdown("### Solar System")
    capacity_kwp = st.number_input("PV Capacity (kWp)", min_value=0.0, value=15.0, step=0.5)
    tilt = st.number_input("Panel Tilt (°)", min_value=0.0, max_value=90.0, value=35.0, step=1.0)
    azimuth = st.number_input("Panel Azimuth (°)", min_value=-180.0, max_value=180.0, value=0.0, step=1.0)
    yield_factor = st.slider("Yield Factor", min_value=0.10, max_value=1.00, value=0.70, step=0.01)

    st.markdown("### Battery")
    battery_capacity_kwh = st.number_input("Battery Capacity (kWh)", min_value=0.1, value=13.5, step=0.5)
    initial_soc_kwh = st.number_input("Initial Battery SoC (kWh)", min_value=0.0, value=6.0, step=0.1)
    min_reserve_kwh = st.number_input("Minimum Battery Reserve (kWh)", min_value=0.0, value=1.35, step=0.1)
    max_charge_kw = st.number_input("Max Charge Power (kW)", min_value=0.0, value=5.0, step=0.5)
    max_discharge_kw = st.number_input("Max Discharge Power (kW)", min_value=0.0, value=5.0, step=0.5)
    charge_efficiency = st.slider("Charge Efficiency", min_value=0.50, max_value=1.00, value=0.90, step=0.01)
    discharge_efficiency = st.slider("Discharge Efficiency", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

    st.markdown("### Market / Strategy")
    sell_price_cent_kwh = st.number_input("Default Sell Price (cent/kWh)", min_value=0.0, value=8.0, step=0.5)
    allow_grid_charging = st.checkbox("Allow Grid Charging", value=True)
    grid_charge_price_threshold = st.number_input("Grid Charge Threshold (cent/kWh)", min_value=0.0, value=10.0, step=0.5)
    cycle_penalty = st.number_input("Cycle Penalty (cent/kWh)", min_value=0.0, value=0.001, step=0.001, format="%.3f")
    enforce_solar_first_in_lp = st.checkbox("Enforce Solar First in LP", value=True)
    terminal_soc_value = st.number_input("Terminal SoC Value (cent/kWh)", min_value=0.0, value=0.0, step=0.1)
    min_end_soc_kwh = st.number_input("Minimum End SoC (kWh)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("### Advanced Paths")
    feature_dataset_path = st.text_input("Feature Dataset Path", value=DEFAULT_FEATURE_DATASET_PATH)
    load_model_path = st.text_input("Load Model Path", value=DEFAULT_LOAD_MODEL_PATH)

    run_button = st.button("🚀 Run Optimization", use_container_width=True)

# =========================================================
# HERO
# =========================================================

if cover_b64:
    st.markdown(
        f"""
<div class="hero-cover">
    <img src="data:image/png;base64,{cover_b64}">
</div>
""",
        unsafe_allow_html=True,
    )

hero_logo_html = ""
if logo_b64:
    hero_logo_html = f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Camelectrix Logo">'

st.markdown(
    f"""
<div class="hero-card">
    <div class="brand-row">
        {hero_logo_html}
        <div>
            <div class="hero-title">Camelectrix – HEMS Control Center</div>
            <div class="hero-subtitle">
                Day-ahead energy planning for smart homes using live solar forecast,
                live electricity price forecast, baseline household load forecasting,
                and battery dispatch optimization.
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

pill_cols = st.columns(5)
pill_labels = [
    "Live Solar Forecast",
    "Live Price Forecast",
    "Baseline Load Model",
    "Rule-Based Dispatch",
    "LP Cost Optimization",
]

for col, label in zip(pill_cols, pill_labels):
    with col:
        st.markdown(
            f"""
<div class="hero-pill" style="text-align:center; width:100%;">
    {label}
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

col_intro1, col_intro2 = st.columns([1.35, 1])

with col_intro1:
    st.markdown(
        f"""
<div class="glass-card">
    <div class="section-title">Planning Objective</div>
    <div class="muted">
        Generate a cost-minimizing 24-hour dispatch schedule for <b>{planning_date}</b> at 15-minute resolution.
        The app combines PV forecast, electricity price forecast, and baseline household demand prediction
        to determine when to consume solar locally, charge the battery, discharge the battery, buy from the grid,
        or export excess energy.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_intro2:
    st.markdown(
        """
<div class="glass-card">
    <div class="section-title">Model Transparency</div>
    <div class="muted">
        Solar and electricity prices are retrieved from online APIs.<br><br>
        Household load is forecast using a trained baseline household model and mapped onto the selected planning horizon.
        Live end-user smart meter data is not yet integrated in this prototype.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

if not run_button:
    st.info("Set the planning date and system parameters in the sidebar, then run the optimization.")
    st.stop()

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

    # solar + price: real day-ahead APIs
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

    # load: baseline model from notebook-derived artifact
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

    rule_summary = summarize_dispatch(rule_dispatch_df)
    opt_summary = summarize_dispatch(optimized_dispatch_df)
    savings_eur = rule_summary["cost_eur"] - opt_summary["cost_eur"]

    rule_checks = run_balance_checks(rule_dispatch_df, PARAMS)
    opt_checks = run_balance_checks(optimized_dispatch_df, PARAMS)

except Exception as exc:
    st.error(f"Pipeline execution failed: {exc}")
    st.stop()

# =========================================================
# SOURCE STATUS
# =========================================================
def source_card(title: str, source_value: str) -> str:
    if source_value in {"forecast_api", "historical_api", "entsoe_api", "historical_simulation_forecast"}:
        css = "ok"
        label = "Connected"
    elif source_value in {"fallback_model", "not_published", "fallback_unavailable"}:
        css = "warn"
        label = "Fallback / Limited"
    else:
        css = "warn"
        label = "Unknown"

    return f"""
    <div class="glass-card">
        <div class="section-title">{title}</div>
        <div class="{css}">{label}</div>
        <div class="muted" style="margin-top:0.35rem;">{source_value}</div>
    </div>
    """

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(source_card("Solar Source", str(solar_raw["source"].iloc[0])), unsafe_allow_html=True)
with s2:
    st.markdown(source_card("Price Source", str(price_raw["source"].iloc[0])), unsafe_allow_html=True)
with s3:
    st.markdown(source_card("Load Source", str(load_raw["source"].iloc[0])), unsafe_allow_html=True)

st.markdown("")

# =========================================================
# TOP KPI CARDS
# =========================================================
self_sufficiency = 0.0
if opt_summary["pv_to_load_kwh"] + opt_summary["battery_to_load_kwh"] > 0:
    self_sufficiency = (
        (opt_summary["pv_to_load_kwh"] + opt_summary["battery_to_load_kwh"])
        / max(optimized_dispatch_df["household_load_kwh"].sum(), 1e-9)
    ) * 100.0

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.metric("Optimized Cost", f"€ {opt_summary['cost_eur']:.2f}", delta=f"€ {savings_eur:.2f} vs rule")
with summary2:
    st.metric("Grid Import", f"{opt_summary['grid_import_kwh']:.2f} kWh")
with summary3:
    st.metric("Grid Export", f"{opt_summary['grid_export_kwh']:.2f} kWh")
with summary4:
    st.metric("Final Battery SoC", f"{opt_summary['final_soc_percent']:.1f}%")

st.markdown("")

exec1, exec2, exec3 = st.columns([1.15, 1.15, 1])

with exec1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">Planning Date</div>
        <div style="font-size:1.65rem; font-weight:800; color:#f8fafc;">{planning_date}</div>
        <div class="muted" style="margin-top:0.35rem;">24-hour schedule · 96 intervals · 15-minute resolution</div>
    </div>
    """, unsafe_allow_html=True)

with exec2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">Self-Sufficiency</div>
        <div style="font-size:1.65rem; font-weight:800; color:#f8fafc;">{self_sufficiency:.1f}%</div>
        <div class="muted" style="margin-top:0.35rem;">PV + battery coverage of household demand</div>
    </div>
    """, unsafe_allow_html=True)

economic_label = "Optimized Advantage" if savings_eur > 0 else "Near Baseline"
economic_css = "ok" if savings_eur > 0 else "warn"

with exec3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">Optimization Status</div>
        <div class="{economic_css}" style="font-size:1.25rem;">{economic_label}</div>
        <div class="muted" style="margin-top:0.35rem;">Estimated saving vs rule-based dispatch: € {savings_eur:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Forecast Inputs",
    "Dispatch Comparison",
    "Optimized Schedule",
    "Decision Timeline",
    "Configuration & Assumptions",
])

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Integrated Energy Flow (Optimized)")
        overview_df = optimized_dispatch_df.set_index("utc_timestamp")[[
            "household_load_kwh",
            "pv_generation_kwh",
            "grid_to_load_kwh",
            "battery_to_load_kwh",
            "pv_to_battery_kwh",
            "grid_to_battery_kwh",
            "export_to_grid_kwh",
        ]]
        st.area_chart(overview_df)

    with right:
        st.subheader("Battery SoC")
        st.line_chart(optimized_dispatch_df.set_index("utc_timestamp")[["soc_percent"]])

    bottom1, bottom2 = st.columns([1, 1])

    with bottom1:
        st.subheader("Cost Comparison")
        comparison_df = pd.DataFrame({
            "Method": ["Rule-Based", "Optimized LP"],
            "Cost EUR": [rule_summary["cost_eur"], opt_summary["cost_eur"]],
            "Grid Import kWh": [rule_summary["grid_import_kwh"], opt_summary["grid_import_kwh"]],
            "Grid Export kWh": [rule_summary["grid_export_kwh"], opt_summary["grid_export_kwh"]],
        }).set_index("Method")
        st.bar_chart(comparison_df[["Cost EUR"]])

    with bottom2:
        st.subheader("Recommended Operating Strategy")
        for rec in build_recommendation_text(opt_summary, rule_summary):
            st.markdown(f"- {rec}")

# =========================================================
# TAB 2 - FORECAST INPUTS
# =========================================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Solar Generation Forecast")
        solar_plot = (
            dispatch_input_df[["utc_timestamp", "pv_generation_kwh"]]
            .set_index("utc_timestamp")
        )
        st.line_chart(solar_plot)

        st.markdown("#### Household Load Forecast")
        load_plot = (
            dispatch_input_df[["utc_timestamp", "household_load_kwh"]]
            .set_index("utc_timestamp")
        )
        st.line_chart(load_plot)

    with c2:
        st.markdown("#### Electricity Price Forecast")
        price_plot = (
            dispatch_input_df[["utc_timestamp", "energy_price_buy_cent_kwh"]]
            .set_index("utc_timestamp")
        )
        st.line_chart(price_plot)

        st.markdown("#### PV vs Load")
        pv_vs_load = (
            dispatch_input_df[["utc_timestamp", "pv_generation_kwh", "household_load_kwh"]]
            .set_index("utc_timestamp")
        )
        st.line_chart(pv_vs_load)

    st.markdown("#### Dispatch Input Table")
    st.dataframe(dispatch_input_df, use_container_width=True)

# =========================================================
# TAB 3 - DISPATCH COMPARISON
# =========================================================
with tab3:
    comp1, comp2 = st.columns(2)

    with comp1:
        st.subheader("Rule-Based Dispatch")
        st.line_chart(
            rule_dispatch_df.set_index("utc_timestamp")[[
                "grid_to_load_kwh",
                "battery_to_load_kwh",
                "pv_to_battery_kwh",
                "grid_to_battery_kwh",
                "export_to_grid_kwh",
            ]]
        )

    with comp2:
        st.subheader("Optimized LP Dispatch")
        st.line_chart(
            optimized_dispatch_df.set_index("utc_timestamp")[[
                "grid_to_load_kwh",
                "battery_to_load_kwh",
                "pv_to_battery_kwh",
                "grid_to_battery_kwh",
                "export_to_grid_kwh",
            ]]
        )

    st.markdown("#### Method Comparison")
    compare_table = pd.DataFrame({
        "Metric": [
            "Total Cost (EUR)",
            "Grid Import (kWh)",
            "Grid Export (kWh)",
            "PV to Battery (kWh)",
            "Battery to Load (kWh)",
            "Final SoC (kWh)",
        ],
        "Rule-Based": [
            round(rule_summary["cost_eur"], 3),
            round(rule_summary["grid_import_kwh"], 3),
            round(rule_summary["grid_export_kwh"], 3),
            round(rule_summary["pv_to_battery_kwh"], 3),
            round(rule_summary["battery_to_load_kwh"], 3),
            round(rule_summary["final_soc_kwh"], 3),
        ],
        "Optimized LP": [
            round(opt_summary["cost_eur"], 3),
            round(opt_summary["grid_import_kwh"], 3),
            round(opt_summary["grid_export_kwh"], 3),
            round(opt_summary["pv_to_battery_kwh"], 3),
            round(opt_summary["battery_to_load_kwh"], 3),
            round(opt_summary["final_soc_kwh"], 3),
        ],
    })
    compare_table["Improvement"] = compare_table["Rule-Based"] - compare_table["Optimized LP"]
    st.dataframe(compare_table, use_container_width=True)

# =========================================================
# TAB 4 - OPTIMIZED SCHEDULE
# =========================================================
with tab4:
    st.subheader("Optimized Dispatch Schedule")

    sched1, sched2 = st.columns([1.25, 1])

    with sched1:
        st.line_chart(
            optimized_dispatch_df.set_index("utc_timestamp")[[
                "household_load_kwh",
                "pv_generation_kwh",
                "grid_to_load_kwh",
                "battery_to_load_kwh",
                "export_to_grid_kwh",
            ]]
        )

    with sched2:
        st.line_chart(
            optimized_dispatch_df.set_index("utc_timestamp")[[
                "soc_percent",
                "energy_price_buy_cent_kwh",
            ]]
        )

    display_schedule = optimized_dispatch_df.copy()
    display_schedule["utc_timestamp"] = display_schedule["utc_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_schedule, use_container_width=True)

    export_df = optimized_dispatch_df.copy()
    export_df["utc_timestamp"] = export_df["utc_timestamp"].astype(str)
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Optimized Schedule as CSV",
        data=csv_data,
        file_name=f"hems_optimized_dispatch_{planning_date}.csv",
        mime="text/csv",
    )

# =========================================================
# TAB 5 - DECISION TIMELINE
# =========================================================
with tab5:
    st.subheader("Interval-Level Decision Timeline")

    decision_df = optimized_dispatch_df.copy()

    def classify_action(row):
        labels = []
        if row["pv_to_load_kwh"] > 1e-6:
            labels.append("PV→Load")
        if row["pv_to_battery_kwh"] > 1e-6:
            labels.append("PV→Battery")
        if row["battery_to_load_kwh"] > 1e-6:
            labels.append("Battery→Load")
        if row["grid_to_load_kwh"] > 1e-6:
            labels.append("Grid→Load")
        if row["grid_to_battery_kwh"] > 1e-6:
            labels.append("Grid→Battery")
        if row["export_to_grid_kwh"] > 1e-6:
            labels.append("Export")
        return " | ".join(labels) if labels else "Idle"

    decision_df["action_label"] = decision_df.apply(classify_action, axis=1)

    decision_counts = decision_df["action_label"].value_counts().rename_axis("Action").reset_index(name="Count")
    st.bar_chart(decision_counts.set_index("Action"))

    timeline_view = decision_df[[
        "utc_timestamp",
        "action_label",
        "pv_generation_kwh",
        "household_load_kwh",
        "energy_price_buy_cent_kwh",
        "grid_to_load_kwh",
        "battery_to_load_kwh",
        "pv_to_battery_kwh",
        "grid_to_battery_kwh",
        "export_to_grid_kwh",
        "soc_kwh",
        "interval_cost_cent",
    ]].copy()

    timeline_view["utc_timestamp"] = timeline_view["utc_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(timeline_view, use_container_width=True)

# =========================================================
# TAB 6 - CONFIGURATION
# =========================================================
with tab6:
    conf1, conf2 = st.columns(2)

    with conf1:
        st.subheader("Simulation Configuration")
        config_df = pd.DataFrame({
            "Parameter": [
                "Planning Date",
                "Latitude",
                "Longitude",
                "PV Capacity (kWp)",
                "Panel Tilt (°)",
                "Panel Azimuth (°)",
                "Yield Factor",
                "Battery Capacity (kWh)",
                "Initial SoC (kWh)",
                "Minimum Reserve (kWh)",
                "Maximum Effective SoC (kWh)",
                "Max Charge Power (kW)",
                "Max Discharge Power (kW)",
                "Charge Efficiency",
                "Discharge Efficiency",
                "Default Sell Price (cent/kWh)",
                "Allow Grid Charging",
                "Grid Charge Threshold (cent/kWh)",
                "Cycle Penalty (cent/kWh)",
                "Enforce Solar First in LP",
                "Terminal SoC Value (cent/kWh)",
                "Minimum End SoC (kWh)",
                "Feature Dataset Path",
                "Load Model Path",
                "Load Baseline Timestamp",
            ],
            "Value": [
                str(planning_date),
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
                str(last_load_ts),
            ],
        })
        st.dataframe(config_df, use_container_width=True)

    with conf2:
        st.subheader("Validation & Assumptions")
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
        st.dataframe(validation_df, use_container_width=True)

        st.markdown("""
        <div class="footer-box">
            <b>Assumption Note</b><br><br>
            Solar forecast is generated from the project solar utility using online weather data.<br><br>
            Electricity price forecast is generated from the project price utility using online market data.<br><br>
            Household load forecast is generated from the trained baseline household model and mapped onto the selected planning date.
            It is not yet personalized using live smart-meter measurements from the end user.
        </div>
        """, unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class="footer-box">
    <b>HEMS Dashboard</b><br>
    A professional decision-support interface for residential energy forecasting, battery dispatch scheduling, and daily cost optimization.
</div>
""", unsafe_allow_html=True)