import json
import tempfile
from pathlib import Path
from datetime import timedelta, date

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
    page_title="Optimized Schedule",
    page_icon="🧭",
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

    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 0.3rem 0.35rem 0.1rem 0.35rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 0.85rem 1rem;
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


def build_base_layout(fig: go.Figure, title: str, y_title: str, start_time, end_time) -> go.Figure:
    fig.update_layout(
        height=410,
        margin=dict(l=20, r=20, t=60, b=72),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20)),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title=None,
        ),
        xaxis=dict(
            range=[start_time, end_time],
            tickformat="%H:%M",
            dtick=60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
    )
    return fig


def build_energy_stack_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("utc_timestamp")
    start_time = plot_df["utc_timestamp"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"], y=plot_df["household_load_kwh"],
        mode="lines", name="Household Load",
        line=dict(width=3, color="#f59e0b")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"], y=plot_df["pv_generation_kwh"],
        mode="lines", name="PV Generation",
        line=dict(width=3, color="#10b981"),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.14)"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"], y=plot_df["battery_to_load_kwh"],
        mode="lines", name="Battery → Load",
        line=dict(width=2.8, color="#2563eb")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"], y=plot_df["grid_to_load_kwh"],
        mode="lines", name="Grid → Load",
        line=dict(width=2.8, color="#dc2626")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"], y=plot_df["export_to_grid_kwh"],
        mode="lines", name="Export → Grid",
        line=dict(width=2.5, color="#7c3aed")
    ))

    return build_base_layout(fig, "Optimized Energy Flow Schedule", "Energy (kWh)", start_time, end_time)


def build_soc_price_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("utc_timestamp")
    start_time = plot_df["utc_timestamp"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"],
        y=plot_df["soc_percent"],
        mode="lines",
        name="Battery SoC (%)",
        line=dict(width=3.5, color="#2563eb"),
        yaxis="y1",
        hovertemplate="<b>Battery SoC</b><br>Time: %{x|%H:%M}<br>SoC: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"],
        y=plot_df["energy_price_buy_cent_kwh"],
        mode="lines",
        name="Buy Price (cent/kWh)",
        line=dict(width=3, dash="dash", color="#0f172a"),
        yaxis="y2",
        hovertemplate="<b>Buy Price</b><br>Time: %{x|%H:%M}<br>Price: %{y:.2f} cent/kWh<extra></extra>",
    ))

    fig.update_layout(
        height=410,
        margin=dict(l=20, r=20, t=60, b=72),
        title=dict(text="Battery SoC vs Electricity Price", x=0.02, xanchor="left", font=dict(size=20)),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title=None,
        ),
        xaxis=dict(
            range=[start_time, end_time],
            tickformat="%H:%M",
            dtick=60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis=dict(
            title="State of Charge (%)",
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Price (cent/kWh)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def build_charge_discharge_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("utc_timestamp")
    start_time = plot_df["utc_timestamp"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_df["utc_timestamp"],
        y=plot_df["pv_to_battery_kwh"],
        name="PV → Battery",
        marker_color="rgba(16,185,129,0.70)",
        hovertemplate="<b>PV → Battery</b><br>Time: %{x|%H:%M}<br>Energy: %{y:.3f} kWh<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=plot_df["utc_timestamp"],
        y=plot_df["grid_to_battery_kwh"],
        name="Grid → Battery",
        marker_color="rgba(239,68,68,0.70)",
        hovertemplate="<b>Grid → Battery</b><br>Time: %{x|%H:%M}<br>Energy: %{y:.3f} kWh<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=plot_df["utc_timestamp"],
        y=-plot_df["battery_to_load_kwh"],
        name="Battery → Load",
        marker_color="rgba(37,99,235,0.75)",
        hovertemplate="<b>Battery → Load</b><br>Time: %{x|%H:%M}<br>Energy: %{customdata:.3f} kWh<extra></extra>",
        customdata=plot_df["battery_to_load_kwh"],
    ))

    fig = build_base_layout(fig, "Battery Charging and Discharging Pattern", "Energy (kWh)", start_time, end_time)
    fig.update_layout(barmode="relative")
    return fig


def build_interval_cost_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("utc_timestamp")
    start_time = plot_df["utc_timestamp"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_df["utc_timestamp"],
        y=plot_df["interval_cost_cent"] / 100.0,
        name="Interval Cost (€)",
        hovertemplate="<b>Interval Cost</b><br>Time: %{x|%H:%M}<br>Cost: €%{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["utc_timestamp"],
        y=plot_df["cumulative_cost_cent"] / 100.0,
        mode="lines",
        name="Cumulative Cost (€)",
        line=dict(width=3.5, color="#16a34a"),
        yaxis="y2",
        hovertemplate="<b>Cumulative Cost</b><br>Time: %{x|%H:%M}<br>Cost: €%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=410,
        margin=dict(l=20, r=20, t=60, b=72),
        title=dict(text="Interval Cost and Cumulative Daily Cost", x=0.02, xanchor="left", font=dict(size=20)),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title=None,
        ),
        xaxis=dict(
            range=[start_time, end_time],
            tickformat="%H:%M",
            dtick=60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Interval Cost (€)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Cumulative Cost (€)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def build_action_timeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    out["action_label"] = out.apply(classify_action, axis=1)
    return out


def build_insights(summary: dict, df: pd.DataFrame) -> list[tuple[str, str]]:
    peak_export_idx = df["export_to_grid_kwh"].idxmax()
    peak_discharge_idx = df["battery_to_load_kwh"].idxmax()
    peak_charge_idx = (df["pv_to_battery_kwh"] + df["grid_to_battery_kwh"]).idxmax()

    peak_export_time = df.loc[peak_export_idx, "utc_timestamp"].strftime("%H:%M")
    peak_discharge_time = df.loc[peak_discharge_idx, "utc_timestamp"].strftime("%H:%M")
    peak_charge_time = df.loc[peak_charge_idx, "utc_timestamp"].strftime("%H:%M")

    return [
        (
            "Economic Schedule Outcome",
            f"The optimized schedule ends with a daily operating cost of €{summary['cost_eur']:.2f}. "
            f"This is the economically selected dispatch under the current battery constraints and forecast inputs."
        ),
        (
            "Battery Operating Logic",
            f"The battery ends the day at {summary['final_soc_percent']:.1f}% SoC, which shows how aggressively the optimizer uses stored energy. "
            f"Peak battery discharge appears around {peak_discharge_time}."
        ),
        (
            "Charging Window Interpretation",
            f"The strongest battery charging activity appears around {peak_charge_time}. "
            f"This reflects where the optimizer sees the highest future value in stored energy."
        ),
        (
            "Export Behavior",
            f"Peak export occurs around {peak_export_time}. "
            f"That typically indicates either solar surplus beyond load and charging capacity, or low marginal battery value at that interval."
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
    st.caption("This page shows only the optimized LP dispatch schedule and its operational interpretation.")


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
    optimized_dispatch_df = run_lp_dispatch(prepared_forecast_df, PARAMS, initial_soc_kwh)
    opt_summary = summarize_dispatch(optimized_dispatch_df)
    action_df = build_action_timeline(optimized_dispatch_df)

except Exception as exc:
    st.error(f"Optimized schedule pipeline failed: {exc}")
    st.stop()


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">Optimized Schedule Center</div>
    <div class="hero-subtitle">
        This page presents the final LP-optimized dispatch plan for the selected day. It is not a raw dump of interval values.
        It is an operational view of how the controller allocates PV, battery, grid import, charging, discharge, export, and cost progression across the full day.
    </div>
    <div style="margin-top:0.65rem;">
        <span class="pill">LP-Optimized Dispatch</span>
        <span class="pill">24-Hour Timeline</span>
        <span class="pill">Battery + Price Coupling</span>
        <span class="pill">Action-Level Interpretation</span>
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
    <div class="kpi-label">Optimized Daily Cost</div>
    <div class="kpi-value">€ {opt_summary['cost_eur']:.2f}</div>
    <div class="kpi-sub">Full 24-hour optimized operating cost</div>
</div>
""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Grid Import</div>
    <div class="kpi-value">{opt_summary['grid_import_kwh']:.2f} kWh</div>
    <div class="kpi-sub">Total imported energy across the day</div>
</div>
""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Grid Export</div>
    <div class="kpi-value">{opt_summary['grid_export_kwh']:.2f} kWh</div>
    <div class="kpi-sub">Total exported solar surplus</div>
</div>
""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
<div class="kpi-card">
    <div class="kpi-label">Final Battery SoC</div>
    <div class="kpi-value">{opt_summary['final_soc_percent']:.1f}%</div>
    <div class="kpi-sub">{opt_summary['final_soc_kwh']:.2f} kWh remaining at end-of-day</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")


# =========================================================
# SUMMARY
# =========================================================
status_text = "Economically Coherent Dispatch" if opt_summary["cost_eur"] <= 0 or opt_summary["grid_import_kwh"] < opt_summary["grid_export_kwh"] else "Optimized Under Current Constraints"
status_css = "ok" if "Coherent" in status_text else "warn"

st.markdown(
    f"""
<div class="glass-card">
    <div class="section-title">Schedule Interpretation</div>
    <div class="{status_css}" style="font-size:1.18rem; margin-bottom:0.5rem;">{status_text}</div>
    <div class="muted">
        The optimized schedule should be read as a control policy over time, not just a final score.
        A strong schedule reduces expensive imports, uses the battery when marginal value is high,
        and accepts export only when local use or storage is no longer economically preferable.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")


# =========================================================
# CHARTS
# =========================================================
c1, c2 = st.columns(2)

with c1:
    flow_fig = build_energy_stack_chart(optimized_dispatch_df)
    st.plotly_chart(flow_fig, use_container_width=True)

with c2:
    soc_price_fig = build_soc_price_chart(optimized_dispatch_df)
    st.plotly_chart(soc_price_fig, use_container_width=True)

st.markdown("")

c3, c4 = st.columns(2)

with c3:
    battery_fig = build_charge_discharge_chart(optimized_dispatch_df)
    st.plotly_chart(battery_fig, use_container_width=True)

with c4:
    cost_fig = build_interval_cost_chart(optimized_dispatch_df)
    st.plotly_chart(cost_fig, use_container_width=True)

st.markdown("")


# =========================================================
# ACTION SUMMARY
# =========================================================
st.subheader("Optimized Action Profile")

action_counts = (
    action_df["action_label"]
    .value_counts()
    .rename_axis("Action")
    .reset_index(name="Count")
)

bar_fig = go.Figure()
bar_fig.add_trace(
    go.Bar(
        x=action_counts["Action"],
        y=action_counts["Count"],
        text=action_counts["Count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Intervals: %{y}<extra></extra>",
    )
)
bar_fig.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=40, b=40),
    title=dict(text="Action Frequency Across 96 Intervals", x=0.02, xanchor="left", font=dict(size=20)),
    plot_bgcolor="rgba(255,255,255,0)",
    paper_bgcolor="rgba(255,255,255,0)",
    xaxis=dict(showgrid=False),
    yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(148,163,184,0.20)", zeroline=False),
)

st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("")


# =========================================================
# INTERPRETATION CARDS
# =========================================================
st.markdown(
    """
<div class="glass-card">
    <div class="section-title">Operational Interpretation</div>
    <div class="muted">
        These diagnostics explain how the optimized controller actually behaves through the day:
        when it relies on PV, when it charges or discharges the battery, when it imports from the grid,
        and how those choices map into cost accumulation and terminal battery state.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

insights = build_insights(opt_summary, optimized_dispatch_df)
i1, i2 = st.columns(2)
i3, i4 = st.columns(2)

for col, (title, text) in zip([i1, i2, i3, i4], insights):
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

st.markdown("")


# =========================================================
# CSV EXPORT ONLY
# =========================================================
export_df = optimized_dispatch_df.copy()
export_df["utc_timestamp"] = export_df["utc_timestamp"].astype(str)
csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Optimized Schedule as CSV",
    data=csv_data,
    file_name=f"hems_optimized_schedule_{planning_date}.csv",
    mime="text/csv",
)