import json
import tempfile
from pathlib import Path
from datetime import timedelta, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    page_title="Forecast Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
        max-width: 1500px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.05;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 900px;
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
        font-size: 1.7rem;
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
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 0.6rem;
    }

    .muted {
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.7;
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

    .status-ok {
        color: #16a34a;
        font-weight: 700;
    }

    .status-warn {
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
        font-weight: 750;
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

    required_keys = ["lat", "lon", "kwp", "tilt", "azimuth", "yield_factor"]
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


def build_forecast_input_table(
    solar_raw: pd.DataFrame,
    price_raw: pd.DataFrame,
    load_raw: pd.DataFrame,
) -> pd.DataFrame:
    solar = solar_raw.copy()
    solar["time"] = pd.to_datetime(solar["time"], utc=True)
    solar = solar.rename(columns={
        "predicted_kwh": "pv_generation_kwh",
        "predicted_kw": "pv_generation_kw",
        "source": "solar_source",
    })
    solar = solar[["time", "pv_generation_kwh", "pv_generation_kw", "solar_source"]].copy()

    price = price_raw.copy()
    price["time"] = pd.to_datetime(price["time"], utc=True)
    price = price.rename(columns={
        "price_cent_kwh": "energy_price_buy_cent_kwh",
        "price_eur_mwh": "energy_price_buy_eur_mwh",
        "source": "price_source",
    })
    price = price[["time", "energy_price_buy_cent_kwh", "energy_price_buy_eur_mwh", "price_source"]].copy()

    load = load_raw.copy()
    load["time"] = pd.to_datetime(load["time"], utc=True)
    load = load.rename(columns={
        "predicted_kwh": "household_load_kwh",
        "predicted_kw": "household_load_kw",
        "source": "load_source",
    })

    keep_cols = ["time", "household_load_kwh", "household_load_kw"]
    if "household" in load.columns:
        keep_cols.append("household")
    if "load_source" in load.columns:
        keep_cols.append("load_source")
    load = load[keep_cols].copy()

    df = (
        solar.merge(price, on="time", how="inner")
             .merge(load, on="time", how="inner")
             .sort_values("time")
             .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Merged forecast dataframe is empty. Time alignment across sources failed.")

    df["net_load_kwh"] = df["household_load_kwh"] - df["pv_generation_kwh"]
    df["solar_surplus_kwh"] = np.maximum(df["pv_generation_kwh"] - df["household_load_kwh"], 0.0)
    df["residual_demand_kwh"] = np.maximum(df["household_load_kwh"] - df["pv_generation_kwh"], 0.0)

    return df


def source_status_html(title: str, source_value: str) -> str:
    if source_value in {"forecast_api", "historical_api", "entsoe_api", "historical_simulation_forecast"}:
        status_css = "status-ok"
        status_label = "Connected"
    elif source_value in {"fallback_model", "not_published", "fallback_unavailable"}:
        status_css = "status-warn"
        status_label = "Fallback / Limited"
    else:
        status_css = "status-warn"
        status_label = "Unknown"

    return f"""
    <div class="glass-card">
        <div class="section-title">{title}</div>
        <div class="{status_css}">{status_label}</div>
        <div class="muted" style="margin-top:0.35rem;">{source_value}</div>
    </div>
    """


def hour_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def compute_forecast_summary(df: pd.DataFrame) -> dict:
    peak_pv_idx = df["pv_generation_kwh"].idxmax()
    peak_load_idx = df["household_load_kwh"].idxmax()
    cheapest_idx = df["energy_price_buy_cent_kwh"].idxmin()
    expensive_idx = df["energy_price_buy_cent_kwh"].idxmax()

    total_pv = float(df["pv_generation_kwh"].sum())
    total_load = float(df["household_load_kwh"].sum())
    direct_solar_coverage = min(total_pv, total_load) / max(total_load, 1e-9) * 100.0

    return {
        "total_pv_kwh": total_pv,
        "total_load_kwh": total_load,
        "avg_price_cent_kwh": float(df["energy_price_buy_cent_kwh"].mean()),
        "max_price_cent_kwh": float(df["energy_price_buy_cent_kwh"].max()),
        "min_price_cent_kwh": float(df["energy_price_buy_cent_kwh"].min()),
        "peak_pv_value_kwh": float(df.loc[peak_pv_idx, "pv_generation_kwh"]),
        "peak_pv_time": df.loc[peak_pv_idx, "time"],
        "peak_load_value_kwh": float(df.loc[peak_load_idx, "household_load_kwh"]),
        "peak_load_time": df.loc[peak_load_idx, "time"],
        "cheapest_time": df.loc[cheapest_idx, "time"],
        "expensive_time": df.loc[expensive_idx, "time"],
        "direct_solar_coverage_pct": float(direct_solar_coverage),
        "surplus_intervals": int((df["solar_surplus_kwh"] > 1e-6).sum()),
        "high_price_intervals": int(
            (df["energy_price_buy_cent_kwh"] >= df["energy_price_buy_cent_kwh"].quantile(0.80)).sum()
        ),
    }


def build_layout_base(fig: go.Figure, title: str, yaxis_title: str, start_time, end_time) -> go.Figure:
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=70),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20)),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            title=None,
        ),
        xaxis=dict(
            title=None,
            range=[start_time, end_time],
            tickformat="%H:%M",
            dtick=60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.20)",
            zeroline=False,
        ),
    )
    return fig


def build_solar_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("time")
    start_time = plot_df["time"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    peak_idx = plot_df["pv_generation_kwh"].idxmax()
    peak_row = plot_df.loc[peak_idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["pv_generation_kwh"],
            mode="lines",
            name="Solar Output",
            line=dict(color="#f59e0b", width=3.5),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.22)",
            hovertemplate=(
                "<b>Solar Generation</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_row["time"]],
            y=[peak_row["pv_generation_kwh"]],
            mode="markers+text",
            name="Peak Solar",
            text=[f"Peak {peak_row['pv_generation_kwh']:.2f}"],
            textposition="top center",
            marker=dict(size=10, color="#f59e0b"),
            hovertemplate=(
                "<b>Peak Solar</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vrect(
        x0=peak_row["time"] - pd.Timedelta(hours=1),
        x1=peak_row["time"] + pd.Timedelta(hours=1),
        fillcolor="rgba(245,158,11,0.08)",
        line_width=0,
        layer="below",
    )

    return build_layout_base(fig, "Solar Generation Profile", "Energy (kWh)", start_time, end_time)


def build_price_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("time")
    start_time = plot_df["time"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    q20 = plot_df["energy_price_buy_cent_kwh"].quantile(0.20)
    q80 = plot_df["energy_price_buy_cent_kwh"].quantile(0.80)

    cheap_df = plot_df[plot_df["energy_price_buy_cent_kwh"] <= q20]
    expensive_df = plot_df[plot_df["energy_price_buy_cent_kwh"] >= q80]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["energy_price_buy_cent_kwh"],
            mode="lines",
            name="Buy Price",
            line=dict(color="#2563eb", width=3.2),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
            hovertemplate=(
                "<b>Electricity Price</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Price: %{y:.3f} cent/kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cheap_df["time"],
            y=cheap_df["energy_price_buy_cent_kwh"],
            mode="markers",
            name="Cheap Window",
            marker=dict(size=8, color="#16a34a"),
            hovertemplate=(
                "<b>Cheap Window</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Price: %{y:.3f} cent/kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=expensive_df["time"],
            y=expensive_df["energy_price_buy_cent_kwh"],
            mode="markers",
            name="Expensive Window",
            marker=dict(size=8, color="#dc2626"),
            hovertemplate=(
                "<b>Expensive Window</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Price: %{y:.3f} cent/kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=q20,
        line_dash="dot",
        line_color="rgba(22,163,74,0.7)",
        annotation_text="Low-price threshold",
        annotation_position="bottom right",
    )

    fig.add_hline(
        y=q80,
        line_dash="dot",
        line_color="rgba(220,38,38,0.7)",
        annotation_text="High-price threshold",
        annotation_position="top right",
    )

    return build_layout_base(fig, "Electricity Price Curve", "Price (cent/kWh)", start_time, end_time)


def build_load_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("time")
    start_time = plot_df["time"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    avg_load = float(plot_df["household_load_kwh"].mean())
    peak_idx = plot_df["household_load_kwh"].idxmax()
    peak_row = plot_df.loc[peak_idx]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["household_load_kwh"],
            mode="lines",
            name="Household Load",
            line=dict(color="#7c3aed", width=3.2),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.14)",
            hovertemplate=(
                "<b>Household Load</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Load: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=avg_load,
        line_dash="dash",
        line_color="rgba(71,85,105,0.8)",
        annotation_text=f"Average {avg_load:.2f} kWh",
        annotation_position="top right",
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_row["time"]],
            y=[peak_row["household_load_kwh"]],
            mode="markers+text",
            name="Peak Load",
            text=[f"Peak {peak_row['household_load_kwh']:.2f}"],
            textposition="top center",
            marker=dict(size=10, color="#7c3aed"),
            hovertemplate=(
                "<b>Peak Load</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Load: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    return build_layout_base(fig, "Household Load Curve", "Energy (kWh)", start_time, end_time)


def build_balance_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy().sort_values("time")
    start_time = plot_df["time"].min().floor("D")
    end_time = start_time + pd.Timedelta(hours=24)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["pv_generation_kwh"],
            mode="lines",
            name="PV Generation",
            line=dict(color="#f59e0b", width=3),
            hovertemplate=(
                "<b>PV Generation</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["time"],
            y=plot_df["household_load_kwh"],
            mode="lines",
            name="Household Load",
            line=dict(color="#2563eb", width=3),
            hovertemplate=(
                "<b>Household Load</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=plot_df["time"],
            y=plot_df["net_load_kwh"],
            name="Net Load",
            marker_color=np.where(plot_df["net_load_kwh"] >= 0, "rgba(37,99,235,0.24)", "rgba(34,197,94,0.24)"),
            hovertemplate=(
                "<b>Net Load</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Net: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_width=1.2,
        line_color="rgba(71,85,105,0.7)",
    )

    fig = build_layout_base(fig, "PV vs Load and Net Demand", "Energy (kWh)", start_time, end_time)
    fig.update_layout(barmode="relative")
    return fig


def build_insight_cards(summary: dict) -> list[tuple[str, str]]:
    return [
        (
            "Solar Opportunity Window",
            f"Peak solar generation is expected around {hour_label(summary['peak_pv_time'])}, reaching "
            f"{summary['peak_pv_value_kwh']:.2f} kWh. This is the strongest local-generation window of the day."
        ),
        (
            "Market Price Signal",
            f"The cheapest energy window is around {hour_label(summary['cheapest_time'])}, while the most expensive period is near "
            f"{hour_label(summary['expensive_time'])}. This spread matters for charge/discharge timing."
        ),
        (
            "Demand Pressure",
            f"Peak household demand occurs around {hour_label(summary['peak_load_time'])} at "
            f"{summary['peak_load_value_kwh']:.2f} kWh. This is the interval where imported energy risk is highest if PV is weak."
        ),
        (
            "Solar Coverage Potential",
            f"Direct daily solar-to-demand coverage is approximately {summary['direct_solar_coverage_pct']:.1f}% "
            f"before battery shifting effects are considered."
        ),
    ]


# =========================================================
# LOAD CONFIG
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


# =========================================================
# SIDEBAR
# =========================================================
tomorrow_default = pd.Timestamp.now(tz="UTC").date() + timedelta(days=1)

with st.sidebar:
    planning_date = st.date_input("Planning Date", value=tomorrow_default)
    st.caption("Forecast page runs the data layer only. No battery optimization is executed here.")


# =========================================================
# DATA PIPELINE
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

    forecast_df = build_forecast_input_table(
        solar_raw=solar_raw,
        price_raw=price_raw,
        load_raw=load_raw,
    )

    summary = compute_forecast_summary(forecast_df)

except Exception as exc:
    st.error(f"Forecast pipeline execution failed: {exc}")
    st.stop()


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">Forecast Center</div>
    <div class="hero-subtitle">
        A dedicated day-ahead forecast view for solar generation, electricity prices, and household demand.
        This page isolates the forecast layer from the dispatch layer, so you can see whether tomorrow’s operating
        conditions are driven by solar availability, demand shape, or market price volatility.
    </div>
    <div style="margin-top:0.65rem;">
        <span class="pill">Day-Ahead Outlook</span>
        <span class="pill">24-Hour Horizon</span>
        <span class="pill">15-Minute Resolution</span>
        <span class="pill">Interactive Visual Analytics</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SOURCE STATUS
# =========================================================
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(source_status_html("Solar Source", str(forecast_df["solar_source"].iloc[0])), unsafe_allow_html=True)
with s2:
    st.markdown(source_status_html("Price Source", str(forecast_df["price_source"].iloc[0])), unsafe_allow_html=True)
with s3:
    load_source_value = str(forecast_df["load_source"].iloc[0]) if "load_source" in forecast_df.columns else "model_forecast"
    st.markdown(source_status_html("Load Source", load_source_value), unsafe_allow_html=True)

st.markdown("")


# =========================================================
# KPI ROW
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Total Solar Forecast</div>
    <div class="kpi-value">{summary['total_pv_kwh']:.2f} kWh</div>
    <div class="kpi-sub">Peak around {hour_label(summary['peak_pv_time'])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Total Household Demand</div>
    <div class="kpi-value">{summary['total_load_kwh']:.2f} kWh</div>
    <div class="kpi-sub">Peak around {hour_label(summary['peak_load_time'])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Average Electricity Price</div>
    <div class="kpi-value">{summary['avg_price_cent_kwh']:.2f} c/kWh</div>
    <div class="kpi-sub">Range: {summary['min_price_cent_kwh']:.2f} → {summary['max_price_cent_kwh']:.2f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">Direct Solar Coverage Potential</div>
    <div class="kpi-value">{summary['direct_solar_coverage_pct']:.1f}%</div>
    <div class="kpi-sub">Before battery shifting and control decisions</div>
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
    solar_fig = build_solar_chart(forecast_df)
    st.plotly_chart(solar_fig, use_container_width=True)

with c2:
    price_fig = build_price_chart(forecast_df)
    st.plotly_chart(price_fig, use_container_width=True)

st.markdown("")

c3, c4 = st.columns(2)

with c3:
    load_fig = build_load_chart(forecast_df)
    st.plotly_chart(load_fig, use_container_width=True)

with c4:
    balance_fig = build_balance_chart(forecast_df)
    st.plotly_chart(balance_fig, use_container_width=True)

st.markdown("")


# =========================================================
# INSIGHTS
# =========================================================
st.markdown(
    """
<div class="glass-card">
    <div class="section-title">Forecast Interpretation</div>
    <div class="muted">
        The charts below are intended to support operational reasoning before optimization.
        They help answer three questions: when solar availability is strongest, when market prices are most favorable,
        and when household demand is most likely to create import pressure.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

insights = build_insight_cards(summary)
i1, i2 = st.columns(2)
i3, i4 = st.columns(2)

cards = [i1, i2, i3, i4]
for col, (title, text) in zip(cards, insights):
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
# OPTIONAL DATA PREVIEW
# =========================================================
with st.expander("Show processed forecast dataset"):
    preview_df = forecast_df.copy()
    preview_df["time"] = preview_df["time"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(preview_df, use_container_width=True, height=320)

