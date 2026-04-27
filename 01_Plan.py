import json
import base64
from pathlib import Path
from datetime import date
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Plan",
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
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    max-width: 1500px;
    }

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
    padding: 1.6rem 1.8rem;
    border-radius: 26px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 18px 42px rgba(0,0,0,0.18);
    margin-top: 1.2rem;
    margin-bottom: 1rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .brand-logo {
        width: 76px;
        height: 76px;
        object-fit: contain;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        padding: 0.35rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        flex-shrink: 0;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 850;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
        max-width: 980px;
    }

    .hero-pill {
        display: inline-block;
        padding: 0.46rem 0.78rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #e2e8f0;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-top: 0.65rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .glass-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.08);
        height: 100%;
    }

    .kpi-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        height: 100%;
    }

    .kpi-label {
        color: #475569;
        font-size: 0.92rem;
        margin-bottom: 0.28rem;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 1.75rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .kpi-sub {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
        color: #0f172a;
    }

    .muted {
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.7;
    }

    .big-decision {
        font-size: 1.35rem;
        font-weight: 850;
        color: #0f172a;
        line-height: 1.35;
        margin-bottom: 0.45rem;
    }

    .decision-support {
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.7;
    }

    .status-good {
        color: #16a34a;
        font-weight: 800;
    }

    .status-warn {
        color: #d97706;
        font-weight: 800;
    }

    .status-bad {
        color: #dc2626;
        font-weight: 800;
    }

    .mini-label {
        color: #64748b;
        font-size: 0.84rem;
        margin-bottom: 0.18rem;
    }

    .mini-value {
        color: #0f172a;
        font-size: 1.08rem;
        font-weight: 760;
    }

    .action-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.07);
        height: 100%;
    }

    .action-title {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }

    .action-main {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        line-height: 1.35;
    }

    .action-sub {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .timeline-wrap {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
    }

    .timeline-item {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.07);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }

    .timeline-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 0.3rem;
    }

    .timeline-time {
        color: #0f172a;
        font-size: 0.93rem;
        font-weight: 800;
    }

    .timeline-action {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0f172a;
    }

    .timeline-why {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        border: 1px solid rgba(15,23,42,0.08);
    }

    .badge-buy { background: rgba(59,130,246,0.10); color: #1d4ed8; }
    .badge-battery { background: rgba(245,158,11,0.12); color: #b45309; }
    .badge-store { background: rgba(16,185,129,0.12); color: #047857; }
    .badge-export { background: rgba(139,92,246,0.12); color: #6d28d9; }
    .badge-hold { background: rgba(71,85,105,0.12); color: #334155; }

    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 0.28rem 0.35rem 0.12rem 0.35rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.08);
    }

    table {
        font-size: 0.93rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "logo.png"

USER_CONFIG_PATH = Path("user_config.json")
SYSTEM_CONFIG_PATH = Path("system_config.json")

DATA_DIR = Path("data")
RUNTIME_DIR = DATA_DIR / "runtime"

AGGREGATED_TABLE_PATH = RUNTIME_DIR / "aggregated_table.csv"
DISPATCH_TABLE_PATH = RUNTIME_DIR / "dispatch_table.csv"


# =========================================================
# HELPERS
# =========================================================
def image_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    return base64.b64encode(image_path.read_bytes()).decode()


def load_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_user_config() -> dict:
    config = load_json_config(USER_CONFIG_PATH)
    required_keys = [
        "battery_capacity_kwh",
        "soc_min_kwh",
        "soc_max_kwh",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing configuration keys in user_config.json: {missing}")
    return config


def load_system_config() -> dict:
    config = load_json_config(SYSTEM_CONFIG_PATH)
    defaults = {
        "interval_minutes": 15,
        "optimization_horizon_hours": 48,
        "action_horizon_hours": 48,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def find_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a column for '{label}'. Available columns: {list(df.columns)}")


def load_runtime_aggregated_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Runtime forecast file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Runtime forecast file is empty: {csv_path}")

    time_col = find_first_existing_column(
        df,
        ["utc_timestamp", "time", "timestamp", "datetime", "interval_start"],
        "timestamp",
    )
    pv_col = find_first_existing_column(
        df,
        ["pv_generation_kwh", "pv_kwh", "solar_kwh", "solar_generation_kwh"],
        "pv_generation_kwh",
    )
    load_col = find_first_existing_column(
        df,
        ["household_load_kwh", "load_kwh", "demand_kwh", "household_demand_kwh"],
        "household_load_kwh",
    )
    price_col = find_first_existing_column(
        df,
        ["energy_price_buy_cent_kwh", "price_cent_kwh", "buy_price_cent_kwh"],
        "energy_price_buy_cent_kwh",
    )

    out = df.copy()
    out["utc_timestamp"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out["pv_generation_kwh"] = pd.to_numeric(out[pv_col], errors="coerce")
    out["household_load_kwh"] = pd.to_numeric(out[load_col], errors="coerce")
    out["energy_price_buy_cent_kwh"] = pd.to_numeric(out[price_col], errors="coerce")

    if "energy_price_sell_cent_kwh" in out.columns:
        out["energy_price_sell_cent_kwh"] = pd.to_numeric(out["energy_price_sell_cent_kwh"], errors="coerce")
    else:
        out["energy_price_sell_cent_kwh"] = np.nan

    out = out.sort_values("utc_timestamp").reset_index(drop=True)

    return out[
        [
            "utc_timestamp",
            "pv_generation_kwh",
            "household_load_kwh",
            "energy_price_buy_cent_kwh",
            "energy_price_sell_cent_kwh",
        ]
    ].copy()


def load_runtime_dispatch_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Runtime dispatch file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Runtime dispatch file is empty: {csv_path}")

    if "utc_timestamp" not in df.columns:
        raise ValueError("dispatch_table.csv must contain 'utc_timestamp' column.")

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    return df


def select_runtime_window(df: pd.DataFrame, planning_date: date, interval_minutes: int, window_hours: int) -> pd.DataFrame:
    rows_needed = int(window_hours * 60 / interval_minutes)
    work = df.sort_values("utc_timestamp").reset_index(drop=True).copy()
    work["row_date"] = work["utc_timestamp"].dt.date

    matching_indices = work.index[work["row_date"] == planning_date].tolist()
    if not matching_indices:
        available_dates = sorted({str(d) for d in work["row_date"].unique()})
        raise ValueError(
            f"No rows found for planning date {planning_date}. "
            f"Available dates in runtime file: {available_dates}"
        )

    start_idx = matching_indices[0]
    end_idx = start_idx + rows_needed

    if end_idx > len(work):
        raise ValueError(
            f"Not enough runtime rows after {planning_date} to build a {window_hours}-hour window. "
            f"Needed {rows_needed} rows, available {len(work) - start_idx}."
        )

    out = work.iloc[start_idx:end_idx].copy().drop(columns=["row_date"])

    expected_step = pd.Timedelta(minutes=interval_minutes)
    diffs = out["utc_timestamp"].diff().dropna()
    if not diffs.eq(expected_step).all():
        raise ValueError(
            f"Runtime data is not continuous at {interval_minutes}-minute resolution "
            f"for the selected {window_hours}-hour window."
        )

    return out.reset_index(drop=True)


def extract_dispatch_variant(dispatch_df: pd.DataFrame, suffix: str, battery_capacity_kwh: float) -> pd.DataFrame:
    base_cols = [
        "utc_timestamp",
        "pv_generation_kwh",
        "household_load_kwh",
        "energy_price_buy_cent_kwh",
        "energy_price_sell_cent_kwh",
    ]

    required_variant_cols = [
        f"grid_to_load_kwh_{suffix}",
        f"pv_to_load_kwh_{suffix}",
        f"pv_to_battery_kwh_{suffix}",
        f"battery_to_load_kwh_{suffix}",
        f"grid_to_battery_kwh_{suffix}",
        f"export_to_grid_kwh_{suffix}",
        f"soc_kwh_{suffix}",
        f"interval_cost_cent_{suffix}",
        f"cumulative_cost_cent_{suffix}",
        f"decision_rule_{suffix}",
    ]

    missing = [c for c in required_variant_cols if c not in dispatch_df.columns]
    if missing:
        raise ValueError(f"Missing dispatch columns for '{suffix}': {missing}")

    out = dispatch_df[base_cols].copy()
    out["grid_to_load_kwh"] = pd.to_numeric(dispatch_df[f"grid_to_load_kwh_{suffix}"], errors="coerce")
    out["pv_to_load_kwh"] = pd.to_numeric(dispatch_df[f"pv_to_load_kwh_{suffix}"], errors="coerce")
    out["pv_to_battery_kwh"] = pd.to_numeric(dispatch_df[f"pv_to_battery_kwh_{suffix}"], errors="coerce")
    out["battery_to_load_kwh"] = pd.to_numeric(dispatch_df[f"battery_to_load_kwh_{suffix}"], errors="coerce")
    out["grid_to_battery_kwh"] = pd.to_numeric(dispatch_df[f"grid_to_battery_kwh_{suffix}"], errors="coerce")
    out["export_to_grid_kwh"] = pd.to_numeric(dispatch_df[f"export_to_grid_kwh_{suffix}"], errors="coerce")
    out["soc_kwh"] = pd.to_numeric(dispatch_df[f"soc_kwh_{suffix}"], errors="coerce")
    out["interval_cost_cent"] = pd.to_numeric(dispatch_df[f"interval_cost_cent_{suffix}"], errors="coerce")
    out["cumulative_cost_cent"] = pd.to_numeric(dispatch_df[f"cumulative_cost_cent_{suffix}"], errors="coerce")
    out["decision_rule"] = dispatch_df[f"decision_rule_{suffix}"].astype(str)
    out["method"] = suffix

    out["soc_percent"] = (out["soc_kwh"] / battery_capacity_kwh) * 100.0
    out["total_import_kwh"] = out["grid_to_load_kwh"] + out["grid_to_battery_kwh"]
    out["total_export_kwh"] = out["export_to_grid_kwh"]

    pv_used = out["pv_to_load_kwh"] + out["pv_to_battery_kwh"] + out["export_to_grid_kwh"]
    out["curtailed_pv_kwh"] = np.maximum(0.0, out["pv_generation_kwh"] - pv_used)

    return out


def summarize_dispatch(df: pd.DataFrame) -> dict:
    total_cost_cent = float(df["interval_cost_cent"].sum())
    total_cost_eur = total_cost_cent / 100.0

    return {
        "cost_cent": total_cost_cent,
        "cost_eur": total_cost_eur,
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


def format_net_result(cost_eur: float) -> tuple[str, str]:
    if cost_eur < 0:
        return "48h Net Result", f"Profit € {abs(cost_eur):.2f}"
    elif cost_eur > 0:
        return "48h Net Result", f"Cost € {cost_eur:.2f}"
    return "48h Net Result", "Break-even € 0.00"


def format_savings_delta(savings_eur: float) -> str:
    if savings_eur > 0:
        return f"Saved € {savings_eur:.2f}"
    elif savings_eur < 0:
        return f"Extra Cost € {abs(savings_eur):.2f}"
    return "No cost change"


def split_into_days(df: pd.DataFrame, interval_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_per_day = int(24 * 60 / interval_minutes)
    return df.iloc[:rows_per_day].copy(), df.iloc[rows_per_day:rows_per_day * 2].copy()


def build_day_summary(df: pd.DataFrame) -> dict:
    total_cost_eur = float(df["interval_cost_cent"].sum()) / 100.0
    buy_price_mean = float(df["energy_price_buy_cent_kwh"].mean())

    return {
        "cost_eur": total_cost_eur,
        "final_soc_percent": float(df["soc_percent"].iloc[-1]),
        "grid_import_kwh": float(df["total_import_kwh"].sum()),
        "grid_export_kwh": float(df["total_export_kwh"].sum()),
        "pv_to_battery_kwh": float(df["pv_to_battery_kwh"].sum()),
        "battery_to_load_kwh": float(df["battery_to_load_kwh"].sum()),
        "avg_price": buy_price_mean,
    }


def get_dominant_message(opt_summary: dict, rule_summary: dict, opt_df: pd.DataFrame) -> tuple[str, str]:
    savings = rule_summary["cost_eur"] - opt_summary["cost_eur"]
    export_kwh = opt_summary["grid_export_kwh"]
    final_soc = opt_summary["final_soc_percent"]
    grid_charge = opt_summary["grid_to_battery_kwh"]
    battery_use = opt_summary["battery_to_load_kwh"]

    if export_kwh > 3 and battery_use > 3:
        title = "Store energy for peak hours and export only when solar surplus is strong."
        subtitle = "The plan expects both useful battery shifting and meaningful solar surplus across the 48-hour window."
    elif export_kwh > 3:
        title = "Strong solar surplus expected — likely export opportunities around the brightest hours."
        subtitle = "Local demand and battery charging cannot absorb all solar production, so export becomes valuable."
    elif grid_charge > 2 and savings >= 0:
        title = "Use lower-price hours to prepare the battery for more expensive periods."
        subtitle = "The schedule charges economically when prices are weaker and spends stored energy later."
    elif final_soc < 20:
        title = "Battery is used aggressively to reduce cost over the next 48 hours."
        subtitle = "The plan leaves little reserve by the end of the horizon, which means cost reduction is prioritized."
    else:
        title = "Hold battery energy for higher-value periods and reduce expensive imports."
        subtitle = "The optimizer behaves conservatively and tries to shift energy into the more valuable parts of the horizon."

    return title, subtitle


def summarize_action_windows(opt_df: pd.DataFrame) -> dict:
    df = opt_df.copy()

    def best_window(mask: pd.Series, value_col: str, label_if_empty: str) -> str:
        if not mask.any():
            return label_if_empty
        work = df.loc[mask, ["utc_timestamp", value_col]].copy()
        best_idx = work[value_col].idxmax()
        best_time = df.loc[best_idx, "utc_timestamp"]
        return best_time.strftime("%a %H:%M")

    buy_mask = (df["grid_to_battery_kwh"] + df["grid_to_load_kwh"]) > 0.01
    battery_mask = df["battery_to_load_kwh"] > 0.01
    store_mask = df["pv_to_battery_kwh"] > 0.01
    export_mask = df["export_to_grid_kwh"] > 0.01

    return {
        "buy": best_window(buy_mask, "total_import_kwh", "Not recommended"),
        "battery": best_window(battery_mask, "battery_to_load_kwh", "Low priority"),
        "store": best_window(store_mask, "pv_to_battery_kwh", "Limited"),
        "export": best_window(export_mask, "export_to_grid_kwh", "No major window"),
    }


def classify_hourly_action(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()

    def classify(row):
        price = row["energy_price_buy_cent_kwh"]
        low_price = row["price_q25"]
        high_price = row["price_q75"]

        export = row["export_to_grid_kwh"]
        batt_use = row["battery_to_load_kwh"]
        pv_charge = row["pv_to_battery_kwh"]
        grid_charge = row["grid_to_battery_kwh"]
        grid_load = row["grid_to_load_kwh"]
        pv = row["pv_generation_kwh"]
        load = row["household_load_kwh"]

        if export > 0.2:
            return "Export Opportunity", "Solar production is higher than local demand and battery absorption.", "export"
        if batt_use > 0.2 and price >= high_price:
            return "Use Battery", "Stored energy is most valuable during higher-price or higher-demand hours.", "battery"
        if pv_charge > 0.2:
            return "Store Solar", "Solar production is being shifted into the battery for later use.", "store"
        if grid_charge > 0.15 and price <= low_price:
            return "Buy & Charge", "Grid energy is relatively cheap in this period.", "buy"
        if grid_load > 0.25 and price <= low_price:
            return "Buy from Grid", "This is one of the cheaper import windows.", "buy"
        if pv >= load * 0.8:
            return "Run on Solar", "Local solar is covering most of the demand.", "store"
        return "Hold / Normal Use", "No strong action signal is dominant in this period.", "hold"

    labels = df.apply(classify, axis=1)
    df["action"] = [x[0] for x in labels]
    df["reason"] = [x[1] for x in labels]
    df["badge"] = [x[2] for x in labels]
    return df


def build_action_timeline(opt_df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        opt_df.set_index("utc_timestamp")
        .resample("1H")
        .agg({
            "pv_generation_kwh": "sum",
            "household_load_kwh": "sum",
            "energy_price_buy_cent_kwh": "mean",
            "grid_to_load_kwh": "sum",
            "pv_to_battery_kwh": "sum",
            "battery_to_load_kwh": "sum",
            "grid_to_battery_kwh": "sum",
            "export_to_grid_kwh": "sum",
        })
        .reset_index()
    )

    hourly["price_q25"] = hourly["energy_price_buy_cent_kwh"].quantile(0.25)
    hourly["price_q75"] = hourly["energy_price_buy_cent_kwh"].quantile(0.75)

    hourly = classify_hourly_action(hourly)

    segments = []
    if hourly.empty:
        return pd.DataFrame()

    start_idx = 0
    for i in range(1, len(hourly) + 1):
        end_of_segment = (
            i == len(hourly)
            or hourly.loc[i, "action"] != hourly.loc[start_idx, "action"]
        )
        if end_of_segment:
            seg = hourly.iloc[start_idx:i].copy()
            segments.append({
                "start": seg["utc_timestamp"].iloc[0],
                "end": seg["utc_timestamp"].iloc[-1] + pd.Timedelta(hours=1),
                "action": seg["action"].iloc[0],
                "reason": seg["reason"].iloc[0],
                "badge": seg["badge"].iloc[0],
            })
            start_idx = i

    out = pd.DataFrame(segments)

    # حذف segmentهای خیلی ریز و بی‌اثر
    out["duration_hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600.0
    out = out[out["duration_hours"] >= 2].copy()

    if out.empty:
        return pd.DataFrame(segments).head(6)

    return out.reset_index(drop=True)


def build_price_soc_chart(opt_df: pd.DataFrame) -> go.Figure:
    plot_df = opt_df.copy().sort_values("utc_timestamp")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["soc_percent"],
            mode="lines",
            name="Battery Level",
            line=dict(width=3.6, color="#2563eb"),
            yaxis="y1",
            hovertemplate=(
                "<b>Battery Level</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "SoC: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["energy_price_buy_cent_kwh"],
            mode="lines",
            name="Grid Price",
            line=dict(width=3, dash="dash", color="#0f172a"),
            yaxis="y2",
            hovertemplate=(
                "<b>Grid Price</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Price: %{y:.2f} cent/kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=110),
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
            title="Time",
            tickformat="%a %H:%M",
            dtick=6 * 60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Battery SoC (%)",
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
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


def build_solar_load_chart(runtime_df: pd.DataFrame, opt_df: pd.DataFrame) -> go.Figure:
    plot_df = runtime_df.copy().sort_values("utc_timestamp")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["household_load_kwh"],
            mode="lines",
            name="Home Load",
            line=dict(width=3, color="#f97316"),
            hovertemplate=(
                "<b>Home Load</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["pv_generation_kwh"],
            mode="lines",
            name="Solar",
            line=dict(width=3.4, color="#16a34a"),
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.10)",
            hovertemplate=(
                "<b>Solar</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    export_points = opt_df[opt_df["export_to_grid_kwh"] > 0.1]
    if not export_points.empty:
        fig.add_trace(
            go.Scatter(
                x=export_points["utc_timestamp"],
                y=export_points["pv_generation_kwh"],
                mode="markers",
                name="Export Points",
                marker=dict(size=8, color="#7c3aed", line=dict(width=1.5, color="white"),),
                hovertemplate=(
                    "<b>Export Opportunity</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M}<br>"
                    "Solar: %{y:.3f} kWh"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=70),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
            title=None,
        ),
        xaxis=dict(
            title="Time",
            tickformat="%a %H:%M",
            dtick=6 * 60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Energy (kWh)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        ),
    )
    return fig

def build_forecast_inputs_chart(runtime_df: pd.DataFrame) -> go.Figure:
    plot_df = runtime_df.copy().sort_values("utc_timestamp")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["pv_generation_kwh"],
            mode="lines",
            name="PV Forecast",
            line=dict(width=3.2, color="#16a34a"),
            hovertemplate=(
                "<b>PV Forecast</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["household_load_kwh"],
            mode="lines",
            name="Load Forecast",
            line=dict(width=3.0, color="#f97316"),
            hovertemplate=(
                "<b>Load Forecast</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Energy: %{y:.3f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["utc_timestamp"],
            y=plot_df["energy_price_buy_cent_kwh"],
            mode="lines",
            name="Buy Price",
            line=dict(width=2.8, color="#1d4ed8", dash="dash"),
            yaxis="y2",
            hovertemplate=(
                "<b>Buy Price</b><br>"
                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                "Price: %{y:.2f} cent/kWh"
                "<extra></extra>"
            ),
        )
    )

    if plot_df["energy_price_sell_cent_kwh"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["utc_timestamp"],
                y=plot_df["energy_price_sell_cent_kwh"],
                mode="lines",
                name="Sell Price",
                line=dict(width=2.8, color="#7c3aed", dash="dot"),
                yaxis="y2",
                hovertemplate=(
                    "<b>Sell Price</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M}<br>"
                    "Price: %{y:.2f} cent/kWh"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=95),
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
            title="Time",
            tickformat="%a %H:%M",
            dtick=6 * 60 * 60 * 1000,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Energy (kWh per interval)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
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

def day_main_action(day_df: pd.DataFrame) -> tuple[str, str]:
    export = float(day_df["export_to_grid_kwh"].sum())
    battery_use = float(day_df["battery_to_load_kwh"].sum())
    pv_charge = float(day_df["pv_to_battery_kwh"].sum())
    grid_charge = float(day_df["grid_to_battery_kwh"].sum())
    final_soc = float(day_df["soc_percent"].iloc[-1])

    if export > 2.5:
        return "Export surplus solar", "There is enough solar generation to cover local use and still leave meaningful surplus."
    if battery_use > max(pv_charge, grid_charge) and battery_use > 2:
        return "Use stored energy in peak periods", "Battery discharge plays a strong role in reducing expensive imports."
    if pv_charge > 2:
        return "Store midday solar", "A large part of the value comes from shifting solar into the battery."
    if grid_charge > 1.5:
        return "Charge during cheaper grid hours", "Low-price periods are used to prepare for later demand."
    if final_soc > 45:
        return "Keep battery reserve", "The schedule stays conservative and holds energy for later flexibility."
    return "Balanced operation", "No single action dominates; the controller spreads value across the day."


def render_timeline_items(timeline_df: pd.DataFrame):
    badge_map = {
        "buy": ("🔵", "Buy"),
        "battery": ("🟠", "Use Battery"),
        "store": ("🟢", "Store Energy"),
        "export": ("🟣", "Export"),
        "hold": ("⚪", "Hold"),
    }

    if timeline_df.empty:
        st.info("No strong action blocks were found for this horizon.")
        return

    for _, row in timeline_df.iterrows():
        icon, short_label = badge_map.get(row["badge"], ("⚪", "Hold"))

        start_label = row["start"].strftime("%a %H:%M")
        end_label = row["end"].strftime("%a %H:%M")

        st.markdown(
            f"""
<div class="glass-card" style="margin-bottom:0.75rem;">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:0.35rem;">
        <div style="font-weight:800; color:#0f172a;">{start_label} → {end_label}</div>
        <div style="font-weight:800; color:#334155;">{icon} {short_label}</div>
    </div>
    <div style="font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:0.25rem;">{row['action']}</div>
    <div class="muted">{row['reason']}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# =========================================================
# LOAD CONFIG
# =========================================================
logo_b64 = image_to_base64(LOGO_PATH)

try:
    user_config = load_user_config()
    system_config = load_system_config()
except Exception as exc:
    st.error(f"Failed to load configuration: {exc}")
    st.stop()

battery_capacity_kwh = float(user_config["battery_capacity_kwh"])
soc_min_kwh = float(user_config["soc_min_kwh"])
soc_max_kwh = float(user_config["soc_max_kwh"])

interval_minutes = int(system_config["interval_minutes"])
optimization_horizon_hours = int(system_config["optimization_horizon_hours"])
action_horizon_hours = int(system_config["action_horizon_hours"])

expected_rows = int(optimization_horizon_hours * 60 / interval_minutes)


# =========================================================
# SIDEBAR
# =========================================================
default_start_date = pd.Timestamp.now(tz="UTC").date()

with st.sidebar:
    planning_date = st.date_input("Planning Date", value=default_start_date)
    run_button = st.button("Load 48h Plan", use_container_width=True)
    st.caption(f"{optimization_horizon_hours}-hour horizon · {interval_minutes}-minute resolution · {expected_rows} intervals")

if "auto_load_48h_plan" not in st.session_state:
    st.session_state["auto_load_48h_plan"] = True

run_button = run_button or st.session_state.pop("auto_load_48h_plan", False)


# =========================================================
# HERO
# =========================================================
hero_logo_html = ""
if logo_b64:
    hero_logo_html = f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Logo">'

st.markdown(
    dedent(
        f"""
<div class="hero-card">
<div class="brand-row">
{hero_logo_html}
<div>
<div class="hero-title">Your 48-Hour Energy Plan</div>
<div class="hero-subtitle">
See the best times to buy electricity, store solar energy, use the battery, or export surplus energy
across the next 48 hours.
</div>
<div>
<span class="hero-pill">48-Hour Plan</span>
<span class="hero-pill">Action-Focused</span>
<span class="hero-pill">Buy · Store · Use · Export</span>
</div>
</div>
</div>
</div>
"""
    ).strip(),
    unsafe_allow_html=True,
)

if not run_button:
    st.info("Pick the planning date and load the 48-hour plan.")
    st.stop()


# =========================================================
# PIPELINE
# =========================================================
try:
    planning_date = pd.Timestamp(planning_date).date()

    aggregated_runtime_df = load_runtime_aggregated_table(AGGREGATED_TABLE_PATH)
    dispatch_runtime_df = load_runtime_dispatch_table(DISPATCH_TABLE_PATH)

    runtime_window_df = select_runtime_window(
        aggregated_runtime_df,
        planning_date=planning_date,
        interval_minutes=interval_minutes,
        window_hours=optimization_horizon_hours,
    )

    dispatch_window_df = select_runtime_window(
        dispatch_runtime_df,
        planning_date=planning_date,
        interval_minutes=interval_minutes,
        window_hours=optimization_horizon_hours,
    )

    rule_dispatch_df = extract_dispatch_variant(
        dispatch_window_df,
        suffix="rule_based",
        battery_capacity_kwh=battery_capacity_kwh,
    )

    optimized_dispatch_df = extract_dispatch_variant(
        dispatch_window_df,
        suffix="lp_optimized",
        battery_capacity_kwh=battery_capacity_kwh,
    )

    rule_summary = summarize_dispatch(rule_dispatch_df)
    opt_summary = summarize_dispatch(optimized_dispatch_df)

    savings_eur = rule_summary["cost_eur"] - opt_summary["cost_eur"]

    day1_df, day2_df = split_into_days(optimized_dispatch_df, interval_minutes)
    day1_summary = build_day_summary(day1_df)
    day2_summary = build_day_summary(day2_df)

    action_windows = summarize_action_windows(optimized_dispatch_df)
    timeline_df = build_action_timeline(optimized_dispatch_df)

    main_title, main_subtitle = get_dominant_message(opt_summary, rule_summary, optimized_dispatch_df)

    load_sum = max(float(optimized_dispatch_df["household_load_kwh"].sum()), 1e-9)
    local_supply = opt_summary["pv_to_load_kwh"] + opt_summary["battery_to_load_kwh"]
    self_sufficiency = (local_supply / load_sum) * 100.0

    day1_action, day1_reason = day_main_action(day1_df)
    day2_action, day2_reason = day_main_action(day2_df)

except Exception as exc:
    st.error(f"Plan generation failed: {exc}")
    st.stop()


# =========================================================
# TOP KPI ROW
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    label, value = format_net_result(opt_summary["cost_eur"])
    st.metric(
        label,
        value,
        delta=format_savings_delta(savings_eur),
        delta_color="normal" if savings_eur >= 0 else "inverse",
    )

with k2:
    st.metric("Final Battery Level", f"{opt_summary['final_soc_percent']:.1f}%")

with k3:
    st.metric("Grid Export", f"{opt_summary['grid_export_kwh']:.2f} kWh")

with k4:
    st.metric("Self-Sufficiency", f"{self_sufficiency:.1f}%")

st.markdown("")

# =========================================================
# MAIN DECISION CARD
# =========================================================
status_text = "Good value across the 48h plan" if savings_eur > 0 else ("Neutral outcome" if savings_eur == 0 else "Weak optimization outcome")
status_css = "status-good" if savings_eur > 0 else ("status-warn" if savings_eur == 0 else "status-bad")

st.markdown(
    f"""
<div class="glass-card">
    <div class="section-title">Main Decision</div>
    <div class="big-decision">{main_title}</div>
    <div class="decision-support">{main_subtitle}</div>
    <div class="decision-support" style="margin-top:0.6rem;">
        <span class="{status_css}">{status_text}</span> ·
        Planning date: <b>{planning_date}</b> ·
        Horizon: <b>{optimization_horizon_hours} hours</b> ·
        Resolution: <b>{interval_minutes} minutes</b>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")

# =========================================================
# FORECAST INPUTS
# =========================================================
st.markdown(
    """
<div class="glass-card" style="margin-bottom:0.8rem;">
    <div class="section-title">Forecast Inputs</div>
    <div class="muted">
        This chart shows the main forecast signals driving the plan:
        PV generation forecast, household load forecast, buy price, and sell price.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.plotly_chart(build_forecast_inputs_chart(runtime_window_df), use_container_width=True)

st.markdown("")

# =========================================================
# ACTION CARDS
# =========================================================
a1, a2, a3, a4 = st.columns(4)

with a1:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-title">Best Time to Buy</div>
        <div class="action-main">{action_windows['buy']}</div>
        <div class="action-sub">Use lower-price periods for grid import or low-cost charging.</div>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-title">Best Time to Use Battery</div>
        <div class="action-main">{action_windows['battery']}</div>
        <div class="action-sub">Stored energy matters most when prices or demand pressure rise.</div>
    </div>
    """, unsafe_allow_html=True)

with a3:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-title">Best Time to Store Energy</div>
        <div class="action-main">{action_windows['store']}</div>
        <div class="action-sub">Solar charging is most useful when local generation becomes strong.</div>
    </div>
    """, unsafe_allow_html=True)

with a4:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-title">Export Opportunity</div>
        <div class="action-main">{action_windows['export']}</div>
        <div class="action-sub">Export appears only when solar surplus remains after demand and storage.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")


# =========================================================
# DAY 1 / DAY 2 SUMMARY
# =========================================================
d1, d2 = st.columns(2)

with d1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">Day 1 Summary</div>
        <div class="big-decision" style="font-size:1.12rem;">{day1_action}</div>
        <div class="muted" style="margin-bottom:0.8rem;">{day1_reason}</div>
        <div class="mini-label">Expected Cost</div>
        <div class="mini-value">€ {day1_summary['cost_eur']:.2f}</div>
        <div style="height:0.55rem;"></div>
        <div class="mini-label">End of Day Battery</div>
        <div class="mini-value">{day1_summary['final_soc_percent']:.1f}%</div>
        <div style="height:0.55rem;"></div>
        <div class="mini-label">Import / Export</div>
        <div class="mini-value">{day1_summary['grid_import_kwh']:.2f} / {day1_summary['grid_export_kwh']:.2f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">Day 2 Summary</div>
        <div class="big-decision" style="font-size:1.12rem;">{day2_action}</div>
        <div class="muted" style="margin-bottom:0.8rem;">{day2_reason}</div>
        <div class="mini-label">Expected Cost</div>
        <div class="mini-value">€ {day2_summary['cost_eur']:.2f}</div>
        <div style="height:0.55rem;"></div>
        <div class="mini-label">End of Day Battery</div>
        <div class="mini-value">{day2_summary['final_soc_percent']:.1f}%</div>
        <div style="height:0.55rem;"></div>
        <div class="mini-label">Import / Export</div>
        <div class="mini-value">{day2_summary['grid_import_kwh']:.2f} / {day2_summary['grid_export_kwh']:.2f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")


# =========================================================
# TIMELINE
# =========================================================
st.markdown(
    """
<div class="glass-card" style="margin-bottom:0.8rem;">
    <div class="section-title">48-Hour Action Timeline</div>
    <div class="muted">
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if timeline_df.empty:
    st.info("No strong timeline blocks were found for this horizon.")
else:
    render_timeline_items(timeline_df)

st.markdown("")


# =========================================================
# CHARTS
# =========================================================
st.markdown(
    """
<div class="glass-card" style="margin-bottom:0.8rem;">
    <div class="section-title">Visual Signals</div>
    <div class="muted">
        This chart explains how the battery state of charge evolves across the selected horizon and how it aligns with the grid buy price signal.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.plotly_chart(build_price_soc_chart(optimized_dispatch_df), use_container_width=True)


# =========================================================
# OPTIONAL DETAILS
# =========================================================
# with st.expander("Show detailed 48h action table"):
#     timeline_table = timeline_df.copy()
#     if not timeline_table.empty:
#         timeline_table["Start"] = timeline_table["start"].dt.strftime("%Y-%m-%d %H:%M")
#         timeline_table["End"] = timeline_table["end"].dt.strftime("%Y-%m-%d %H:%M")
#         timeline_table = timeline_table[["Start", "End", "action", "reason"]].rename(
#             columns={
#                 "action": "Action",
#                 "reason": "Why",
#             }
#         )
#         st.dataframe(timeline_table, use_container_width=True, hide_index=True)
#     else:
#         st.write("No grouped actions available.")
#
# with st.expander("Show processed optimized schedule dataset"):
#     preview_df = optimized_dispatch_df.copy()
#     preview_df["utc_timestamp"] = preview_df["utc_timestamp"].dt.strftime("%Y-%m-%d %H:%M")
#     st.dataframe(preview_df, use_container_width=True, height=320)