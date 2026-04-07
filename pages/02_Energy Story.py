import json
import base64
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Energy Story",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown(
    """
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
        overflow: hidden;
        position: relative;
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

    .section-label {
        color: #0f172a;
        font-size: 1.02rem;
        font-weight: 800;
        margin: 0.3rem 0 0.7rem 0;
    }

    .story-card {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        min-height: 112px;
        height: 100%;
    }

    .story-head {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }

    .icon-badge {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }

    .icon-solar { background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%); }
    .icon-home { background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); }
    .icon-battery { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
    .icon-grid { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }

    .story-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 850;
    }

    .story-main {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 850;
        line-height: 1.32;
        margin-bottom: 0.2rem;
    }

    .story-sub {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .soft-note {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        color: #475569;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "logo.png"

USER_CONFIG_PATH = Path("user_config.json")
SYSTEM_CONFIG_PATH = Path("system_config.json")

DATA_DIR = Path("data")
RUNTIME_DIR = DATA_DIR / "runtime"
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
    required_keys = ["battery_capacity_kwh"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing configuration keys in user_config.json: {missing}")
    return config


def load_system_config() -> dict:
    config = load_json_config(SYSTEM_CONFIG_PATH)
    config.setdefault("interval_minutes", 15)
    config.setdefault("optimization_horizon_hours", 48)
    return config


def load_runtime_dispatch_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Runtime dispatch file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Runtime dispatch file is empty: {csv_path}")

    if "utc_timestamp" not in df.columns:
        raise ValueError("dispatch_table.csv must contain 'utc_timestamp' column.")

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    if df["utc_timestamp"].isna().any():
        raise ValueError("dispatch_table.csv contains invalid utc_timestamp values.")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    return df


def select_runtime_window(
    df: pd.DataFrame,
    planning_date: date,
    interval_minutes: int,
    window_hours: int,
) -> pd.DataFrame:
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


def extract_dispatch_variant(
    dispatch_df: pd.DataFrame,
    suffix: str,
    battery_capacity_kwh: float,
) -> pd.DataFrame:
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
        "cost_eur": total_cost_eur,
        "pv_generation_kwh": float(df["pv_generation_kwh"].sum()),
        "household_load_kwh": float(df["household_load_kwh"].sum()),
        "pv_to_load_kwh": float(df["pv_to_load_kwh"].sum()),
        "pv_to_battery_kwh": float(df["pv_to_battery_kwh"].sum()),
        "battery_to_load_kwh": float(df["battery_to_load_kwh"].sum()),
        "grid_to_load_kwh": float(df["grid_to_load_kwh"].sum()),
        "grid_to_battery_kwh": float(df["grid_to_battery_kwh"].sum()),
        "export_to_grid_kwh": float(df["export_to_grid_kwh"].sum()),
        "curtailed_pv_kwh": float(df["curtailed_pv_kwh"].sum()),
        "final_soc_percent": float(df["soc_percent"].iloc[-1]),
    }


def clean_energy(value: float, tol: float = 1e-6) -> float:
    value = float(value)
    return 0.0 if abs(value) < tol else value


def pct(value: float, total: float) -> float:
    total = max(total, 1e-9)
    return (value / total) * 100.0


def role_card(icon_css: str, icon: str, title: str, main: str, sub: str) -> str:
    return f"""
<div class="story-card">
    <div class="story-head">
        <div class="icon-badge {icon_css}">{icon}</div>
        <div class="story-title">{title}</div>
    </div>
    <div class="story-main">{main}</div>
    <div class="story-sub">{sub}</div>
</div>
"""


def build_flow_summary_html(
    solar_to_home: float,
    solar_to_battery: float,
    solar_to_sell: float,
    grid_to_battery: float,
    battery_to_home: float,
) -> str:
    flows = [
        (
            "Solar → Home",
            solar_to_home,
            "#22c55e",
            "Solar energy that was used immediately by the home.",
        ),
        (
            "Solar → Battery",
            solar_to_battery,
            "#3b82f6",
            "Solar energy stored first so it could be used later.",
        ),
        (
            "Solar → Sold",
            solar_to_sell,
            "#8b5cf6",
            "Surplus solar that could not be used locally and was exported.",
        ),
        (
            "Grid → Battery",
            grid_to_battery,
            "#38bdf8",
            "Grid charging used to prepare the battery for later demand.",
        ),
        (
            "Battery → Home",
            battery_to_home,
            "#2563eb",
            "Stored energy returned later from the battery to the home.",
        ),
    ]

    flows = [(label, value, color, desc) for label, value, color, desc in flows if value > 0.1]
    max_flow = max([v for _, v, _, _ in flows], default=1.0)

    base_duration = 1.6
    extra_duration = 2.8

    rows = []
    for idx, (label, value, color, desc) in enumerate(flows):
        ratio = value / max_flow if max_flow > 0 else 0
        width_pct = max(8.0, ratio * 100.0)
        duration = base_duration + (1.0 - ratio) * extra_duration
        delay = idx * 0.14

        rows.append(
            f"""
            <div class="flow-row">
                <div class="flow-label">{label}</div>
                <div class="flow-bar-wrap">
                    <div class="flow-bar-shell">
                        <div
                            class="flow-bar"
                            style="
                                --target-width:{width_pct:.2f}%;
                                --bar-color:{color};
                                --duration:{duration:.2f}s;
                                --delay:{delay:.2f}s;
                            "
                        >
                            <div class="flow-glow"></div>
                            <div class="flow-stream"></div>
                        </div>
                    </div>
                </div>
                <div class="flow-value">{value:.1f} kWh</div>
                <div class="flow-desc">{desc}</div>
            </div>
            """
        )

    rows_html = "\n".join(rows)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        :root {{
            color-scheme: light;
        }}

        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        * {{
            box-sizing: border-box;
        }}

        .flow-card {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1.05rem 1.2rem;
            box-shadow: 0 10px 24px rgba(0,0,0,0.05);
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
            overflow: hidden;
        }}

        .flow-title {{
            font-size: 1.34rem;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: 0.16rem;
        }}

        .flow-subtitle {{
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1rem;
            line-height: 1.55;
        }}

        .flow-grid {{
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }}

        .flow-row {{
            display: grid;
            grid-template-columns: 180px minmax(320px, 1fr) 110px;
            grid-template-areas:
                "label bar value"
                "desc  desc desc";
            column-gap: 14px;
            row-gap: 0.42rem;
            align-items: center;
            padding: 0.1rem 0;
        }}

        .flow-label {{
            grid-area: label;
            font-size: 0.98rem;
            font-weight: 850;
            color: #0f172a;
        }}

        .flow-bar-wrap {{
            grid-area: bar;
            width: 100%;
        }}

        .flow-bar-shell {{
            width: 100%;
            height: 18px;
            border-radius: 999px;
            background: linear-gradient(180deg, #e2e8f0 0%, #cfd8e3 100%);
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.7), inset 0 0 0 1px rgba(15,23,42,0.04);
            overflow: hidden;
            position: relative;
        }}

        .flow-bar {{
            position: relative;
            height: 100%;
            width: 0%;
            border-radius: 999px;
            background: linear-gradient(90deg, color-mix(in srgb, var(--bar-color) 84%, white 16%) 0%, var(--bar-color) 100%);
            box-shadow:
                0 6px 16px color-mix(in srgb, var(--bar-color) 32%, transparent 68%),
                inset 0 1px 1px rgba(255,255,255,0.45);
            animation: fillBar var(--duration) cubic-bezier(0.22, 1, 0.36, 1) forwards;
            animation-delay: var(--delay);
            overflow: hidden;
        }}

        .flow-glow {{
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.32) 0%, rgba(255,255,255,0.02) 100%);
            pointer-events: none;
        }}

        .flow-stream {{
            position: absolute;
            top: 2px;
            bottom: 2px;
            left: -24%;
            width: 22%;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.92) 48%, rgba(255,255,255,0.0) 100%);
            filter: blur(0.2px);
            opacity: 0.95;
            animation:
                fillBar var(--duration) cubic-bezier(0.22, 1, 0.36, 1) forwards,
                travelStream calc(var(--duration) * 0.92) linear infinite;
            animation-delay: var(--delay), calc(var(--delay) + 0.08s);
        }}

        .flow-value {{
            grid-area: value;
            font-size: 0.96rem;
            font-weight: 850;
            color: #0f172a;
            text-align: right;
            white-space: nowrap;
        }}

        .flow-desc {{
            grid-area: desc;
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.52;
            padding-left: 0;
        }}

        @keyframes fillBar {{
            from {{ width: 0%; }}
            to {{ width: var(--target-width); }}
        }}

        @keyframes travelStream {{
            0% {{ transform: translateX(-10%); opacity: 0.0; }}
            8% {{ opacity: 0.95; }}
            100% {{ transform: translateX(560%); opacity: 0.0; }}
        }}

        @media (max-width: 1100px) {{
            .flow-row {{
                grid-template-columns: 160px minmax(240px, 1fr) 95px;
            }}
        }}

        @media (max-width: 820px) {{
            .flow-row {{
                grid-template-columns: 1fr;
                grid-template-areas:
                    "label"
                    "bar"
                    "value"
                    "desc";
            }}

            .flow-value {{
                text-align: left;
            }}
        }}
    </style>
    </head>
    <body>
        <div class="flow-card">
            <div class="flow-title">Energy Flow Summary</div>
            <div class="flow-subtitle">
                The chart below shows only the main energy paths over the planning horizon.
            </div>
            <div class="flow-grid">
                {rows_html}
            </div>
        </div>
    </body>
    </html>
    """


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
interval_minutes = int(system_config["interval_minutes"])
optimization_horizon_hours = int(system_config["optimization_horizon_hours"])
expected_rows = int(optimization_horizon_hours * 60 / interval_minutes)


# =========================================================
# SIDEBAR
# =========================================================
default_start_date = pd.Timestamp.now(tz="UTC").date()

with st.sidebar:
    planning_date = st.date_input("Planning Date", value=default_start_date)
    run_button = st.button("Load Energy Story", use_container_width=True)
    st.caption(
        f"{optimization_horizon_hours}-hour horizon · {interval_minutes}-minute resolution · {expected_rows} intervals"
    )


# =========================================================
# HERO
# =========================================================
hero_logo_html = ""
if logo_b64:
    hero_logo_html = f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Logo">'

st.markdown(
    f"""
<div class="hero-card">
    <div class="brand-row">
        {hero_logo_html}
        <div>
            <div class="hero-title">Energy Story</div>
            <div class="hero-subtitle">
                A simplified view of what solar produced, how the battery helped, and how much surplus was exported.
            </div>
            <div>
                <span class="hero-pill">Solar Role</span>
                <span class="hero-pill">Battery Role</span>
                <span class="hero-pill">Export Story</span>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not run_button:
    st.markdown(
        """
        <div class="soft-note">
            Select a planning date from the sidebar and load the energy story.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# =========================================================
# PIPELINE
# =========================================================
try:
    planning_date = pd.Timestamp(planning_date).date()

    dispatch_runtime_df = load_runtime_dispatch_table(DISPATCH_TABLE_PATH)
    dispatch_window_df = select_runtime_window(
        dispatch_runtime_df,
        planning_date=planning_date,
        interval_minutes=interval_minutes,
        window_hours=optimization_horizon_hours,
    )

    optimized_dispatch_df = extract_dispatch_variant(
        dispatch_window_df,
        suffix="lp_optimized",
        battery_capacity_kwh=battery_capacity_kwh,
    )

    summary = summarize_dispatch(optimized_dispatch_df)

    solar_total = clean_energy(summary["pv_generation_kwh"])
    home_total = clean_energy(summary["household_load_kwh"])

    solar_to_home = clean_energy(summary["pv_to_load_kwh"])
    solar_to_battery = clean_energy(summary["pv_to_battery_kwh"])
    battery_to_home = clean_energy(summary["battery_to_load_kwh"])
    grid_to_battery = clean_energy(summary["grid_to_battery_kwh"])
    sold_to_grid = clean_energy(summary["export_to_grid_kwh"])

    battery_charge = clean_energy(solar_to_battery + grid_to_battery)
    battery_discharge = clean_energy(battery_to_home)

    solar_sold_share = pct(sold_to_grid, solar_total)
    battery_home_share = pct(battery_to_home, home_total)
    local_direct_solar_share = pct(solar_to_home, solar_total)

    if sold_to_grid > max(solar_to_home, battery_to_home):
        solar_role = "Solar acts mainly as a producer"
        solar_sub = (
            f"{solar_sold_share:.1f}% of solar leaves as export, while "
            f"{local_direct_solar_share:.1f}% directly supports home demand."
        )
    elif battery_to_home > 0.25 * home_total:
        solar_role = "Solar and storage work together"
        solar_sub = (
            f"{local_direct_solar_share:.1f}% of solar serves demand directly and "
            f"storage later returns {battery_home_share:.1f}% of home demand."
        )
    else:
        solar_role = "Solar acts mainly as direct supply"
        solar_sub = (
            f"{local_direct_solar_share:.1f}% of solar directly supports the home "
            f"before storage or export."
        )

except Exception as exc:
    st.error(f"Energy Story page failed: {exc}")
    st.stop()


# =========================================================
# SUMMARY CARDS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        role_card(
            "icon-solar",
            "☀️",
            "Solar role",
            solar_role,
            solar_sub,
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        role_card(
            "icon-battery",
            "🔋",
            "Battery role",
            f"{battery_discharge:.1f} kWh supplied later",
            f"The battery covers {battery_home_share:.1f}% of household demand over the {optimization_horizon_hours}-hour window.",
        ),
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        role_card(
            "icon-grid",
            "💸",
            "Export role",
            f"{sold_to_grid:.1f} kWh exported",
            "Export becomes dominant when solar production is higher than direct household demand and storage need.",
        ),
        unsafe_allow_html=True,
    )


# =========================================================
# FLOW SUMMARY
# =========================================================
components.html(
    build_flow_summary_html(
        solar_to_home=solar_to_home,
        solar_to_battery=solar_to_battery,
        solar_to_sell=sold_to_grid,
        grid_to_battery=grid_to_battery,
        battery_to_home=battery_to_home,
    ),
    height=470,
    scrolling=False,
)


# =========================================================
# KEY TAKEAWAYS
# =========================================================
st.markdown('<div class="section-label">Key takeaways</div>', unsafe_allow_html=True)

k1, k2 = st.columns(2)

with k1:
    st.markdown(
        role_card(
            "icon-home",
            "🏠",
            "Direct supply",
            f"{solar_to_home:.1f} kWh used immediately",
            "This is the solar energy consumed by the home without going through the battery first.",
        ),
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        role_card(
            "icon-battery",
            "🔋",
            "Battery charging",
            f"{battery_charge:.1f} kWh stored",
            "This is the total energy that entered the battery from solar and grid charging over the full horizon.",
        ),
        unsafe_allow_html=True,
    )