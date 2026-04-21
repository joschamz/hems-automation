import json
import base64
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="System Center",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
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

KEY_ARTIFACTS = {
    "Aggregated Forecast Table": RUNTIME_DIR / "aggregated_table.csv",
    "Dispatch Table": RUNTIME_DIR / "dispatch_table.csv",
    "Load Training Dataset": DATA_DIR / "load_training_dataset.csv",
}


# =========================================================
# FIELD DEFINITIONS
# =========================================================
USER_CONFIG_FIELDS = {
    "lat": {"label": "Latitude", "step": 0.000001, "format": "%.6f"},
    "lon": {"label": "Longitude", "step": 0.000001, "format": "%.6f"},
    "kwp": {"label": "PV Size (kWp)", "step": 0.1, "format": "%.2f"},
    "tilt": {"label": "Tilt (deg)", "step": 1.0, "format": "%.1f"},
    "azimuth": {"label": "Azimuth (deg)", "step": 1.0, "format": "%.1f"},
    "yield_factor": {"label": "Yield Factor", "step": 0.01, "format": "%.2f"},
    "battery_capacity_kwh": {"label": "Battery Capacity (kWh)", "step": 0.1, "format": "%.2f"},
    "max_charge_kw": {"label": "Max Charge Power (kW)", "step": 0.1, "format": "%.2f"},
    "max_discharge_kw": {"label": "Max Discharge Power (kW)", "step": 0.1, "format": "%.2f"},
    "charge_efficiency": {"label": "Charge Efficiency", "step": 0.01, "format": "%.2f"},
    "discharge_efficiency": {"label": "Discharge Efficiency", "step": 0.01, "format": "%.2f"},
    "soc_min_kwh": {"label": "Minimum SoC (kWh)", "step": 0.1, "format": "%.2f"},
    "soc_max_kwh": {"label": "Maximum SoC (kWh)", "step": 0.1, "format": "%.2f"},
}

SYSTEM_CONFIG_FIELDS = {
    "interval_minutes": {"label": "Interval Length (minutes)", "step": 1, "format": "%d", "type": "int"},
    "default_sell_price_cent_kwh": {"label": "Default Sell Price (cent/kWh)", "step": 0.1, "format": "%.2f"},
    "allow_grid_charging": {"label": "Allow Grid Charging", "type": "bool"},
    "grid_charge_price_threshold_cent_kwh": {"label": "Grid Charge Threshold (cent/kWh)", "step": 0.1, "format": "%.2f"},
    "cycle_penalty_cent_per_kwh": {"label": "Cycle Penalty (cent/kWh)", "step": 0.001, "format": "%.3f"},
    "enforce_solar_first_in_lp": {"label": "Enforce Solar First in LP", "type": "bool"},
    "terminal_soc_value_cent_kwh": {"label": "Terminal SoC Value (cent/kWh)", "step": 0.1, "format": "%.2f"},
    "min_end_soc_kwh": {"label": "Minimum End SoC (kWh)", "step": 0.1, "format": "%.2f", "type": "nullable_float"},
    "optimization_horizon_hours": {"label": "Optimization Horizon (hours)", "step": 1, "format": "%d", "type": "int"},
    "action_horizon_hours": {"label": "Action Horizon (hours)", "step": 1, "format": "%d", "type": "int"},
    "update_frequency_hours": {"label": "Update Frequency (hours)", "step": 1, "format": "%d", "type": "int"},
}


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
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_json_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle, indent=4)
        file_handle.write("\n")


def coerce_float(config: dict, key: str, default: float = 0.0) -> float:
    value = config.get(key, default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_int(config: dict, key: str, default: int = 0) -> int:
    value = config.get(key, default)
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_bool(value, default=False) -> bool:
    if value is None:
        return default
    return bool(value)


def artifact_state(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "Missing", "bad"
    size_kb = path.stat().st_size / 1024
    if size_kb <= 1:
        return "Very small", "warn"
    return "Ready", "ok"


def render_config_form(config: dict, fields: dict, form_key: str, submit_label: str):
    with st.form(form_key):
        left_column, right_column = st.columns(2)
        updated_config = dict(config)

        for index, (key, metadata) in enumerate(fields.items()):
            column = left_column if index % 2 == 0 else right_column

            with column:
                field_type = metadata.get("type", "float")

                if field_type == "bool":
                    updated_config[key] = st.checkbox(
                        metadata["label"],
                        value=safe_bool(config.get(key, False)),
                        key=f"{form_key}_{key}",
                    )

                elif field_type == "int":
                    updated_config[key] = st.number_input(
                        metadata["label"],
                        value=coerce_int(config, key),
                        step=int(metadata["step"]),
                        format=metadata["format"],
                        key=f"{form_key}_{key}",
                    )

                elif field_type == "nullable_float":
                    raw_value = config.get(key, None)
                    use_null = raw_value is None

                    is_null = st.checkbox(
                        f"{metadata['label']} = null",
                        value=use_null,
                        key=f"{form_key}_{key}_is_null",
                    )

                    if is_null:
                        updated_config[key] = None
                        st.number_input(
                            metadata["label"],
                            value=0.0 if raw_value is None else coerce_float(config, key),
                            step=metadata["step"],
                            format=metadata["format"],
                            disabled=True,
                            key=f"{form_key}_{key}_value",
                        )
                    else:
                        updated_config[key] = st.number_input(
                            metadata["label"],
                            value=0.0 if raw_value is None else coerce_float(config, key),
                            step=metadata["step"],
                            format=metadata["format"],
                            key=f"{form_key}_{key}_value",
                        )

                else:
                    updated_config[key] = st.number_input(
                        metadata["label"],
                        value=coerce_float(config, key),
                        step=metadata["step"],
                        format=metadata["format"],
                        key=f"{form_key}_{key}",
                    )

        submitted = st.form_submit_button(submit_label, use_container_width=True)

    if submitted:
        return updated_config
    return None


def collect_csv_files() -> dict[str, Path]:
    csv_files = sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []
    file_options: dict[str, Path] = {}

    for path in csv_files:
        try:
            label = str(path.relative_to(Path.cwd()))
        except ValueError:
            label = str(path)
        file_options[label] = path

    return file_options


def build_artifact_status_rows() -> list[dict]:
    rows = []
    for label, path in KEY_ARTIFACTS.items():
        exists = path.exists()
        size_kb = round(path.stat().st_size / 1024, 2) if exists else None
        status, _ = artifact_state(path)
        rows.append(
            {
                "Artifact": label,
                "Status": status,
                "Path": str(path),
                "Exists": exists,
                "Size (KB)": size_kb,
            }
        )
    return rows


def read_csv_preview(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def info_box(title: str, body: str, icon: str = "📌"):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_block(logo_b64: str):
    hero_logo_html = ""
    if logo_b64:
        hero_logo_html = f'<img class="brand-logo" src="data:image/png;base64,{logo_b64}" alt="Camelectrix Logo">'

    st.markdown(
        f"""
<div class="hero-card">
<div class="brand-row">
{hero_logo_html}
<div>
<div class="hero-title">System Center</div>
<div class="hero-subtitle">
A clean control and documentation layer for the HEMS dashboard prototype.
This page combines system overview, configuration management, and runtime artifact inspection
in one place so the product stays understandable, testable, and operationally grounded.
</div>
<div>
<span class="hero-pill">🧭 System Overview</span>
<span class="hero-pill">⚙️ Configuration Control</span>
<span class="hero-pill">📁 Artifact Inspection</span>
<span class="hero-pill">🧠 Runtime Transparency</span>
</div>
</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def snapshot_item(label: str, value: str):
    st.markdown(
        f"""
        <div class="snapshot-item">
            <div class="snapshot-label">{label}</div>
            <div class="snapshot-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_chip_html(label: str, tone: str) -> str:
    klass = {
        "ok": "chip-ok",
        "warn": "chip-warn",
        "bad": "chip-bad",
    }.get(tone, "chip-warn")
    return f'<span class="status-chip {klass}">{label}</span>'


# =========================================================
# LOAD SNAPSHOT
# =========================================================
logo_b64 = image_to_base64(LOGO_PATH)

try:
    user_config_snapshot = load_json_config(USER_CONFIG_PATH)
except Exception:
    user_config_snapshot = {}

try:
    system_config_snapshot = load_json_config(SYSTEM_CONFIG_PATH)
except Exception:
    system_config_snapshot = {}

interval_minutes = int(system_config_snapshot.get("interval_minutes", 15))
optimization_horizon_hours = int(system_config_snapshot.get("optimization_horizon_hours", 48))
action_horizon_hours = int(system_config_snapshot.get("action_horizon_hours", 48))
expected_rows = int(optimization_horizon_hours * 60 / max(interval_minutes, 1))

pv_capacity_kwp = float(user_config_snapshot.get("kwp", 0.0))
battery_capacity_kwh = float(user_config_snapshot.get("battery_capacity_kwh", 0.0))
yield_factor = float(user_config_snapshot.get("yield_factor", 0.0))
allow_grid_charging = safe_bool(system_config_snapshot.get("allow_grid_charging", False))
artifact_count = sum(1 for _, p in KEY_ARTIFACTS.items() if p.exists())

aggregated_preview = read_csv_preview(KEY_ARTIFACTS["Aggregated Forecast Table"])
dispatch_preview = read_csv_preview(KEY_ARTIFACTS["Dispatch Table"])


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        max-width: 1500px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
        padding: 1.55rem 1.8rem;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 42px rgba(0,0,0,0.18);
        margin-top: 0.4rem;
        margin-bottom: 1rem;
        overflow: hidden;
        position: relative;
    }

    .hero-card:before {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -80px;
        background: radial-gradient(circle, rgba(59,130,246,0.14) 0%, rgba(59,130,246,0.0) 72%);
        pointer-events: none;
    }

    .hero-card:after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        left: -70px;
        bottom: -90px;
        background: radial-gradient(circle, rgba(16,185,129,0.10) 0%, rgba(16,185,129,0.0) 72%);
        pointer-events: none;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
        z-index: 1;
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
        font-size: 2.15rem;
        font-weight: 850;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        line-height: 1.1;
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
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 22px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 26px rgba(0,0,0,0.07);
        height: 100%;
    }

    .feature-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        height: 100%;
    }

    .feature-icon {
        font-size: 1.45rem;
        margin-bottom: 0.45rem;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.58;
    }

    .section-header {
        color: #0f172a;
        font-size: 1.06rem;
        font-weight: 850;
        margin: 0.1rem 0 0.75rem 0;
    }

    .muted {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.68;
    }

    .snapshot-item {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        padding: 0.62rem 0;
        border-bottom: 1px dashed rgba(148,163,184,0.28);
    }

    .snapshot-item:last-child {
        border-bottom: none;
    }

    .snapshot-label {
        color: #475569;
        font-size: 0.92rem;
    }

    .snapshot-value {
        color: #0f172a;
        font-size: 0.94rem;
        font-weight: 800;
        text-align: right;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.68rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 800;
        border: 1px solid rgba(15,23,42,0.08);
    }

    .chip-ok {
        background: rgba(220,252,231,0.9);
        color: #166534;
    }

    .chip-warn {
        background: rgba(254,249,195,0.9);
        color: #92400e;
    }

    .chip-bad {
        background: rgba(254,226,226,0.9);
        color: #991b1b;
    }

    .artifact-name {
        font-weight: 800;
        color: #0f172a;
    }

    .artifact-path {
        color: #64748b;
        font-size: 0.84rem;
        word-break: break-word;
    }

    .step-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        height: 100%;
    }

    .step-number {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #0f172a;
        color: white;
        font-weight: 850;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.7rem;
    }

    .step-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.3rem;
    }

    .step-text {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.58;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(15,23,42,0.08);
        padding: 0.85rem 1rem;
        border-radius: 18px;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.65rem;
        flex-wrap: wrap;
    }

    button[role="tab"] {
        border-radius: 999px !important;
        padding: 0.58rem 1rem !important;
        border: 1px solid rgba(15,23,42,0.08) !important;
        background: rgba(255,255,255,0.96) !important;
        font-weight: 800 !important;
    }

    button[role="tab"][aria-selected="true"] {
        background: #eef2ff !important;
        border: 1px solid rgba(79,70,229,0.28) !important;
        color: #312e81 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================
hero_block(logo_b64)


# =========================================================
# TOP KPIS
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Optimization Horizon", f"{optimization_horizon_hours}h")

with k2:
    st.metric(f"{interval_minutes}-Min Intervals", f"{expected_rows}")

with k3:
    st.metric("Grid Charging", "ON" if allow_grid_charging else "OFF")

with k4:
    st.metric("Key Artifacts Found", f"{artifact_count}/{len(KEY_ARTIFACTS)}")

st.markdown("")


# =========================================================
# OVERVIEW ROW
# =========================================================
left, right = st.columns([1.18, 0.82])

with left:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">🧩 What this system is</div>
            <div class="muted">
                The HEMS dashboard is a runtime-facing decision-support prototype for residential energy planning.
                It does not pretend to be a closed-loop production controller yet.
                Its current strength is that it reads upstream forecast and dispatch outputs directly,
                instead of rebuilding a second truth inside the UI.
                <br><br>
                In practice, the platform brings together three layers:
                forecast inputs, dispatch outputs, and operator-facing explanations.
                That structure is already strong enough for engineering review, scenario inspection,
                optimization validation, and runtime artifact sanity checks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    pv_text = f"{pv_capacity_kwp:.1f} kWp" if pv_capacity_kwp > 0 else "Not set"
    batt_text = f"{battery_capacity_kwh:.1f} kWh" if battery_capacity_kwh > 0 else "Not set"
    yield_text = f"{yield_factor:.2f}" if yield_factor > 0 else "Not set"
    forecast_rows = len(aggregated_preview) if aggregated_preview is not None else 0
    dispatch_rows = len(dispatch_preview) if dispatch_preview is not None else 0

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">📌 Current snapshot</div>
        """,
        unsafe_allow_html=True,
    )
    snapshot_item("PV Capacity", pv_text)
    snapshot_item("Battery Capacity", batt_text)
    snapshot_item("Yield Factor", yield_text)
    snapshot_item("Forecast Rows", f"{forecast_rows:,}" if forecast_rows else "Unavailable")
    snapshot_item("Dispatch Rows", f"{dispatch_rows:,}" if dispatch_rows else "Unavailable")
    snapshot_item("Action Horizon", f"{action_horizon_hours} h")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")


# =========================================================
# WORKFLOW
# =========================================================
st.markdown('<div class="section-header">🪜 System workflow</div>', unsafe_allow_html=True)

wf1, wf2, wf3, wf4 = st.columns(4)

workflow_cards = [
    (
        "1",
        "Configure",
        "Set the physical system, battery bounds, and optimization behavior through user and system configuration files.",
    ),
    (
        "2",
        "Generate runtime inputs",
        "Forecast artifacts are produced upstream and stored in runtime files for solar, load, and electricity price signals.",
    ),
    (
        "3",
        "Produce dispatch outputs",
        "Rule-based and optimized schedules are generated upstream and stored in dispatch artifacts.",
    ),
    (
        "4",
        "Inspect and explain",
        "The dashboard turns those runtime outputs into operational pages, comparisons, and explainable narratives.",
    ),
]

for col, (num, title, text) in zip([wf1, wf2, wf3, wf4], workflow_cards):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")


# =========================================================
# ARTIFACT HEALTH
# =========================================================
st.markdown('<div class="section-header">📦 Key artifact health</div>', unsafe_allow_html=True)
artifact_cols = st.columns(3)

for col, (label, path) in zip(artifact_cols, KEY_ARTIFACTS.items()):
    status_text, tone = artifact_state(path)
    with col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="artifact-name">{label}</div>
                <div class="artifact-path">{path}</div>
                <div style="margin-top:0.8rem;">
                    {status_chip_html(status_text, tone)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")


# =========================================================
# TABS
# =========================================================
tab_overview, tab_user, tab_system, tab_data = st.tabs(
    ["System Notes", "User Config", "System Config", "Artifacts & Data"]
)


# =========================================================
# TAB 1 - SYSTEM NOTES
# =========================================================
with tab_overview:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">🧱 What still needs work</div>
            <div class="muted">
                It still needs stronger resilience around missing artifacts, tighter provenance tracking,
                clearer execution feedback, and eventually a better bridge between planning output and real operational control.
                The architecture is promising; it is not finished.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 2 - USER CONFIG
# =========================================================
with tab_user:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:0.9rem;">
            <div class="section-header">⚙️ User-side physical settings</div>
            <div class="muted">
                These values describe the household system itself: site location, PV setup, battery size,
                charge and discharge limits, efficiencies, and SoC bounds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        user_config = load_json_config(USER_CONFIG_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read {USER_CONFIG_PATH.name}: {error}")
    else:
        a, b, c = st.columns(3)
        with a:
            st.metric("PV Size", f"{coerce_float(user_config, 'kwp'):.2f} kWp")
        with b:
            st.metric("Battery Capacity", f"{coerce_float(user_config, 'battery_capacity_kwh'):.2f} kWh")
        with c:
            st.metric(
                "SoC Range",
                f"{coerce_float(user_config, 'soc_min_kwh'):.2f} → {coerce_float(user_config, 'soc_max_kwh'):.2f} kWh",
            )

        with st.expander("Show current user_config.json", expanded=False):
            st.json(user_config)

        updated_user_config = render_config_form(
            config=user_config,
            fields=USER_CONFIG_FIELDS,
            form_key="user_config_form",
            submit_label="Save user configuration",
        )

        if updated_user_config is not None:
            try:
                save_json_config(USER_CONFIG_PATH, updated_user_config)
            except OSError as error:
                st.error(f"Could not save {USER_CONFIG_PATH.name}: {error}")
            else:
                st.success(f"Saved configuration to {USER_CONFIG_PATH.name}.")
                st.rerun()


# =========================================================
# TAB 3 - SYSTEM CONFIG
# =========================================================
with tab_system:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:0.9rem;">
            <div class="section-header">🧠 Optimization and runtime settings</div>
            <div class="muted">
                These values control the scheduling logic and runtime behavior:
                interval length, sell-price assumptions, grid-charging policy, LP behavior,
                horizon lengths, and update frequency.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        system_config = load_json_config(SYSTEM_CONFIG_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read {SYSTEM_CONFIG_PATH.name}: {error}")
    else:
        a, b, c = st.columns(3)
        with a:
            st.metric("Interval", f"{coerce_int(system_config, 'interval_minutes')} min")
        with b:
            st.metric("Optimization Horizon", f"{coerce_int(system_config, 'optimization_horizon_hours')} h")
        with c:
            st.metric("Update Frequency", f"{coerce_int(system_config, 'update_frequency_hours')} h")

        with st.expander("Show current system_config.json", expanded=False):
            st.json(system_config)

        updated_system_config = render_config_form(
            config=system_config,
            fields=SYSTEM_CONFIG_FIELDS,
            form_key="system_config_form",
            submit_label="Save system configuration",
        )

        if updated_system_config is not None:
            try:
                save_json_config(SYSTEM_CONFIG_PATH, updated_system_config)
            except OSError as error:
                st.error(f"Could not save {SYSTEM_CONFIG_PATH.name}: {error}")
            else:
                st.success(f"Saved configuration to {SYSTEM_CONFIG_PATH.name}.")
                st.rerun()


# =========================================================
# TAB 4 - ARTIFACTS & DATA
# =========================================================
with tab_data:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:0.9rem;">
            <div class="section-header">📁 Runtime and historical data browser</div>
            <div class="muted">
                Use this section to inspect CSV files from both the historical data folder and the runtime folder.
                This is where you verify whether the dashboard is reading the artifacts you think it is reading.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_df = pd.DataFrame(build_artifact_status_rows())
    st.markdown("**Key runtime artifacts**")
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.markdown("")

    file_options = collect_csv_files()

    if not file_options:
        st.warning("No CSV files were found in the data/ folder.")
    else:
        selected_file_label = st.selectbox("Select a CSV file", list(file_options.keys()))
        selected_file = file_options[selected_file_label]

        try:
            df = pd.read_csv(selected_file)
        except Exception as error:
            st.error(f"Could not read selected CSV: {error}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows", f"{len(df):,}")
            with c2:
                st.metric("Columns", f"{len(df.columns)}")
            with c3:
                st.metric("Folder", selected_file.parent.name)
            with c4:
                size_kb = round(selected_file.stat().st_size / 1024, 2)
                st.metric("Size", f"{size_kb} KB")

            st.markdown(f"**Selected file:** `{selected_file_label}`")
            st.dataframe(df, use_container_width=True, height=420)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            if numeric_cols:
                st.markdown("")
                st.markdown("**Quick numeric preview**")

                chosen_numeric = st.multiselect(
                    "Numeric columns to plot",
                    options=numeric_cols,
                    default=numeric_cols[: min(3, len(numeric_cols))],
                )

                if chosen_numeric:
                    plot_df = df[chosen_numeric].copy()
                    st.line_chart(plot_df)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download selected CSV",
                data=csv_bytes,
                file_name=selected_file.name,
                mime="text/csv",
            )