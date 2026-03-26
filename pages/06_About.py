import streamlit as st


st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
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
        padding: 1.8rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 16px 38px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.35rem;
        line-height: 1.05;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 1rem;
        max-width: 900px;
    }

    .hero-pill {
        display: inline-block;
        padding: 0.5rem 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 999px;
        color: #0f172a;
        font-size: 0.84rem;
        font-weight: 700;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .glass-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1.05rem 1.1rem 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        height: 100%;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
        color: #0f172a;
    }

    .muted {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        text-align: center;
        height: 100%;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }

    .metric-label {
        margin-top: 0.35rem;
        color: #475569;
        font-size: 0.9rem;
    }

    .feature-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        height: 100%;
    }

    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .step-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        height: 100%;
    }

    .step-number {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #0f172a;
        color: white;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.7rem;
    }
            
</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">About the HEMS Control Center</div>
    <div class="hero-subtitle">
        A decision-support prototype for residential energy management that combines solar forecasting,
        day-ahead electricity pricing, baseline household demand prediction, and battery dispatch optimization
        into one operational dashboard.
    </div>
    <div>
        <span class="hero-pill">⚡ Smart Home Energy Management</span>
        <span class="hero-pill">☀️ Solar + Battery Coordination</span>
        <span class="hero-pill">📈 Forecast-Driven Planning</span>
        <span class="hero-pill">🧠 Rule-Based + LP Optimization</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# OVERVIEW + KPI
# =========================================================
left, right = st.columns([1.25, 1])

with left:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">What this system does</div>
        <div class="muted">
            The HEMS Control Center is designed to produce a day-ahead residential energy plan at 15-minute resolution.
            It integrates three core forecast layers - PV generation, electricity prices, and household demand - and
            computes a battery dispatch schedule that determines when to consume locally, charge the battery, discharge
            the battery, import from the grid, or export surplus energy.<br><br>
            The current prototype is focused on operational transparency and controllable experimentation rather than
            full production deployment. It provides both a rule-based baseline and a linear-programming optimizer so
            scheduling quality can be compared under the same scenario.
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    k1, k2 = st.columns(2)

    with k1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">24h</div>
            <div class="metric-label">Planning Horizon</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">96</div>
            <div class="metric-label">15-Min Intervals</div>
        </div>
        """, unsafe_allow_html=True)
        

st.markdown("")

# =========================================================
# WORKFLOW
# =========================================================
st.subheader("System Workflow")

wf1, wf2, wf3, wf4 = st.columns(4)

workflow_cards = [
    ("1", "⚙️ Configuration", "System settings, PV setup, geographic location, and model paths are managed centrally in Admin."),
    ("2", "☀️ Forecast Inputs", "Solar generation, electricity prices, and household demand forecasts are prepared for the selected planning date."),
    ("3", "🧮 Optimization", "A rule-based baseline and an LP optimizer compute feasible energy flows across PV, battery, household demand, and grid."),
    ("4", "📊 Decision Support", "The dashboard visualizes schedules, compares strategies, validates balances, and exports optimized plans."),
]

for col, (num, title, text) in zip([wf1, wf2, wf3, wf4], workflow_cards):
        with col:
            st.markdown(f"""
            <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# =========================================================
# FEATURES
# =========================================================
st.subheader("Core Capabilities")

feature_rows = [
    [
        ("🔋", "Battery Dispatch Scheduling", "Optimizes the battery as an active flexibility asset instead of treating it as passive storage."),
        ("🧠", "LP Cost Optimization", "Computes a cost-oriented dispatch plan under battery, power, SoC, and operational constraints."),
    ],
    [
        ("🛠️", "Centralized Configuration", "Moves critical system configuration to Admin for cleaner operational control."),
        ("⬇️", "Exportable Schedules", "Supports CSV export of optimized schedules for downstream use or reporting."),
    ],
]

for row in feature_rows:
    cols = st.columns(2)
    for col, (icon, title, text) in zip(cols, row):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")