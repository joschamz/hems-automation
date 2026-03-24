import json
from pathlib import Path

import streamlit as st


CONFIG_PATH = Path(__file__).resolve().parent / "user_config.json"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def render_home() -> None:
    st.title("HEMS Automation")
    st.caption("Day-level forecasting and configuration for the residential energy workflow.")

    left_column, right_column = st.columns([1.3, 1.0])

    with left_column:
        st.subheader("Pages")
        st.write("Use the sidebar to move between the dashboard and configuration pages.")
        st.markdown(
            """
            - Forecast: combines solar production, electricity prices, and household load forecasts.
            - Config: edits the PV system parameters stored in user_config.json.
            """
        )

    with right_column:
        st.subheader("Current Configuration")
        try:
            st.json(_load_config())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Could not read configuration: {error}")


st.set_page_config(page_title="HEMS Automation", page_icon="⚡", layout="wide")

navigation = st.navigation(
    [
        st.Page(render_home, title="Welcome", icon="👋", default=True),
        st.Page("pages/01_forecast.py", title="Forecast", icon="📈"),
        st.Page("pages/02_hist.py", title="Historical and EDA", icon="📊"),
        st.Page("pages/09_config.py", title="Config", icon="⚙️"),

    ],
    position="sidebar",
    expanded=True,
)

navigation.run()