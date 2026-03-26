import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Admin Center", page_icon="📂", layout="wide")

CONFIG_PATH = Path("user_config.json")
SYSTEM_CONFIG_FIELDS = {
    "lat": {"label": "Latitude", "step": 0.000001, "format": "%.6f"},
    "lon": {"label": "Longitude", "step": 0.000001, "format": "%.6f"},
    "kwp": {"label": "PV Size (kWp)", "step": 0.1, "format": "%.2f"},
    "tilt": {"label": "Tilt (deg)", "step": 1.0, "format": "%.1f"},
    "azimuth": {"label": "Azimuth (deg)", "step": 1.0, "format": "%.1f"},
    "yield_factor": {"label": "Yield Factor", "step": 0.01, "format": "%.2f"},
    "feature_dataset_path": {"label": "Feature Dataset Path", "type": "text"},
    "load_model_path": {"label": "Load Model Path", "type": "text"},
}

BATTERY_STRATEGY_FIELDS = {
    "battery_capacity_kwh": {"label": "Battery Capacity (kWh)", "step": 0.5, "format": "%.2f"},
    "initial_soc_kwh": {"label": "Initial Battery SoC (kWh)", "step": 0.1, "format": "%.2f"},
    "min_reserve_kwh": {"label": "Minimum Battery Reserve (kWh)", "step": 0.1, "format": "%.2f"},
    "max_charge_kw": {"label": "Max Charge Power (kW)", "step": 0.5, "format": "%.2f"},
    "max_discharge_kw": {"label": "Max Discharge Power (kW)", "step": 0.5, "format": "%.2f"},
    "charge_efficiency": {"label": "Charge Efficiency", "step": 0.01, "format": "%.2f"},
    "discharge_efficiency": {"label": "Discharge Efficiency", "step": 0.01, "format": "%.2f"},
    "sell_price_cent_kwh": {"label": "Default Sell Price (cent/kWh)", "step": 0.5, "format": "%.2f"},
    "allow_grid_charging": {"label": "Allow Grid Charging", "type": "bool"},
    "grid_charge_price_threshold": {"label": "Grid Charge Threshold (cent/kWh)", "step": 0.5, "format": "%.2f"},
    "cycle_penalty": {"label": "Cycle Penalty (cent/kWh)", "step": 0.001, "format": "%.3f"},
    "enforce_solar_first_in_lp": {"label": "Enforce Solar First in LP", "type": "bool"},
    "terminal_soc_value": {"label": "Terminal SoC Value (cent/kWh)", "step": 0.1, "format": "%.2f"},
    "min_end_soc_kwh": {"label": "Minimum End SoC (kWh)", "step": 0.1, "format": "%.2f"},
}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle, indent=4)
        file_handle.write("\n")


def coerce_number(config: dict, key: str) -> float:
    value = config.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
    
def render_config_form(config: dict, fields: dict, form_key: str, submit_label: str):
    with st.form(form_key):
        left_column, right_column = st.columns(2)
        updated_config = dict(config)

        field_items = list(fields.items())
        for index, (key, metadata) in enumerate(field_items):
            column = left_column if index % 2 == 0 else right_column
            with column:
                field_type = metadata.get("type", "number")

                if field_type == "text":
                    updated_config[key] = st.text_input(
                        metadata["label"],
                        value=str(config.get(key, "")),
                        key=f"{form_key}_{key}",
                    )
                elif field_type == "bool":
                    updated_config[key] = st.checkbox(
                        metadata["label"],
                        value=bool(config.get(key, False)),
                        key=f"{form_key}_{key}",
                    )
                else:
                    updated_config[key] = st.number_input(
                        metadata["label"],
                        value=coerce_number(config, key),
                        step=metadata["step"],
                        format=metadata["format"],
                        key=f"{form_key}_{key}",
                    )

        submitted = st.form_submit_button(submit_label, use_container_width=True)

    if submitted:
        return updated_config
    return None


st.title("Admin Center")
st.caption("Manage system configuration and browse historical project data.")

tab_system, tab_battery, tab_data = st.tabs(
    ["System Configuration", "Battery & Strategy", "Historical Data"]
)
with tab_system:
    st.subheader("System Configuration")

    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read configuration: {error}")
    else:
        with st.expander("Current JSON", expanded=False):
            st.json(config)

        updated = render_config_form(
            config=config,
            fields=SYSTEM_CONFIG_FIELDS,
            form_key="system_config_form",
            submit_label="Save system configuration",
        )

        if updated is not None:
            try:
                save_config(updated)
            except OSError as error:
                st.error(f"Could not save configuration: {error}")
            else:
                st.success(f"Saved configuration to {CONFIG_PATH.name}.")


with tab_battery:
    st.subheader("Battery & Strategy")

    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read configuration: {error}")
    else:
        updated = render_config_form(
            config=config,
            fields=BATTERY_STRATEGY_FIELDS,
            form_key="battery_strategy_form",
            submit_label="Save battery & strategy settings",
        )

        if updated is not None:
            try:
                save_config(updated)
            except OSError as error:
                st.error(f"Could not save configuration: {error}")
            else:
                st.success(f"Saved configuration to {CONFIG_PATH.name}.")

with tab_data:
    st.subheader("Historical Data")

    data_dir = Path("data")

    if not data_dir.exists():
        st.error("The data directory does not exist.")
        st.stop()

    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        st.warning("No CSV files were found in the data directory.")
        st.stop()

    file_options = {str(path.relative_to(data_dir.parent)): path for path in csv_files}

    selected_file_label = st.selectbox("Select a CSV file", list(file_options.keys()))
    selected_file = file_options[selected_file_label]

    df = pd.read_csv(selected_file)

    st.markdown(f"**Selected file:** `{selected_file_label}`")
    st.markdown(f"**Rows:** {len(df):,} &nbsp;&nbsp; **Columns:** {len(df.columns)}")

    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
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