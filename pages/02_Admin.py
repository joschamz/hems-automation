import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Admin Center", page_icon="📂", layout="wide")

CONFIG_PATH = Path("user_config.json")
CONFIG_FIELDS = {
    "lat": {"label": "Latitude", "step": 0.000001, "format": "%.6f"},
    "lon": {"label": "Longitude", "step": 0.000001, "format": "%.6f"},
    "kwp": {"label": "PV Size (kWp)", "step": 0.1, "format": "%.2f"},
    "tilt": {"label": "Tilt (deg)", "step": 1.0, "format": "%.1f"},
    "azimuth": {"label": "Azimuth (deg)", "step": 1.0, "format": "%.1f"},
    "yield_factor": {"label": "Yield Factor", "step": 0.01, "format": "%.2f"},
    "feature_dataset_path": {"label": "Feature Dataset Path","type": "text"},
"load_model_path": {"label": "Load Model Path","type": "text"},
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


st.title("Admin Center")
st.caption("Manage system configuration and browse historical project data.")

tab_config, tab_data = st.tabs(["Configuration", "Historical Data"])

with tab_config:
    st.subheader("System Configuration")

    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read configuration: {error}")
    else:
        with st.expander("Current JSON", expanded=False):
            st.json(config)

        with st.form("config_form"):
            left_column, right_column = st.columns(2)
            updated_config = dict(config)

            field_items = list(CONFIG_FIELDS.items())
            for index, (key, metadata) in enumerate(field_items):
                column = left_column if index % 2 == 0 else right_column
                with column:
                    if metadata.get("type") == "text":
                        updated_config[key] = st.text_input(
                            metadata["label"],
                            value=str(config.get(key, "")),
                            key=f"config_{key}",
                        )
                    else:
                        updated_config[key] = st.number_input(
                            metadata["label"],
                            value=coerce_number(config, key),
                            step=metadata["step"],
                            format=metadata["format"],
                            key=f"config_{key}",
                      )

            submitted = st.form_submit_button(
                "Save configuration",
                use_container_width=True,
            )

        if submitted:
            try:
                save_config(updated_config)
            except OSError as error:
                st.error(f"Could not save configuration: {error}")
            else:
                st.success(f"Saved configuration to {CONFIG_PATH.name}.")
                st.json(updated_config)

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