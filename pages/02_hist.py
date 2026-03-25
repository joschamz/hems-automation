from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Historical Data", page_icon="📂", layout="wide")

st.title("Historical Data")
st.caption("Browse saved CSV outputs and historical artifacts from the project.")

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
        default=numeric_cols[: min(3, len(numeric_cols))]
    )

    if chosen_numeric:
        plot_df = df[chosen_numeric].copy()
        st.line_chart(plot_df)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download selected CSV",
    data=csv_bytes,
    file_name=selected_file.name,
    mime="text/csv"
)