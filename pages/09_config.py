import json
from pathlib import Path

import streamlit as st


CONFIG_PATH = Path(__file__).resolve().parents[1] / "user_config.json"
CONFIG_FIELDS = {
	"lat": {"label": "Latitude", "step": 0.000001, "format": "%.6f"},
	"lon": {"label": "Longitude", "step": 0.000001, "format": "%.6f"},
	"kwp": {"label": "PV Size (kWp)", "step": 0.1, "format": "%.2f"},
	"tilt": {"label": "Tilt (deg)", "step": 1.0, "format": "%.1f"},
	"azimuth": {"label": "Azimuth (deg)", "step": 1.0, "format": "%.1f"},
	"yield_factor": {"label": "Yield Factor", "step": 0.01, "format": "%.2f"},
}


def _load_config() -> dict:
	with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
		return json.load(file_handle)


def _save_config(config: dict) -> None:
	with CONFIG_PATH.open("w", encoding="utf-8") as file_handle:
		json.dump(config, file_handle, indent=4)
		file_handle.write("\n")


def _coerce_number(config: dict, key: str) -> float:
	value = config.get(key, 0.0)
	try:
		return float(value)
	except (TypeError, ValueError):
		return 0.0


def render_page() -> None:
	st.title("Configuration")
	st.caption("Review and update the values stored in user_config.json.")

	try:
		config = _load_config()
	except (OSError, ValueError, json.JSONDecodeError) as error:
		st.error(f"Could not read configuration: {error}")
		return

	with st.expander("Current JSON", expanded=False):
		st.json(config)

	with st.form("config_form"):
		st.subheader("Edit Parameters")

		left_column, right_column = st.columns(2)
		updated_config = dict(config)

		field_names = list(CONFIG_FIELDS.items())
		for index, (key, metadata) in enumerate(field_names):
			column = left_column if index % 2 == 0 else right_column
			with column:
				updated_config[key] = st.number_input(
					metadata["label"],
					value=_coerce_number(config, key),
					step=metadata["step"],
					format=metadata["format"],
					key=f"config_{key}",
				)

		submitted = st.form_submit_button("Save configuration", use_container_width=True)

	if submitted:
		try:
			_save_config(updated_config)
		except OSError as error:
			st.error(f"Could not save configuration: {error}")
			return

		st.success(f"Saved configuration to {CONFIG_PATH.name}.")
		st.json(updated_config)


render_page()
