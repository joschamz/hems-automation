import json
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
USER_CONFIG_PATH = ROOT_DIR / "user_config.json"
SYSTEM_CONFIG_PATH = ROOT_DIR / "system_config.json"


USER_CONFIG_FIELDS = {
    "lat": {
        "label": "Latitude",
        "type": "float",
        "step": 0.000001,
        "format": "%.6f",
        "help": "Site latitude in decimal degrees used for weather and solar models.",
    },
    "lon": {
        "label": "Longitude",
        "type": "float",
        "step": 0.000001,
        "format": "%.6f",
        "help": "Site longitude in decimal degrees used for weather and solar models.",
    },
    "kwp": {
        "label": "PV Size (kWp)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Installed PV peak power in kWp.",
    },
    "yield_factor": {
        "label": "Yield Factor",
        "type": "float",
        "step": 0.01,
        "format": "%.2f",
        "help": "Overall PV derating factor for non-ideal performance (e.g., 0.7).",
    },
    "tilt": {
        "label": "Tilt (deg)",
        "type": "float",
        "step": 1.0,
        "format": "%.1f",
        "help": "PV panel tilt angle from horizontal in degrees.",
    },
    "azimuth": {
        "label": "Azimuth (deg)",
        "type": "float",
        "step": 1.0,
        "format": "%.1f",
        "help": "PV orientation angle in degrees (0 usually south in this setup).",
    },
    "battery_capacity_kwh": {
        "label": "Battery Capacity (kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Nominal usable battery energy capacity.",
    },
    "max_charge_kw": {
        "label": "Max Charge Power (kW)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Maximum battery charging power.",
    },
    "charge_efficiency": {
        "label": "Charge Efficiency",
        "type": "float",
        "step": 0.01,
        "format": "%.2f",
        "help": "Battery charging efficiency from input energy to stored energy.",
    },
    "discharge_efficiency": {
        "label": "Discharge Efficiency",
        "type": "float",
        "step": 0.01,
        "format": "%.2f",
        "help": "Battery discharging efficiency from stored energy to delivered energy.",
    },
    "max_discharge_kw": {
        "label": "Max Discharge Power (kW)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Maximum battery discharging power.",
    },
    "soc_min_kwh": {
        "label": "SoC Minimum (kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Minimum allowed battery state-of-charge reserve.",
    },
    "soc_max_kwh": {
        "label": "SoC Maximum (kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Maximum allowed battery state-of-charge.",
    },
}


SYSTEM_CONFIG_FIELDS = {
    "interval_minutes": {
        "label": "Interval Minutes",
        "type": "int",
        "step": 1,
        "help": "Dispatch time-step in minutes. Must divide 60 (e.g., 15).",
    },
    "optimization_horizon_hours": {
        "label": "Optimization Horizon (hours)",
        "type": "int",
        "step": 1,
        "help": "Look-ahead window used by rule-based and LP dispatch calculations.",
    },
    "action_horizon_hours": {
        "label": "Action Horizon (hours)",
        "type": "int",
        "step": 1,
        "help": "Only this initial part of the optimization horizon is exported/actioned.",
    },
    "update_frequency_hours": {
        "label": "Update Frequency (hours)",
        "type": "int",
        "step": 1,
        "help": "Planned rerun cadence for rolling forecast updates.",
    },
    "default_sell_price_cent_kwh": {
        "label": "Default Sell Price (cent/kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Fallback export tariff if sell price is missing in input data.",
    },
    "allow_grid_charging": {
        "label": "Allow Grid Charging",
        "type": "bool",
        "help": "If enabled, battery may charge from grid under configured rules.",
    },
    "grid_charge_price_threshold_cent_kwh": {
        "label": "Grid Charge Price Threshold (cent/kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Only charge from grid when buy price is at or below this threshold.",
    },
    "cycle_penalty_cent_per_kwh": {
        "label": "Cycle Penalty (cent/kWh)",
        "type": "float",
        "step": 0.001,
        "format": "%.3f",
        "help": "Degradation proxy cost added per kWh charged/discharged.",
    },
    "enforce_solar_first_in_lp": {
        "label": "Enforce Solar-First In LP",
        "type": "bool",
        "help": "If enabled, LP prefers direct PV-to-load before battery/grid supply.",
    },
    "terminal_soc_value_cent_kwh": {
        "label": "Terminal SoC Value (cent/kWh)",
        "type": "float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Residual value assigned to ending SoC at optimization horizon.",
    },
    "min_end_soc_kwh": {
        "label": "Min End SoC (kWh)",
        "type": "nullable_float",
        "step": 0.1,
        "format": "%.2f",
        "help": "Optional hard lower bound for SoC at the end of horizon. Set None to disable.",
    },
}


def render_field(config: dict, key: str, field: dict, widget_key: str):
    field_type = field["type"]
    value = config.get(key)

    if field_type == "float":
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return st.number_input(
            field["label"],
            value=value,
            step=float(field.get("step", 0.1)),
            format=field.get("format", "%.2f"),
            help=field.get("help"),
            key=widget_key,
        )

    if field_type == "int":
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        return int(
            st.number_input(
                field["label"],
                value=value,
                step=int(field.get("step", 1)),
                help=field.get("help"),
                key=widget_key,
            )
        )

    if field_type == "bool":
        return st.checkbox(
            field["label"],
            value=bool(value),
            help=field.get("help"),
            key=widget_key,
        )

    none_key = f"{widget_key}_none"
    use_none = st.checkbox(
        f"{field['label']} is None",
        value=value is None,
        help=field.get("help"),
        key=none_key,
    )
    if use_none:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    return st.number_input(
        field["label"],
        value=value,
        step=float(field.get("step", 0.1)),
        format=field.get("format", "%.2f"),
        help=field.get("help"),
        key=widget_key,
    )


def render_installation_tab() -> None:
    st.subheader("Installation Configuration")
    st.caption("Values of the HEMS setup (site and battery characteristics).")

    try:
        config = json.loads(USER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read {USER_CONFIG_PATH.name}: {error}")
        return

    updated = dict(config)

    with st.form("user_config_form"):
        with st.expander("Site Location", expanded=True):
            left_column, right_column = st.columns(2)
            with left_column:
                updated["lat"] = render_field(updated, "lat", USER_CONFIG_FIELDS["lat"], "user_config_form_lat")
            with right_column:
                updated["lon"] = render_field(updated, "lon", USER_CONFIG_FIELDS["lon"], "user_config_form_lon")

        with st.expander("PV System", expanded=True):
            left_column, divider_col, right_column = st.columns([1, 0.04, 1])
            with left_column:
                st.markdown("**Performance**")
                updated["kwp"] = render_field(updated, "kwp", USER_CONFIG_FIELDS["kwp"], "user_config_form_kwp")
                updated["yield_factor"] = render_field(
                    updated,
                    "yield_factor",
                    USER_CONFIG_FIELDS["yield_factor"],
                    "user_config_form_yield_factor",
                )
            with divider_col:
                st.markdown(
                    '<div style="border-left:2px solid #e0e0e0; height:100%; min-height:200px; margin:0 auto;"></div>',
                    unsafe_allow_html=True,
                )
            with right_column:
                st.markdown("**Orientation**")
                updated["tilt"] = render_field(updated, "tilt", USER_CONFIG_FIELDS["tilt"], "user_config_form_tilt")
                updated["azimuth"] = render_field(
                    updated,
                    "azimuth",
                    USER_CONFIG_FIELDS["azimuth"],
                    "user_config_form_azimuth",
                )

        with st.expander("Battery Hardware", expanded=True):
            left_column, divider_col, right_column = st.columns([1, 0.04, 1])
            with left_column:
                st.markdown("**Capacity**")
                updated["battery_capacity_kwh"] = render_field(
                    updated,
                    "battery_capacity_kwh",
                    USER_CONFIG_FIELDS["battery_capacity_kwh"],
                    "user_config_form_battery_capacity_kwh",
                )
                updated["soc_min_kwh"] = render_field(
                    updated,
                    "soc_min_kwh",
                    USER_CONFIG_FIELDS["soc_min_kwh"],
                    "user_config_form_soc_min_kwh",
                )
                updated["soc_max_kwh"] = render_field(
                    updated,
                    "soc_max_kwh",
                    USER_CONFIG_FIELDS["soc_max_kwh"],
                    "user_config_form_soc_max_kwh",
                )
            with divider_col:
                st.markdown(
                    '<div style="border-left:2px solid #e0e0e0; height:100%; min-height:360px; margin:0 auto;"></div>',
                    unsafe_allow_html=True,
                )
            with right_column:
                st.markdown("**Performance**")
                updated["max_charge_kw"] = render_field(
                    updated,
                    "max_charge_kw",
                    USER_CONFIG_FIELDS["max_charge_kw"],
                    "user_config_form_max_charge_kw",
                )
                updated["charge_efficiency"] = render_field(
                    updated,
                    "charge_efficiency",
                    USER_CONFIG_FIELDS["charge_efficiency"],
                    "user_config_form_charge_efficiency",
                )

                updated["max_discharge_kw"] = render_field(
                    updated,
                    "max_discharge_kw",
                    USER_CONFIG_FIELDS["max_discharge_kw"],
                    "user_config_form_max_discharge_kw",
                )

                updated["discharge_efficiency"] = render_field(
                    updated,
                    "discharge_efficiency",
                    USER_CONFIG_FIELDS["discharge_efficiency"],
                    "user_config_form_discharge_efficiency",
                )

        submitted = st.form_submit_button("Save", use_container_width=True)

    if submitted:
        try:
            USER_CONFIG_PATH.write_text(
                json.dumps(updated, indent=4, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            st.error(f"Could not save {USER_CONFIG_PATH.name}: {error}")
            return
        st.success(f"Saved {USER_CONFIG_PATH.name}")


def render_dispatch_tab() -> None:
    st.subheader("Dispatch System Configuration")
    st.caption("Tunable optimization and rule-dispatch behavior for user/admin.")

    try:
        config = json.loads(SYSTEM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not read {SYSTEM_CONFIG_PATH.name}: {error}")
        return

    updated = dict(config)

    with st.form("system_config_form"):
        with st.expander("Dispatch Timing", expanded=True):
            left_column, right_column = st.columns(2)
            with left_column:
                updated["interval_minutes"] = render_field(
                    updated,
                    "interval_minutes",
                    SYSTEM_CONFIG_FIELDS["interval_minutes"],
                    "system_config_form_interval_minutes",
                )
                updated["optimization_horizon_hours"] = render_field(
                    updated,
                    "optimization_horizon_hours",
                    SYSTEM_CONFIG_FIELDS["optimization_horizon_hours"],
                    "system_config_form_optimization_horizon_hours",
                )
            with right_column:
                updated["action_horizon_hours"] = render_field(
                    updated,
                    "action_horizon_hours",
                    SYSTEM_CONFIG_FIELDS["action_horizon_hours"],
                    "system_config_form_action_horizon_hours",
                )
                updated["update_frequency_hours"] = render_field(
                    updated,
                    "update_frequency_hours",
                    SYSTEM_CONFIG_FIELDS["update_frequency_hours"],
                    "system_config_form_update_frequency_hours",
                )

        with st.expander("Prices and Grid Charging Policy", expanded=True):
            left_column, divider_col, right_column = st.columns([1, 0.04, 1])

            with left_column:
                updated["default_sell_price_cent_kwh"] = render_field(
                    updated,
                    "default_sell_price_cent_kwh",
                    SYSTEM_CONFIG_FIELDS["default_sell_price_cent_kwh"],
                    "system_config_form_default_sell_price_cent_kwh",
                )
            with divider_col:
                st.markdown(
                    '<div style="border-left:2px solid #e0e0e0; height:100%; min-height:120px; margin:0 auto;"></div>',
                    unsafe_allow_html=True,
                )
            with right_column:
                updated["allow_grid_charging"] = render_field(
                        updated,
                        "allow_grid_charging",
                        SYSTEM_CONFIG_FIELDS["allow_grid_charging"],
                        "system_config_form_allow_grid_charging",
                    )
                updated["grid_charge_price_threshold_cent_kwh"] = render_field(
                    updated,
                    "grid_charge_price_threshold_cent_kwh",
                    SYSTEM_CONFIG_FIELDS["grid_charge_price_threshold_cent_kwh"],
                    "system_config_form_grid_charge_price_threshold_cent_kwh",
                )

        with st.expander("LP Objective", expanded=True):
            left_column, right_column = st.columns(2)
            with left_column:
                updated["cycle_penalty_cent_per_kwh"] = render_field(
                    updated,
                    "cycle_penalty_cent_per_kwh",
                    SYSTEM_CONFIG_FIELDS["cycle_penalty_cent_per_kwh"],
                    "system_config_form_cycle_penalty_cent_per_kwh",
                )
                updated["enforce_solar_first_in_lp"] = render_field(
                    updated,
                    "enforce_solar_first_in_lp",
                    SYSTEM_CONFIG_FIELDS["enforce_solar_first_in_lp"],
                    "system_config_form_enforce_solar_first_in_lp",
                )
            with right_column:
                updated["terminal_soc_value_cent_kwh"] = render_field(
                    updated,
                    "terminal_soc_value_cent_kwh",
                    SYSTEM_CONFIG_FIELDS["terminal_soc_value_cent_kwh"],
                    "system_config_form_terminal_soc_value_cent_kwh",
                )
                updated["min_end_soc_kwh"] = render_field(
                    updated,
                    "min_end_soc_kwh",
                    SYSTEM_CONFIG_FIELDS["min_end_soc_kwh"],
                    "system_config_form_min_end_soc_kwh",
                )

        submitted = st.form_submit_button("Save", use_container_width=True)

    if submitted:
        try:
            SYSTEM_CONFIG_PATH.write_text(
                json.dumps(updated, indent=4, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            st.error(f"Could not save {SYSTEM_CONFIG_PATH.name}: {error}")
            return
        st.success(f"Saved {SYSTEM_CONFIG_PATH.name}")


def render_page() -> None:
    st.title("Configuration")

    installation_tab, dispatch_tab = st.tabs(
        ["Installation Configuration", "Dispatch System Configuration"]
    )

    with installation_tab:
        render_installation_tab()

    with dispatch_tab:
        render_dispatch_tab()


render_page()
