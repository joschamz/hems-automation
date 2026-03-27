from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog

PathLike = Union[str, Path]

REQUIRED_INPUT_COLUMNS = [
    "utc_timestamp",
    "pv_generation_kwh",
    "energy_price_buy_cent_kwh",
    "household_load_kwh",
]

OPTIONAL_INPUT_COLUMNS = [
    "energy_price_sell_cent_kwh",
    "soc_min_dynamic_kwh",
]

FLOW_COLUMNS = [
    "grid_to_load_kwh",
    "pv_to_load_kwh",
    "pv_to_battery_kwh",
    "battery_to_load_kwh",
    "grid_to_battery_kwh",
    "export_to_grid_kwh",
]

OUTPUT_COLUMNS = [
    "utc_timestamp",
    "grid_to_load_kwh",
    "pv_to_load_kwh",
    "pv_to_battery_kwh",
    "battery_to_load_kwh",
    "grid_to_battery_kwh",
    "export_to_grid_kwh",
    "soc_kwh",
    "soc_min_dynamic_kwh",
    "energy_price_buy_cent_kwh",
    "energy_price_sell_cent_kwh",
    "interval_cost_cent",
    "cumulative_cost_cent",
    "decision_rule",
    "method",
]


def _resolve_path(path_like: PathLike) -> Path:
    path = Path(path_like)

    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    found = next((candidate for candidate in candidates if candidate.exists()), None)
    if found is not None:
        return found

    raise FileNotFoundError(f"Could not find file: {path_like}")


def load_dispatch_params(
    user_config_path: PathLike = "user_config.json",
    system_config_path: PathLike = "system_config.json",
) -> dict[str, Any]:
    user_config = json.loads(_resolve_path(user_config_path).read_text(encoding="utf-8"))
    system_config = json.loads(_resolve_path(system_config_path).read_text(encoding="utf-8"))

    required_user_keys = [
        "battery_capacity_kwh",
        "max_charge_kw",
        "max_discharge_kw",
        "charge_efficiency",
        "discharge_efficiency",
        "soc_min_kwh",
        "soc_max_kwh",
    ]
    required_system_keys = [
        "interval_minutes",
        "default_sell_price_cent_kwh",
        "allow_grid_charging",
        "grid_charge_price_threshold_cent_kwh",
        "cycle_penalty_cent_per_kwh",
        "enforce_solar_first_in_lp",
        "terminal_soc_value_cent_kwh",
        "min_end_soc_kwh",
        "optimization_horizon_hours",
        "action_horizon_hours",
        "update_frequency_hours",
    ]

    missing_user = [k for k in required_user_keys if k not in user_config]
    missing_system = [k for k in required_system_keys if k not in system_config]
    if missing_user or missing_system:
        raise KeyError(
            f"Missing config keys - user_config: {missing_user}, system_config: {missing_system}"
        )

    return {
        "interval_minutes": int(system_config["interval_minutes"]),
        "battery_capacity_kwh": float(user_config["battery_capacity_kwh"]),
        "soc_min_kwh": float(user_config["soc_min_kwh"]),
        "soc_max_kwh": float(user_config["soc_max_kwh"]),
        "max_charge_kw": float(user_config["max_charge_kw"]),
        "max_discharge_kw": float(user_config["max_discharge_kw"]),
        "charge_efficiency": float(user_config["charge_efficiency"]),
        "discharge_efficiency": float(user_config["discharge_efficiency"]),
        "default_sell_price_cent_kwh": float(system_config["default_sell_price_cent_kwh"]),
        "allow_grid_charging": bool(system_config["allow_grid_charging"]),
        "grid_charge_price_threshold_cent_kwh": float(system_config["grid_charge_price_threshold_cent_kwh"]),
        "cycle_penalty_cent_per_kwh": float(system_config["cycle_penalty_cent_per_kwh"]),
        "enforce_solar_first_in_lp": bool(system_config["enforce_solar_first_in_lp"]),
        "terminal_soc_value_cent_kwh": float(system_config["terminal_soc_value_cent_kwh"]),
        "min_end_soc_kwh": (
            None
            if system_config["min_end_soc_kwh"] is None
            else float(system_config["min_end_soc_kwh"])
        ),
        "optimization_horizon_hours": int(system_config["optimization_horizon_hours"]),
        "action_horizon_hours": int(system_config["action_horizon_hours"]),
        "update_frequency_hours": int(system_config["update_frequency_hours"]),
    }


def compute_horizon_intervals(params: dict[str, Any]) -> dict[str, int]:
    interval_minutes = int(params["interval_minutes"])
    if interval_minutes <= 0 or 60 % interval_minutes != 0:
        raise ValueError(f"interval_minutes={interval_minutes} must be a positive divisor of 60.")

    intervals_per_hour = 60 // interval_minutes
    optimization_intervals = int(params["optimization_horizon_hours"] * intervals_per_hour)
    action_intervals = int(params["action_horizon_hours"] * intervals_per_hour)

    return {
        "interval_minutes": interval_minutes,
        "intervals_per_hour": intervals_per_hour,
        "optimization_intervals": optimization_intervals,
        "action_intervals": action_intervals,
        "update_frequency_hours": int(params["update_frequency_hours"]),
    }


def prepare_forecast_input(input_df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    missing_required = [col for col in REQUIRED_INPUT_COLUMNS if col not in input_df.columns]
    if missing_required:
        raise ValueError(f"Missing required input columns: {missing_required}")

    df = input_df.copy()
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    if df["utc_timestamp"].isna().any():
        bad_rows = df.index[df["utc_timestamp"].isna()].tolist()[:5]
        raise ValueError(f"Invalid utc_timestamp values at rows: {bad_rows}")

    if df["utc_timestamp"].duplicated().any():
        duplicate_count = int(df["utc_timestamp"].duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate utc_timestamp rows.")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    numeric_cols = [
        "pv_generation_kwh",
        "energy_price_buy_cent_kwh",
        "household_load_kwh",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN after numeric conversion.")

    if "energy_price_sell_cent_kwh" not in df.columns:
        df["energy_price_sell_cent_kwh"] = params["default_sell_price_cent_kwh"]
    else:
        df["energy_price_sell_cent_kwh"] = pd.to_numeric(df["energy_price_sell_cent_kwh"], errors="coerce")
        df["energy_price_sell_cent_kwh"] = df["energy_price_sell_cent_kwh"].fillna(
            params["default_sell_price_cent_kwh"]
        )

    if "soc_min_dynamic_kwh" not in df.columns:
        df["soc_min_dynamic_kwh"] = float(params["soc_min_kwh"])
    else:
        df["soc_min_dynamic_kwh"] = pd.to_numeric(df["soc_min_dynamic_kwh"], errors="coerce")
        df["soc_min_dynamic_kwh"] = df["soc_min_dynamic_kwh"].fillna(float(params["soc_min_kwh"]))

    df["soc_min_dynamic_kwh"] = df["soc_min_dynamic_kwh"].clip(
        lower=float(params["soc_min_kwh"]),
        upper=float(params["soc_max_kwh"]),
    )

    for non_negative_col in ["pv_generation_kwh", "household_load_kwh"]:
        if (df[non_negative_col] < 0).any():
            min_value = float(df[non_negative_col].min())
            raise ValueError(
                f"Column '{non_negative_col}' must be non-negative. Found min={min_value}"
            )

    if df.empty:
        raise ValueError("Input table is empty.")

    horizon = compute_horizon_intervals(params)
    min_required = horizon["optimization_intervals"]
    if len(df) < min_required:
        raise ValueError(
            f"Input forecast must cover at least {params['optimization_horizon_hours']}h ({min_required} rows). "
            f"Got {len(df)} rows. The full optimization horizon is always required even though only "
            f"{params['action_horizon_hours']}h are actioned."
        )

    expected_step = pd.Timedelta(minutes=horizon["interval_minutes"])
    wrong_step = df["utc_timestamp"].diff().dropna() != expected_step
    if wrong_step.any():
        wrong_positions = wrong_step[wrong_step].index.tolist()[:5]
        raise ValueError(
            f"Timestamp spacing is not {horizon['interval_minutes']}-minute UTC at positions: {wrong_positions}"
        )

    return df


def validate_initial_soc(initial_soc_kwh: float, params: dict[str, Any], tolerance: float = 1e-9) -> None:
    soc_min = float(params["soc_min_kwh"])
    soc_max = float(params["soc_max_kwh"])
    if not (soc_min - tolerance <= float(initial_soc_kwh) <= soc_max + tolerance):
        raise ValueError(
            f"initial_soc_kwh={initial_soc_kwh} is outside [{soc_min}, {soc_max}]"
        )


def finalize_dispatch_output(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    cycle_penalty = float(params.get("cycle_penalty_cent_per_kwh", 0.0))
    output = df.copy()
    output["soc_percent"] = (output["soc_kwh"] / float(params["battery_capacity_kwh"])) * 100.0
    pv_used = output["pv_to_load_kwh"] + output["pv_to_battery_kwh"] + output["export_to_grid_kwh"]
    output["curtailed_pv_kwh"] = np.maximum(0.0, output["pv_generation_kwh"] - pv_used)
    output["total_import_kwh"] = output["grid_to_load_kwh"] + output["grid_to_battery_kwh"]
    output["total_export_kwh"] = output["export_to_grid_kwh"]
    output["interval_cost_cent"] = (
        output["energy_price_buy_cent_kwh"] * output["total_import_kwh"]
        - output["energy_price_sell_cent_kwh"] * output["total_export_kwh"]
        + cycle_penalty
        * (
            output["pv_to_battery_kwh"]
            + output["grid_to_battery_kwh"]
            + output["battery_to_load_kwh"]
        )
    )
    output["cumulative_cost_cent"] = output["interval_cost_cent"].cumsum()
    return output


def run_rule_based_dispatch(
    forecast_df: pd.DataFrame,
    params: dict[str, Any],
    initial_soc_kwh: float,
) -> pd.DataFrame:
    dt_hours = params["interval_minutes"] / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours
    eta_c = params["charge_efficiency"]
    eta_d = params["discharge_efficiency"]
    soc_min = params["soc_min_kwh"]
    soc_max = params["soc_max_kwh"]
    allow_grid_charging = bool(params.get("allow_grid_charging", False))
    grid_charge_threshold = params.get("grid_charge_price_threshold_cent_kwh")

    soc = float(initial_soc_kwh)
    rows: list[dict[str, Any]] = []

    for row in forecast_df.itertuples(index=False):
        timestamp = row.utc_timestamp
        load_remaining = float(row.household_load_kwh)
        pv_remaining = float(row.pv_generation_kwh)
        buy_price = float(row.energy_price_buy_cent_kwh)
        sell_price = float(row.energy_price_sell_cent_kwh)
        interval_soc_min = float(np.clip(getattr(row, "soc_min_dynamic_kwh", soc_min), soc_min, soc_max))

        pv_to_load_kwh = min(load_remaining, pv_remaining)
        load_remaining -= pv_to_load_kwh
        pv_remaining -= pv_to_load_kwh

        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        pv_to_battery_kwh = min(pv_remaining, charge_limit_kwh, charge_headroom_kwh)
        soc += pv_to_battery_kwh * eta_c
        pv_remaining -= pv_to_battery_kwh

        available_discharge_kwh = max(0.0, (soc - interval_soc_min) * eta_d)
        battery_to_load_kwh = min(load_remaining, discharge_limit_kwh, available_discharge_kwh)
        soc -= battery_to_load_kwh / eta_d
        load_remaining -= battery_to_load_kwh

        grid_to_load_kwh = load_remaining

        charge_limit_left_kwh = max(0.0, charge_limit_kwh - pv_to_battery_kwh)
        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        should_grid_charge = (
            allow_grid_charging
            and grid_charge_threshold is not None
            and buy_price <= float(grid_charge_threshold)
        )
        grid_to_battery_kwh = min(charge_limit_left_kwh, charge_headroom_kwh) if should_grid_charge else 0.0
        soc += grid_to_battery_kwh * eta_c

        export_to_grid_kwh = pv_remaining

        if soc < interval_soc_min - 1e-6 or soc > soc_max + 1e-6:
            raise ValueError(
                f"SoC out of bounds at {timestamp}: soc={soc:.6f}, bounds=[{interval_soc_min}, {soc_max}]"
            )
        soc = float(np.clip(soc, interval_soc_min, soc_max))

        flow_names = [
            "pv_to_load",
            "pv_to_battery",
            "battery_to_load",
            "grid_to_load",
            "grid_to_battery",
            "export_to_grid",
        ]
        flow_vals = [
            pv_to_load_kwh,
            pv_to_battery_kwh,
            battery_to_load_kwh,
            grid_to_load_kwh,
            grid_to_battery_kwh,
            export_to_grid_kwh,
        ]
        decision_rule = " | ".join(n for n, v in zip(flow_names, flow_vals) if v > 1e-9) or "idle"

        rows.append(
            {
                "utc_timestamp": timestamp,
                "pv_generation_kwh": row.pv_generation_kwh,
                "household_load_kwh": row.household_load_kwh,
                "energy_price_buy_cent_kwh": buy_price,
                "energy_price_sell_cent_kwh": sell_price,
                "grid_to_load_kwh": grid_to_load_kwh,
                "pv_to_load_kwh": pv_to_load_kwh,
                "pv_to_battery_kwh": pv_to_battery_kwh,
                "battery_to_load_kwh": battery_to_load_kwh,
                "grid_to_battery_kwh": grid_to_battery_kwh,
                "export_to_grid_kwh": export_to_grid_kwh,
                "soc_kwh": soc,
                "soc_min_dynamic_kwh": interval_soc_min,
                "decision_rule": decision_rule,
                "method": "rule_based",
            }
        )

    return finalize_dispatch_output(pd.DataFrame(rows), params)


def run_lp_dispatch(
    forecast_df: pd.DataFrame,
    params: dict[str, Any],
    initial_soc_kwh: float,
    enforce_solar_first: bool | None = None,
) -> pd.DataFrame:
    n = len(forecast_df)
    dt_hours = params["interval_minutes"] / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours

    eta_c = float(params["charge_efficiency"])
    eta_d = float(params["discharge_efficiency"])
    soc_min = float(params["soc_min_kwh"])
    soc_max = float(params["soc_max_kwh"])

    validate_initial_soc(initial_soc_kwh, params)

    load = forecast_df["household_load_kwh"].to_numpy(dtype=float)
    pv = forecast_df["pv_generation_kwh"].to_numpy(dtype=float)
    buy = forecast_df["energy_price_buy_cent_kwh"].to_numpy(dtype=float)
    sell = forecast_df["energy_price_sell_cent_kwh"].to_numpy(dtype=float)

    if "soc_min_dynamic_kwh" in forecast_df.columns:
        soc_min_dynamic = forecast_df["soc_min_dynamic_kwh"].to_numpy(dtype=float)
    else:
        soc_min_dynamic = np.full(shape=n, fill_value=soc_min, dtype=float)
    soc_min_dynamic = np.clip(soc_min_dynamic, soc_min, soc_max)

    cycle_penalty = float(params.get("cycle_penalty_cent_per_kwh", 0.0))
    if enforce_solar_first is None:
        enforce_solar_first = bool(params.get("enforce_solar_first_in_lp", True))
    terminal_soc_value = float(params.get("terminal_soc_value_cent_kwh", 0.0))
    min_end_soc_kwh = params.get("min_end_soc_kwh")

    idx_gl = np.arange(0, n)
    idx_pl = np.arange(n, 2 * n)
    idx_pb = np.arange(2 * n, 3 * n)
    idx_bl = np.arange(3 * n, 4 * n)
    idx_gb = np.arange(4 * n, 5 * n)
    idx_ex = np.arange(5 * n, 6 * n)
    idx_soc = np.arange(6 * n, 7 * n + 1)
    num_vars = 7 * n + 1

    c = np.zeros(num_vars, dtype=float)
    c[idx_gl] = buy
    c[idx_gb] = buy
    c[idx_ex] = -sell
    c[idx_pb] += cycle_penalty
    c[idx_gb] += cycle_penalty
    c[idx_bl] += cycle_penalty
    c[idx_soc[-1]] -= terminal_soc_value

    a_eq_rows = []
    b_eq = []

    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_gl[t]] = 1.0
        row[idx_pl[t]] = 1.0
        row[idx_bl[t]] = 1.0
        a_eq_rows.append(row)
        b_eq.append(load[t])

        row = np.zeros(num_vars, dtype=float)
        row[idx_soc[t + 1]] = 1.0
        row[idx_soc[t]] = -1.0
        row[idx_pb[t]] = -eta_c
        row[idx_gb[t]] = -eta_c
        row[idx_bl[t]] = 1.0 / eta_d
        a_eq_rows.append(row)
        b_eq.append(0.0)

    a_ub_rows = []
    b_ub = []

    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_pl[t]] = 1.0
        row[idx_pb[t]] = 1.0
        row[idx_ex[t]] = 1.0
        a_ub_rows.append(row)
        b_ub.append(pv[t])

        row = np.zeros(num_vars, dtype=float)
        row[idx_pb[t]] = 1.0
        row[idx_gb[t]] = 1.0
        a_ub_rows.append(row)
        b_ub.append(charge_limit_kwh)

    if enforce_solar_first:
        for t in range(n):
            row = np.zeros(num_vars, dtype=float)
            row[idx_gl[t]] = 1.0
            row[idx_bl[t]] = 1.0
            a_ub_rows.append(row)
            b_ub.append(max(0.0, load[t] - pv[t]))

    bounds = [(0.0, None)] * num_vars
    for idx in idx_pb:
        bounds[idx] = (0.0, charge_limit_kwh)
    for idx in idx_bl:
        bounds[idx] = (0.0, discharge_limit_kwh)
    for idx in idx_gb:
        bounds[idx] = (0.0, charge_limit_kwh)

    bounds[idx_soc[0]] = (float(initial_soc_kwh), float(initial_soc_kwh))
    for t in range(n):
        bounds[idx_soc[t + 1]] = (float(soc_min_dynamic[t]), soc_max)

    if min_end_soc_kwh is not None:
        min_end_soc = float(min_end_soc_kwh)
        if min_end_soc > soc_max:
            raise ValueError(f"min_end_soc_kwh={min_end_soc} exceeds soc_max_kwh={soc_max}.")
        min_end_soc = max(soc_min, min_end_soc)
        current_lb, current_ub = bounds[idx_soc[-1]]
        bounds[idx_soc[-1]] = (max(current_lb, min_end_soc), current_ub)

    result = linprog(
        c=c,
        A_ub=np.array(a_ub_rows, dtype=float) if a_ub_rows else None,
        b_ub=np.array(b_ub, dtype=float) if b_ub else None,
        A_eq=np.array(a_eq_rows, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP optimization failed: {result.message}")

    x = result.x
    output_df = forecast_df.copy()
    output_df["grid_to_load_kwh"] = x[idx_gl]
    output_df["pv_to_load_kwh"] = x[idx_pl]
    output_df["pv_to_battery_kwh"] = x[idx_pb]
    output_df["battery_to_load_kwh"] = x[idx_bl]
    output_df["grid_to_battery_kwh"] = x[idx_gb]
    output_df["export_to_grid_kwh"] = x[idx_ex]
    output_df["soc_kwh"] = x[idx_soc[1:]]
    output_df["soc_min_dynamic_kwh"] = soc_min_dynamic
    output_df["decision_rule"] = "optimizer_lp"
    output_df["method"] = "lp_optimized"

    return finalize_dispatch_output(output_df, params)


def add_rule_activity_columns(df: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    output = df.copy()

    output["rule_pv_to_load"] = output["pv_to_load_kwh"] > tolerance
    output["rule_pv_to_battery"] = output["pv_to_battery_kwh"] > tolerance
    output["rule_battery_to_load"] = output["battery_to_load_kwh"] > tolerance
    output["rule_grid_to_load"] = output["grid_to_load_kwh"] > tolerance
    output["rule_grid_to_battery"] = output["grid_to_battery_kwh"] > tolerance
    output["rule_export_to_grid"] = output["export_to_grid_kwh"] > tolerance

    flow_rule_cols = [
        "rule_pv_to_load",
        "rule_pv_to_battery",
        "rule_battery_to_load",
        "rule_grid_to_load",
        "rule_grid_to_battery",
        "rule_export_to_grid",
    ]
    output["rule_idle"] = ~output[flow_rule_cols].any(axis=1)
    output["rule_optimizer_lp"] = output["method"].eq("lp_optimized")

    return output


def run_balance_checks(
    dispatch_df: pd.DataFrame,
    params: dict[str, Any],
    tolerance: float = 1e-6,
) -> pd.Series:
    load_balance_error = (
        dispatch_df["pv_to_load_kwh"]
        + dispatch_df["battery_to_load_kwh"]
        + dispatch_df["grid_to_load_kwh"]
        - dispatch_df["household_load_kwh"]
    ).abs().max()

    pv_balance_error = (
        dispatch_df["pv_to_load_kwh"]
        + dispatch_df["pv_to_battery_kwh"]
        + dispatch_df["export_to_grid_kwh"]
        + dispatch_df.get("curtailed_pv_kwh", 0.0)
        - dispatch_df["pv_generation_kwh"]
    ).abs().max()

    soc_ok = dispatch_df["soc_kwh"].between(
        float(params["soc_min_kwh"]) - tolerance,
        float(params["soc_max_kwh"]) + tolerance,
    ).all()

    return pd.Series(
        {
            "window_rows": len(dispatch_df),
            "max_load_balance_error_kwh": float(load_balance_error),
            "max_pv_balance_error_kwh": float(pv_balance_error),
            "soc_within_bounds": bool(soc_ok),
        }
    )


def slice_to_action_window(df_full: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    horizon = compute_horizon_intervals(params)
    action_intervals = horizon["action_intervals"]
    return df_full.iloc[:action_intervals].copy().reset_index(drop=True)


def run_dispatch_pipeline(
    forecast_df: pd.DataFrame,
    initial_soc_kwh: float,
    user_config_path: PathLike = "user_config.json",
    system_config_path: PathLike = "system_config.json",
    enforce_solar_first: bool | None = None,
    output_path: Path = Path(__file__).resolve().parents[1] / "data/runtime/dispatch_table.csv",
    save_output: bool = True,
) -> pd.DataFrame:
    params = load_dispatch_params(
        user_config_path=user_config_path,
        system_config_path=system_config_path,
    )
    prepared_forecast_df = prepare_forecast_input(forecast_df, params)
    validate_initial_soc(initial_soc_kwh, params)

    horizon = compute_horizon_intervals(params)
    optimization_intervals = horizon["optimization_intervals"]
    optimization_df = prepared_forecast_df.iloc[:optimization_intervals].copy().reset_index(drop=True)

    rule_df = run_rule_based_dispatch(
        forecast_df=optimization_df,
        params=params,
        initial_soc_kwh=initial_soc_kwh,
    )
    lp_df = run_lp_dispatch(
        forecast_df=optimization_df,
        params=params,
        initial_soc_kwh=initial_soc_kwh,
        enforce_solar_first=enforce_solar_first,
    )

    rule_action_df = slice_to_action_window(rule_df, params)
    lp_action_df = slice_to_action_window(lp_df, params)

    rule_action_df = add_rule_activity_columns(rule_action_df)
    lp_action_df = add_rule_activity_columns(lp_action_df)

    # Prepare columns to merge
    cols_to_merge = FLOW_COLUMNS + ["soc_kwh", "interval_cost_cent", "cumulative_cost_cent", "decision_rule"] + [
        "rule_pv_to_load", "rule_pv_to_battery", "rule_battery_to_load",
        "rule_grid_to_load", "rule_grid_to_battery", "rule_export_to_grid", "rule_idle", "rule_optimizer_lp"
    ]
    
    # Rename method columns with suffixes
    rule_merge = rule_action_df[["utc_timestamp"] + cols_to_merge].copy()
    lp_merge = lp_action_df[["utc_timestamp"] + cols_to_merge].copy()
    
    rule_merge_cols = {col: f"{col}_rule_based" for col in cols_to_merge}
    rule_merge.rename(columns=rule_merge_cols, inplace=True)
    
    lp_merge_cols = {col: f"{col}_lp_optimized" for col in cols_to_merge}
    lp_merge.rename(columns=lp_merge_cols, inplace=True)
    
    # Merge on timestamp
    combined_df = rule_merge.merge(lp_merge, on="utc_timestamp", how="inner")
    
    # Add back input columns
    combined_df = combined_df.merge(
        prepared_forecast_df[["utc_timestamp", "pv_generation_kwh", "household_load_kwh", 
                             "energy_price_buy_cent_kwh", "energy_price_sell_cent_kwh", "soc_min_dynamic_kwh"]],
        on="utc_timestamp",
        how="left"
    )
    
    # Reorder columns: timestamp first, then inputs, then rule-based, then LP
    input_cols = ["utc_timestamp", "pv_generation_kwh", "household_load_kwh", 
                  "energy_price_buy_cent_kwh", "energy_price_sell_cent_kwh", "soc_min_dynamic_kwh"]
    rule_cols = [c for c in combined_df.columns if c.endswith("_rule_based")]
    lp_cols = [c for c in combined_df.columns if c.endswith("_lp_optimized")]
    
    final_col_order = input_cols + rule_cols + lp_cols
    combined_df = combined_df[final_col_order]
    
    result_df = combined_df.reset_index(drop=True)
    
    # Optionally persist to CSV
    if save_output:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"✓ Saved dispatch output to: {output_path.resolve()}")
    
    return result_df


__all__ = [
    "REQUIRED_INPUT_COLUMNS",
    "OPTIONAL_INPUT_COLUMNS",
    "FLOW_COLUMNS",
    "OUTPUT_COLUMNS",
    "load_dispatch_params",
    "compute_horizon_intervals",
    "prepare_forecast_input",
    "validate_initial_soc",
    "finalize_dispatch_output",
    "run_rule_based_dispatch",
    "run_lp_dispatch",
    "add_rule_activity_columns",
    "run_balance_checks",
    "slice_to_action_window",
    "run_dispatch_pipeline",
]
 #Example usage in main.py:
# input_path = Path('../data/runtime/aggregated_table.csv')
# forecast_df = pd.read_csv(input_path)

# INITIAL_SOC_KWH = 5.0  # Replace with measured battery SoC before running

# result_df = run_dispatch_pipeline(
#     forecast_df=forecast_df,
#     initial_soc_kwh=INITIAL_SOC_KWH,
# )