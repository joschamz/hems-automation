from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog

PathLike = Union[str, Path]

# Fixed dispatch constants: 48 h horizon at 15-min intervals
INTERVAL_MINUTES: int = 15
HORIZON_INTERVALS: int = 192  # 48 h × 4 intervals/h

FLOW_COLUMNS = [
    "grid_to_load_kwh",
    "pv_to_load_kwh",
    "pv_to_battery_kwh",
    "battery_to_load_kwh",
    "grid_to_battery_kwh",
    "export_to_grid_kwh",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_AGGREGATED_CSV = _REPO_ROOT / "data/runtime/aggregated_table.csv"
_DEFAULT_DISPATCH_CSV = _REPO_ROOT / "data/runtime/dispatch_table.csv"


def _load_params(
    user_config_path: PathLike = "user_config.json",
    system_config_path: PathLike = "system_config.json",
) -> dict[str, Any]:
    def _resolve(p: PathLike) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        for candidate in (Path.cwd() / path, _REPO_ROOT / path):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find config file: {p}")

    user_config = json.loads(_resolve(user_config_path).read_text(encoding="utf-8"))
    system_config = json.loads(_resolve(system_config_path).read_text(encoding="utf-8"))

    required_user_keys = [
        "battery_capacity_kwh", "max_charge_kw", "max_discharge_kw",
        "charge_efficiency", "discharge_efficiency", "soc_min_kwh", "soc_max_kwh",
    ]
    required_system_keys = [
        "default_sell_price_cent_kwh", "allow_grid_charging",
        "grid_charge_price_threshold_cent_kwh", "cycle_penalty_cent_per_kwh",
        "enforce_solar_first_in_lp", "terminal_soc_value_cent_kwh", "min_end_soc_kwh",
    ]
    missing_user = [k for k in required_user_keys if k not in user_config]
    missing_system = [k for k in required_system_keys if k not in system_config]
    if missing_user or missing_system:
        raise KeyError(
            f"Missing config keys — user_config: {missing_user}, system_config: {missing_system}"
        )

    return {
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
            None if system_config["min_end_soc_kwh"] is None
            else float(system_config["min_end_soc_kwh"])
        ),
    }


def _load_aggregated_table(csv_path: PathLike) -> pd.DataFrame:
    required_cols = [
        "utc_timestamp",
        "pv_generation_kwh",
        "energy_price_buy_cent_kwh",
        "energy_price_sell_cent_kwh",
        "household_load_kwh",
    ]
    df = pd.read_csv(csv_path, usecols=lambda c: c in required_cols)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"aggregated_table.csv is missing columns: {missing}")

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    if df["utc_timestamp"].isna().any():
        raise ValueError("aggregated_table.csv contains unparseable utc_timestamp values.")
    if df["utc_timestamp"].duplicated().any():
        raise ValueError(f"aggregated_table.csv has {df['utc_timestamp'].duplicated().sum()} duplicate timestamps.")

    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    for col in ["pv_generation_kwh", "energy_price_buy_cent_kwh", "energy_price_sell_cent_kwh", "household_load_kwh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values.")

    for col in ["pv_generation_kwh", "household_load_kwh"]:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' must be non-negative. Found min={df[col].min()}")

    if len(df) < HORIZON_INTERVALS:
        raise ValueError(
            f"aggregated_table.csv must have at least {HORIZON_INTERVALS} rows (48 h at 15 min). Got {len(df)}."
        )

    expected_step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    wrong_step = df["utc_timestamp"].diff().dropna() != expected_step
    if wrong_step.any():
        bad = wrong_step[wrong_step].index.tolist()[:5]
        raise ValueError(f"Timestamp spacing is not 15-min UTC at positions: {bad}")

    return df.iloc[:HORIZON_INTERVALS].reset_index(drop=True)


def _finalize_dispatch_output(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    cycle_penalty = float(params["cycle_penalty_cent_per_kwh"])
    out = df.copy()
    out["soc_percent"] = (out["soc_kwh"] / float(params["battery_capacity_kwh"])) * 100.0
    pv_used = out["pv_to_load_kwh"] + out["pv_to_battery_kwh"] + out["export_to_grid_kwh"]
    out["curtailed_pv_kwh"] = np.maximum(0.0, out["pv_generation_kwh"] - pv_used)
    out["total_import_kwh"] = out["grid_to_load_kwh"] + out["grid_to_battery_kwh"]
    out["total_export_kwh"] = out["export_to_grid_kwh"]
    out["interval_cost_cent"] = (
        out["energy_price_buy_cent_kwh"] * out["total_import_kwh"]
        - out["energy_price_sell_cent_kwh"] * out["total_export_kwh"]
        + cycle_penalty * (out["pv_to_battery_kwh"] + out["grid_to_battery_kwh"] + out["battery_to_load_kwh"])
    )
    out["cumulative_cost_cent"] = out["interval_cost_cent"].cumsum()
    return out


def run_rule_based_dispatch(
    forecast_df: pd.DataFrame,
    params: dict[str, Any],
    actual_soc_kwh: float,
) -> pd.DataFrame:
    dt_hours = INTERVAL_MINUTES / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours
    eta_c = params["charge_efficiency"]
    eta_d = params["discharge_efficiency"]
    soc_min = params["soc_min_kwh"]
    soc_max = params["soc_max_kwh"]
    allow_grid_charging = bool(params["allow_grid_charging"])
    grid_charge_threshold = params["grid_charge_price_threshold_cent_kwh"]

    soc = float(actual_soc_kwh)
    rows: list[dict[str, Any]] = []

    for row in forecast_df.itertuples(index=False):
        load_remaining = float(row.household_load_kwh)
        pv_remaining = float(row.pv_generation_kwh)
        buy_price = float(row.energy_price_buy_cent_kwh)
        sell_price = float(row.energy_price_sell_cent_kwh)

        pv_to_load_kwh = min(load_remaining, pv_remaining)
        load_remaining -= pv_to_load_kwh
        pv_remaining -= pv_to_load_kwh

        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        pv_to_battery_kwh = min(pv_remaining, charge_limit_kwh, charge_headroom_kwh)
        soc += pv_to_battery_kwh * eta_c
        pv_remaining -= pv_to_battery_kwh

        available_discharge_kwh = max(0.0, (soc - soc_min) * eta_d)
        battery_to_load_kwh = min(load_remaining, discharge_limit_kwh, available_discharge_kwh)
        soc -= battery_to_load_kwh / eta_d
        load_remaining -= battery_to_load_kwh

        grid_to_load_kwh = load_remaining

        charge_limit_left_kwh = max(0.0, charge_limit_kwh - pv_to_battery_kwh)
        charge_headroom_kwh = max(0.0, (soc_max - soc) / eta_c)
        should_grid_charge = allow_grid_charging and buy_price <= float(grid_charge_threshold)
        grid_to_battery_kwh = min(charge_limit_left_kwh, charge_headroom_kwh) if should_grid_charge else 0.0
        soc += grid_to_battery_kwh * eta_c

        export_to_grid_kwh = pv_remaining

        if soc < soc_min - 1e-6 or soc > soc_max + 1e-6:
            raise ValueError(
                f"SoC out of bounds at {row.utc_timestamp}: soc={soc:.6f}, bounds=[{soc_min}, {soc_max}]"
            )
        soc = float(np.clip(soc, soc_min, soc_max))

        flow_names = ["pv_to_load", "pv_to_battery", "battery_to_load", "grid_to_load", "grid_to_battery", "export_to_grid"]
        flow_vals = [pv_to_load_kwh, pv_to_battery_kwh, battery_to_load_kwh, grid_to_load_kwh, grid_to_battery_kwh, export_to_grid_kwh]
        decision_rule = " | ".join(n for n, v in zip(flow_names, flow_vals) if v > 1e-9) or "idle"

        rows.append({
            "utc_timestamp": row.utc_timestamp,
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
            "decision_rule": decision_rule,
            "method": "rule_based",
        })

    return _finalize_dispatch_output(pd.DataFrame(rows), params)


def run_lp_dispatch(
    forecast_df: pd.DataFrame,
    params: dict[str, Any],
    actual_soc_kwh: float,
    enforce_solar_first: bool | None = None,
) -> pd.DataFrame:
    n = len(forecast_df)
    dt_hours = INTERVAL_MINUTES / 60.0
    charge_limit_kwh = params["max_charge_kw"] * dt_hours
    discharge_limit_kwh = params["max_discharge_kw"] * dt_hours
    eta_c = float(params["charge_efficiency"])
    eta_d = float(params["discharge_efficiency"])
    soc_min = float(params["soc_min_kwh"])
    soc_max = float(params["soc_max_kwh"])
    cycle_penalty = float(params["cycle_penalty_cent_per_kwh"])
    terminal_soc_value = float(params["terminal_soc_value_cent_kwh"])
    min_end_soc_kwh = params["min_end_soc_kwh"]
    if enforce_solar_first is None:
        enforce_solar_first = bool(params["enforce_solar_first_in_lp"])

    load = forecast_df["household_load_kwh"].to_numpy(dtype=float)
    pv = forecast_df["pv_generation_kwh"].to_numpy(dtype=float)
    buy = forecast_df["energy_price_buy_cent_kwh"].to_numpy(dtype=float)
    sell = forecast_df["energy_price_sell_cent_kwh"].to_numpy(dtype=float)

    # Variable index blocks: gl, pl, pb, bl, gb, ex, soc (n+1)
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

    a_eq_rows, b_eq = [], []
    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_gl[t]] = row[idx_pl[t]] = row[idx_bl[t]] = 1.0
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

    a_ub_rows, b_ub = [], []
    for t in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[idx_pl[t]] = row[idx_pb[t]] = row[idx_ex[t]] = 1.0
        a_ub_rows.append(row)
        b_ub.append(pv[t])

        row = np.zeros(num_vars, dtype=float)
        row[idx_pb[t]] = row[idx_gb[t]] = 1.0
        a_ub_rows.append(row)
        b_ub.append(charge_limit_kwh)

    if enforce_solar_first:
        for t in range(n):
            row = np.zeros(num_vars, dtype=float)
            row[idx_gl[t]] = row[idx_bl[t]] = 1.0
            a_ub_rows.append(row)
            b_ub.append(max(0.0, load[t] - pv[t]))

    bounds = [(0.0, None)] * num_vars
    for i in idx_pb:
        bounds[i] = (0.0, charge_limit_kwh)
    for i in idx_bl:
        bounds[i] = (0.0, discharge_limit_kwh)
    for i in idx_gb:
        bounds[i] = (0.0, charge_limit_kwh)
    bounds[idx_soc[0]] = (float(actual_soc_kwh), float(actual_soc_kwh))
    for t in range(n):
        bounds[idx_soc[t + 1]] = (soc_min, soc_max)

    if min_end_soc_kwh is not None:
        min_end = max(soc_min, float(min_end_soc_kwh))
        if min_end > soc_max:
            raise ValueError(f"min_end_soc_kwh={min_end} exceeds soc_max_kwh={soc_max}.")
        lb, ub = bounds[idx_soc[-1]]
        bounds[idx_soc[-1]] = (max(lb, min_end), ub)

    result = linprog(
        c=c,
        A_ub=np.array(a_ub_rows, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=np.array(a_eq_rows, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP optimization failed: {result.message}")

    x = result.x
    out = forecast_df.copy()
    out["grid_to_load_kwh"] = x[idx_gl]
    out["pv_to_load_kwh"] = x[idx_pl]
    out["pv_to_battery_kwh"] = x[idx_pb]
    out["battery_to_load_kwh"] = x[idx_bl]
    out["grid_to_battery_kwh"] = x[idx_gb]
    out["export_to_grid_kwh"] = x[idx_ex]
    out["soc_kwh"] = x[idx_soc[1:]]
    out["decision_rule"] = "optimizer_lp"
    out["method"] = "lp_optimized"

    return _finalize_dispatch_output(out, params)


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

    return pd.Series({
        "window_rows": len(dispatch_df),
        "max_load_balance_error_kwh": float(load_balance_error),
        "max_pv_balance_error_kwh": float(pv_balance_error),
        "soc_within_bounds": bool(soc_ok),
    })


def run_dispatch(
    actual_soc_kwh: float,
    aggregated_csv: PathLike = _DEFAULT_AGGREGATED_CSV,
    user_config_path: PathLike = "user_config.json",
    system_config_path: PathLike = "system_config.json",
    enforce_solar_first: bool | None = None,
    output_path: PathLike = _DEFAULT_DISPATCH_CSV,
    save_output: bool = True,
) -> pd.DataFrame:
    params = _load_params(user_config_path, system_config_path)

    soc_min = float(params["soc_min_kwh"])
    soc_max = float(params["soc_max_kwh"])
    if not (soc_min - 1e-9 <= float(actual_soc_kwh) <= soc_max + 1e-9):
        raise ValueError(f"actual_soc_kwh={actual_soc_kwh} is outside [{soc_min}, {soc_max}]")

    forecast_df = _load_aggregated_table(aggregated_csv)

    rule_df = run_rule_based_dispatch(forecast_df, params, actual_soc_kwh)
    lp_df = run_lp_dispatch(forecast_df, params, actual_soc_kwh, enforce_solar_first)

    cols_to_merge = FLOW_COLUMNS + [
        "soc_kwh", "interval_cost_cent", "cumulative_cost_cent", "decision_rule",
    ]
    # Add boolean activity flags directly
    for df in (rule_df, lp_df):
        for col in FLOW_COLUMNS:
            df[f"rule_{col.replace('_kwh', '')}"] = df[col] > 1e-9
        flow_flag_cols = [f"rule_{c.replace('_kwh', '')}" for c in FLOW_COLUMNS]
        df["rule_idle"] = ~df[flow_flag_cols].any(axis=1)
        df["rule_optimizer_lp"] = df["method"].eq("lp_optimized")

    flag_cols = [f"rule_{c.replace('_kwh', '')}" for c in FLOW_COLUMNS] + ["rule_idle", "rule_optimizer_lp"]
    cols_to_merge = cols_to_merge + flag_cols

    rule_merge = rule_df[["utc_timestamp"] + cols_to_merge].rename(columns={c: f"{c}_rule_based" for c in cols_to_merge})
    lp_merge = lp_df[["utc_timestamp"] + cols_to_merge].rename(columns={c: f"{c}_lp_optimized" for c in cols_to_merge})

    combined_df = rule_merge.merge(lp_merge, on="utc_timestamp", how="inner")
    combined_df = combined_df.merge(
        forecast_df[["utc_timestamp", "pv_generation_kwh", "household_load_kwh",
                     "energy_price_buy_cent_kwh", "energy_price_sell_cent_kwh"]],
        on="utc_timestamp",
        how="left",
    )

    input_cols = ["utc_timestamp", "pv_generation_kwh", "household_load_kwh",
                  "energy_price_buy_cent_kwh", "energy_price_sell_cent_kwh"]
    rule_cols = [c for c in combined_df.columns if c.endswith("_rule_based")]
    lp_cols = [c for c in combined_df.columns if c.endswith("_lp_optimized")]
    result_df = combined_df[input_cols + rule_cols + lp_cols].reset_index(drop=True)

    if save_output:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out, index=False)
        print(f"✓ Saved dispatch output to: {out.resolve()}")

    return result_df


__all__ = [
    "FLOW_COLUMNS",
    "INTERVAL_MINUTES",
    "HORIZON_INTERVALS",
    "run_rule_based_dispatch",
    "run_lp_dispatch",
    "run_balance_checks",
    "run_dispatch",
]
