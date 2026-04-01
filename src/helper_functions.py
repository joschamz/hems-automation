from pathlib import Path
from datetime import datetime
import re
import pandas as pd

def cleanup_old_models(models_dir: Path, keep_last_n: int = 3):
    """
    Keep the latest N model files (based on timestamp in filename)
    and delete the rest.

    Expected filename format:
    load_forecast_model_YYYY-MM-DD_HH-MM-SS.pkl
    """
    pattern = re.compile(
        r"load_forecast_model_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.pkl"
    )

    candidates = []

    # Collect valid model files with parsed timestamps
    for file in models_dir.glob("load_forecast_model_*.pkl"):
        match = pattern.match(file.name)
        if match:
            date_part, time_part = match.groups()
            timestamp_str = f"{date_part} {time_part.replace('-', ':')}"
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            candidates.append((timestamp, file))

    if not candidates:
        print("No valid model files found.")
        return

    # Sort newest → oldest
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Split keep vs delete
    to_keep = candidates[:keep_last_n]
    to_delete = candidates[keep_last_n:]

    print("Keeping:")
    for _, f in to_keep:
        print(f"  {f}")

    print("Deleting:")
    for _, f in to_delete:
        print(f"  {f}")
        f.unlink()


def get_latest_model_path(models_dir: Path) -> Path:
    pattern = re.compile(r"load_forecast_model_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.pkl")

    candidates = []

    for file in models_dir.glob("load_forecast_model_*.pkl"):
        match = pattern.match(file.name)
        if match:
            date_part, time_part = match.groups()
            timestamp_str = f"{date_part} {time_part.replace('-', ':')}"
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            candidates.append((timestamp, file))

    if not candidates:
        raise FileNotFoundError("No valid model files found")

    # Sort by timestamp descending → closest to now (latest)
    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][1]

def save_history(df: pd.DataFrame, history_path: Path) -> None:
    """
    Append only new timestamps from df into history CSV.

    Assumes df contains column 'utc_timestamp'.
    """

    if df.empty:
        print("No data to save.")
        return

    # ensure correct dtype
    df = df.copy()
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)

    history_path.parent.mkdir(parents=True, exist_ok=True)

    # --- first run ---
    if not history_path.exists():
        df.to_csv(history_path, index=False)
        print(f"[HISTORY] Created new history file with {len(df)} rows.")
        return

    # --- load only timestamps (efficient) ---
    existing = pd.read_csv(
        history_path,
        usecols=["utc_timestamp"],
        parse_dates=["utc_timestamp"]
    )

    last_ts = existing["utc_timestamp"].max()

    # --- filter only new rows ---
    new_rows = df[df["utc_timestamp"] > last_ts]

    if new_rows.empty:
        print("[HISTORY] No new rows to append.")
        return

    # --- append ---
    new_rows.to_csv(
        history_path,
        mode="a",
        header=False,
        index=False,
    )

    print(f"[HISTORY] Appended {len(new_rows)} new rows.")
