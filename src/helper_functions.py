from pathlib import Path
from datetime import datetime
import re

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
