from pathlib import Path
import sys
import time
from datetime import datetime, timedelta
import pandas as pd

# ---------------------------------
# Setup paths
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
LAST_SOC_KWH = None

from utils.dispatch_utils import run_dispatch
from utils.train_module_HH import train_module
from utils.aggregated_utils import build_aggregated_table
from helper_functions import cleanup_old_models
from helper_functions import save_history
INPUT_FILE = BASE_DIR / "data/input/shifted-date-residential1_feature_engineered_full.csv"
OUTPUT_FILE = BASE_DIR / "data/load_training_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
history_path = BASE_DIR / "data/history/history_dispatch_table.csv"



# ---------------------------------
# Data preparation
# ---------------------------------
def prepare_data():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    now = datetime.now()

    # round down to nearest 15 min
    minute = (now.minute // 15) * 15
    cutoff_time = now.replace(minute=minute, second=0, microsecond=0)

    df_filtered = df[df["timestamp"] <= cutoff_time]
    df_filtered.to_csv(OUTPUT_FILE, index=False)

    print(f"[DATA] Prepared until {cutoff_time} ({len(df_filtered)} rows)")
    return cutoff_time


# ---------------------------------
# Training task
# ---------------------------------
def run_training():
    print("[TRAIN] Starting training...")

    prepare_data()

    success = train_module(
        inputCSVFile=str(OUTPUT_FILE),
        outputDirectoryTrainedModule=str(MODEL_DIR)
    )
    cleanup_old_models(Path("models"), keep_last_n=3)

    if success:
        print("[TRAIN] Training completed successfully.")
    else:
        print("[TRAIN] Training failed.")

    return success


# ---------------------------------
# Forecast task
# ---------------------------------
def run_forecast(current_slot):    
    global LAST_SOC_KWH
   
    print(f"[FORECAST] Running forecast for {current_slot}")
    df = build_aggregated_table()
    #forecast_df = pd.read_csv(input_path)

    INITIAL_SOC_KWH = 5.0  # Replace with measured battery SoC before running

    input_path = Path(BASE_DIR / "data/runtime/aggregated_table.csv")
    
        # --- determine SOC ---
    if LAST_SOC_KWH is None:
        ACTUAL_SOC_KWH = 5.0  # initial fallback
    else:
        ACTUAL_SOC_KWH = LAST_SOC_KWH
    
    result_df = run_dispatch(
    actual_soc_kwh=ACTUAL_SOC_KWH,
    aggregated_csv=input_path,
    save_output=True,
    )
    
    save_history(result_df, history_path)
    
    # --- update SOC for next run ---
    LAST_SOC_KWH = result_df.loc[0, "soc_kwh_rule_based"]    
    
    
    # Example Forecast logic (to be implemented):
    # call function to create aggregated table for forecast
    # call function to create dispatch table using the aggregated table
     
    print("[FORECAST] Done.")


# ---------------------------------
# Main orchestrator loop
# ---------------------------------
last_training_day = None
last_forecast_slot = None

print("Orchestrator started...\n")

while True:
    now = datetime.now()

    # ---------------------------------
    # 1. DAILY TRAINING (once per day)
    # ---------------------------------
    # for production, we have to use these lines to run training once per day --------- to be uncommented in production
    if last_training_day != now.date():
         run_training()
         last_training_day = now.date()
    

    # ---------------------------------
    # 2. FORECAST EVERY 15 MINUTES
    # ---------------------------------
    current_slot = now.replace(
        minute=(now.minute // 15) * 15,
        second=0,
        microsecond=0
    )

    if last_forecast_slot != current_slot:
        run_forecast(current_slot)
        last_forecast_slot = current_slot

    # ---------------------------------
    # Sleep (light polling)
    # ---------------------------------
    time.sleep(30)