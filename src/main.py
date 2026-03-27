# importing the data until date(now) and store it in the "data/load_training_dataset.csv"
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from datetime import datetime, timezone
import time
from utils.train_module_HH import train_module



# Input / output paths
input_file = BASE_DIR / "data/input/shifted-date-residential1_feature_engineered_full.csv"
output_file = BASE_DIR / "data/load_training_dataset.csv"

# Load CSV
df = pd.read_csv(input_file)

# Parse timestamp and remove timezone
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

# Get current local time (Germany assumed from system)
now = datetime.now()

# Round down to nearest 15 minutes (e.g., 11:33 -> 11:30)
minute = (now.minute // 15) * 15
cutoff_time = now.replace(minute=minute, second=0, microsecond=0)

# Filter rows up to cutoff
df_filtered = df[df["timestamp"] <= cutoff_time]

# Save result
df_filtered.to_csv(output_file, index=False)

print(f"Saved {len(df_filtered)} rows up to {cutoff_time}")

# will output load_forecast_model.pkl in the "models" folder
trained_module = train_module(
inputCSVFile=str(input_file),
outputDirectoryTrainedModule= str(BASE_DIR / "models"))






# now the "data/load_training_dataset.csv" file contains all the data up to the current date and time, ready for training the model.
#-----------------------------------------------------------------------------------------------------------------------------------2024-07-20 15:45:00

# the main loop to train the model every day at 00:00 and save the model in the "models" folder with the name "load_forecast_model_YYYY-MM-DD_HH-MM-SS.pkl"     
while True:
    print("Running task...")

# --- your logic here ---

    #append the new data to the existing "data/load_training_dataset.csv" file
    #load the actual dataset from actuals_[yearmonth].csv
    #append it to the existing "data/load_training_dataset.csv" file
    #save the updated dataset to "data/load_training_dataset.csv"
    #continue with training the model using the updated "data/load_training_dataset.csv" file

# will output load_forecast_model.pkl in the "models" folder
    #trained_module = train_module(
    #inputCSVFile=str(input_file),
    #outputDirectoryTrainedModule= str(BASE_DIR / "models"))

    #if not trained_module:
    #    print("Training failed. Will retry in 24 hours.")
    #else:
    #    print("Training succeeded. Model saved.")
    
    print("Done. Sleeping for 24 hours...\n")
    time.sleep(24 * 60 * 60)  # 24 hours