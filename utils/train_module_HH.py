from pathlib import Path
import traceback
import joblib
import numpy as np
import pandas as pd
import holidays
from datetime import datetime

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor


# ------------------------------------------------------------
# Helper: prepare base dataframe
# ------------------------------------------------------------
def _load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain 'timestamp' column")

    if "load" not in df.columns:
        raise ValueError("Input CSV must contain 'load' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    return df


# ------------------------------------------------------------
# Helper: create required engineered features if missing
# ------------------------------------------------------------
def _ensure_required_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    de_holidays = holidays.Germany()

    # calendar features
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df.index.dayofweek
    if "month" not in df.columns:
        df["month"] = df.index.month
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = (
            df.index.tz_localize(None).normalize().map(lambda d: int(d in de_holidays))
        )

    # lag features
    if "lag_1" not in df.columns:
        df["lag_1"] = df["load"].shift(1)
    if "lag_2" not in df.columns:
        df["lag_2"] = df["load"].shift(2)
    if "lag_3" not in df.columns:
        df["lag_3"] = df["load"].shift(3)
    if "lag_96" not in df.columns:
        df["lag_96"] = df["load"].shift(96)
    if "lag_192" not in df.columns:
        df["lag_192"] = df["load"].shift(192)

    # delta / rolling
    if "delta_1" not in df.columns:
        df["delta_1"] = df["load"] - df["lag_1"]
    if "delta_2" not in df.columns:
        df["delta_2"] = df["lag_1"] - df["lag_2"]
    if "rolling_std_1h" not in df.columns:
        df["rolling_std_1h"] = df["load"].rolling(window=4).std()
    if "rolling_mean_1h" not in df.columns:
        df["rolling_mean_1h"] = df["load"].rolling(window=4).mean()
    if "rolling_mean_24h" not in df.columns:
        df["rolling_mean_24h"] = df["load"].rolling(window=96).mean()

    return df


# ------------------------------------------------------------
# Main train function
# ------------------------------------------------------------
def train_module(inputCSVFile, outputDirectoryTrainedModule) -> bool:
    """
    Train a 24h-ahead direct multi-output load forecasting model
    and save it as a PKL bundle.

    Parameters
    ----------
    inputCSVFile : str
        Path to input CSV file
    outputDirectoryTrainedModule : str
        Directory where trained PKL model will be saved

    Returns
    -------
    bool
        True if training + saving succeeded, otherwise False
    """
    try:
        input_path = Path(inputCSVFile)
        output_dir = Path(outputDirectoryTrainedModule)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # 1) Load data
        # ----------------------------------------------------
        df = _load_dataset(str(input_path))
        df = _ensure_required_features(df)

        # ----------------------------------------------------
        # 2) Define feature groups
        # ----------------------------------------------------
        fixed_feature_cols = [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_96",
            "lag_192",
            "delta_1",
            "delta_2",
            "rolling_mean_1h",
            "rolling_mean_24h",
            "rolling_std_1h",
        ]

        weather_feature_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "snow_depth",
            "weather_code",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "global_tilted_irradiance",
            "sunshine_duration",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit",
        ]

        missing_cols = [c for c in fixed_feature_cols + weather_feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        feature_cols = fixed_feature_cols + weather_feature_cols

        # ----------------------------------------------------
        # 3) Build multi-step targets
        # ----------------------------------------------------
        horizon = 96  # 24h with 15-min resolution

        y_multi = pd.DataFrame(index=df.index)
        for h in range(1, horizon + 1):
            y_multi[f"target_t+{h}"] = df["load"].shift(-h)

        data_multi = pd.concat([df[feature_cols], y_multi], axis=1).dropna()

        if data_multi.empty:
            raise ValueError("Training dataset is empty after feature/target alignment")

        X_multi = data_multi[feature_cols].copy()
        y_multi = data_multi[[c for c in data_multi.columns if c.startswith("target_t+")]].copy()

        # ----------------------------------------------------
        # 4) Train final model
        # ----------------------------------------------------
        best_params = {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "max_depth": 5,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
        }

        model = MultiOutputRegressor(
            LGBMRegressor(
                random_state=42,
                verbosity=-1,
                **best_params
            )
        )

        model.fit(X_multi, y_multi)

        # ----------------------------------------------------
        # 5) Save bundle
        # ----------------------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_model_path = output_dir / f"load_forecast_model_{timestamp}.pkl"
        bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "fixed_feature_cols": fixed_feature_cols,
            "weather_feature_cols": weather_feature_cols,
            "horizon": horizon,
            "train_start": str(X_multi.index.min()),
            "train_end": str(X_multi.index.max()),
            "n_train_rows": len(X_multi),
            "best_params": best_params,
        }

        joblib.dump(bundle, output_model_path)

        print(f"Model trained and saved successfully: {output_model_path}")
        return True

    except Exception as e:
        print("Training failed.")
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    

#---------------------------------Example usage---------------------------------
#ok = train_module(
#    inputCSVFile="../data/input/shifted-date-residential1_feature_engineered_full.csv",
#    outputDirectoryTrainedModule="../models"
#)

#print("Train status:", ok)