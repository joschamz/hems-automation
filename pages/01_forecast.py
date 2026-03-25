import json
import tempfile
from datetime import timedelta

import pandas as pd
import streamlit as st

from utils import get_daily_solar_kwh, get_daily_prices, get_daily_load_forecast
from utils.load_utils import load_feature_engineered_dataset

st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")

st.title("Forecast")
st.caption("Solar, electricity price, and household load forecasts for the selected planning date.")

def write_temp_solar_config(lat, lon, kwp, tilt, azimuth, yield_factor) -> str:
    config = {
        "lat": float(lat),
        "lon": float(lon),
        "kwp": float(kwp),
        "tilt": float(tilt),
        "azimuth": float(azimuth),
        "yield_factor": float(yield_factor),
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, tmp, indent=2)
    tmp.close()
    return tmp.name

@st.cache_data(show_spinner=False)
def get_last_available_load_timestamp(feature_dataset_path: str) -> pd.Timestamp:
    df_all = load_feature_engineered_dataset(feature_dataset_path)
    return df_all.index.max()

def remap_load_forecast_to_target_day(load_raw: pd.DataFrame, planning_date) -> pd.DataFrame:
    out = load_raw.copy()
    target_start = pd.Timestamp(planning_date).tz_localize("UTC")
    out["time"] = pd.date_range(start=target_start, periods=len(out), freq="15min", tz="UTC")
    return out

tomorrow_default = pd.Timestamp.now(tz="UTC").date() + timedelta(days=1)

with st.sidebar:
    planning_date = st.date_input("Planning Date", value=tomorrow_default)
    latitude = st.number_input("Latitude", value=47.659216, format="%.6f")
    longitude = st.number_input("Longitude", value=9.175072, format="%.6f")
    capacity_kwp = st.number_input("PV Capacity (kWp)", min_value=0.0, value=15.0, step=0.5)
    tilt = st.number_input("Tilt (°)", min_value=0.0, max_value=90.0, value=35.0, step=1.0)
    azimuth = st.number_input("Azimuth (°)", min_value=-180.0, max_value=180.0, value=0.0, step=1.0)
    yield_factor = st.slider("Yield Factor", min_value=0.10, max_value=1.00, value=0.70, step=0.01)
    feature_dataset_path = st.text_input("Feature Dataset Path", value="data/input/shifted-date-residential1_feature_engineered_full.csv")
    load_model_path = st.text_input("Load Model Path", value="models/load_forecast_model.pkl")
    run_button = st.button("Load Forecasts", use_container_width=True)

if not run_button:
    st.info("Set the parameters in the sidebar and click Load Forecasts.")
    st.stop()

planning_date = pd.Timestamp(planning_date).date()
solar_config_path = write_temp_solar_config(
    lat=latitude,
    lon=longitude,
    kwp=capacity_kwp,
    tilt=tilt,
    azimuth=azimuth,
    yield_factor=yield_factor,
)

solar_mode = "forecast" if planning_date >= pd.Timestamp.now().date() else "historical"
price_mode = "forecast" if planning_date >= pd.Timestamp.now(tz="UTC").date() else "historical"

solar_raw = get_daily_solar_kwh(
    target_date=planning_date,
    mode=solar_mode,
    config_path=solar_config_path,
    allow_fallback=True,
)

price_raw = get_daily_prices(
    target_date=planning_date,
    mode=price_mode,
    secrets_dir="secrets",
    allow_fallback=True,
)

last_load_ts = get_last_available_load_timestamp(feature_dataset_path)
load_raw = get_daily_load_forecast(
    forecast_time=last_load_ts,
    household_id=1,
    model_path=load_model_path,
    feature_dataset_path=feature_dataset_path,
)
load_raw = remap_load_forecast_to_target_day(load_raw, planning_date)

solar_df = solar_raw.copy()
price_df = price_raw.copy()
load_df = load_raw.copy()

solar_df["time"] = pd.to_datetime(solar_df["time"], utc=True)
price_df["time"] = pd.to_datetime(price_df["time"], utc=True)
load_df["time"] = pd.to_datetime(load_df["time"], utc=True)

agg_df = (
    solar_df[["time", "predicted_kwh", "predicted_kw", "source"]]
    .rename(columns={"predicted_kwh": "solar_kwh", "predicted_kw": "solar_kw", "source": "solar_source"})
    .merge(
        price_df[["time", "price_cent_kwh", "price_eur_mwh", "source"]]
        .rename(columns={"source": "price_source"}),
        on="time",
        how="inner",
    )
    .merge(
        load_df[["time", "predicted_kwh", "predicted_kw", "source"]]
        .rename(columns={"predicted_kwh": "load_kwh", "predicted_kw": "load_kw", "source": "load_source"}),
        on="time",
        how="inner",
    )
)

k1, k2, k3 = st.columns(3)
k1.metric("Total Solar Forecast", f"{solar_df['predicted_kwh'].sum():.2f} kWh")
k2.metric("Average Buy Price", f"{price_df['price_cent_kwh'].mean():.2f} cent/kWh")
k3.metric("Total Load Forecast", f"{load_df['predicted_kwh'].sum():.2f} kWh")

tab1, tab2, tab3, tab4 = st.tabs(["Solar", "Price", "Load", "Aggregated"])

with tab1:
    st.subheader("Solar Forecast")
    st.line_chart(solar_df.set_index("time")[["predicted_kwh"]])
    st.dataframe(solar_df, use_container_width=True)

with tab2:
    st.subheader("Electricity Price Forecast")
    st.line_chart(price_df.set_index("time")[["price_cent_kwh"]])
    st.dataframe(price_df, use_container_width=True)

with tab3:
    st.subheader("Household Load Forecast")
    st.line_chart(load_df.set_index("time")[["predicted_kwh"]])
    st.dataframe(load_df, use_container_width=True)

with tab4:
    st.subheader("Aggregated Forecast Table")
    st.dataframe(agg_df, use_container_width=True)