# -*- coding: utf-8 -*-
"""
Cached data/model loading for the dashboard. Connections and registered
models are cached for the app's lifetime; feature/forecast pulls are cached
with a TTL matching the hourly pipeline's cadence.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import config
import hopsworks_utils
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import fetch_combined
from training_pipeline.build_dataset import FEATURE_COLUMNS

RAW_COLUMNS = [
    "city",
    "event_time",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]

LOOKBACK_DAYS_FOR_TREND = 14


@st.cache_resource
def get_project():
    return hopsworks_utils.get_project()


@st.cache_resource
def load_models(city: str) -> dict:
    """Returns {horizon_hours: fitted sklearn model} for each configured horizon, for one city."""
    registry = hopsworks_utils.get_model_registry(get_project())
    models = {}

    for horizon in config.FORECAST_HORIZONS_HOURS:
        name = config.MODEL_REGISTRY_NAME_TEMPLATE.format(city=city, horizon=horizon // 24)
        hw_model = registry.get_best_model(name, "rmse", "min")
        model_dir = hw_model.download()
        models[horizon] = joblib.load(Path(model_dir) / "model.pkl")

    return models


@st.cache_data(ttl=3600)
def load_recent_actual_features(
    city: str, lookback_days: int = LOOKBACK_DAYS_FOR_TREND
) -> pd.DataFrame:
    """Latest stored feature rows from Hopsworks for one city, used for the trend
    chart and as inference context (their raw columns get stitched onto the
    forecast window)."""
    fg = hopsworks_utils.get_feature_group()
    df = fg.read()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df[df["city"] == city].sort_values("event_time")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    return df[df["event_time"] >= cutoff].reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_forecast_raw(city: str) -> pd.DataFrame:
    forecast_days = max(config.FORECAST_HORIZONS_HOURS) // 24 + 1
    return fetch_combined(forecast_days=forecast_days, city=city)


def build_inference_dataset(city: str) -> pd.DataFrame:
    """Stitches recent actual rows + forecast rows into one continuous hourly
    series, then re-runs feature_engineering so derived features (lags,
    rolling means, change rate) are computed consistently across the
    actual/forecast boundary — the same transform used at training time."""
    actual_raw = load_recent_actual_features(city)[RAW_COLUMNS]
    forecast_raw = load_forecast_raw(city)[RAW_COLUMNS]

    combined = (
        pd.concat([actual_raw, forecast_raw], ignore_index=True)
        .drop_duplicates(subset=["city", "event_time"], keep="first")
        .sort_values("event_time")
    )
    return engineer_features(combined)


def get_horizon_predictions(models: dict, city: str) -> pd.DataFrame:
    engineered = build_inference_dataset(city)
    now = pd.Timestamp.now(tz="UTC").floor("h")

    rows = []
    for horizon in config.FORECAST_HORIZONS_HOURS:
        target_time = now + pd.Timedelta(hours=horizon)
        nearest_idx = (engineered["event_time"] - target_time).abs().idxmin()
        candidate = engineered.loc[[nearest_idx]]

        X = candidate[FEATURE_COLUMNS]
        predicted = float(models[horizon].predict(X)[0])

        rows.append(
            {
                "horizon_days": horizon // 24,
                "event_time": candidate["event_time"].iloc[0],
                "predicted_us_aqi": predicted,
                "feature_row": candidate,
            }
        )

    return pd.DataFrame(rows)
