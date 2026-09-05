# -*- coding: utf-8 -*-
"""
Cached data/model loading for the dashboard. Connections and registered
models are cached for the app's lifetime; actual-feature pulls (Hopsworks)
are cached with a TTL matching the hourly pipeline's cadence. Forecast
pulls (Open-Meteo) use a longer TTL - see FORECAST_CACHE_TTL_SECONDS.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import config
import hopsworks_utils
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import fetch_combined
from training_pipeline.build_dataset import (
    FEATURE_COLUMNS,
    TARGET_WEATHER_COLUMNS,
    feature_columns,
    target_weather_column_name,
)

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

# Open-Meteo hit sustained 429 rate limiting today (its free tier is shared
# across every app on Streamlit Community Cloud's egress IPs, not just this
# one). A 3-day-ahead forecast doesn't need refreshing every hour, so a
# longer TTL trades some freshness for far fewer requests against a limit we
# don't fully control.
FORECAST_CACHE_TTL_SECONDS = 6 * 3600


@st.cache_resource
def get_project():
    return hopsworks_utils.get_project()


# Estimator class name -> the name a reader would recognise. Taken from the
# loaded object rather than the registry description, because models registered
# before the description fix (commit 6fce628) name only the registry entry, not
# the winning algorithm.
ALGORITHM_LABELS = {
    "RandomForestRegressor": "Random Forest",
    "XGBRegressor": "XGBoost",
    "Ridge": "Ridge Regression",
    "PersistenceRegressor": "Persistence baseline",
    "LSTMRegressor": "LSTM",
}


def algorithm_label(model) -> str:
    return ALGORITHM_LABELS.get(type(model).__name__, type(model).__name__)


def latest_registered_model(registry, name):
    """Highest version of `name` in the registry.

    Deliberately NOT get_best_model(name, "rmse", "min"). Every nightly run
    registers a new version, and each version's RMSE is measured on its own
    chronological hold-out — a different, growing slice of history. Those
    numbers are not comparable across versions: a model trained on 2026-08-11,
    when the feature store held only a few days and its test window barely
    moved, scored RMSE 7.65 and won permanently, while 34 later versions
    trained on a full year of data were never served (that version's R2 was
    -0.935, the worst of all of them).

    The newest version is the one trained on the most complete data, so that is
    what gets served.
    """
    versions = registry.get_models(name)
    if not versions:
        raise ValueError(f"No registered model versions for '{name}'.")
    return max(versions, key=lambda m: m.version)


@st.cache_resource
def load_models(city: str) -> dict:
    """Returns {horizon_hours: fitted sklearn model} for each configured horizon, for one city."""
    registry = hopsworks_utils.get_model_registry(get_project())
    models = {}

    for horizon in config.FORECAST_HORIZONS_HOURS:
        name = config.MODEL_REGISTRY_NAME_TEMPLATE.format(city=city, horizon=horizon // 24)
        hw_model = latest_registered_model(registry, name)
        model_dir = hw_model.download()
        models[horizon] = joblib.load(Path(model_dir) / "model.pkl")

    return models


@st.cache_data(ttl=3600)
def load_model_metrics(city: str) -> pd.DataFrame:
    """Held-out metrics recorded at training time, one row per horizon.

    Read from the Model Registry rather than recomputed, so the figures shown
    are exactly the ones the winning model was selected on.
    """
    registry = hopsworks_utils.get_model_registry(get_project())
    rows = []

    for horizon in config.FORECAST_HORIZONS_HOURS:
        name = config.MODEL_REGISTRY_NAME_TEMPLATE.format(city=city, horizon=horizon // 24)
        try:
            hw_model = latest_registered_model(registry, name)
        except Exception:
            continue

        metrics = hw_model.training_metrics or {}
        rows.append(
            {
                "horizon_days": horizon // 24,
                "registry_name": name,
                "version": hw_model.version,
                "rmse": _as_float(metrics.get("rmse")),
                "mae": _as_float(metrics.get("mae")),
                "r2": _as_float(metrics.get("r2")),
            }
        )

    return pd.DataFrame(rows)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


@st.cache_data(ttl=FORECAST_CACHE_TTL_SECONDS)
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


def model_feature_columns(model, horizon: int) -> list:
    """The exact feature list `model` was fitted on.

    Read off the fitted estimator rather than assumed, so a model registered
    before target-hour weather existed keeps working against the current code
    instead of erroring on an unexpected column count. Falls back to this
    horizon's full list for estimators that don't record their inputs (the
    persistence baseline).
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return feature_columns(horizon)
    return list(names)


def build_features_for_horizon(engineered: pd.DataFrame, row_index, horizon: int) -> pd.DataFrame:
    """One row of model inputs: the row's own features plus the weather at the
    hour being predicted, which is simply the engineered row at that hour."""
    candidate = engineered.loc[[row_index]].copy()

    source_time = candidate["event_time"].iloc[0]
    target_time = source_time + pd.Timedelta(hours=horizon)
    at_target = engineered[engineered["event_time"] == target_time]

    for column in TARGET_WEATHER_COLUMNS:
        name = target_weather_column_name(column, horizon)
        candidate[name] = (
            float(at_target[column].iloc[0]) if not at_target.empty else float("nan")
        )

    return candidate


def get_horizon_predictions(models: dict, city: str) -> pd.DataFrame:
    engineered = build_inference_dataset(city)
    now = pd.Timestamp.now(tz="UTC").floor("h")

    rows = []
    for horizon in config.FORECAST_HORIZONS_HOURS:
        # The row we predict FROM is now; the row we predict FOR is now+horizon.
        # Target-hour weather is taken from the forecast rows already stitched
        # into `engineered` by build_inference_dataset().
        nearest_idx = (engineered["event_time"] - now).abs().idxmin()
        candidate = build_features_for_horizon(engineered, nearest_idx, horizon)

        X = candidate[model_feature_columns(models[horizon], horizon)]
        predicted = float(models[horizon].predict(X)[0])

        rows.append(
            {
                "horizon_days": horizon // 24,
                # The hour being forecast, not the hour forecast from.
                "event_time": candidate["event_time"].iloc[0]
                + pd.Timedelta(hours=horizon),
                "predicted_us_aqi": predicted,
                "feature_row": candidate,
            }
        )

    return pd.DataFrame(rows)
