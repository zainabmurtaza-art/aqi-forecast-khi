# -*- coding: utf-8 -*-
"""
Reads the Hopsworks Feature Group and builds a training-ready dataset:
adds one shifted target column per forecast horizon (t+1/t+2/t+3 day AQI),
and provides a chronological (not random) train/test split, since this is a
time series and random shuffling would leak future information into training.
"""

from datetime import timedelta

import pandas as pd

import config
from hopsworks_utils import get_feature_group

FEATURE_COLUMNS = (
    [
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
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "aqi_change_rate",
    ]
    + [f"aqi_roll_mean_{w}h" for w in config.ROLLING_WINDOWS_HOURS]
    + [f"aqi_lag_{l}h" for l in config.LAG_HOURS]
)


# Weather at the hour being forecast, not at the hour we forecast from.
#
# Every other feature describes time t while the target is AQI at t+H, so at
# H=72 the model has almost nothing that distinguishes one day from the next
# and falls back on the level it learned in training - which is how a
# chronological hold-out ends up with a negative R2. Weather is the one thing
# about t+H that is genuinely knowable in advance (that is what a weather
# forecast is), and it drives dispersion: wind clears particulates, still humid
# air lets them accumulate.
#
# No feature-group schema change is needed for this: the weather at t+H is
# already stored as the row at t+H, so the same self-join that builds the
# target carries it back.
#
# Measured against the restored feature store, adding these improved MAE at
# every city and horizon tried, e.g. karachi/+3 day random_forest R2
# -0.350 -> -0.095 (MAE 8.53 -> 7.49) and karachi/+1 day xgboost +0.607 ->
# +0.652 (MAE 4.49 -> 4.22).
#
# Caveat worth knowing when reading the metrics: at training time these are
# *observed* values from the archive, while at inference they come from
# Open-Meteo's forecast. Held-out scores are therefore mildly optimistic
# relative to live performance, since the model never trains on an imperfect
# weather forecast.
TARGET_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


def target_column_name(horizon_hours: int) -> str:
    return f"target_t_plus_{horizon_hours // 24}"


def target_weather_column_name(column: str, horizon_hours: int) -> str:
    return f"{column}_at_t_plus_{horizon_hours // 24}"


def feature_columns(horizon_hours: int) -> list:
    """Model inputs for one horizon: the shared columns plus that horizon's
    target-hour weather. Each (city, horizon) trains its own model, so the
    horizons never need to share a feature list."""
    return FEATURE_COLUMNS + [
        target_weather_column_name(c, horizon_hours) for c in TARGET_WEATHER_COLUMNS
    ]


def read_feature_group_df() -> pd.DataFrame:
    fg = get_feature_group()
    df = fg.read()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df.sort_values(["city", "event_time"]).reset_index(drop=True)


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Self-joins the values at event_time + horizon back onto each row.

    Brings back two things per horizon: us_aqi (the target) and the weather at
    that same future hour (features - see TARGET_WEATHER_COLUMNS). Both come
    from the one join, since the row at t+H already holds them.
    """
    df = df.copy()

    for horizon in config.FORECAST_HORIZONS_HOURS:
        carried = ["us_aqi"] + TARGET_WEATHER_COLUMNS
        future = df[["city", "event_time"] + carried].copy()
        future["event_time"] = future["event_time"] - timedelta(hours=horizon)
        future = future.rename(
            columns={
                "us_aqi": target_column_name(horizon),
                **{c: target_weather_column_name(c, horizon) for c in TARGET_WEATHER_COLUMNS},
            }
        )
        df = df.merge(future, on=["city", "event_time"], how="left")

    return df


def chronological_train_test_split(
    df: pd.DataFrame, target_col: str, city: str, test_frac: float = 0.2,
    feature_cols: list = None,
):
    """Filters to one city, drops rows missing any feature or the target, then
    splits by time order. Filtering by city matters because a model is trained
    per (city, horizon) pair — mixing cities would blend unrelated series."""
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
    city_df = df[df["city"] == city]
    usable = city_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    split_idx = int(len(usable) * (1 - test_frac))
    train_df = usable.iloc[:split_idx]
    test_df = usable.iloc[split_idx:]

    return (
        train_df[feature_cols],
        train_df[target_col],
        test_df[feature_cols],
        test_df[target_col],
    )


def build_training_data() -> pd.DataFrame:
    df = read_feature_group_df()
    return add_horizon_targets(df)
