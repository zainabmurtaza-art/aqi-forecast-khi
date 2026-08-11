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


def target_column_name(horizon_hours: int) -> str:
    return f"target_t_plus_{horizon_hours // 24}"


def read_feature_group_df() -> pd.DataFrame:
    fg = get_feature_group()
    df = fg.read()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df.sort_values(["city", "event_time"]).reset_index(drop=True)


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Self-joins us_aqi at event_time + horizon back onto each row, per horizon."""
    df = df.copy()

    for horizon in config.FORECAST_HORIZONS_HOURS:
        future = df[["city", "event_time", "us_aqi"]].copy()
        future["event_time"] = future["event_time"] - timedelta(hours=horizon)
        future = future.rename(columns={"us_aqi": target_column_name(horizon)})
        df = df.merge(future, on=["city", "event_time"], how="left")

    return df


def chronological_train_test_split(df: pd.DataFrame, target_col: str, test_frac: float = 0.2):
    """Drops rows missing any feature or the target, then splits by time order."""
    usable = df.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)

    split_idx = int(len(usable) * (1 - test_frac))
    train_df = usable.iloc[:split_idx]
    test_df = usable.iloc[split_idx:]

    return (
        train_df[FEATURE_COLUMNS],
        train_df[target_col],
        test_df[FEATURE_COLUMNS],
        test_df[target_col],
    )


def build_training_data() -> pd.DataFrame:
    df = read_feature_group_df()
    return add_horizon_targets(df)
