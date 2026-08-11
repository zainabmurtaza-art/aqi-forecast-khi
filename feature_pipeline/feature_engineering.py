# -*- coding: utf-8 -*-
"""
Feature engineering shared by backfill, the hourly pipeline, and dashboard
inference. Takes a raw hourly DataFrame (city, event_time, pollutants,
us_aqi, weather fields — see open_meteo_client.fetch_combined) and adds:

  - time features: hour, day, month, day_of_week, is_weekend
  - derived features: aqi_change_rate, rolling means, lag features

All derived features are computed per-city (groupby) and assume the input
rows for each city are contiguous hourly readings sorted by event_time —
true for both a fresh Open-Meteo pull and a Hopsworks lookback window, since
both come from the same hourly source.
"""

import pandas as pd

import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # pandas' .dt accessor yields int32; Hopsworks' declared schema expects
    # int64 ("bigint"), so cast explicitly rather than relying on inference.
    df["hour"] = df["event_time"].dt.hour.astype("int64")
    df["day"] = df["event_time"].dt.day.astype("int64")
    df["month"] = df["event_time"].dt.month.astype("int64")
    df["day_of_week"] = df["event_time"].dt.dayofweek.astype("int64")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int64")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "event_time"]).reset_index(drop=True)
    grouped = df.groupby("city")["us_aqi"]

    df["aqi_change_rate"] = grouped.diff()

    for window in config.ROLLING_WINDOWS_HOURS:
        df[f"aqi_roll_mean_{window}h"] = grouped.transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
        )

    for lag in config.LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = grouped.shift(lag)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: raw hourly rows -> rows with time + derived features."""
    df = add_time_features(df)
    df = add_derived_features(df)
    return df
