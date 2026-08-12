# -*- coding: utf-8 -*-
"""
One-off (re-runnable) job: pull config.BACKFILL_DAYS of history from
Open-Meteo for every city in config.CITIES, engineer features, and
bulk-insert into the Hopsworks Feature Group. Run this once to seed a real
training dataset, and again any time a larger history window is wanted
(inserts are upserts on the (city, event_time) primary key, so re-running
is safe).
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import config
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import MAX_WEATHER_HISTORY_DAYS, fetch_combined
from hopsworks_utils import get_feature_group


def run_backfill(days: int = config.BACKFILL_DAYS, cities: Optional[Iterable[str]] = None) -> int:
    if days > MAX_WEATHER_HISTORY_DAYS:
        raise ValueError(
            f"AQI_BACKFILL_DAYS={days} exceeds the ~{MAX_WEATHER_HISTORY_DAYS}-day "
            "history window Open-Meteo's forecast API serves for weather data. "
            "Lower it, or switch fetch_weather() to archive-api.open-meteo.com "
            "for a longer backfill."
        )

    cities = list(cities) if cities is not None else list(config.CITIES)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    fg = get_feature_group()
    total_rows = 0
    for city in cities:
        print(f"[{city}] Fetching {days} days of history ({start_date} to {end_date})...")
        raw = fetch_combined(start_date=str(start_date), end_date=str(end_date), city=city)
        print(f"[{city}] Fetched {len(raw)} raw hourly rows.")

        features = engineer_features(raw)
        fg.insert(features)
        print(f"[{city}] Inserted {len(features)} rows into Hopsworks feature group "
              f"'{config.FEATURE_GROUP_NAME}' (v{config.FEATURE_GROUP_VERSION}).")
        total_rows += len(features)

    print(f"Backfill complete: {total_rows} rows inserted across {len(cities)} cities.")
    return total_rows


if __name__ == "__main__":
    run_backfill()
