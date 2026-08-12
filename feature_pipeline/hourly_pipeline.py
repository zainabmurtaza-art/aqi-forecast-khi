# -*- coding: utf-8 -*-
"""
Scheduled hourly job (see .github/workflows/hourly_feature_pipeline.yml):
for every city in config.CITIES, pulls a short recent window from
Open-Meteo (enough to compute lag/rolling features), engineers features,
and upserts just the newest row(s) into the Hopsworks Feature Group.

Deliberately pulls a small window directly from Open-Meteo each run rather
than re-reading the whole feature store history (the problem with the
original prototype's full-CSV-recompute approach) — a few days of hourly
data is enough context for the longest lag/rolling feature, and Open-Meteo
is cheap/free to query repeatedly.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import config
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import fetch_combined
from hopsworks_utils import get_feature_group

# Longest lookback any derived feature needs, plus a small safety margin.
LOOKBACK_DAYS = max(config.ROLLING_WINDOWS_HOURS + config.LAG_HOURS) // 24 + 2


def run_hourly_update(rows_to_insert: int = 1, cities: Optional[Iterable[str]] = None) -> int:
    cities = list(cities) if cities is not None else list(config.CITIES)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    now = datetime.now(timezone.utc)

    fg = get_feature_group()
    total_rows = 0
    for city in cities:
        raw = fetch_combined(start_date=str(start_date), end_date=str(end_date), city=city)
        features = engineer_features(raw)

        completed_hours = features[features["event_time"] <= now]
        if completed_hours.empty:
            print(f"[{city}] No completed hourly rows available yet from Open-Meteo; skipping.")
            continue

        newest_rows = completed_hours.tail(rows_to_insert)
        fg.insert(newest_rows)
        print(f"[{city}] Upserted {len(newest_rows)} row(s) up to "
              f"{newest_rows['event_time'].max()} into "
              f"'{config.FEATURE_GROUP_NAME}' (v{config.FEATURE_GROUP_VERSION}).")
        total_rows += len(newest_rows)

    return total_rows


if __name__ == "__main__":
    run_hourly_update()
