# -*- coding: utf-8 -*-
"""
One-off (re-runnable) job: pull config.BACKFILL_DAYS of history from
Open-Meteo for every city in config.CITIES, engineer features, and
bulk-insert into the Hopsworks Feature Group. Run this once to seed a real
training dataset, and again any time a larger history window is wanted
(inserts are upserts on the (city, event_time) primary key, so re-running
is safe).

Weather history comes from Open-Meteo's ERA5 archive, which serves years of
real data, but air-quality history (a separate CAMS-based endpoint) only has
real coverage from ~2022-09 onward for cities in this project - a backfill
window reaching further back than that just yields extra rows with a null
target, which the training pipeline already drops. MAX_BACKFILL_DAYS below
is a sanity ceiling against a typo (e.g. an accidental extra digit in
AQI_BACKFILL_DAYS), not a real API limit.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import config
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import fetch_combined
from hopsworks_utils import get_feature_group

MAX_BACKFILL_DAYS = 1825  # 5 years


def run_backfill(days: int = config.BACKFILL_DAYS, cities: Optional[Iterable[str]] = None) -> int:
    if days > MAX_BACKFILL_DAYS:
        raise ValueError(
            f"AQI_BACKFILL_DAYS={days} exceeds the {MAX_BACKFILL_DAYS}-day sanity ceiling. "
            "If you really want more than 5 years of history, raise MAX_BACKFILL_DAYS."
        )

    cities = list(cities) if cities is not None else list(config.CITIES)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    fg = get_feature_group()
    total_rows = 0
    failed_cities = []
    for city in cities:
        try:
            print(f"[{city}] Fetching {days} days of history ({start_date} to {end_date})...")
            raw = fetch_combined(start_date=str(start_date), end_date=str(end_date), city=city)
            print(f"[{city}] Fetched {len(raw)} raw hourly rows.")

            features = engineer_features(raw)
            fg.insert(features)
            print(f"[{city}] Inserted {len(features)} rows into Hopsworks feature group "
                  f"'{config.FEATURE_GROUP_NAME}' (v{config.FEATURE_GROUP_VERSION}).")
            total_rows += len(features)
        except Exception as exc:
            # Inserts are upserts on (city, event_time), so a failed city just
            # needs a re-run - don't let it take down the other 9 cities too.
            print(f"[{city}] Backfill failed, skipping: {exc}")
            failed_cities.append(city)

    print(f"Backfill complete: {total_rows} rows inserted across "
          f"{len(cities) - len(failed_cities)}/{len(cities)} cities.")
    if failed_cities:
        raise RuntimeError(f"Backfill failed for: {', '.join(failed_cities)} (re-run to retry them).")
    return total_rows


if __name__ == "__main__":
    run_backfill()
