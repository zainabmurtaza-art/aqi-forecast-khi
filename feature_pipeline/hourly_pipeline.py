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

Failure policy: writes are idempotent upserts keyed on (city, event_time) and
this job runs every hour, so a city that misses one hour is picked up by the
next run with no gap left behind. A blip affecting a minority of cities
therefore annotates the run rather than failing it; the job only exits
non-zero when enough cities fail at once to indicate a real outage rather
than noise (see MAX_TOLERATED_FAILURE_RATIO).
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import ci_annotations
import config
from feature_pipeline.feature_engineering import engineer_features
from feature_pipeline.open_meteo_client import fetch_combined
from hopsworks_utils import get_feature_group, retry_on_transient

# Longest lookback any derived feature needs, plus a small safety margin.
LOOKBACK_DAYS = max(config.ROLLING_WINDOWS_HOURS + config.LAG_HOURS) // 24 + 2

# Fail the run only once more than this share of cities failed in one pass.
# Below it, the next hourly run self-heals the gap; above it, something
# systemic is wrong (expired API key, Hopsworks outage) and should go red.
MAX_TOLERATED_FAILURE_RATIO = 0.5


def run_hourly_update(rows_to_insert: int = 1, cities: Optional[Iterable[str]] = None) -> int:
    cities = list(cities) if cities is not None else list(config.CITIES)

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    now = datetime.now(timezone.utc)

    fg = retry_on_transient(get_feature_group, description="Fetch feature group")
    total_rows = 0
    failed = {}
    for city in cities:
        try:
            raw = fetch_combined(start_date=str(start_date), end_date=str(end_date), city=city)
            features = engineer_features(raw)

            completed_hours = features[features["event_time"] <= now]
            if completed_hours.empty:
                print(f"[{city}] No completed hourly rows available yet from Open-Meteo; skipping.")
                continue

            newest_rows = completed_hours.tail(rows_to_insert)
            # The write is the step that actually talks to Hopsworks, and a
            # 5xx here was previously enough on its own to fail the whole run.
            retry_on_transient(
                lambda rows=newest_rows: fg.insert(rows),
                description=f"[{city}] feature group insert",
            )
            print(f"[{city}] Upserted {len(newest_rows)} row(s) up to "
                  f"{newest_rows['event_time'].max()} into "
                  f"'{config.FEATURE_GROUP_NAME}' (v{config.FEATURE_GROUP_VERSION}).")
            total_rows += len(newest_rows)
        except Exception as exc:
            # Don't let one city's failure skip the rest, since this runs
            # unattended every hour for all cities.
            print(f"[{city}] Hourly update failed, skipping: {type(exc).__name__}: {exc}")
            failed[city] = f"{type(exc).__name__}: {exc}"

    _report(failed, attempted=len(cities), total_rows=total_rows)
    return total_rows


def _report(failed: dict, attempted: int, total_rows: int) -> None:
    """Logs the outcome, and decides whether a partial failure fails the run."""
    succeeded = attempted - len(failed)
    print(f"\nHourly update finished: {succeeded}/{attempted} cities updated, "
          f"{total_rows} row(s) written.")

    if not failed:
        return

    detail = "; ".join(f"{city} ({reason})" for city, reason in failed.items())
    ci_annotations.write_summary(
        f"### Hourly feature pipeline\n\n"
        f"- Cities updated: **{succeeded}/{attempted}**\n"
        f"- Rows written: **{total_rows}**\n"
        f"- Failed: {', '.join(failed)}\n"
    )

    if len(failed) > attempted * MAX_TOLERATED_FAILURE_RATIO:
        raise RuntimeError(
            f"Hourly update failed for {len(failed)}/{attempted} cities, which is "
            f"past the tolerated threshold — treating as an outage: {detail}."
        )

    ci_annotations.warn(
        f"Hourly update skipped {len(failed)}/{attempted} cities this run "
        f"({', '.join(failed)}); the next hourly run backfills them. Detail: {detail}"
    )


if __name__ == "__main__":
    run_hourly_update()
