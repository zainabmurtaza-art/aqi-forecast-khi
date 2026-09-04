# -*- coding: utf-8 -*-
"""
Shared Hopsworks connection helpers used by the feature pipeline, training
pipeline, and dashboard, so login/feature-group/model-registry lookup logic
lives in one place instead of being copy-pasted into every entrypoint.

Also owns retry_on_transient(), the one retry policy every Hopsworks call in
this project goes through — see its docstring for why a plain
`requests`-based retry list wasn't enough.
"""

import time

import hopsworks
import requests
from hsfs.feature import Feature

import config

# Hopsworks signals a server-side HTTP failure with RestAPIError, which
# derives from plain Exception rather than from requests.exceptions.
# RequestException — so a retry list built only out of `requests`/OSError
# types silently misses every 500/502/503 the Hopsworks API returns, which is
# exactly the failure that was turning otherwise-fine scheduled runs red.
# Imported defensively so a future SDK reshuffle degrades to "don't retry
# REST errors" instead of breaking the pipelines at import time.
try:
    from hopsworks.client.exceptions import RestAPIError
except ImportError:  # pragma: no cover - depends on installed SDK layout
    try:
        from hopsworks_common.client.exceptions import RestAPIError
    except ImportError:
        RestAPIError = None

FEATURE_GROUP_DESCRIPTION = "Hourly AQI + weather features for 3-day-ahead forecasting."

# Status codes worth another attempt: transient server-side faults, gateway
# blips, and rate limiting. Deliberately excludes 400/401/403/404 — retrying a
# bad request or a bad API key just wastes minutes before failing anyway.
_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Network-level faults. Note builtin ConnectionError/TimeoutError are OSError
# subclasses, and requests' own ConnectionError/Timeout are RequestException
# subclasses; both are listed for clarity rather than out of necessity.
_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.RequestException,
    ConnectionError,
    TimeoutError,
    OSError,
)

_PRIMARY_KEY = ["city", "event_time"]

# Declared explicitly so the schema is always what our code intends, rather
# than whatever pandas happens to infer from the first DataFrame that gets
# written (which caused int-vs-double mismatches between different-sized
# fetch windows in practice).
_SCHEMA = [
    Feature("city", type="string"),
    Feature("event_time", type="timestamp"),
    Feature("pm10", type="double"),
    Feature("pm2_5", type="double"),
    Feature("carbon_monoxide", type="double"),
    Feature("nitrogen_dioxide", type="double"),
    Feature("sulphur_dioxide", type="double"),
    Feature("ozone", type="double"),
    Feature("us_aqi", type="double"),
    Feature("temperature_2m", type="double"),
    Feature("relative_humidity_2m", type="double"),
    Feature("surface_pressure", type="double"),
    Feature("wind_speed_10m", type="double"),
    Feature("hour", type="bigint"),
    Feature("day", type="bigint"),
    Feature("month", type="bigint"),
    Feature("day_of_week", type="bigint"),
    Feature("is_weekend", type="bigint"),
    Feature("aqi_change_rate", type="double"),
    Feature("aqi_roll_mean_3h", type="double"),
    Feature("aqi_roll_mean_24h", type="double"),
    Feature("aqi_lag_24h", type="double"),
    Feature("aqi_lag_48h", type="double"),
]


def is_transient(exc: BaseException) -> bool:
    """True if `exc` looks like a blip worth retrying rather than a real bug.

    Covers network-level faults plus Hopsworks REST errors carrying a
    retryable HTTP status. A RestAPIError with a 4xx status other than 408/425/
    429 (bad payload, bad key, missing model) is reported as non-transient so
    it surfaces immediately instead of after several pointless retries.
    """
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if RestAPIError is not None and isinstance(exc, RestAPIError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in _TRANSIENT_HTTP_STATUS
    return False


def retry_on_transient(
    fn,
    retries: int = 4,
    backoff_seconds: float = 5.0,
    description: str = "Hopsworks call",
    on_retry=None,
):
    """Calls `fn()`, retrying transient failures with exponential backoff.

    `on_retry` runs before each retry — the training pipeline uses it to drop
    the cached login so the next attempt reconnects rather than reusing a
    connection that may itself be what broke.
    """
    delay = backoff_seconds
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries or not is_transient(exc):
                raise
            print(
                f"{description}: transient failure on attempt {attempt}/{retries} "
                f"({type(exc).__name__}: {str(exc)[:200]}). Retrying in {delay:.0f}s..."
            )
            if on_retry is not None:
                on_retry()
            time.sleep(delay)
            delay *= 2


# Logging in is itself a network round trip, and the training pipeline used to
# do one per model registration (30 per run) — each an independent chance to
# hit a blip. Cache the project so a process logs in once.
_cached_project = None


def get_project(force_new: bool = False):
    global _cached_project
    if _cached_project is None or force_new:
        _cached_project = retry_on_transient(
            lambda: hopsworks.login(
                api_key_value=config.HOPSWORKS_API_KEY,
                project=config.HOPSWORKS_PROJECT_NAME,
            ),
            description="Hopsworks login",
        )
    return _cached_project


def reset_project():
    """Drops the cached login so the next get_project() reconnects."""
    global _cached_project
    _cached_project = None


def get_feature_store(project=None):
    project = project or get_project()
    return project.get_feature_store()


def get_feature_group(feature_store=None):
    feature_store = feature_store or get_feature_store()
    return feature_store.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description=FEATURE_GROUP_DESCRIPTION,
        primary_key=_PRIMARY_KEY,
        event_time="event_time",
        # Delta format needs an extra native library we don't install; Hudi is
        # Hopsworks' other built-in option and needs nothing extra.
        time_travel_format="HUDI",
        features=_SCHEMA,
    )


def get_model_registry(project=None):
    project = project or get_project()
    return project.get_model_registry()
