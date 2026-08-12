# -*- coding: utf-8 -*-
"""
Open-Meteo client: free, keyless air-quality and weather data, for both
historical (backfill) and forecast (hourly pipeline + inference) use.

Both Open-Meteo APIs accept the same start_date/end_date windowing, so a
single date range can be requested per call. Pass a past start_date for
history, or leave start_date unset and pass forecast_days for a forecast
that starts today.

Weather history uses two different endpoints depending on the request:
WEATHER_URL (the forecast endpoint) only serves ~92 days of past data, so
any start_date/end_date range instead goes to WEATHER_ARCHIVE_URL, the
ERA5-based reanalysis archive - confirmed by hand to return real (non-null)
data back to 2015 with no gap up to the current hour, unlike a forecast
endpoint's history window. AIR_QUALITY_URL, by contrast, already serves its
own full history directly (confirmed real CAMS coverage from ~2022-09
onward; NaN before that, which the training pipeline already drops), so it
doesn't need a separate archive endpoint.
"""

from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# A 10-city loop makes 20 requests back to back, and the dashboard adds its
# own bursts on top (every city switch is a cache miss on first view); retry
# transient network blips/5xx and rate limiting (429) instead of failing
# immediately. urllib3 honors a Retry-After header if Open-Meteo sends one,
# otherwise backoff_factor gives ~1s/2s/4s between the 3 attempts.
_session = requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retry))

AIR_QUALITY_HOURLY_FIELDS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

WEATHER_HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


def _fetch_hourly(
    url: str,
    hourly_fields: list,
    latitude: float,
    longitude: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_days: Optional[int] = None,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_fields),
        "timezone": "UTC",
    }
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    elif forecast_days:
        params["forecast_days"] = forecast_days
    else:
        raise ValueError("Provide either start_date+end_date or forecast_days.")

    response = _session.get(url, params=params, timeout=(10, 45))
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError(f"Open-Meteo response had no hourly data: {payload}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Force a consistent float dtype for every value column. Without this,
    # pandas infers int64 vs float64 depending on whether this particular
    # batch happens to contain any whole numbers/NaNs, which then clashes
    # with the Hopsworks feature group's schema (fixed from the first write).
    for field in hourly_fields:
        df[field] = df[field].astype(float)

    return df.rename(columns={"time": "event_time"})


def fetch_air_quality(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_days: Optional[int] = None,
    city: str = config.CITY_NAME,
) -> pd.DataFrame:
    """Hourly pm10/pm2_5/co/no2/so2/o3/us_aqi for either a date range or forecast_days ahead."""
    coords = config.CITIES[city]
    return _fetch_hourly(
        AIR_QUALITY_URL,
        AIR_QUALITY_HOURLY_FIELDS,
        coords["lat"],
        coords["lon"],
        start_date,
        end_date,
        forecast_days,
    )


def fetch_weather(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_days: Optional[int] = None,
    city: str = config.CITY_NAME,
) -> pd.DataFrame:
    """Hourly temperature/humidity/pressure/wind for either a date range or forecast_days ahead.
    A date range uses the archive/reanalysis endpoint (years of history); forecast_days
    uses the forecast endpoint, which archive-api doesn't serve."""
    coords = config.CITIES[city]
    url = WEATHER_ARCHIVE_URL if (start_date and end_date) else WEATHER_URL
    return _fetch_hourly(
        url,
        WEATHER_HOURLY_FIELDS,
        coords["lat"],
        coords["lon"],
        start_date,
        end_date,
        forecast_days,
    )


def fetch_combined(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    forecast_days: Optional[int] = None,
    city: str = config.CITY_NAME,
) -> pd.DataFrame:
    """Air quality + weather merged on event_time, with the city column added."""
    air = fetch_air_quality(start_date, end_date, forecast_days, city)
    weather = fetch_weather(start_date, end_date, forecast_days, city)

    merged = air.merge(weather, on="event_time", how="inner")
    merged.insert(0, "city", city)
    return merged


if __name__ == "__main__":
    print(fetch_combined(forecast_days=3).head())
