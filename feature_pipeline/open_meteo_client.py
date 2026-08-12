# -*- coding: utf-8 -*-
"""
Open-Meteo client: free, keyless air-quality and weather data, for both
historical (backfill) and forecast (hourly pipeline + inference) use.

Both Open-Meteo APIs accept the same start_date/end_date windowing, so a
single date range can be requested per call. Pass a past start_date for
history, or leave start_date unset and pass forecast_days for a forecast
that starts today.
"""

from typing import Optional

import pandas as pd
import requests

import config

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WEATHER_URL only serves ~92 days of past data (it's the forecast endpoint's
# recent-history window, not a full archive) - a start_date further back than
# that returns HTTP 400. config.BACKFILL_DAYS must stay under this ceiling;
# a longer backfill would need the separate archive-api.open-meteo.com/v1/archive
# endpoint instead.
MAX_WEATHER_HISTORY_DAYS = 90

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

    response = requests.get(url, params=params, timeout=30)
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
    """Hourly temperature/humidity/pressure/wind for either a date range or forecast_days ahead."""
    coords = config.CITIES[city]
    return _fetch_hourly(
        WEATHER_URL,
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
