# -*- coding: utf-8 -*-
"""
Optional secondary data source: a single live AQICN reading for the
configured city. Superseded as the primary source by open_meteo_client.py,
which has historical and forecast endpoints AQICN's free tier lacks.
"""

from datetime import datetime, timezone

import requests

import config

AQICN_FEED_URL = "https://api.waqi.info/feed/{city}/?token={token}"


def fetch_raw_reading(city: str = config.CITY_NAME) -> dict:
    """Fetches the current AQICN reading for `city` as raw JSON."""
    if not config.AQICN_API_TOKEN:
        raise RuntimeError(
            "AQICN_API_TOKEN is not set. Add it to your .env if you want to "
            "use the AQICN live-reading source (optional; the feature "
            "pipeline runs fine without it via Open-Meteo)."
        )

    url = AQICN_FEED_URL.format(city=city, token=config.AQICN_API_TOKEN)
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN API call failed: {data}")

    return data


def to_live_reading_row(raw: dict, city: str = config.CITY_NAME) -> dict:
    """Extracts a small dict of live pollutant/AQI values from a raw AQICN reply."""
    d = raw["data"]
    iaqi = d.get("iaqi", {})

    return {
        "city": city,
        "event_time": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
        "aqicn_aqi": d.get("aqi"),
        "aqicn_pm25": iaqi.get("pm25", {}).get("v"),
        "aqicn_pm10": iaqi.get("pm10", {}).get("v"),
        "aqicn_temperature": iaqi.get("t", {}).get("v"),
        "aqicn_pressure": iaqi.get("p", {}).get("v"),
        "aqicn_humidity": iaqi.get("h", {}).get("v"),
        "aqicn_wind": iaqi.get("w", {}).get("v"),
    }


def fetch_live_reading(city: str = config.CITY_NAME) -> dict:
    """Convenience wrapper: fetch + extract in one call."""
    return to_live_reading_row(fetch_raw_reading(city), city)


if __name__ == "__main__":
    print(fetch_live_reading())
