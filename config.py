# -*- coding: utf-8 -*-
"""
Central settings for the AQI forecasting project.

Every module imports constants from here instead of hardcoding city names,
coordinates, thresholds, or Hopsworks project names inline.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Location -----------------------------------------------------------

# All cities the pipelines/dashboard support. Keys double as the "city"
# primary-key value in the feature group and as the {city} slot in
# MODEL_REGISTRY_NAME_TEMPLATE, so keep them stable once data has been
# backfilled under them.
CITIES = {
    "karachi": {"label": "Karachi", "lat": 24.8607, "lon": 67.0011},
    "lahore": {"label": "Lahore", "lat": 31.5497, "lon": 74.3436},
    "islamabad": {"label": "Islamabad", "lat": 33.6844, "lon": 73.0479},
    "faisalabad": {"label": "Faisalabad", "lat": 31.4504, "lon": 73.1350},
    "multan": {"label": "Multan", "lat": 30.1575, "lon": 71.5249},
    "peshawar": {"label": "Peshawar", "lat": 34.0151, "lon": 71.5805},
    "quetta": {"label": "Quetta", "lat": 30.1798, "lon": 66.9750},
    "hyderabad": {"label": "Hyderabad", "lat": 25.3960, "lon": 68.3578},
    "abbottabad": {"label": "Abbottabad", "lat": 34.1463, "lon": 73.2117},
    "sukkur": {"label": "Sukkur", "lat": 27.7052, "lon": 68.8574},
}

# Single-city default, kept for scripts/back-compat (e.g. AQICN's one-city
# live reading). Multi-city jobs iterate config.CITIES directly instead.
CITY_NAME = os.getenv("AQI_CITY", "karachi")
_default_coords = CITIES.get(CITY_NAME, CITIES["karachi"])
LATITUDE = float(os.getenv("AQI_LAT", _default_coords["lat"]))
LONGITUDE = float(os.getenv("AQI_LON", _default_coords["lon"]))
TIMEZONE = os.getenv("AQI_TIMEZONE", "Asia/Karachi")

# --- Hopsworks ------------------------------------------------------------

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_karachi_forecast")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

MODEL_REGISTRY_NAME_TEMPLATE = "aqi_{city}_t_plus_{horizon}"

# --- AQICN (secondary/optional live-reading source) -----------------------

AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")

# --- Forecast horizons ------------------------------------------------------

FORECAST_HORIZONS_HOURS = [24, 48, 72]  # t+1 day, t+2 day, t+3 day

# --- Backfill ---------------------------------------------------------------

# Weather history now comes from Open-Meteo's archive (ERA5) endpoint, which
# serves years of data; air-quality history has real (non-null) coverage
# from ~2022-09 onward. 365 days gives training a full seasonal cycle
# without reaching into the null-heavy pre-2022-09 range.
BACKFILL_DAYS = int(os.getenv("AQI_BACKFILL_DAYS", "365"))

# --- Feature engineering windows ---------------------------------------------

ROLLING_WINDOWS_HOURS = [3, 24]
LAG_HOURS = [24, 48]

# --- AQI hazard alert thresholds (US AQI scale) ------------------------------

AQI_ALERT_THRESHOLD_RED = 151   # "Unhealthy" and above
AQI_ALERT_THRESHOLD_AMBER = 101  # "Unhealthy for Sensitive Groups" and above
