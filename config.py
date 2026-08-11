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

CITY_NAME = os.getenv("AQI_CITY", "karachi")
LATITUDE = float(os.getenv("AQI_LAT", "24.8607"))
LONGITUDE = float(os.getenv("AQI_LON", "67.0011"))
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

BACKFILL_DAYS = int(os.getenv("AQI_BACKFILL_DAYS", "90"))

# --- Feature engineering windows ---------------------------------------------

ROLLING_WINDOWS_HOURS = [3, 24]
LAG_HOURS = [24, 48]

# --- AQI hazard alert thresholds (US AQI scale) ------------------------------

AQI_ALERT_THRESHOLD_RED = 151   # "Unhealthy" and above
AQI_ALERT_THRESHOLD_AMBER = 101  # "Unhealthy for Sensitive Groups" and above
