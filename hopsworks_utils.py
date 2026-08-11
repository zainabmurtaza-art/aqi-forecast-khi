# -*- coding: utf-8 -*-
"""
Shared Hopsworks connection helpers used by the feature pipeline, training
pipeline, and dashboard, so login/feature-group/model-registry lookup logic
lives in one place instead of being copy-pasted into every entrypoint.
"""

import hopsworks
from hsfs.feature import Feature

import config

FEATURE_GROUP_DESCRIPTION = "Hourly AQI + weather features for 3-day-ahead forecasting."

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


def get_project():
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )


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
