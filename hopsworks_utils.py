# -*- coding: utf-8 -*-
"""
Shared Hopsworks connection helpers used by the feature pipeline, training
pipeline, and dashboard, so login/feature-group/model-registry lookup logic
lives in one place instead of being copy-pasted into every entrypoint.
"""

import hopsworks

import config

FEATURE_GROUP_DESCRIPTION = "Hourly AQI + weather features for 3-day-ahead forecasting."

_PRIMARY_KEY = ["city", "event_time"]


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
    )


def get_model_registry(project=None):
    project = project or get_project()
    return project.get_model_registry()
