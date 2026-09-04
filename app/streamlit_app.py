# -*- coding: utf-8 -*-
"""
AQI forecast dashboard. Run locally with:
    streamlit run app/streamlit_app.py

Deployed on Streamlit Community Cloud pointing at this same file, with
HOPSWORKS_API_KEY set under the app's own Settings -> Secrets.
"""

import sys
from pathlib import Path

# Streamlit Cloud runs this file with only its own directory (app/) on
# sys.path, not the project root - add the root explicitly so config.py,
# hopsworks_utils.py, feature_pipeline/, and training_pipeline/ are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import config
from app import theme
from app.data_loader import get_horizon_predictions, load_models, load_recent_actual_features
from app.ui_components import (
    render_aqi_key,
    render_alert_banner,
    render_current_readings,
    render_forecast_chart,
    render_manual_prediction_form,
    render_shap_panel,
    render_trend_chart,
)

st.set_page_config(page_title="AQI Forecast — Pakistan", layout="wide")
theme.apply_theme()

view = st.sidebar.radio(
    "View", ["Current Readings", "Forecast", "Manual Prediction", "AQI Key"]
)

if view == "AQI Key":
    st.title("US AQI Colour Key")
    st.markdown(
        '<div class="aqi-caption">The category bands every reading and forecast '
        "on this dashboard is coloured against.</div>",
        unsafe_allow_html=True,
    )
    render_aqi_key()
    st.stop()

city_keys = list(config.CITIES.keys())
default_city = config.CITY_NAME if config.CITY_NAME in config.CITIES else city_keys[0]

selected_city = st.sidebar.selectbox(
    "City",
    options=city_keys,
    format_func=lambda c: config.CITIES[c]["label"],
    index=city_keys.index(default_city),
)
city_label = config.CITIES[selected_city]["label"]

if view == "Current Readings":
    st.title(f"Current Readings — {city_label}")
    st.markdown(
        '<div class="aqi-caption">Today&rsquo;s air quality alongside every '
        "feature the forecast models read, as most recently recorded.</div>",
        unsafe_allow_html=True,
    )
    # Deliberately needs neither the model registry nor Open-Meteo: this page
    # keeps working during a model-registry or forecast-API outage, which is
    # exactly when someone is most likely to want the raw current numbers.
    with st.spinner(f"Loading the latest readings for {city_label}..."):
        try:
            readings_df = load_recent_actual_features(selected_city)
        except Exception:
            st.error(
                "Couldn't reach the Hopsworks feature store right now. "
                "Please reload in a minute or two."
            )
            st.stop()

    if readings_df.empty:
        st.error(
            f"No feature data found in Hopsworks yet for {city_label}. Run "
            "`python -m feature_pipeline.backfill_pipeline` first."
        )
        st.stop()

    render_current_readings(readings_df, city_label)
    st.stop()

if view == "Manual Prediction":
    st.title(f"Manual AQI Prediction — {city_label}")
    with st.spinner(f"Loading models for {city_label}..."):
        try:
            models = load_models(selected_city)
        except Exception:
            st.error(
                f"No trained models found in Hopsworks yet for {city_label}. Run "
                "`python -m feature_pipeline.backfill_pipeline` and "
                "`python -m training_pipeline.train` first."
            )
            st.stop()
    render_manual_prediction_form(models)
    st.stop()

st.title(f"Air Quality Forecast — {city_label}")
st.markdown(
    '<div class="aqi-caption">Three-day US AQI outlook, with the recent trend '
    "behind it. See <strong>Current Readings</strong> for today&rsquo;s full "
    "feature detail.</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Loading models and latest data for {city_label}..."):
    try:
        models = load_models(selected_city)
    except Exception:
        st.error(
            f"No trained models found in Hopsworks yet for {city_label}. Run "
            "`python -m feature_pipeline.backfill_pipeline` and "
            "`python -m training_pipeline.train` first."
        )
        st.stop()

    actual_df = load_recent_actual_features(selected_city)

    try:
        predictions = get_horizon_predictions(models, selected_city)
    except Exception:
        st.error(
            "Couldn't reach Open-Meteo for the live forecast right now (likely a "
            "temporary rate limit) even after retrying with backoff. Please reload "
            "in a minute or two."
        )
        st.stop()

if actual_df.empty:
    st.error(
        f"No feature data found in Hopsworks yet for {city_label}. Run "
        "`python -m feature_pipeline.backfill_pipeline` first."
    )
    st.stop()

current_aqi = float(actual_df["us_aqi"].iloc[-1])

render_alert_banner(current_aqi, predictions)

col1, col2 = st.columns(2)
with col1:
    render_forecast_chart(predictions)
with col2:
    render_trend_chart(actual_df)

st.divider()

st.subheader("Why the model forecast this")
st.markdown(
    '<div class="aqi-caption">Pick a horizon to see a SHAP breakdown of which '
    "pollutant, weather, and time features pushed that day&rsquo;s prediction "
    "up or down.</div>",
    unsafe_allow_html=True,
)

horizon_choice = st.selectbox(
    "Explain which forecast?",
    options=predictions["horizon_days"].tolist(),
    format_func=lambda d: f"+{d} day",
)
selected = predictions[predictions["horizon_days"] == horizon_choice].iloc[0]
horizon_hours = horizon_choice * 24
render_shap_panel(models[horizon_hours], selected["feature_row"], horizon_choice, actual_df)
