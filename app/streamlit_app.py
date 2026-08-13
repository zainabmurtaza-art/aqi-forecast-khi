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
from app.data_loader import get_horizon_predictions, load_models, load_recent_actual_features
from app.ui_components import (
    render_aqi_key,
    render_alert_banner,
    render_forecast_chart,
    render_shap_panel,
    render_trend_chart,
)

st.set_page_config(page_title="AQI Forecast — Pakistan", layout="wide")

# Streamlit's selectbox is a searchable combobox that keeps text-input focus
# after a selection, which otherwise leaves a blinking text cursor sitting in
# the sidebar.
st.markdown(
    """
    <style>
    [data-testid="stSelectbox"] input {
        caret-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

view = st.sidebar.radio("View", ["Forecast", "AQI Key"])

if view == "AQI Key":
    st.title("US AQI Color Key")
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

st.title(f"Air Quality Forecast — {city_label}")

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

st.info(
    "Pick a forecast horizon below to see a SHAP breakdown of which pollutant, weather, "
    "and time features pushed that specific day's prediction up or down."
)

horizon_choice = st.selectbox(
    "Explain which forecast?",
    options=predictions["horizon_days"].tolist(),
    format_func=lambda d: f"+{d} day",
)
selected = predictions[predictions["horizon_days"] == horizon_choice].iloc[0]
horizon_hours = horizon_choice * 24
render_shap_panel(models[horizon_hours], selected["feature_row"], horizon_choice, actual_df)
