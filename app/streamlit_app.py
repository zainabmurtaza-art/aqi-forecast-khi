# -*- coding: utf-8 -*-
"""
AQI forecast dashboard. Run locally with:
    streamlit run app/streamlit_app.py

Deployed on Streamlit Community Cloud pointing at this same file, with
HOPSWORKS_API_KEY set under the app's own Settings -> Secrets.
"""

import streamlit as st

import config
from app.data_loader import get_horizon_predictions, load_models, load_recent_actual_features
from app.ui_components import render_alert_banner, render_forecast_chart, render_shap_panel, render_trend_chart

st.set_page_config(page_title=f"AQI Forecast — {config.CITY_NAME.title()}", layout="wide")
st.title(f"Air Quality Forecast — {config.CITY_NAME.title()}")

with st.spinner("Loading models and latest data..."):
    models = load_models()
    actual_df = load_recent_actual_features()
    predictions = get_horizon_predictions(models)

if actual_df.empty:
    st.error(
        "No feature data found in Hopsworks yet. Run "
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

horizon_choice = st.selectbox(
    "Explain which forecast?",
    options=predictions["horizon_days"].tolist(),
    format_func=lambda d: f"+{d} day",
)
selected = predictions[predictions["horizon_days"] == horizon_choice].iloc[0]
horizon_hours = horizon_choice * 24
render_shap_panel(models[horizon_hours], selected["feature_row"], horizon_choice)
