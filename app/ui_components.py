# -*- coding: utf-8 -*-
"""Dashboard panels: alert banner, forecast chart, trend chart, SHAP panel."""

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

from training_pipeline.build_dataset import FEATURE_COLUMNS


AQI_KEY_RANGES = [
    (0, 50, "Good", "#2e7d32"),
    (51, 100, "Moderate", "#f9a825"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ef6c00"),
    (151, 200, "Unhealthy", "#d32f2f"),
    (201, 300, "Very Unhealthy", "#8e24aa"),
    (301, 500, "Hazardous", "#7e0023"),
]


def _aqi_category(aqi_value: float) -> tuple:
    """Returns (label, color) for a US AQI value, from the same breakpoints
    shown in the AQI Key - so the charts and the key never disagree."""
    for lo, hi, label, color in AQI_KEY_RANGES:
        if aqi_value <= hi:
            return label, color
    return AQI_KEY_RANGES[-1][2], AQI_KEY_RANGES[-1][3]


def render_aqi_key():
    """Static US AQI color/range legend, shown instead of the forecast charts
    when the sidebar is set to the 'AQI Key' view."""
    for lo, hi, label, color in AQI_KEY_RANGES:
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1rem; border-radius:0.5rem;
                        color:white; margin-bottom:0.5rem;">
                <strong>{lo}-{hi}: {label}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alert_banner(current_aqi: float, predictions: pd.DataFrame):
    worst_forecast = predictions["predicted_us_aqi"].max()
    # Severity color/label should escalate if current conditions are already
    # bad even when the forecast improves, but the displayed "worst forecast"
    # number must stay forecast-only - conflating the two previously showed
    # today's current reading mislabeled as a forecast value whenever it was
    # the higher of the two.
    label, color = _aqi_category(max(current_aqi, worst_forecast))

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem; color:white;">
            <strong>AQI status: {label}</strong> — current {current_aqi:.0f},
            worst forecast over next 3 days: {worst_forecast:.0f} (US AQI scale)
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_chart(predictions: pd.DataFrame):
    values = predictions["predicted_us_aqi"]
    mean_val = values.mean()
    # A 0-500 axis makes 3 nearby forecast values look almost equal-height;
    # zoom the visible range to the mean +/- the largest deviation (with a
    # padding floor so 3 near-identical values don't collapse to zero range)
    # so day-to-day differences are visible. Never dips below 0 since AQI can't.
    padding = max((values - mean_val).abs().max() * 1.4, 10)
    y_range = [max(mean_val - padding, 0), mean_val + padding]

    fig = go.Figure(
        go.Bar(
            x=[f"+{d} day" for d in predictions["horizon_days"]],
            y=values,
            marker_color=[_aqi_category(v)[1] for v in values],
        )
    )
    fig.update_layout(
        title="3-day AQI forecast",
        yaxis_title="Predicted US AQI",
        yaxis_range=y_range,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trend_chart(actual_df: pd.DataFrame):
    fig = go.Figure(
        go.Scatter(x=actual_df["event_time"], y=actual_df["us_aqi"], mode="lines")
    )
    fig.update_layout(title="Recent AQI trend", yaxis_title="US AQI", xaxis_title="Time (UTC)")
    st.plotly_chart(fig, use_container_width=True)


FEATURE_DESCRIPTIONS = {
    "pm10": "PM10 — coarse particulate matter (particles ≤10 micrometers), µg/m³",
    "pm2_5": "PM2.5 — fine particulate matter (particles ≤2.5 micrometers, the pollutant US AQI weighs most heavily), µg/m³",
    "carbon_monoxide": "Carbon monoxide (CO) concentration, µg/m³",
    "nitrogen_dioxide": "Nitrogen dioxide (NO₂) concentration, µg/m³ — mainly from vehicle/industrial combustion",
    "sulphur_dioxide": "Sulphur dioxide (SO₂) concentration, µg/m³ — mainly from burning fuel with sulphur in it",
    "ozone": "Ground-level ozone (O₃) concentration, µg/m³ — forms from sunlight reacting with other pollutants",
    "us_aqi": "The current US Air Quality Index reading at this hour",
    "temperature_2m": "Air temperature 2 meters above ground, °C",
    "relative_humidity_2m": "Relative humidity 2 meters above ground, %",
    "surface_pressure": "Atmospheric pressure at the surface, hPa",
    "wind_speed_10m": "Wind speed 10 meters above ground, km/h — higher wind disperses pollutants and tends to lower AQI",
    "hour": "Hour of the day (0-23)",
    "day": "Day of the month (1-31)",
    "month": "Month of the year (1-12)",
    "day_of_week": "Day of the week (0 = Monday ... 6 = Sunday)",
    "is_weekend": "1 if Saturday or Sunday, else 0",
    "aqi_change_rate": "How much AQI changed from the previous hour to this one",
    "aqi_roll_mean_3h": "Average AQI over the past 3 hours",
    "aqi_roll_mean_24h": "Average AQI over the past 24 hours",
    "aqi_lag_24h": "AQI value exactly 24 hours before this reading (same time yesterday)",
    "aqi_lag_48h": "AQI value exactly 48 hours before this reading (same time two days ago)",
}


def render_shap_panel(model, feature_row: pd.DataFrame, horizon_days: int):
    st.subheader(f"Why this +{horizon_days}-day prediction (SHAP)")

    X = feature_row[FEATURE_COLUMNS]
    if isinstance(model, (RandomForestRegressor, XGBRegressor)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, Ridge):
        explainer = shap.LinearExplainer(model, X)
    else:
        st.info("SHAP explanation not available for this model type.")
        return

    shap_values = explainer.shap_values(X)
    contributions = pd.Series(shap_values[0], index=FEATURE_COLUMNS).sort_values()

    fig = go.Figure(go.Bar(x=contributions.values, y=contributions.index, orientation="h"))
    fig.update_layout(title="Feature contribution to this prediction", xaxis_title="SHAP value")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What do these feature names mean?"):
        for col in FEATURE_COLUMNS:
            st.markdown(f"- **{col}** — {FEATURE_DESCRIPTIONS.get(col, '')}")
