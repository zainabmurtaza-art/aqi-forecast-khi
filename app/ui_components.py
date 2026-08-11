# -*- coding: utf-8 -*-
"""Dashboard panels: alert banner, forecast chart, trend chart, SHAP panel."""

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import config
from training_pipeline.build_dataset import FEATURE_COLUMNS


def _aqi_category(aqi_value: float) -> tuple:
    """Returns (label, color) for a US AQI value."""
    if aqi_value >= config.AQI_ALERT_THRESHOLD_RED:
        return "Unhealthy or worse", "#d32f2f"
    if aqi_value >= config.AQI_ALERT_THRESHOLD_AMBER:
        return "Unhealthy for sensitive groups", "#f9a825"
    return "Acceptable", "#2e7d32"


def render_alert_banner(current_aqi: float, predictions: pd.DataFrame):
    worst_aqi = max(current_aqi, predictions["predicted_us_aqi"].max())
    label, color = _aqi_category(worst_aqi)

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:1rem; border-radius:0.5rem; color:white;">
            <strong>AQI status: {label}</strong> — current {current_aqi:.0f},
            worst forecast over next 3 days: {worst_aqi:.0f} (US AQI scale)
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_chart(predictions: pd.DataFrame):
    fig = go.Figure(
        go.Bar(
            x=[f"+{d} day" for d in predictions["horizon_days"]],
            y=predictions["predicted_us_aqi"],
            marker_color=[_aqi_category(v)[1] for v in predictions["predicted_us_aqi"]],
        )
    )
    fig.update_layout(title="3-day AQI forecast", yaxis_title="Predicted US AQI")
    st.plotly_chart(fig, use_container_width=True)


def render_trend_chart(actual_df: pd.DataFrame):
    fig = go.Figure(
        go.Scatter(x=actual_df["event_time"], y=actual_df["us_aqi"], mode="lines")
    )
    fig.update_layout(title="Recent AQI trend", yaxis_title="US AQI", xaxis_title="Time (UTC)")
    st.plotly_chart(fig, use_container_width=True)


def render_shap_panel(model, feature_row: pd.DataFrame, horizon_days: int):
    st.subheader(f"Why this +{horizon_days}-day prediction (SHAP)")

    X = feature_row[FEATURE_COLUMNS]
    if isinstance(model, RandomForestRegressor):
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
