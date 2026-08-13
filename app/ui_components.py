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
