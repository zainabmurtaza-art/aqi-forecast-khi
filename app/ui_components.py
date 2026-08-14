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
from training_pipeline.models import PersistenceRegressor


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


def render_shap_panel(
    model, feature_row: pd.DataFrame, horizon_days: int, background_df: pd.DataFrame
):
    st.subheader(f"Why this +{horizon_days}-day prediction (SHAP)")

    X = feature_row[FEATURE_COLUMNS]
    if isinstance(model, (RandomForestRegressor, XGBRegressor)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, Ridge):
        # LinearExplainer needs a real background sample to compute an expected
        # value against - passing the single row being explained as its own
        # background (the previous bug here) makes every SHAP value exactly 0,
        # since there's nothing to attribute the difference to.
        background = background_df[FEATURE_COLUMNS].dropna()
        explainer = shap.LinearExplainer(model, background)
    elif isinstance(model, PersistenceRegressor):
        st.info(
            "This forecast uses a naive persistence baseline (predicted = current AQI) "
            "because it beat every trained model on this city/horizon's held-out test "
            "data - there's no feature-driven explanation to show for a rule this simple."
        )
        return
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


def render_manual_prediction_form(models: dict):
    """Lets a user type in their own feature values and see what the trained
    model predicts - a 'what-if' calculator rather than a live forecast.
    Only needs already-loaded models (no Open-Meteo/Hopsworks feature-group
    call), so this page works even during a live-data outage."""
    st.write(
        "Enter pollutant, weather, and time values to see what the trained "
        "model predicts for US AQI. Useful for exploring “what if” "
        "scenarios, independent of live forecast data."
    )

    horizon_days_options = sorted(h // 24 for h in models.keys())
    horizon_choice = st.selectbox(
        "Predict using which horizon's model?",
        options=horizon_days_options,
        format_func=lambda d: f"+{d} day model",
    )
    horizon_hours = horizon_choice * 24

    st.subheader("Pollutants")
    col1, col2 = st.columns(2)
    with col1:
        pm10 = st.number_input(
            "PM10 (µg/m³)", min_value=0.0, value=50.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["pm10"],
        )
        pm2_5 = st.number_input(
            "PM2.5 (µg/m³)", min_value=0.0, value=30.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["pm2_5"],
        )
        carbon_monoxide = st.number_input(
            "Carbon monoxide (µg/m³)", min_value=0.0, value=300.0, step=10.0,
            help=FEATURE_DESCRIPTIONS["carbon_monoxide"],
        )
        nitrogen_dioxide = st.number_input(
            "Nitrogen dioxide (µg/m³)", min_value=0.0, value=20.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["nitrogen_dioxide"],
        )
    with col2:
        sulphur_dioxide = st.number_input(
            "Sulphur dioxide (µg/m³)", min_value=0.0, value=10.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["sulphur_dioxide"],
        )
        ozone = st.number_input(
            "Ozone (µg/m³)", min_value=0.0, value=30.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["ozone"],
        )
        us_aqi = st.number_input(
            "Current US AQI", min_value=0.0, max_value=500.0, value=100.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["us_aqi"] + " — the “Recent AQI history” "
            "section below starts equal to this value (the model expects them to be "
            "roughly consistent with each other); use its reset button if you change "
            "this after adjusting those.",
        )

    st.subheader("Weather")
    col3, col4 = st.columns(2)
    with col3:
        temperature_2m = st.number_input(
            "Temperature (°C)", value=25.0, step=0.5,
            help=FEATURE_DESCRIPTIONS["temperature_2m"],
        )
        relative_humidity_2m = st.number_input(
            "Relative humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["relative_humidity_2m"],
        )
    with col4:
        surface_pressure = st.number_input(
            "Surface pressure (hPa)", value=1013.0, step=0.5,
            help=FEATURE_DESCRIPTIONS["surface_pressure"],
        )
        wind_speed_10m = st.number_input(
            "Wind speed (km/h)", min_value=0.0, value=10.0, step=0.5,
            help=FEATURE_DESCRIPTIONS["wind_speed_10m"],
        )

    st.subheader("Date & time")
    col5, col6 = st.columns(2)
    with col5:
        selected_date = st.date_input("Date")
    with col6:
        hour = st.slider("Hour of day", 0, 23, 12, help=FEATURE_DESCRIPTIONS["hour"])

    day = selected_date.day
    month = selected_date.month
    day_of_week = selected_date.weekday()  # Monday=0 ... Sunday=6, same convention as training
    is_weekend = int(day_of_week >= 5)

    history_keys = [
        "manual_pred_roll_3h", "manual_pred_roll_24h",
        "manual_pred_lag_24h", "manual_pred_lag_48h",
    ]
    # setdefault seeds each key's very first value only; passing `value=` AND a
    # `key=` already holding session state to the same widget call just logs a
    # Streamlit warning and is ignored, so session_state has to be the only
    # source of truth from here on (verified live: widgets keep whatever value
    # they were last given across reruns regardless of a later `value=` arg -
    # it's the only way to actually push a new value into a rendered widget).
    for k in history_keys:
        st.session_state.setdefault(k, float(us_aqi))

    with st.expander("Recent AQI history (defaults to Current AQI above until you change them)"):
        if st.button("Reset these to match Current AQI"):
            for k in history_keys:
                st.session_state[k] = float(us_aqi)
            st.rerun()

        aqi_change_rate = st.number_input(
            "AQI change from the previous hour", value=0.0, step=1.0,
            help=FEATURE_DESCRIPTIONS["aqi_change_rate"],
        )
        aqi_roll_mean_3h = st.number_input(
            "Average AQI over the past 3 hours", step=1.0,
            key=history_keys[0], help=FEATURE_DESCRIPTIONS["aqi_roll_mean_3h"],
        )
        aqi_roll_mean_24h = st.number_input(
            "Average AQI over the past 24 hours", step=1.0,
            key=history_keys[1], help=FEATURE_DESCRIPTIONS["aqi_roll_mean_24h"],
        )
        aqi_lag_24h = st.number_input(
            "AQI exactly 24 hours ago", step=1.0,
            key=history_keys[2], help=FEATURE_DESCRIPTIONS["aqi_lag_24h"],
        )
        aqi_lag_48h = st.number_input(
            "AQI exactly 48 hours ago", step=1.0,
            key=history_keys[3], help=FEATURE_DESCRIPTIONS["aqi_lag_48h"],
        )

    if st.button("Predict AQI", type="primary"):
        row = pd.DataFrame([{
            "pm10": pm10,
            "pm2_5": pm2_5,
            "carbon_monoxide": carbon_monoxide,
            "nitrogen_dioxide": nitrogen_dioxide,
            "sulphur_dioxide": sulphur_dioxide,
            "ozone": ozone,
            "us_aqi": us_aqi,
            "temperature_2m": temperature_2m,
            "relative_humidity_2m": relative_humidity_2m,
            "surface_pressure": surface_pressure,
            "wind_speed_10m": wind_speed_10m,
            "hour": hour,
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "aqi_change_rate": aqi_change_rate,
            "aqi_roll_mean_3h": aqi_roll_mean_3h,
            "aqi_roll_mean_24h": aqi_roll_mean_24h,
            "aqi_lag_24h": aqi_lag_24h,
            "aqi_lag_48h": aqi_lag_48h,
        }])[FEATURE_COLUMNS]

        model = models[horizon_hours]
        predicted = float(model.predict(row)[0])
        label, color = _aqi_category(predicted)

        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1.5rem; border-radius:0.5rem; color:white;">
                <strong style="font-size:1.3rem;">Predicted US AQI: {predicted:.0f}</strong><br/>
                Category: {label}
            </div>
            """,
            unsafe_allow_html=True,
        )
