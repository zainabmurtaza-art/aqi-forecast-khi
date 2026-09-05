# -*- coding: utf-8 -*-
"""
AQI forecast dashboard. Run locally with:
    streamlit run app/streamlit_app.py

Deployed on Streamlit Community Cloud pointing at this same file, with
HOPSWORKS_API_KEY set under the app's own Settings -> Secrets.

Each view is a function in VIEW_FUNCTIONS, rendered into a single slot so the
body can be cleared on navigation — see the note above `body` at the bottom.
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
from app.data_loader import (
    get_horizon_predictions,
    load_model_metrics,
    load_models,
    load_recent_actual_features,
)
from app.ui_components import (
    render_alert_banner,
    render_copilot,
    render_current_readings,
    render_forecast_chart,
    render_health_guidelines,
    render_manual_prediction_form,
    render_model_metrics,
    render_shap_panel,
    render_trend_chart,
)

st.set_page_config(page_title="AQI Forecast — Pakistan", layout="wide")
theme.apply_theme()

VIEWS = [
    "Current Readings",
    "Forecast",
    "Model & Explanation",
    "Manual Prediction",
    "Health & AQI Key",
    "AQI Copilot",
]

view = st.sidebar.radio("View", VIEWS)

city_keys = list(config.CITIES.keys())
default_city = config.CITY_NAME if config.CITY_NAME in config.CITIES else city_keys[0]

selected_city = st.sidebar.selectbox(
    "City",
    options=city_keys,
    format_func=lambda c: config.CITIES[c]["label"],
    index=city_keys.index(default_city),
)
city_label = config.CITIES[selected_city]["label"]


def _no_models_error():
    st.error(
        f"No trained models found in Hopsworks yet for {city_label}. Run "
        "`python -m feature_pipeline.backfill_pipeline` and "
        "`python -m training_pipeline.train` first."
    )
    st.stop()


def _no_features_error():
    st.error(
        f"No feature data found in Hopsworks yet for {city_label}. Run "
        "`python -m feature_pipeline.backfill_pipeline` first."
    )
    st.stop()


def view_current_readings():
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
        _no_features_error()

    render_current_readings(readings_df, city_label)


def view_health_and_key():
    st.title("Health Guidance & AQI Key")
    st.markdown(
        '<div class="aqi-caption">What today&rsquo;s air quality means for different '
        "people, and the colour bands every number on this dashboard is measured "
        "against.</div>",
        unsafe_allow_html=True,
    )
    # Current AQI is a nicety here, not a requirement — the band-by-band guidance
    # is still worth showing if the feature store is down.
    current_aqi = None
    try:
        readings_df = load_recent_actual_features(selected_city)
        if not readings_df.empty:
            current_aqi = float(readings_df.sort_values("event_time")["us_aqi"].iloc[-1])
    except Exception:
        st.warning(
            "Couldn't load the current reading, so the guidance below is shown "
            "without today's status."
        )

    # The band-by-band cards inside render_health_guidelines already carry each
    # category's colour and range, so the standalone colour key that used to sit
    # below them was showing the same information twice.
    render_health_guidelines(current_aqi, city_label)


def view_copilot():
    st.title("AQI Copilot")
    render_copilot(
        get_readings=load_recent_actual_features,
        get_forecast=lambda city: get_horizon_predictions(load_models(city), city),
        get_metrics=load_model_metrics,
        get_models=load_models,
        default_city=selected_city,
    )


def view_model_and_explanation():
    st.title(f"Model & Explanation — {city_label}")
    st.markdown(
        '<div class="aqi-caption">Which model is producing each forecast, how '
        "accurate it has been, and what drove today&rsquo;s numbers.</div>",
        unsafe_allow_html=True,
    )
    with st.spinner(f"Loading models for {city_label}..."):
        try:
            models = load_models(selected_city)
            metrics_df = load_model_metrics(selected_city)
        except Exception:
            _no_models_error()

    render_model_metrics(models, metrics_df, city_label)

    st.divider()

    actual_df = load_recent_actual_features(selected_city)
    try:
        predictions = get_horizon_predictions(models, selected_city)
    except Exception:
        st.warning(
            "Couldn't reach Open-Meteo for the live forecast, so there's no "
            "individual prediction to explain right now. The metrics above are "
            "unaffected. Please reload in a minute or two."
        )
        st.stop()

    horizon_choice = st.selectbox(
        "Explain which forecast?",
        options=predictions["horizon_days"].tolist(),
        format_func=lambda d: f"+{d} day",
    )
    selected = predictions[predictions["horizon_days"] == horizon_choice].iloc[0]
    render_shap_panel(
        models[horizon_choice * 24], selected["feature_row"], horizon_choice, actual_df
    )


def view_manual_prediction():
    st.title(f"Manual AQI Prediction — {city_label}")
    with st.spinner(f"Loading models for {city_label}..."):
        try:
            models = load_models(selected_city)
        except Exception:
            _no_models_error()
    render_manual_prediction_form(models)


def view_forecast():
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
            _no_models_error()

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
        _no_features_error()

    render_alert_banner(float(actual_df["us_aqi"].iloc[-1]), predictions)

    col1, col2 = st.columns(2)
    with col1:
        render_forecast_chart(predictions)
    with col2:
        render_trend_chart(actual_df)

    st.divider()

    st.markdown(
        '<div class="aqi-caption">Want to know why these numbers came out the way they '
        "did, or how accurate the model has been? See <strong>Model &amp; "
        "Explanation</strong>. For what today&rsquo;s air quality means for you, see "
        "<strong>Health &amp; AQI Key</strong>.</div>",
        unsafe_allow_html=True,
    )


VIEW_FUNCTIONS = {
    "Current Readings": view_current_readings,
    "Forecast": view_forecast,
    "Model & Explanation": view_model_and_explanation,
    "Manual Prediction": view_manual_prediction,
    "Health & AQI Key": view_health_and_key,
    "AQI Copilot": view_copilot,
}

# Navigating to a page that loads slowly used to leave the *previous* page's
# body on screen underneath the new page's title and spinner — the Forecast
# banner showing through "Loading models for Karachi...".
#
# Streamlit reuses a container across runs and replaces its children by index,
# pruning any surplus only once the script finishes. So a new view that has
# written two elements so far sits on top of the old view's remaining ten.
# Clearing the slot in the same run does not help: re-entering the container
# immediately re-establishes it at the same path, inheriting that child list.
#
# The clear has to be the last thing a run does, so the frontend prunes before
# the slow work starts. Hence two passes: this run empties the slot and stops;
# the rerun draws the new page from a blank slate.
body = st.empty()
selection = (view, selected_city)

if "_rendered_selection" not in st.session_state:
    # First load — nothing on screen to clear, so don't spend a rerun on it.
    st.session_state["_rendered_selection"] = selection
elif st.session_state["_rendered_selection"] != selection:
    st.session_state["_rendered_selection"] = selection
    body.empty()
    st.rerun()

with body.container():
    VIEW_FUNCTIONS[view]()
