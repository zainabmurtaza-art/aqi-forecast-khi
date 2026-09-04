# -*- coding: utf-8 -*-
"""Dashboard panels: current readings, alert banner, forecast chart, trend
chart, SHAP panel."""

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

import config
from app import copilot, health_guidance, theme
from app.data_loader import algorithm_label
from training_pipeline.build_dataset import FEATURE_COLUMNS
from training_pipeline.models import PersistenceRegressor

# Reader-facing names for the model's feature columns, used anywhere a feature
# is named in prose or on a chart axis. Raw column names (pm2_5, aqi_lag_24h)
# are precise but mean nothing to someone who hasn't read the pipeline.
SHAP_PLAIN_NAMES = {
    "pm10": "PM10 dust level",
    "pm2_5": "PM2.5 fine particles",
    "carbon_monoxide": "Carbon monoxide",
    "nitrogen_dioxide": "Nitrogen dioxide",
    "sulphur_dioxide": "Sulphur dioxide",
    "ozone": "Ozone",
    "us_aqi": "Current AQI",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "surface_pressure": "Air pressure",
    "wind_speed_10m": "Wind speed",
    "hour": "Hour of day",
    "day": "Day of month",
    "month": "Month of year",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend or weekday",
    "aqi_change_rate": "AQI change in the last hour",
    "aqi_roll_mean_3h": "Average AQI over 3 hours",
    "aqi_roll_mean_24h": "Average AQI over 24 hours",
    "aqi_lag_24h": "AQI this time yesterday",
    "aqi_lag_48h": "AQI this time two days ago",
}


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


def _to_local(series: pd.Series) -> pd.Series:
    """UTC event times -> the project's local timezone. Everything is stored
    and modelled in UTC, but a reader in Pakistan reads 'today' in PKT, so
    every displayed timestamp is converted at the edge rather than earlier."""
    times = pd.to_datetime(series, utc=True)
    try:
        return times.dt.tz_convert(config.TIMEZONE)
    except Exception:
        # An unrecognised tz name shouldn't blank the whole dashboard.
        return times


# Units for the model's raw feature columns, shown beside each value in the
# Current Readings panel. Keys match FEATURE_COLUMNS exactly.
FEATURE_UNITS = {
    "pm10": "µg/m³",
    "pm2_5": "µg/m³",
    "carbon_monoxide": "µg/m³",
    "nitrogen_dioxide": "µg/m³",
    "sulphur_dioxide": "µg/m³",
    "ozone": "µg/m³",
    "us_aqi": "AQI",
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "aqi_change_rate": "AQI/h",
    "aqi_roll_mean_3h": "AQI",
    "aqi_roll_mean_24h": "AQI",
    "aqi_lag_24h": "AQI",
    "aqi_lag_48h": "AQI",
}

# Human-readable names, grouped into the cards the panel renders. Between them
# these groups cover every entry in FEATURE_COLUMNS, so "all feature readings"
# on the panel means literally all of the model's inputs.
READING_GROUPS = [
    ("Pollutants", [
        ("pm2_5", "PM2.5"),
        ("pm10", "PM10"),
        ("carbon_monoxide", "Carbon monoxide"),
        ("nitrogen_dioxide", "Nitrogen dioxide"),
        ("sulphur_dioxide", "Sulphur dioxide"),
        ("ozone", "Ozone"),
    ]),
    ("Weather", [
        ("temperature_2m", "Temperature"),
        ("relative_humidity_2m", "Relative humidity"),
        ("surface_pressure", "Surface pressure"),
        ("wind_speed_10m", "Wind speed"),
    ]),
    ("Recent AQI history (derived)", [
        ("aqi_change_rate", "Change vs. last hour"),
        ("aqi_roll_mean_3h", "3-hour average"),
        ("aqi_roll_mean_24h", "24-hour average"),
        ("aqi_lag_24h", "24 hours ago"),
        ("aqi_lag_48h", "48 hours ago"),
    ]),
]

# Time features are formatted rather than printed raw (a "day_of_week" of 2
# means nothing to a reader), so they get their own renderer below.
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


def _format_value(column: str, value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if column in ("carbon_monoxide", "surface_pressure"):
        return f"{float(value):,.0f}"
    if column == "aqi_change_rate":
        return f"{float(value):+.1f}"
    if column in ("us_aqi", "relative_humidity_2m"):
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


def render_aqi_key():
    """Static US AQI color/range legend, shown instead of the forecast charts
    when the sidebar is set to the 'AQI Key' view."""
    rows = "".join(
        f'<div style="background-color:{color}; padding:0.85rem 1.1rem; '
        f'border-radius:5px; color:white; margin-bottom:0.5rem; '
        f'display:flex; justify-content:space-between; align-items:baseline;">'
        f'<strong style="font-size:1.05rem;">{label}</strong>'
        f'<span style="opacity:0.92;">{lo}&ndash;{hi}</span>'
        f"</div>"
        for lo, hi, label, color in AQI_KEY_RANGES
    )
    st.markdown(theme.card("US AQI categories", rows), unsafe_allow_html=True)


def render_health_guidelines(current_aqi=None, city_label: str = ""):
    """Health advisories for the current reading, then the full band-by-band table."""
    if current_aqi is not None:
        category, color = _aqi_category(current_aqi)
        st.markdown(
            f"""
            <div class="aqi-hero" style="background-color:{color};">
                <div>
                    <div class="aqi-hero-label">Current status &mdash; {city_label}</div>
                    <div class="aqi-hero-value">{current_aqi:.0f}</div>
                </div>
                <div>
                    <div class="aqi-hero-label">Category</div>
                    <div class="aqi-hero-category">{category}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="aqi-caption">{health_guidance.headline_for(category)}</div>',
            unsafe_allow_html=True,
        )

        advice = health_guidance.guidance_for(category)
        columns = st.columns(2)
        for i, group in enumerate(health_guidance.GROUPS):
            with columns[i % 2]:
                st.markdown(
                    theme.card(group, f"<div>{advice[group]}</div>"),
                    unsafe_allow_html=True,
                )

    st.subheader("Guidance at every AQI level")
    st.markdown(
        '<div class="aqi-caption">What each band means for different people, so you can '
        "read ahead to the forecast as well as today.</div>",
        unsafe_allow_html=True,
    )

    for lo, hi, label, color in AQI_KEY_RANGES:
        advice = health_guidance.guidance_for(label)
        rows = "".join(
            f'<div style="margin-bottom:0.5rem;"><strong>{group}:</strong> {advice[group]}</div>'
            for group in health_guidance.GROUPS
        )
        st.markdown(
            f"""
            <div class="aqi-card" style="border-left:6px solid {color};">
                <div class="aqi-card-title" style="color:{color};">
                    {label} &nbsp;&middot;&nbsp; AQI {lo}&ndash;{hi}
                </div>
                {rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="aqi-caption">{health_guidance.DISCLAIMER}</div>',
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
        <div class="aqi-hero" style="background-color:{color};">
            <div>
                <div class="aqi-hero-label">Current US AQI</div>
                <div class="aqi-hero-value">{current_aqi:.0f}</div>
            </div>
            <div>
                <div class="aqi-hero-label">Status</div>
                <div class="aqi-hero-category">{label}</div>
            </div>
            <div class="aqi-hero-meta">
                <div class="aqi-hero-label">Worst forecast, next 3 days</div>
                <div class="aqi-hero-category">{worst_forecast:.0f}</div>
            </div>
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
            marker_line=dict(color=theme.NAVY, width=1),
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            textfont=dict(family=theme.SERIF_STACK, color=theme.NAVY, size=14),
        )
    )
    fig.update_layout(
        title="3-day AQI forecast",
        yaxis_title="Predicted US AQI",
        yaxis_range=y_range,
    )
    st.plotly_chart(theme.style_figure(fig), use_container_width=True)


def render_trend_chart(actual_df: pd.DataFrame):
    local = _to_local(actual_df["event_time"])
    fig = go.Figure(
        go.Scatter(
            x=local,
            y=actual_df["us_aqi"],
            mode="lines",
            line=dict(color=theme.MAGENTA, width=2),
            fill="tozeroy",
            fillcolor="rgba(123, 45, 94, 0.10)",
        )
    )
    fig.update_layout(
        title="Recent AQI trend",
        yaxis_title="US AQI",
        xaxis_title=f"Local time ({config.TIMEZONE})",
    )
    st.plotly_chart(theme.style_figure(fig), use_container_width=True)


def render_current_readings(actual_df: pd.DataFrame, city_label: str):
    """Every model feature as observed right now, next to the current AQI.

    Reads the newest row the feature store holds for this city (written by the
    hourly pipeline) and lays its features out in grouped cards, plus a
    summary of how AQI has moved across the current local day. Falls back
    gracefully — and says so — when the newest stored row predates today,
    which happens if the hourly pipeline is behind.
    """
    if actual_df.empty:
        st.warning("No stored readings available for this city yet.")
        return

    df = actual_df.sort_values("event_time").reset_index(drop=True)
    local_times = _to_local(df["event_time"])
    latest = df.iloc[-1]
    latest_local = local_times.iloc[-1]

    today = pd.Timestamp.now(tz=latest_local.tz).date() if latest_local.tz else latest_local.date()
    today_mask = local_times.dt.date == today
    today_df = df[today_mask]
    is_stale = today_df.empty

    if is_stale:
        # Nothing from today yet: report the latest day we do have, clearly
        # labelled, rather than showing an empty panel.
        fallback_day = latest_local.date()
        today_df = df[local_times.dt.date == fallback_day]
        day_label = fallback_day.strftime("%A, %d %B %Y")
    else:
        day_label = today.strftime("%A, %d %B %Y")

    current_aqi = float(latest["us_aqi"])
    category, color = _aqi_category(current_aqi)

    st.markdown(
        f"""
        <div class="aqi-hero" style="background-color:{color};">
            <div>
                <div class="aqi-hero-label">Current US AQI &mdash; {city_label}</div>
                <div class="aqi-hero-value">{current_aqi:.0f}</div>
            </div>
            <div>
                <div class="aqi-hero-label">Category</div>
                <div class="aqi-hero-category">{category}</div>
            </div>
            <div class="aqi-hero-meta">
                <div class="aqi-hero-label">Observation</div>
                <div class="aqi-hero-category">{latest_local.strftime('%H:%M')}</div>
                <div>{latest_local.strftime('%d %b %Y')} ({config.TIMEZONE})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_stale:
        st.warning(
            f"The feature store has no readings for today yet — showing the most "
            f"recent stored hour ({latest_local.strftime('%d %b %Y, %H:%M')}). "
            "The hourly pipeline may be running behind."
        )

    # --- the day's AQI envelope -------------------------------------------
    day_aqi = today_df["us_aqi"].dropna()
    if not day_aqi.empty:
        summary = theme.reading_grid([
            theme.reading("Readings so far", f"{len(day_aqi)}", "hours"),
            theme.reading("Lowest", _format_value("us_aqi", day_aqi.min()), "AQI"),
            theme.reading("Average", _format_value("us_aqi", day_aqi.mean()), "AQI"),
            theme.reading("Highest", _format_value("us_aqi", day_aqi.max()), "AQI"),
        ])
        st.markdown(
            theme.card(f"Air quality on {day_label}", summary), unsafe_allow_html=True
        )

    # --- every model feature, grouped -------------------------------------
    for title, columns in READING_GROUPS:
        rows = [
            theme.reading(label, _format_value(col, latest.get(col)), FEATURE_UNITS.get(col, ""))
            for col, label in columns
            if col in latest.index
        ]
        if rows:
            st.markdown(theme.card(title, theme.reading_grid(rows)), unsafe_allow_html=True)

    # --- time features, formatted for a human ------------------------------
    time_rows = []
    if "hour" in latest.index and not pd.isna(latest["hour"]):
        time_rows.append(theme.reading("Hour of day", f"{int(latest['hour']):02d}:00", "UTC"))
    if "day_of_week" in latest.index and not pd.isna(latest["day_of_week"]):
        time_rows.append(
            theme.reading("Day of week", _WEEKDAY_NAMES[int(latest["day_of_week"]) % 7])
        )
    if "day" in latest.index and not pd.isna(latest["day"]):
        time_rows.append(theme.reading("Day of month", f"{int(latest['day'])}"))
    if "month" in latest.index and not pd.isna(latest["month"]):
        time_rows.append(theme.reading("Month", _MONTH_NAMES[(int(latest["month"]) - 1) % 12]))
    if "is_weekend" in latest.index and not pd.isna(latest["is_weekend"]):
        time_rows.append(
            theme.reading("Weekend", "Yes" if int(latest["is_weekend"]) else "No")
        )
    if time_rows:
        st.markdown(
            theme.card("Time features (as the model sees them)", theme.reading_grid(time_rows)),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="aqi-caption">Readings come from the Hopsworks feature store, '
        "written hourly by the feature pipeline. These are the exact inputs the "
        "forecast models consume.</div>",
        unsafe_allow_html=True,
    )


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


def _metric_quality(r2) -> tuple:
    """(verdict, plain-language meaning) for an R2 value.

    R2 below 0 is not a rounding artefact — it means the model does worse on
    held-out data than always guessing the average would. That is worth saying
    outright rather than presenting a negative number without comment.
    """
    if r2 is None:
        return "unknown", "No score was recorded for this model."
    if r2 < 0:
        return "worse than guessing the average", (
            "A negative R² means this model's held-out predictions were less accurate "
            "than simply always predicting the average AQI. Treat this horizon's "
            "forecast with caution."
        )
    if r2 < 0.3:
        return "weak", "The model explains only a small share of the variation in AQI."
    if r2 < 0.6:
        return "moderate", "The model captures a fair share of the variation in AQI."
    if r2 < 0.85:
        return "good", "The model explains most of the variation in AQI."
    return "strong", "The model explains almost all of the variation in held-out AQI."


def render_model_metrics(models: dict, metrics_df: pd.DataFrame, city_label: str):
    """Which algorithm is deployed per horizon, and how well it scored.

    Metrics come from the Model Registry (recorded at training time on a
    chronological hold-out), so they describe the model actually in use.
    """
    st.subheader("Which model is making these forecasts")

    if metrics_df is None or metrics_df.empty:
        st.warning("No registered model metrics found for this city yet.")
        return

    for _, row in metrics_df.sort_values("horizon_days").iterrows():
        horizon_hours = int(row["horizon_days"]) * 24
        model = models.get(horizon_hours)
        algorithm = algorithm_label(model) if model is not None else "unknown"
        verdict, meaning = _metric_quality(row["r2"])

        def fmt(value, digits=3):
            return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"

        cells = theme.reading_grid([
            theme.reading("Algorithm", algorithm),
            theme.reading("RMSE", fmt(row["rmse"], 2), "AQI"),
            theme.reading("MAE", fmt(row["mae"], 2), "AQI"),
            theme.reading("R²", fmt(row["r2"])),
        ])
        st.markdown(
            theme.card(f"+{int(row['horizon_days'])}-day forecast &mdash; {city_label}", cells),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="aqi-caption"><strong>Fit: {verdict}.</strong> {meaning}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("What do RMSE, MAE and R² mean?"):
        st.markdown(
            "- **MAE (mean absolute error)** — the average size of the miss, in AQI points. "
            "An MAE of 5 means the forecast is typically about 5 AQI points off.\n"
            "- **RMSE (root mean squared error)** — the same idea, but large misses count "
            "for much more. RMSE well above MAE means the model is occasionally badly wrong.\n"
            "- **R² (coefficient of determination)** — the share of the variation in AQI the "
            "model explains. 1.0 is perfect, 0 is no better than always guessing the "
            "average, and below 0 is *worse* than that.\n\n"
            "All three are measured on a chronological hold-out — the most recent slice of "
            "history, which the model never saw during training."
        )

    st.markdown(
        '<div class="aqi-caption">Each city and horizon is trained separately, and the '
        "algorithm with the lowest RMSE is the one deployed — which is why different "
        "horizons can use different models.</div>",
        unsafe_allow_html=True,
    )


def _shap_sentence(feature: str, value, contribution: float) -> str:
    """One plain-language line: what this feature was, and which way it pushed."""
    direction = "raised" if contribution >= 0 else "lowered"
    label = SHAP_PLAIN_NAMES.get(feature, feature)
    unit = FEATURE_UNITS.get(feature, "")
    shown = _format_value(feature, value)

    # Don't append "AQI" to a value whose label already says AQI ("Current AQI
    # was 62 AQI"), and drop the unit entirely for a missing value.
    if shown == "—" or (unit == "AQI" and "AQI" in label):
        unit = ""
    value_part = f"{shown}{(' ' + unit) if unit else ''}"

    magnitude = f"{abs(contribution):.1f}"
    return (
        f"<strong>{label}</strong> was {value_part} &mdash; this {direction} the forecast "
        f"by <strong>{magnitude}</strong> AQI point{'' if magnitude == '1.0' else 's'}."
    )


def render_shap_panel(
    model, feature_row: pd.DataFrame, horizon_days: int, background_df: pd.DataFrame
):
    st.subheader(f"Why the model forecast this for +{horizon_days} day")
    st.markdown(
        '<div class="aqi-caption">The model starts from a baseline (its average '
        "prediction) and then adjusts up or down for each thing it measured today. "
        "Below is what moved this particular forecast, and by how much.</div>",
        unsafe_allow_html=True,
    )

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
    contributions = pd.Series(shap_values[0], index=FEATURE_COLUMNS)
    values = X.iloc[0]

    # --- the plain-language version, first ---------------------------------
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    pushed_up = [f for f in ranked.index if ranked[f] > 0][:3]
    pushed_down = [f for f in ranked.index if ranked[f] < 0][:3]

    col_up, col_down = st.columns(2)
    with col_up:
        rows = "".join(
            f'<div style="margin-bottom:0.55rem;">{_shap_sentence(f, values[f], ranked[f])}</div>'
            for f in pushed_up
        ) or '<div class="aqi-caption">Nothing pushed this forecast up.</div>'
        st.markdown(theme.card("What pushed the forecast UP", rows), unsafe_allow_html=True)
    with col_down:
        rows = "".join(
            f'<div style="margin-bottom:0.55rem;">{_shap_sentence(f, values[f], ranked[f])}</div>'
            for f in pushed_down
        ) or '<div class="aqi-caption">Nothing pushed this forecast down.</div>'
        st.markdown(theme.card("What pushed the forecast DOWN", rows), unsafe_allow_html=True)

    net = contributions.sum()
    st.markdown(
        f'<div class="aqi-caption">Together these adjustments moved the forecast '
        f'<strong>{"up" if net >= 0 else "down"} {abs(net):.1f} AQI points</strong> '
        "from the model's baseline prediction.</div>",
        unsafe_allow_html=True,
    )

    # --- then the full chart, for anyone who wants it ----------------------
    ordered = contributions.sort_values()
    fig = go.Figure(
        go.Bar(
            x=ordered.values,
            y=[SHAP_PLAIN_NAMES.get(f, f) for f in ordered.index],
            orientation="h",
            marker_color=[
                theme.MAGENTA if v >= 0 else theme.NAVY_SOFT for v in ordered.values
            ],
            hovertemplate="%{y}<br>%{x:+.2f} AQI points<extra></extra>",
        )
    )
    fig.update_layout(
        title="Every feature's effect on this forecast",
        xaxis_title="← lowers the forecast    |    raises the forecast →   (AQI points)",
    )
    st.plotly_chart(theme.style_figure(fig, height=560), use_container_width=True)

    with st.expander("How do I read this chart?"):
        st.markdown(
            "Each bar is one thing the model looked at, and how far it moved **this "
            "particular forecast** away from the model's usual prediction.\n\n"
            "- **Bars to the right** (magenta) pushed the predicted AQI **up** — worse air.\n"
            "- **Bars to the left** (navy) pushed it **down** — cleaner air.\n"
            "- **Bar length** is how much: the units are AQI points, so a bar at +4 "
            "added 4 points to the forecast.\n"
            "- **Features near the top and bottom** mattered most; ones bunched near zero "
            "barely affected this forecast at all.\n\n"
            "These are SHAP values — a method that fairly splits a prediction's total "
            "movement between the features that caused it, so the bars add up to the "
            "gap between this forecast and the model's baseline."
        )

    with st.expander("What do these feature names mean?"):
        for col in FEATURE_COLUMNS:
            label = SHAP_PLAIN_NAMES.get(col, col)
            st.markdown(f"- **{label}** (`{col}`) — {FEATURE_DESCRIPTIONS.get(col, '')}")


def render_copilot(get_readings, get_forecast, get_metrics, get_models, default_city):
    """Chat panel over the grounded copilot in app/copilot.py."""
    st.markdown(
        '<div class="aqi-caption">Ask about current air quality, forecasts, pollutants, '
        "or the models. Every answer is read from this dashboard&rsquo;s own stored data, "
        "so the numbers are real ones &mdash; not generated text.</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.setdefault("copilot_history", [])

    with st.expander("Try an example question"):
        for i, example in enumerate(copilot.EXAMPLE_QUESTIONS):
            if st.button(example, key=f"copilot_example_{i}"):
                st.session_state["copilot_pending"] = example
                st.rerun()

    typed = st.chat_input("Ask about air quality…")
    question = st.session_state.pop("copilot_pending", None) or typed

    if question:
        reply = copilot.answer(
            question,
            get_readings=get_readings,
            get_forecast=get_forecast,
            get_metrics=get_metrics,
            get_models=get_models,
            default_city=default_city,
        )
        history.append({"question": question, "answer": reply})

    if not history:
        st.info(
            "No questions yet — try one of the examples above, or type your own below."
        )
        return

    # Newest exchange first, so the latest answer is visible without scrolling.
    for exchange in reversed(history):
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            st.markdown(exchange["answer"])

    if st.button("Clear conversation"):
        st.session_state["copilot_history"] = []
        st.rerun()


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
            <div class="aqi-hero" style="background-color:{color};">
                <div>
                    <div class="aqi-hero-label">Predicted US AQI (+{horizon_choice} day)</div>
                    <div class="aqi-hero-value">{predicted:.0f}</div>
                </div>
                <div>
                    <div class="aqi-hero-label">Category</div>
                    <div class="aqi-hero-category">{label}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
