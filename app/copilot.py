# -*- coding: utf-8 -*-
"""
AQI Copilot: a grounded question-answering layer over this dashboard's own data.

Deliberately NOT an LLM. Every answer here is assembled from values actually
read out of the feature store or the forecast models, so it cannot invent an
AQI reading, and it needs no API key, no per-question cost, and no new secret
in the Streamlit deployment. When a question falls outside what the data can
answer, it says so instead of guessing.

The trade-off is that phrasing is templated rather than free-form. If free-form
conversation is wanted later, the grounding helpers here (`resolve_city`,
`classify`, and the `_answer_*` functions) are the right things to hand to a
model as tools — the retrieval stays honest either way.
"""

import re

import pandas as pd

import config
from app import health_guidance

# --- intents ---------------------------------------------------------------

CURRENT = "current"
FORECAST = "forecast"
POLLUTANT = "pollutant"
COMPARE = "compare"
ACTIVITY = "activity"
MODEL = "model"
HELP = "help"
UNKNOWN = "unknown"

EXAMPLE_QUESTIONS = [
    "What is the AQI in Karachi today?",
    "Can I go for a jog in Lahore today?",
    "What is the forecast for Islamabad?",
    "How much PM2.5 is in the air in Multan?",
    "Compare Karachi and Lahore",
    "Which model is used for Karachi?",
]

# Words that signal each intent. Ordered by specificity when scanning.
_ACTIVITY_WORDS = (
    "jog", "jogging", "run", "running", "walk", "walking", "exercise", "workout",
    "cycle", "cycling", "outdoor", "outside", "play", "sport", "gym", "safe",
    "should i", "can i", "is it ok", "is it okay",
)
_FORECAST_WORDS = ("forecast", "tomorrow", "next few days", "next 3 days", "coming days",
                   "predict", "prediction", "will it", "later this week", "3 day", "three day")
_COMPARE_WORDS = ("compare", "versus", " vs ", "vs.", "better than", "worse than",
                  "cleaner", "dirtier", "which city")
_MODEL_WORDS = ("model", "algorithm", "accuracy", "rmse", "mae", "r2", "r²",
                "how does it work", "shap", "trained")
_HELP_WORDS = ("help", "what can you do", "what can i ask", "how do i use")

_POLLUTANT_ALIASES = {
    "pm2_5": ("pm2.5", "pm2_5", "pm25", "fine particle", "fine particulate"),
    "pm10": ("pm10", "pm 10", "coarse particle", "dust"),
    "ozone": ("ozone", "o3"),
    "nitrogen_dioxide": ("nitrogen dioxide", "no2", "no₂"),
    "sulphur_dioxide": ("sulphur dioxide", "sulfur dioxide", "so2", "so₂"),
    "carbon_monoxide": ("carbon monoxide", "co level", " co ", "carbon-monoxide"),
    "temperature_2m": ("temperature", "how hot", "how warm"),
    "relative_humidity_2m": ("humidity", "humid"),
    "wind_speed_10m": ("wind", "windy"),
    "surface_pressure": ("pressure",),
}

_UNITS = {
    "pm2_5": "µg/m³", "pm10": "µg/m³", "ozone": "µg/m³", "nitrogen_dioxide": "µg/m³",
    "sulphur_dioxide": "µg/m³", "carbon_monoxide": "µg/m³", "temperature_2m": "°C",
    "relative_humidity_2m": "%", "wind_speed_10m": "km/h", "surface_pressure": "hPa",
}

_POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "ozone": "ozone",
    "nitrogen_dioxide": "nitrogen dioxide", "sulphur_dioxide": "sulphur dioxide",
    "carbon_monoxide": "carbon monoxide", "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity", "wind_speed_10m": "wind speed",
    "surface_pressure": "surface pressure",
}


# --- grounding helpers -----------------------------------------------------


def resolve_cities(question: str) -> list:
    """Every configured city named in the question, in the order mentioned."""
    text = question.lower()
    hits = []
    for key, meta in config.CITIES.items():
        for token in {key, meta["label"].lower()}:
            idx = text.find(token)
            if idx != -1:
                hits.append((idx, key))
                break
    return [key for _, key in sorted(hits)]


def resolve_pollutant(question: str):
    text = f" {question.lower()} "
    for column, aliases in _POLLUTANT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return column
    return None


def classify(question: str) -> str:
    text = f" {question.lower()} "
    if any(w in text for w in _HELP_WORDS):
        return HELP
    if any(w in text for w in _MODEL_WORDS):
        return MODEL
    if any(w in text for w in _COMPARE_WORDS) or len(resolve_cities(question)) > 1:
        return COMPARE
    if any(w in text for w in _ACTIVITY_WORDS):
        return ACTIVITY
    if any(w in text for w in _FORECAST_WORDS):
        return FORECAST
    if resolve_pollutant(question):
        return POLLUTANT
    if "aqi" in text or "air quality" in text or "pollution" in text:
        return CURRENT
    return UNKNOWN


def _latest(readings: pd.DataFrame):
    """(row, timestamp) for the newest stored reading, or (None, None)."""
    if readings is None or readings.empty:
        return None, None
    df = readings.sort_values("event_time")
    row = df.iloc[-1]
    return row, pd.to_datetime(row["event_time"], utc=True)


def _when(timestamp) -> str:
    if timestamp is None:
        return ""
    local = timestamp.tz_convert(config.TIMEZONE)
    age_h = (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds() / 3600
    stamp = local.strftime("%d %b, %H:%M")
    if age_h > 3:
        return f" (as of {stamp} local — {age_h:.0f} hours ago, the most recent reading stored)"
    return f" (as of {stamp} local)"


def _category(aqi: float) -> str:
    # Imported lazily to avoid a circular import with ui_components.
    from app.ui_components import _aqi_category

    return _aqi_category(aqi)[0]


# --- answers ---------------------------------------------------------------


def _answer_current(city, get_readings):
    label = config.CITIES[city]["label"]
    row, ts = _latest(get_readings(city))
    if row is None:
        return f"I don't have any stored readings for {label} yet."

    aqi = float(row["us_aqi"])
    return (
        f"The air quality in {label} is **{_category(aqi).lower()}**. "
        f"The US AQI is **{aqi:.0f}**{_when(ts)}. "
        f"{health_guidance.headline_for(_category(aqi))}"
    )


def _answer_pollutant(city, column, get_readings):
    label = config.CITIES[city]["label"]
    row, ts = _latest(get_readings(city))
    if row is None:
        return f"I don't have any stored readings for {label} yet."
    if column not in row.index or pd.isna(row[column]):
        return f"I don't have a {_POLLUTANT_LABELS.get(column, column)} reading for {label}."

    value = float(row[column])
    unit = _UNITS.get(column, "")
    name = _POLLUTANT_LABELS.get(column, column)
    extra = ""
    if column in ("pm2_5", "pm10"):
        extra = (f" For context, the overall AQI there is {float(row['us_aqi']):.0f} "
                 f"({_category(float(row['us_aqi'])).lower()}).")
    return f"{name.capitalize()} in {label} is **{value:.1f} {unit}**{_when(ts)}.{extra}"


def _answer_forecast(city, get_readings, get_forecast):
    label = config.CITIES[city]["label"]
    try:
        forecast = get_forecast(city)
    except Exception:
        return (f"I couldn't load the forecast models for {label} just now. "
                "The Forecast page will show the same information once they're available.")

    if forecast is None or forecast.empty:
        return f"I don't have a forecast for {label} right now."

    parts = []
    for _, r in forecast.sort_values("horizon_days").iterrows():
        value = float(r["predicted_us_aqi"])
        parts.append(f"**+{int(r['horizon_days'])} day:** {value:.0f} ({_category(value).lower()})")

    worst = forecast["predicted_us_aqi"].max()
    return (
        f"Forecast for {label} — " + ", ".join(parts) + ". "
        f"The worst of the three is {worst:.0f}, which is {_category(worst).lower()}. "
        f"{health_guidance.headline_for(_category(worst))}"
    )


def _answer_activity(city, question, get_readings):
    label = config.CITIES[city]["label"]
    row, ts = _latest(get_readings(city))
    if row is None:
        return f"I don't have any stored readings for {label}, so I can't advise on that."

    aqi = float(row["us_aqi"])
    verdict, explanation = health_guidance.is_safe_for_outdoor_exercise(aqi)

    activity = "outdoor activity"
    for word in ("jog", "run", "walk", "cycle", "exercise", "workout", "play"):
        if word in question.lower():
            activity = f"{word}ging" if word == "jog" else f"{word}ning" if word == "run" else word
            break

    return (
        f"**{verdict.capitalize()}** — for {activity} in {label} right now. "
        f"The AQI is **{aqi:.0f}**, which is {_category(aqi).lower()}{_when(ts)}. "
        f"{explanation}"
        f"\n\n_{health_guidance.DISCLAIMER}_"
    )


def _answer_compare(cities, get_readings):
    rows = []
    for city in cities[:4]:
        row, ts = _latest(get_readings(city))
        if row is not None:
            rows.append((config.CITIES[city]["label"], float(row["us_aqi"]), ts))

    if len(rows) < 2:
        return "I need readings for at least two cities to compare, and I don't have them."

    rows.sort(key=lambda r: r[1])
    listing = ", ".join(f"**{name}** {aqi:.0f} ({_category(aqi).lower()})" for name, aqi, _ in rows)
    best, worst = rows[0], rows[-1]
    return (
        f"{listing}.\n\n{best[0]} has the cleanest air of those right now at {best[1]:.0f}; "
        f"{worst[0]} the worst at {worst[1]:.0f}, a difference of {worst[1] - best[1]:.0f} "
        f"AQI points.{_when(best[2])}"
    )


def _answer_model(city, get_metrics, get_models):
    label = config.CITIES[city]["label"]
    try:
        metrics = get_metrics(city)
        models = get_models(city)
    except Exception:
        return f"I couldn't reach the model registry for {label} just now."

    if metrics is None or metrics.empty:
        return f"No trained models are registered for {label} yet."

    from app.data_loader import algorithm_label

    lines = []
    for _, r in metrics.sort_values("horizon_days").iterrows():
        model = models.get(int(r["horizon_days"]) * 24)
        name = algorithm_label(model) if model is not None else "unknown"
        rmse = "—" if pd.isna(r["rmse"]) else f"{r['rmse']:.2f}"
        r2 = "—" if pd.isna(r["r2"]) else f"{r['r2']:.2f}"
        lines.append(f"**+{int(r['horizon_days'])} day:** {name} (RMSE {rmse}, R² {r2})")

    return (
        f"Models deployed for {label} — " + "; ".join(lines) + ". "
        "Each horizon is trained separately and the algorithm with the lowest RMSE wins, "
        "so different horizons can use different models. The Model & Explanation page "
        "breaks down what drove any individual forecast."
    )


def _answer_help() -> str:
    return (
        "I answer from this dashboard's own stored readings and forecasts, so everything "
        "I say is a real number rather than a guess. You can ask me about:\n\n"
        "- **Current conditions** — \"What's the AQI in Karachi?\"\n"
        "- **Forecasts** — \"What's the forecast for Lahore?\"\n"
        "- **A specific pollutant** — \"How much PM2.5 is in Multan?\"\n"
        "- **Comparisons** — \"Compare Karachi and Islamabad\"\n"
        "- **Whether to go outside** — \"Can I go for a run in Peshawar today?\"\n"
        "- **The models** — \"Which model is used for Quetta?\"\n\n"
        f"Cities I cover: {', '.join(m['label'] for m in config.CITIES.values())}."
    )


def answer(question, get_readings, get_forecast, get_metrics, get_models,
           default_city=None) -> str:
    """Routes one question to a grounded answer.

    The get_* callables are injected so this module stays free of Streamlit
    caching and Hopsworks imports, and so it can be tested with plain stubs.
    """
    question = (question or "").strip()
    if not question:
        return _answer_help()

    intent = classify(question)
    if intent == HELP:
        return _answer_help()

    cities = resolve_cities(question)
    city = cities[0] if cities else default_city

    if city is None:
        return (
            "Which city did you mean? I cover "
            f"{', '.join(m['label'] for m in config.CITIES.values())}."
        )

    if intent == COMPARE and len(cities) >= 2:
        return _answer_compare(cities, get_readings)
    if intent == MODEL:
        return _answer_model(city, get_metrics, get_models)
    if intent == ACTIVITY:
        return _answer_activity(city, question, get_readings)
    if intent == FORECAST:
        return _answer_forecast(city, get_readings, get_forecast)
    if intent == POLLUTANT:
        return _answer_pollutant(city, resolve_pollutant(question), get_readings)
    if intent == CURRENT:
        return _answer_current(city, get_readings)

    # Unknown intent: answer the most useful thing we can rather than nothing,
    # and be explicit that the question wasn't understood.
    return (
        "I'm not sure what you're asking, so here's the current picture instead.\n\n"
        + _answer_current(city, get_readings)
        + "\n\nAsk me \"what can you do?\" to see the kinds of question I can answer."
    )
