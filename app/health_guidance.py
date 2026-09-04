# -*- coding: utf-8 -*-
"""
Public-health guidance keyed to the US AQI categories.

Wording follows the US EPA's own AQI advisories, which is the scale this
project forecasts against (Open-Meteo's `us_aqi` field), so the advice matches
the number being shown rather than being invented alongside it.

This is general population-level guidance, not medical advice — DISCLAIMER
below is rendered wherever these strings are.
"""

DISCLAIMER = (
    "General guidance based on the US EPA Air Quality Index. It is not medical "
    "advice — anyone with a heart or lung condition should follow their own "
    "doctor's plan."
)

# Audience groups, in the order they're shown.
GROUPS = ["Sensitive groups", "Children & elderly", "Outdoor workers", "Everyone else"]

# (upper_bound, category) -> {group: advice}. Bounds match AQI_KEY_RANGES in
# ui_components, so a reading can never fall between two sets of advice.
GUIDANCE = {
    "Good": {
        "Sensitive groups": "Air quality poses little or no risk. No precautions needed.",
        "Children & elderly": "Ideal conditions for outdoor play and exercise.",
        "Outdoor workers": "No restrictions. Normal outdoor work is fine.",
        "Everyone else": "A good day to be outside. Enjoy it.",
    },
    "Moderate": {
        "Sensitive groups": "Unusually sensitive people should watch for symptoms such as "
                            "coughing or shortness of breath, and consider shortening intense "
                            "outdoor activity.",
        "Children & elderly": "Fine for normal activity. Take breaks if a child becomes "
                              "breathless during hard play.",
        "Outdoor workers": "No restrictions, but take normal rest breaks during heavy exertion.",
        "Everyone else": "Air quality is acceptable. No precautions needed for most people.",
    },
    "Unhealthy for Sensitive Groups": {
        "Sensitive groups": "Cut back on prolonged or intense outdoor exertion. Keep quick-relief "
                            "medicine (such as a rescue inhaler) to hand.",
        "Children & elderly": "Reduce long or intense outdoor activity; move strenuous play "
                              "indoors or to a less busy time of day.",
        "Outdoor workers": "Take more frequent breaks and lower the intensity of heavy work "
                           "where possible.",
        "Everyone else": "Most people are unaffected, but stop if you feel symptoms.",
    },
    "Unhealthy": {
        "Sensitive groups": "Avoid prolonged or intense outdoor exertion. Move activity indoors "
                            "or reschedule it.",
        "Children & elderly": "Avoid long or intense outdoor activity. Keep strenuous play indoors.",
        "Outdoor workers": "Reduce heavy outdoor work, rotate tasks, and use a well-fitted "
                           "particulate mask (N95/KN95) near heavy traffic or dust.",
        "Everyone else": "Cut back on prolonged or intense outdoor exertion.",
    },
    "Very Unhealthy": {
        "Sensitive groups": "Avoid all outdoor physical activity. Stay indoors and keep windows "
                            "closed where you can.",
        "Children & elderly": "Keep outdoor time to a minimum; move all activity indoors.",
        "Outdoor workers": "Limit outdoor work to essential tasks, with respiratory protection "
                           "and frequent indoor breaks.",
        "Everyone else": "Avoid prolonged or intense outdoor exertion; consider moving activity "
                         "indoors.",
    },
    "Hazardous": {
        "Sensitive groups": "Stay indoors and keep exertion low. Seek medical help for chest "
                            "tightness or breathing difficulty.",
        "Children & elderly": "Remain indoors with windows shut. Avoid any outdoor exertion.",
        "Outdoor workers": "Outdoor work should be suspended other than emergency tasks, with "
                           "full respiratory protection.",
        "Everyone else": "Health warning of emergency conditions — everyone should avoid outdoor "
                         "exertion.",
    },
}

# Short one-line summary per category, used by the alert banner and the copilot.
HEADLINE = {
    "Good": "Air quality is good — no precautions needed.",
    "Moderate": "Air quality is acceptable; unusually sensitive people should watch for symptoms.",
    "Unhealthy for Sensitive Groups": "Sensitive groups should cut back on outdoor exertion.",
    "Unhealthy": "Everyone should reduce prolonged outdoor exertion; sensitive groups should avoid it.",
    "Very Unhealthy": "Health alert — avoid outdoor physical activity.",
    "Hazardous": "Health emergency — everyone should stay indoors.",
}


def guidance_for(category: str) -> dict:
    """{group: advice} for a US AQI category label, falling back to the worst
    band so an unrecognised label never yields reassuring advice."""
    return GUIDANCE.get(category, GUIDANCE["Hazardous"])


def headline_for(category: str) -> str:
    return HEADLINE.get(category, HEADLINE["Hazardous"])


def is_safe_for_outdoor_exercise(aqi: float) -> tuple:
    """(verdict, explanation) for 'can I exercise outside?' style questions.

    Threshold is the EPA's own: at 101+ the index is defined as unhealthy for
    sensitive groups, which is where advice starts to diverge by person.
    """
    if aqi <= 50:
        return "yes", "Air quality is good, so outdoor exercise is fine."
    if aqi <= 100:
        return "yes, with care", (
            "Air quality is moderate. Fine for most people; if you are unusually "
            "sensitive, keep an eye on how you feel during hard effort."
        )
    if aqi <= 150:
        return "not ideal", (
            "This is unhealthy for sensitive groups. If you have asthma, a heart or "
            "lung condition, or are very young or elderly, shorten or reschedule "
            "intense outdoor exercise."
        )
    if aqi <= 200:
        return "no", (
            "Air quality is unhealthy. Prolonged or intense outdoor exertion should be "
            "cut back by everyone, and avoided by sensitive groups."
        )
    return "no", (
        "Air quality is very unhealthy or worse. Outdoor physical activity should be "
        "avoided entirely."
    )
