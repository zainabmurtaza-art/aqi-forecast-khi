# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:31:52 2026

@author: zmurt
"""

"""
LEGACY PROTOTYPE - superseded by feature_pipeline/aqicn_client.py,
feature_pipeline/hourly_pipeline.py, and feature_pipeline/backfill_pipeline.py.

Kept only for reference. Do not run this as-is: the API token that used to be
hardcoded below has been removed (it was exposed in plaintext and must be
rotated at aqicn.org before reuse). Set AQICN_API_TOKEN in your .env instead
and use feature_pipeline/aqicn_client.py, which reads it from the environment.

WHAT THIS SCRIPT ORIGINALLY DID, IN PLAIN ENGLISH:
1. Asked the AQICN website for Karachi's current air quality data.
2. Pulled out the useful numbers (pollution levels, weather, time).
3. Saved that as one row in a spreadsheet file (aqi_features.csv),
   which was a "Feature Store" stand-in.
4. Optionally repeated this once every hour, appending rows over time.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime


# =========================================================================
# SETTINGS - the only section you should need to edit
# =========================================================================

# Token removed - it was hardcoded here and has been exposed, treat as
# compromised. See feature_pipeline/aqicn_client.py for the env-based version.
API_TOKEN = None

# Which city to track
CITY = "karachi"

# Where to save the data. This will be created in the SAME folder you
# run this script from.
OUTPUT_FILE = "aqi_features.csv"

# "once"  -> fetch one row of data, then stop.
# "loop"  -> fetch one row every hour, forever, until you press Ctrl+C.
RUN_MODE = "loop"

# Only used if RUN_MODE = "loop". 3600 seconds = 1 hour.
INTERVAL_SECONDS = 3600


# =========================================================================
# LOGIC - you don't need to edit anything below this line
# =========================================================================

def fetch_raw_data(city: str, token: str) -> dict:
    """
    Sends a request to the AQICN API and returns the raw JSON reply
    as a Python dictionary. This is where we get our "raw data".
    """
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"API call failed: {data}")

    return data


def compute_features(raw: dict) -> dict:
    """
    Turns the raw API reply into a clean row of features (model inputs)
    plus the target (the AQI value we eventually want to predict).
    """
    d = raw["data"]
    iaqi = d.get("iaqi", {})  # breakdown of individual pollutants

    now = datetime.now()

    features = {
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "temperature": iaqi.get("t", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "wind": iaqi.get("w", {}).get("v"),
        "aqi": d.get("aqi"),  # TARGET column
    }
    return features


def save_features(new_row: dict, output_file: str):
    """
    Adds the new row to our CSV file. If the file doesn't exist yet,
    creates it. If it does exist, loads it and appends the new row.
    """
    new_df = pd.DataFrame([new_row])

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Derived feature your project brief asked for: how fast AQI is
    # changing between consecutive rows.
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
    combined_df["aqi_change_rate"] = combined_df["aqi"].diff()

    try:
        combined_df.to_csv(output_file, index=False)
    except PermissionError:
        print(
            f"Could not write to {output_file} - it's probably open in "
            "Excel or another program. Close it and try again."
        )
        return

    print(f"Saved. Total rows in feature store: {len(combined_df)}")


def run_once():
    """Does one full fetch -> compute -> save cycle."""
    raw = fetch_raw_data(CITY, API_TOKEN)
    features = compute_features(raw)
    print("Fetched features:", features)
    save_features(features, OUTPUT_FILE)


def run_loop():
    """Repeats run_once() every INTERVAL_SECONDS, forever."""
    print(f"Starting automated collection every {INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            print(f"[{datetime.now().isoformat()}] Fetching...")
            run_once()
        except Exception as e:
            print(f"Fetch failed this round: {e}")

        print(f"Sleeping for {INTERVAL_SECONDS} seconds...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    if RUN_MODE == "once":
        run_once()
    elif RUN_MODE == "loop":
        run_loop()
    else:
        raise ValueError('RUN_MODE must be "once" or "loop"')