# -*- coding: utf-8 -*-
"""
One-off diagnostic: prints row count, duplicate-key count, and the most
recent rows in the feature group. Not part of the regular pipeline - run
manually via .github/workflows/inspect.yml when you need to sanity-check
what's actually stored in Hopsworks.
"""

import pandas as pd

from hopsworks_utils import get_feature_group


def inspect():
    fg = get_feature_group()
    df = fg.read()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values("event_time")

    print(f"Total rows: {len(df)}")

    duplicate_count = df.duplicated(subset=["city", "event_time"]).sum()
    print(f"Duplicate (city, event_time) pairs: {duplicate_count}")

    print("\nMost recent 5 rows (event_time, us_aqi):")
    print(df[["event_time", "us_aqi"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    inspect()
