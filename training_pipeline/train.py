# -*- coding: utf-8 -*-
"""
Trains RandomForest, Ridge, XGBoost, a persistence baseline, and an LSTM
for each (city, forecast horizon) pair, evaluates all five on a
chronological hold-out split, and registers the lower-RMSE *deployable*
candidate per city/horizon to the Hopsworks Model Registry with its
metrics attached (see DEPLOYABLE_MODELS in models.py for why the LSTM is
compared but never deployed).

The full comparison (all 5 candidates x 10 cities x 3 horizons, with the
winner and each candidate's deployable flag) is written to
outputs/training_summary.csv and uploaded as a workflow artifact by
.github/workflows/daily_training_pipeline.yml — see that run's
"Artifacts" section to check which model won where.

Run manually: python -m training_pipeline.train
  (needs requirements-train.txt installed too, for the LSTM's TensorFlow)
Run by CI: .github/workflows/daily_training_pipeline.yml (daily cron)
"""

import shutil
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd
import requests

import config
from hopsworks_utils import get_model_registry
from training_pipeline.build_dataset import (
    build_training_data,
    chronological_train_test_split,
    target_column_name,
)
from training_pipeline.evaluate import evaluate
from training_pipeline.models import DEPLOYABLE_MODELS, MODEL_FACTORY

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUMMARY_CSV_PATH = OUTPUT_DIR / "training_summary.csv"

# Each register_model() call re-logs-in to Hopsworks (get_model_registry()
# with no cached project) and does several HTTP round trips to upload +
# register a model; a single transient reset anywhere in that chain
# shouldn't fail the whole run when the equivalent Open-Meteo calls already
# tolerate the same kind of blip via urllib3 retries.
_RETRYABLE_EXCEPTIONS = (requests.exceptions.RequestException, ConnectionError, TimeoutError, OSError)


def _retry_transient(fn, retries=3, backoff_seconds=5):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt == retries:
                raise
            print(f"Transient error on attempt {attempt}/{retries}: {exc}. "
                  f"Retrying in {backoff_seconds}s...")
            time.sleep(backoff_seconds)
            backoff_seconds *= 2


def train_and_select_best(X_train, y_train, X_test, y_test):
    """Trains every candidate model, returns (best_name, best_model, best_metrics, comparison_df).

    The winner is chosen only among DEPLOYABLE_MODELS (see models.py) - some
    candidates (currently the LSTM) are evaluated and reported for comparison
    purposes but are never eligible to actually be registered/deployed, so a
    candidate outperforming the trained-and-deployable ones doesn't silently
    change what ships. Every candidate is trained in its own try/except so one
    failing (e.g. LSTM needing more rows than a city/horizon has) doesn't
    lose the rest of the comparison."""
    results = []
    fitted = {}

    for name, build_model in MODEL_FACTORY.items():
        try:
            model = build_model()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            results.append({"model": name, **metrics})
            fitted[name] = model
        except Exception as exc:
            print(f"[{name}] Failed to train/evaluate, excluding from this comparison: {exc}")

    comparison = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    comparison["deployable"] = comparison["model"].isin(DEPLOYABLE_MODELS)

    deployable_rows = comparison[comparison["deployable"]]
    if deployable_rows.empty:
        raise RuntimeError("No deployable candidate trained successfully for this city/horizon.")

    best_name = deployable_rows.iloc[0]["model"]
    best_metrics = deployable_rows.iloc[0].to_dict()
    return best_name, fitted[best_name], best_metrics, comparison


def register_model(model, registry_name: str, metrics: dict, city: str):
    def _attempt():
        registry = get_model_registry()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            joblib.dump(model, tmp_dir / "model.pkl")
            hw_model = registry.python.create_model(
                name=registry_name,
                metrics={"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]},
                description=f"AQI forecaster for {city} — winning algorithm: {metrics['model']}.",
                input_example=None,
            )
            hw_model.save(str(tmp_dir))
            return hw_model
        finally:
            # hw_model.save() moves files out of tmp_dir on success, but a
            # failed attempt may leave it non-empty - always start the next
            # retry from a clean directory.
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return _retry_transient(_attempt)


def run_training():
    df = build_training_data()
    print(f"Loaded {len(df)} feature rows with horizon targets attached "
          f"across {df['city'].nunique()} cities.")

    all_comparisons = []
    failed = []

    for city in config.CITIES:
        city_row_count = int((df["city"] == city).sum())
        if city_row_count == 0:
            print(f"[{city}] 0 rows in the feature group — run "
                  "feature_pipeline.backfill_pipeline for this city first. Skipping.")
            continue

        for horizon in config.FORECAST_HORIZONS_HOURS:
            target_col = target_column_name(horizon)
            try:
                X_train, y_train, X_test, y_test = chronological_train_test_split(
                    df, target_col, city
                )

                if len(X_train) < 10 or len(X_test) < 3:
                    print(
                        f"[{city}/{target_col}] {city_row_count} raw rows but only "
                        f"{len(X_train)} train / {len(X_test)} test usable after dropping "
                        "rows with missing lag/rolling features or target — skipping this "
                        "horizon until more history has been backfilled/collected."
                    )
                    continue

                best_name, best_model, best_metrics, comparison = train_and_select_best(
                    X_train, y_train, X_test, y_test
                )
                print(f"\n[{city}/{target_col}] Model comparison:\n{comparison.to_string(index=False)}")
                print(f"[{city}/{target_col}] Selected: {best_name} (RMSE={best_metrics['rmse']:.2f})")

                comparison = comparison.copy()
                comparison.insert(0, "city", city)
                comparison.insert(1, "horizon_days", horizon // 24)
                comparison["selected"] = comparison["model"] == best_name
                all_comparisons.append(comparison)

                registry_name = config.MODEL_REGISTRY_NAME_TEMPLATE.format(
                    city=city, horizon=horizon // 24
                )
                register_model(best_model, registry_name, best_metrics, city)
                print(f"[{city}/{target_col}] Registered to Hopsworks Model Registry as '{registry_name}'.")
            except Exception as exc:
                # Don't let one bad (city, horizon) pair abort the whole run and
                # lose every other city's results along with it.
                print(f"[{city}/{target_col}] Training failed, skipping: {exc}")
                failed.append(f"{city}/{target_col}")

    if all_comparisons:
        summary = pd.concat(all_comparisons, ignore_index=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        summary.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"\nFull model comparison ({len(summary)} rows, "
              f"{summary['selected'].sum()} winners):\n{summary.to_string(index=False)}")
        print(f"\nWrote comparison summary to {SUMMARY_CSV_PATH}")
    else:
        print("\nNo city/horizon had enough rows to train on this run.")

    if failed:
        raise RuntimeError(f"Training failed for: {', '.join(failed)}.")


if __name__ == "__main__":
    run_training()
