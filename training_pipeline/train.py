# -*- coding: utf-8 -*-
"""
Trains RandomForest, Ridge, and XGBoost for each (city, forecast horizon)
pair, evaluates all three on a chronological hold-out split, and registers
the lower-RMSE model per city/horizon to the Hopsworks Model Registry with
its metrics attached.

The full comparison (all 3 candidates x 10 cities x 3 horizons, with the
winner flagged) is written to outputs/training_summary.csv and uploaded as
a workflow artifact by .github/workflows/daily_training_pipeline.yml — see
that run's "Artifacts" section to check which model won where.

Run manually: python -m training_pipeline.train
Run by CI: .github/workflows/daily_training_pipeline.yml (daily cron)
"""

import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd

import config
from hopsworks_utils import get_model_registry
from training_pipeline.build_dataset import (
    build_training_data,
    chronological_train_test_split,
    target_column_name,
)
from training_pipeline.evaluate import evaluate
from training_pipeline.models import MODEL_FACTORY

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SUMMARY_CSV_PATH = OUTPUT_DIR / "training_summary.csv"


def train_and_select_best(X_train, y_train, X_test, y_test):
    """Trains every candidate model, returns (best_name, best_model, best_metrics, comparison_df)."""
    results = []
    fitted = {}

    for name, build_model in MODEL_FACTORY.items():
        model = build_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        results.append({"model": name, **metrics})
        fitted[name] = model

    comparison = pd.DataFrame(results).sort_values("rmse")
    best_name = comparison.iloc[0]["model"]
    best_metrics = comparison.iloc[0].to_dict()
    return best_name, fitted[best_name], best_metrics, comparison


def register_model(model, model_name: str, metrics: dict, city: str):
    registry = get_model_registry()

    tmp_dir = Path(tempfile.mkdtemp())
    model_path = tmp_dir / "model.pkl"
    joblib.dump(model, model_path)

    hw_model = registry.python.create_model(
        name=model_name,
        metrics={"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]},
        description=f"AQI forecaster ({model_name}) for {city}.",
        input_example=None,
    )
    hw_model.save(str(tmp_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return hw_model


def run_training():
    df = build_training_data()
    print(f"Loaded {len(df)} feature rows with horizon targets attached "
          f"across {df['city'].nunique()} cities.")

    all_comparisons = []

    for city in config.CITIES:
        for horizon in config.FORECAST_HORIZONS_HOURS:
            target_col = target_column_name(horizon)
            X_train, y_train, X_test, y_test = chronological_train_test_split(
                df, target_col, city
            )

            if len(X_train) < 10 or len(X_test) < 3:
                print(
                    f"[{city}/{target_col}] Not enough rows yet ({len(X_train)} train / "
                    f"{len(X_test)} test) — skipping this horizon until more "
                    "history has been backfilled/collected."
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

    if all_comparisons:
        summary = pd.concat(all_comparisons, ignore_index=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        summary.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"\nFull model comparison ({len(summary)} rows, "
              f"{summary['selected'].sum()} winners):\n{summary.to_string(index=False)}")
        print(f"\nWrote comparison summary to {SUMMARY_CSV_PATH}")
    else:
        print("\nNo city/horizon had enough rows to train on this run.")


if __name__ == "__main__":
    run_training()
