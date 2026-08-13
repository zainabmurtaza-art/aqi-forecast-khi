# AQI Forecasting — Pakistan (Serverless)

Predicts the Air Quality Index 3 days ahead for 10 Pakistani cities — Karachi, Lahore,
Islamabad, Faisalabad, Multan, Peshawar, Quetta, Hyderabad, Abbottabad, and Rawalpindi — using a
serverless stack: an hourly feature pipeline, a [Hopsworks](https://www.hopsworks.ai/) feature
store and model registry, a daily retraining job, and a Streamlit dashboard with a city
selector.

## Architecture

```
Open-Meteo (air quality + weather, historical & forecast, free & keyless)
AQICN (optional secondary live-reading source)
        │
        ▼
feature_pipeline/  ──► Hopsworks Feature Group (aqi_features, keyed by city + event_time)
  backfill_pipeline.py   (one-off: N days of history, all 10 cities)
  hourly_pipeline.py     (scheduled hourly via GitHub Actions, all 10 cities)
        │
        ▼
training_pipeline/  ──► Hopsworks Model Registry
  build_dataset.py, models.py, evaluate.py, train.py
  (RandomForest + Ridge, 3 separate models for t+1/t+2/t+3 day horizons
   per city — 30 models total, scheduled daily via GitHub Actions)
        │
        ▼
app/streamlit_app.py  ──► Streamlit Community Cloud
  city selector + current AQI + 3-day forecast + historical trend + SHAP + hazard alert
```

Cities and coordinates live in `config.CITIES`; add or remove a city there and every stage
(backfill, hourly updates, training, dashboard) picks it up automatically.

## One-time setup

1. **Python environment**
   ```bash
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   pip install -r requirements.txt
   ```

2. **Hopsworks account**
   - Create a free account at [hopsworks.ai](https://www.hopsworks.ai/).
   - Create a project (e.g. `aqi_karachi_forecast`).
   - Generate an API key (Account Settings → API Keys).

3. **Local secrets**
   ```bash
   cp .env.example .env
   ```
   Fill in `HOPSWORKS_API_KEY` (and `AQICN_API_TOKEN` if you want the secondary live-reading
   source — rotate any token that has ever been shared/committed before reusing it).

4. **Verify Hopsworks connectivity**
   ```bash
   python -c "import hopsworks; hopsworks.login()"
   ```

## Running each stage manually

```bash
# 1. Backfill historical training data (run once)
python -m feature_pipeline.backfill_pipeline

# 2. Hourly feature update (what the scheduled workflow runs)
python -m feature_pipeline.hourly_pipeline

# 3. Train models and register the best one per horizon
python -m training_pipeline.train

# 4. Run the dashboard locally
streamlit run app/streamlit_app.py
```

## CI/CD

Two GitHub Actions workflows (`.github/workflows/`) run the hourly feature pipeline and the
daily training pipeline on a schedule, using repo secrets `HOPSWORKS_API_KEY` (and
`AQICN_API_TOKEN` if used). Trigger each manually once via `workflow_dispatch` before trusting
the cron schedule.

## Deployment

Push this repo to GitHub, then create an app on
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app/streamlit_app.py`, and
add `HOPSWORKS_API_KEY` under the app's own Settings → Secrets (separate from GitHub Actions
secrets).

## Repository layout

- `config.py` — all settings (cities + coordinates, thresholds, Hopsworks project name).
- `feature_pipeline/` — Open-Meteo/AQICN clients, feature engineering, backfill + hourly jobs.
- `training_pipeline/` — dataset construction, model training/evaluation, registry push.
- `app/` — Streamlit dashboard.
- `notebooks/eda.ipynb` — exploratory analysis on backfilled data.
- `legacy/` — original prototype script, kept for reference.
