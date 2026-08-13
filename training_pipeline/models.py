# -*- coding: utf-8 -*-
"""
Model factory. RandomForest, Ridge, XGBoost, and a naive persistence
baseline are the four candidates trained and compared per (city, horizon).
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


class PersistenceRegressor:
    """Naive baseline: predicts 'tomorrow looks like today', i.e. the
    target is just the row's current us_aqi feature value. Not fit to
    anything - it exists so every city/horizon comparison includes a
    trivial reference point. If this wins (lowest RMSE) for a given
    city/horizon, that tells us the trained models aren't capturing that
    horizon's AQI dynamics at all, not just that they're under-tuned -
    and it's the honest model to deploy there until that's fixed."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["us_aqi"].to_numpy()


MODEL_FACTORY = {
    # max_depth was 12 with no leaf-size floor - on ~90 days of history per
    # city that memorizes training noise and produces negative test R2. Capped
    # depth + a minimum leaf size trades train-set fit for held-out generalization.
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "ridge": lambda: Ridge(alpha=1.0),
    # subsample/colsample_bytree add row/column randomness per tree (same
    # overfitting-control idea as RandomForest's bagging); reg_lambda adds
    # L2 shrinkage on leaf weights.
    "xgboost": lambda: XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    ),
    "persistence": lambda: PersistenceRegressor(),
}


def build_lstm_model():
    """
    Not implemented. A TensorFlow/PyTorch sequence model (e.g. LSTM) is a
    reasonable next step once the feature store has enough history to
    support it — deep sequence models need far more training rows than a
    fresh backfill (weeks/months) provides to beat RandomForest/Ridge on
    RMSE. Revisit once config.BACKFILL_DAYS-scale history has accumulated
    over several retraining cycles.
    """
    raise NotImplementedError("Deep learning horizon models are a stretch goal, not yet built.")
