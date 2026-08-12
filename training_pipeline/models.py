# -*- coding: utf-8 -*-
"""
Model factory. RandomForest, Ridge, and XGBoost are the three candidates
actually trained and compared per (city, horizon).
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

MODEL_FACTORY = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "ridge": lambda: Ridge(alpha=1.0),
    "xgboost": lambda: XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
    ),
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
