# -*- coding: utf-8 -*-
"""
Model factory. RandomForest, Ridge, XGBoost, a naive persistence baseline,
and an LSTM are the five candidates trained and compared per (city, horizon).

TensorFlow is only imported lazily inside LSTMRegressor's own methods, and
only ever installed in the training CI job (see requirements-train.txt) -
not in the app's requirements.txt. See DEPLOYABLE_MODELS below for why.
"""

import numpy as np
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


class LSTMRegressor:
    """Small sequence model: predicts each row's target from the trailing
    `lookback` hours of that row's own (already feature-engineered) values,
    treating each engineered feature row as one timestep. Unlike every other
    candidate here, which treats each row independently, this one actually
    sees short-term temporal shape - the point of including a genuine deep
    learning model rather than just another row-wise regressor.

    fit()/predict() use the same (X, y) DataFrame/Series interface as every
    other candidate, so it drops into MODEL_FACTORY/train_and_select_best
    with no special-casing. X_train and X_test are always chronologically
    contiguous (chronological_train_test_split slices one sorted frame), so
    predict() prepends the tail of the fitted training data to reconstruct
    a continuous sequence across the train/test boundary instead of needing
    the first `lookback` test rows to have their own history discarded.
    """

    def __init__(self, lookback=48, units=32, epochs=30, batch_size=64, random_state=42):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _make_sequences(self, X_2d, y_1d=None):
        n = len(X_2d)
        if n <= self.lookback:
            raise ValueError(
                f"Need more than lookback={self.lookback} rows to build any "
                f"sequences, got {n}."
            )
        starts = range(self.lookback, n)
        seqs = np.stack([X_2d[i - self.lookback:i] for i in starts])
        if y_1d is None:
            return seqs
        targets = np.array([y_1d[i] for i in starts])
        return seqs, targets

    def fit(self, X, y):
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        X_values = X.to_numpy(dtype="float32")
        y_values = y.to_numpy(dtype="float32")

        self._feature_mean = X_values.mean(axis=0)
        self._feature_std = X_values.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        X_scaled = (X_values - self._feature_mean) / self._feature_std

        # AQI targets sit around 60-200; a freshly-initialized Dense(1) head
        # starts near output 0, and with an unscaled target of that
        # magnitude it never fully closes that gap in a handful of epochs -
        # verified this collapses every prediction to one near-constant
        # value even on a trivially learnable clean signal. Scaling the
        # target the same way as the features fixes it completely.
        self._target_mean = y_values.mean()
        self._target_std = y_values.std() or 1.0
        y_scaled = (y_values - self._target_mean) / self._target_std

        X_seq, y_seq = self._make_sequences(X_scaled, y_scaled)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, X_scaled.shape[1])),
            tf.keras.layers.LSTM(self.units),
            tf.keras.layers.Dense(1),
        ])
        self._model.compile(optimizer="adam", loss="mse")
        self._model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
        )

        # Stored so predict() can prepend it to X_test and reconstruct
        # sequences that span the train/test boundary correctly.
        self._train_tail = X_values[-self.lookback:]
        return self

    def predict(self, X):
        X_values = X.to_numpy(dtype="float32")
        combined = np.vstack([self._train_tail, X_values])
        combined_scaled = (combined - self._feature_mean) / self._feature_std
        X_seq = self._make_sequences(combined_scaled)
        preds_scaled = self._model.predict(X_seq, verbose=0).flatten()
        return preds_scaled * self._target_std + self._target_mean


# Candidates eligible to actually be registered/deployed to the Streamlit
# app. LSTM is trained and reported like every other candidate (to satisfy
# "experiment with deep learning models"), but deliberately excluded here:
# TensorFlow is a large dependency (a couple hundred MB just to import),
# which risks out-of-memory crashes on Streamlit Community Cloud's free
# tier if it ever became a runtime requirement of the live dashboard - for
# marginal expected benefit, since the persistence-baseline diagnostic
# already showed simpler models beating fancier ones in several cities.
DEPLOYABLE_MODELS = {"random_forest", "ridge", "xgboost", "persistence"}


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
    "lstm": lambda: LSTMRegressor(),
}
