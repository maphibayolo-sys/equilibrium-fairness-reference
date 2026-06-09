"""baseline.py — logistic-regression baseline model.

This is intentionally simple. The point of the reference
implementation is the architecture around the model, not the model
itself. Any classifier with the same fit/predict interface can be
substituted; the runtime loop is agnostic to model choice.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def fit_baseline(X_train, y_train, seed: int = 42):
    """Train a logistic-regression baseline on the E₀-equalised inputs."""
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_rolling(model, X_test, y_test, group_test, window_size: int):
    """Yield successive rolling windows of predictions.

    Each yielded dict has the structure expected by monitor.compute_drift:

        {
            "y_pred":  np.ndarray   model predictions for the window
            "y_true":  np.ndarray   ground-truth labels for the window
            "group":   np.ndarray   protected-attribute labels for the window
            "start":   int          starting index of the window
            "end":     int          ending index of the window (exclusive)
        }
    """
    y_all = np.asarray(model.predict(X_test))
    n = len(y_all)
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        yield {
            "y_pred": y_all[start:end],
            "y_true": np.asarray(y_test[start:end]),
            "group": np.asarray(group_test[start:end]),
            "start": start,
            "end": end,
        }
