"""Tests for the monitor / δ stage."""
import numpy as np
import pytest

from src.monitor import (
    compute_drift,
    demographic_parity_gap,
    equalised_odds_gap,
    METRICS,
)


def test_demographic_parity_zero_when_groups_identical():
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    group = np.array(["A", "B", "A", "B", "A", "B"])
    gap, n_a, n_b = demographic_parity_gap(y_pred, group, "A", "B")
    assert gap == pytest.approx(1.0)  # A all 1, B all 0
    assert n_a == 3 and n_b == 3


def test_demographic_parity_balanced():
    # Both groups predict 50% positive
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    gap, n_a, n_b = demographic_parity_gap(y_pred, group, "A", "B")
    assert gap == pytest.approx(0.0)
    assert n_a == 4 and n_b == 4


def test_demographic_parity_handles_missing_group():
    y_pred = np.array([1, 1, 1, 1])
    group = np.array(["A", "A", "A", "A"])
    gap, n_a, n_b = demographic_parity_gap(y_pred, group, "A", "B")
    assert gap == 0.0
    assert n_a == 4 and n_b == 0


def test_equalised_odds_zero_when_identical():
    y_pred = np.array([1, 0, 1, 0])
    y_true = np.array([1, 0, 1, 0])
    group = np.array(["A", "A", "B", "B"])
    gap, _, _ = equalised_odds_gap(y_pred, y_true, group, "A", "B")
    assert gap == pytest.approx(0.0)


def test_equalised_odds_detects_tpr_gap():
    # Group A: perfect predictions. Group B: misses all positives.
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.array([1, 1, 1, 1])
    group = np.array(["A", "A", "B", "B"])
    gap, _, _ = equalised_odds_gap(y_pred, y_true, group, "A", "B")
    # TPR_A = 1, TPR_B = 0. FPR has no negatives -> 0.
    # 0.5 * (|1| + |0|) = 0.5
    assert gap == pytest.approx(0.5)


def test_compute_drift_routes_to_correct_metric():
    cfg = {
        "metric": "demographic_parity_gap",
        "group_a": "A", "group_b": "B",
    }
    window = {
        "y_pred": np.array([1, 1, 0, 0]),
        "y_true": np.array([1, 1, 0, 0]),
        "group":  np.array(["A", "A", "B", "B"]),
        "start": 0, "end": 4,
    }
    reading = compute_drift(window, cfg)
    assert reading.metric == "demographic_parity_gap"
    assert reading.magnitude == pytest.approx(1.0)
    assert reading.window_size == 4
    assert reading.n_a == 2 and reading.n_b == 2


def test_compute_drift_unknown_metric_raises():
    cfg = {"metric": "not_a_metric", "group_a": "A", "group_b": "B"}
    window = {
        "y_pred": np.array([1]),
        "y_true": np.array([1]),
        "group":  np.array(["A"]),
        "start": 0, "end": 1,
    }
    with pytest.raises(ValueError):
        compute_drift(window, cfg)


def test_metrics_registry_has_both():
    assert "demographic_parity_gap" in METRICS
    assert "equalised_odds_gap" in METRICS
