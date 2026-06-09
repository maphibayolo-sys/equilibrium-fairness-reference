"""monitor.py — δ: Monitor (stage 2 of the EF architecture).

Computes the drift signal δ over a window of recent predictions.
Two metrics are supported out of the box:

  - demographic_parity_gap  : P(ŷ=1 | A=a) − P(ŷ=1 | A=b)
  - equalised_odds_gap      : 0.5 · (|TPR_a − TPR_b| + |FPR_a − FPR_b|)

Both are pluggable. Adding a third metric requires only:
  (1) writing a function with the same signature
  (2) registering it in METRICS below

The monitoring layer is structurally separate from the decision layer
(the model). This is the architecture's load-bearing constraint:
monitor ≠ decide.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal


Metric = Literal["demographic_parity_gap", "equalised_odds_gap"]


@dataclass
class DriftReading:
    """One drift event — the unit of audit produced by the monitor."""
    metric: str
    group_a: str
    group_b: str
    magnitude: float
    window_size: int
    n_a: int
    n_b: int
    window_start: int
    window_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def demographic_parity_gap(y_pred, group, group_a, group_b):
    """P(ŷ=1 | A=group_a) − P(ŷ=1 | A=group_b).

    Returns the gap, plus the per-group window counts (for audit).
    Returns 0.0 (and zero counts) if either group is unrepresented
    in the window — the monitor reports the absence rather than
    raising an error, so the audit log is complete.
    """
    mask_a = group == group_a
    mask_b = group == group_b
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        return 0.0, n_a, n_b
    p_a = float(np.asarray(y_pred)[mask_a].mean())
    p_b = float(np.asarray(y_pred)[mask_b].mean())
    return p_a - p_b, n_a, n_b


def equalised_odds_gap(y_pred, y_true, group, group_a, group_b):
    """Mean of |TPR gap| and |FPR gap| across the two groups.

    TPR = P(ŷ=1 | y=1, A=a). FPR = P(ŷ=1 | y=0, A=a).
    Returns the gap, plus the per-group window counts (for audit).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    group = np.asarray(group)

    def conditional_rate(mask, label):
        m2 = mask & (y_true == label)
        if m2.sum() == 0:
            return 0.0
        return float(y_pred[m2].mean())

    ma, mb = (group == group_a), (group == group_b)
    n_a, n_b = int(ma.sum()), int(mb.sum())
    if n_a == 0 or n_b == 0:
        return 0.0, n_a, n_b

    tpr_gap = conditional_rate(ma, 1) - conditional_rate(mb, 1)
    fpr_gap = conditional_rate(ma, 0) - conditional_rate(mb, 0)
    return 0.5 * (abs(tpr_gap) + abs(fpr_gap)), n_a, n_b


METRICS = {
    "demographic_parity_gap": demographic_parity_gap,
    "equalised_odds_gap": equalised_odds_gap,
}


def compute_drift(window: dict, cfg: dict) -> DriftReading:
    """Compute δ over one window using the configured metric.

    Parameters
    ----------
    window : dict
        As yielded by baseline.predict_rolling: keys y_pred, y_true,
        group, start, end.
    cfg : dict
        The full deployment config; must contain 'metric', 'group_a',
        'group_b'.
    """
    metric = cfg["metric"]
    a, b = cfg["group_a"], cfg["group_b"]
    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(METRICS)}"
        )

    if metric == "demographic_parity_gap":
        mag, n_a, n_b = demographic_parity_gap(
            window["y_pred"], window["group"], a, b,
        )
    else:  # equalised_odds_gap
        mag, n_a, n_b = equalised_odds_gap(
            window["y_pred"], window["y_true"], window["group"], a, b,
        )

    return DriftReading(
        metric=metric,
        group_a=a, group_b=b,
        magnitude=float(mag),
        window_size=len(window["y_pred"]),
        n_a=n_a, n_b=n_b,
        window_start=window["start"],
        window_end=window["end"],
    )
