"""threshold.py — θ: Threshold check (stage 3 of the EF architecture).

The threshold is the pre-agreed magnitude of |δ| at which correction
becomes mandatory. The threshold-setting decision is exogenous to
this code — by design. θ is owned by an accountable party with the
standing to bind the operating organisation; the implementation only
applies the threshold that party has set.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ThresholdResult:
    """Result of one θ-check."""
    breach: bool
    threshold: float
    magnitude: float
    direction: str  # 'above' or 'within'

    def to_dict(self) -> dict:
        return asdict(self)


def check(magnitude: float, theta: float) -> ThresholdResult:
    """Compare |δ| against the configured θ.

    Returns the breach state. Strictly greater-than is used: a
    magnitude exactly equal to θ is considered within tolerance.
    The choice is a policy one — operators who want '>=' should
    subclass.
    """
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    mag = abs(float(magnitude))
    return ThresholdResult(
        breach=(mag > theta),
        threshold=float(theta),
        magnitude=mag,
        direction=("above" if mag > theta else "within"),
    )
