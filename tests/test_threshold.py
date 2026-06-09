"""Tests for the threshold / θ stage."""
import pytest

from src.threshold import check, ThresholdResult


def test_within_threshold():
    result = check(magnitude=0.03, theta=0.05)
    assert isinstance(result, ThresholdResult)
    assert result.breach is False
    assert result.direction == "within"
    assert result.threshold == 0.05
    assert result.magnitude == 0.03


def test_above_threshold():
    result = check(magnitude=0.09, theta=0.05)
    assert result.breach is True
    assert result.direction == "above"


def test_exactly_at_threshold_is_within():
    # Strict > policy: magnitude == theta is within tolerance.
    result = check(magnitude=0.05, theta=0.05)
    assert result.breach is False
    assert result.direction == "within"


def test_negative_magnitude_uses_absolute_value():
    # δ can be signed; θ is on |δ|
    result = check(magnitude=-0.10, theta=0.05)
    assert result.breach is True
    assert result.magnitude == pytest.approx(0.10)


def test_negative_theta_raises():
    with pytest.raises(ValueError):
        check(magnitude=0.10, theta=-0.01)


def test_zero_theta_breaches_on_any_drift():
    # Operational edge case — a zero-tolerance threshold.
    result = check(magnitude=0.0001, theta=0.0)
    assert result.breach is True
    result2 = check(magnitude=0.0, theta=0.0)
    assert result2.breach is False
