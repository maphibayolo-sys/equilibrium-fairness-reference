"""Tests for the escalate / κ-logging stage."""
import json
from pathlib import Path

import pytest

from src.escalate import dispatch
from src.monitor import DriftReading
from src.threshold import ThresholdResult


@pytest.fixture
def fake_reading():
    return DriftReading(
        metric="demographic_parity_gap",
        group_a="A", group_b="B",
        magnitude=0.12,
        window_size=500, n_a=300, n_b=200,
        window_start=0, window_end=500,
    )


@pytest.fixture
def fake_cfg():
    return {
        "dataset_name": "test_dataset",
        "kappa_owner": "test-owner@example.org",
    }


def test_dispatch_writes_one_event_per_call(tmp_path, fake_reading, fake_cfg):
    log = tmp_path / "events.jsonl"
    result = ThresholdResult(
        breach=True, threshold=0.05, magnitude=0.12, direction="above",
    )
    dispatch(fake_reading, result, fake_cfg, log)
    assert log.exists()
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 1


def test_dispatch_appends_subsequent_events(tmp_path, fake_reading, fake_cfg):
    log = tmp_path / "events.jsonl"
    result = ThresholdResult(
        breach=False, threshold=0.05, magnitude=0.02, direction="within",
    )
    dispatch(fake_reading, result, fake_cfg, log)
    dispatch(fake_reading, result, fake_cfg, log)
    dispatch(fake_reading, result, fake_cfg, log)
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 3


def test_dispatch_logs_all_events_not_just_breaches(tmp_path, fake_reading, fake_cfg):
    """Audit trail must include non-breach windows too."""
    log = tmp_path / "events.jsonl"
    within = ThresholdResult(
        breach=False, threshold=0.05, magnitude=0.02, direction="within",
    )
    dispatch(fake_reading, within, fake_cfg, log)
    line = log.read_text().strip()
    event = json.loads(line)
    assert event["breach"] is False
    assert event["correction_required"] is False


def test_dispatch_event_has_required_fields(tmp_path, fake_reading, fake_cfg):
    log = tmp_path / "events.jsonl"
    result = ThresholdResult(
        breach=True, threshold=0.05, magnitude=0.12, direction="above",
    )
    event = dispatch(fake_reading, result, fake_cfg, log)
    required = {
        "timestamp", "dataset", "metric", "group_a", "group_b",
        "magnitude", "threshold", "breach", "direction",
        "window_size", "kappa_owner", "correction_required",
    }
    assert required.issubset(event.keys())


def test_dispatch_records_kappa_owner(tmp_path, fake_reading, fake_cfg):
    log = tmp_path / "events.jsonl"
    result = ThresholdResult(
        breach=True, threshold=0.05, magnitude=0.12, direction="above",
    )
    event = dispatch(fake_reading, result, fake_cfg, log)
    assert event["kappa_owner"] == "test-owner@example.org"


def test_dispatch_creates_parent_directory(tmp_path, fake_reading, fake_cfg):
    log = tmp_path / "nested" / "deep" / "events.jsonl"
    result = ThresholdResult(
        breach=True, threshold=0.05, magnitude=0.12, direction="above",
    )
    dispatch(fake_reading, result, fake_cfg, log)
    assert log.exists()
