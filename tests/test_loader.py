"""Tests for the loader / E₀ stage."""
import numpy as np
import pytest

from src.loader import (
    load_dataset, equalise_e0, _synthetic_adult, _synthetic_german,
)


@pytest.fixture
def hiring_cfg():
    return {
        "dataset": "adult",
        "dataset_name": "hiring_adult",
        "e0_policy": {
            "drop_columns": ["sex", "race"],
            "protected": "sex",
        },
        "group_a": "Male",
        "group_b": "Female",
        "random_seed": 42,
        "test_fraction": 0.30,
    }


@pytest.fixture
def lending_cfg():
    return {
        "dataset": "german_credit",
        "dataset_name": "lending_german",
        "e0_policy": {
            "drop_columns": ["age_binarised"],
            "protected": "age_binarised",
        },
        "group_a": "young",
        "group_b": "adult",
        "random_seed": 42,
        "test_fraction": 0.30,
    }


def test_synthetic_adult_is_reproducible():
    df1 = _synthetic_adult(n=200, seed=42)
    df2 = _synthetic_adult(n=200, seed=42)
    assert (df1.values == df2.values).all()


def test_synthetic_german_is_reproducible():
    df1 = _synthetic_german(n=200, seed=42)
    df2 = _synthetic_german(n=200, seed=42)
    assert (df1.values == df2.values).all()


def test_e0_drops_protected_attribute_from_inputs(hiring_cfg):
    df = _synthetic_adult(n=500, seed=42)
    X, y, group = equalise_e0(df, hiring_cfg["e0_policy"], "income")

    # Protected attributes must not appear in features (raw or one-hot)
    for col in X.columns:
        assert "sex" not in col.lower()
        assert "race" not in col.lower()

    # Group is retained separately for monitoring
    assert len(group) == len(df)
    assert set(np.unique(group)) <= {"Male", "Female"}


def test_load_dataset_produces_train_test_split(hiring_cfg):
    data = load_dataset(hiring_cfg)
    total = len(data.X_train) + len(data.X_test)
    # Should be close to the synthetic-adult default n
    assert total >= 4000

    # Test fraction approximately respected
    assert 0.25 <= len(data.X_test) / total <= 0.35

    # Labels and groups align with feature rows
    assert len(data.y_train) == len(data.X_train)
    assert len(data.y_test) == len(data.X_test)
    assert len(data.group_train) == len(data.X_train)
    assert len(data.group_test) == len(data.X_test)


def test_load_dataset_records_dropped_features(hiring_cfg):
    data = load_dataset(hiring_cfg)
    assert "sex" in data.e0_dropped
    assert "race" in data.e0_dropped


def test_lending_loader(lending_cfg):
    data = load_dataset(lending_cfg)
    assert set(np.unique(data.group_test)) <= {"young", "adult"}
    # Protected attribute removed from features
    for col in data.X_test.columns:
        assert "age_binarised" not in col


def test_unknown_dataset_raises(hiring_cfg):
    hiring_cfg["dataset"] = "definitely_not_a_real_dataset"
    with pytest.raises(ValueError):
        load_dataset(hiring_cfg)


def test_missing_protected_raises(hiring_cfg):
    hiring_cfg["e0_policy"]["protected"] = "nonexistent_column"
    with pytest.raises(KeyError):
        load_dataset(hiring_cfg)
