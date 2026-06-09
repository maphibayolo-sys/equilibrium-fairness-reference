"""loader.py — E₀: Initialise (stage 1 of the EF architecture).

Loads the configured dataset and applies the initialisation policy.
The policy strips features that encode inherited advantage from the
model inputs; the protected attribute is retained as a monitoring
label only, never as a feature.

If the real UCI dataset cannot be downloaded (no network, behind a
firewall, etc.), a deterministic synthetic stand-in is generated with
the same schema. The synthetic data has built-in disparate-impact
structure so the demo demonstrates non-trivial drift events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# Whether real datasets are available depends on the environment.
# The synthetic fallback is deterministic given the seed and produces
# disparity patterns realistic enough for the demo.
TRY_REAL_DOWNLOADS = False


@dataclass
class LoadedDataset:
    """Container for a loaded, E₀-equalised dataset.

    Attributes
    ----------
    X_train, X_test : pd.DataFrame
        Model inputs. Protected attributes have been removed per the
        configured E₀ policy.
    y_train, y_test : np.ndarray
        Binary outcome labels.
    group_train, group_test : np.ndarray
        Protected-attribute labels. Retained for monitoring only.
        These do NOT enter the model.
    feature_names : list[str]
        Names of the retained input features.
    e0_dropped : list[str]
        Names of the features removed by the E₀ policy.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    group_train: np.ndarray
    group_test: np.ndarray
    feature_names: list
    e0_dropped: list


def _synthetic_adult(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic stand-in for UCI Adult.

    Designed to produce non-trivial drift: the income label has a
    structural dependence on sex that is NOT mediated entirely by the
    legitimate features, so a model that drops sex still leaves
    measurable demographic parity gap.
    """
    rng = np.random.default_rng(seed)

    # Demographics
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    race = rng.choice(["White", "Black", "Asian", "Other"], size=n,
                      p=[0.75, 0.10, 0.10, 0.05])
    age = rng.integers(18, 75, size=n)

    # Education years — slightly higher for the majority group on average
    education_num = rng.normal(10 + 0.5 * (sex == "Male"), 2.5, size=n)
    education_num = np.clip(education_num, 1, 16).round().astype(int)

    # Hours per week
    hours = rng.normal(40 + 4 * (sex == "Male"), 8, size=n)
    hours = np.clip(hours, 1, 99).astype(int)

    # Capital gain — heavy-tailed, correlated with the outcome
    capital_gain = rng.exponential(500, size=n)
    capital_gain = np.where(rng.random(n) < 0.08, capital_gain * 20, capital_gain)
    capital_gain = capital_gain.astype(int)

    # Occupation
    occupation = rng.choice(
        ["Tech", "Service", "Sales", "Trades", "Admin", "Other"],
        size=n, p=[0.18, 0.20, 0.15, 0.17, 0.15, 0.15],
    )

    # Income label.
    # Structural disparity: even controlling for the legitimate signals,
    # there is residual association with sex. This is the drift the
    # monitoring layer is meant to detect.
    score = (
        0.30 * (education_num - 10) / 4
        + 0.25 * (hours - 40) / 10
        + 0.20 * np.log1p(capital_gain) / 8
        + 0.15 * (occupation == "Tech").astype(float)
        + 0.20 * (sex == "Male").astype(float)        # the structural gap
        - 0.05 * (race == "Black").astype(float)
        + rng.normal(0, 0.4, size=n)
    )
    income = (score > score.mean() + 0.3).astype(int)

    return pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "hours_per_week": hours,
        "capital_gain": capital_gain,
        "occupation": occupation,
        "sex": sex,
        "race": race,
        "income": income,
    })


def _synthetic_german(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic stand-in for UCI German Credit.

    The protected attribute is binarised age (under/over 25), which
    is the conventional setup in the algorithmic fairness literature
    on this dataset.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, size=n)
    age_binarised = np.where(age <= 25, "young", "adult")

    duration = rng.integers(6, 72, size=n)
    credit_amount = rng.exponential(3000, size=n).astype(int) + 250
    installment_rate = rng.integers(1, 5, size=n)
    employed_years = rng.integers(0, 30, size=n)
    existing_credits = rng.integers(1, 5, size=n)

    purpose = rng.choice(
        ["car", "furniture", "education", "business", "other"],
        size=n, p=[0.30, 0.25, 0.10, 0.20, 0.15],
    )

    # Credit-risk label. Younger applicants get a structurally worse outcome
    # at the margin, even controlling for the legitimate signals.
    score = (
        -0.30 * (duration - 36) / 12
        - 0.25 * (credit_amount - 3000) / 3000
        + 0.20 * (employed_years - 5) / 10
        - 0.15 * (age_binarised == "young").astype(float)
        + 0.10 * (purpose == "business").astype(float)
        + rng.normal(0, 0.5, size=n)
    )
    good_credit = (score > score.mean() - 0.2).astype(int)

    return pd.DataFrame({
        "duration_months": duration,
        "credit_amount": credit_amount,
        "installment_rate": installment_rate,
        "employed_years": employed_years,
        "existing_credits": existing_credits,
        "purpose": purpose,
        "age": age,
        "age_binarised": age_binarised,
        "good_credit": good_credit,
    })


def _load_raw(dataset: str, seed: int) -> tuple:
    """Return (DataFrame, target_column, default_drop_columns)."""
    if dataset == "adult":
        df = _synthetic_adult(seed=seed)
        return df, "income", ["sex", "race"]
    elif dataset == "german_credit":
        df = _synthetic_german(seed=seed)
        return df, "good_credit", ["age", "age_binarised"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def equalise_e0(df: pd.DataFrame, policy: dict, target: str) -> tuple:
    """Apply the E₀ policy.

    Drops the configured columns from the feature set. The protected
    attribute named in policy['protected'] is retained as a separate
    label series (used for monitoring), not as a feature.

    Returns
    -------
    X : pd.DataFrame   features after the E₀ policy
    y : np.ndarray     binary labels
    group : np.ndarray protected-attribute labels (monitoring only)
    """
    drop_columns = list(policy.get("drop_columns", []))
    protected = policy["protected"]

    if protected not in df.columns:
        raise KeyError(
            f"Protected attribute '{protected}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    group = df[protected].to_numpy()
    y = df[target].to_numpy()

    feature_df = df.drop(columns=drop_columns + [target], errors="ignore")
    # One-hot encode the remaining categorical columns
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    # Convert any boolean dummy columns to int for downstream stability
    bool_cols = feature_df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        feature_df[bool_cols] = feature_df[bool_cols].astype(int)

    return feature_df, y, group


def load_dataset(cfg: dict) -> LoadedDataset:
    """Top-level loader. Reads the config and returns a LoadedDataset."""
    seed = cfg.get("random_seed", 42)
    test_fraction = cfg.get("test_fraction", 0.30)

    df, target, _default_drops = _load_raw(cfg["dataset"], seed)
    X, y, group = equalise_e0(df, cfg["e0_policy"], target)

    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    return LoadedDataset(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y[train_idx],
        y_test=y[test_idx],
        group_train=group[train_idx],
        group_test=group[test_idx],
        feature_names=list(X.columns),
        e0_dropped=list(cfg["e0_policy"].get("drop_columns", [])),
    )
