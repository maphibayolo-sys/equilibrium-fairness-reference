"""loop.py — orchestrates the full Equilibrium Fairness runtime loop.

Stages:
    1. Initialise (E₀) — loader.load_dataset, loader.equalise_e0
    2. Monitor    (δ)  — monitor.compute_drift
    3. Threshold  (θ)  — threshold.check
    4. Correct    (κ)  — escalate.dispatch  (logging stub)

Run:
    python -m src.loop --config configs/hiring_adult.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .loader import load_dataset
from .baseline import fit_baseline, predict_rolling
from .monitor import compute_drift
from .threshold import check
from .escalate import dispatch


def run(config_path: str) -> dict:
    """Run the full EF loop against one config. Return a summary dict."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    log_path = Path(cfg["log_path"])
    # Truncate the log at the start of each run so demos are reproducible.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    # Stage 1: Initialise (E₀)
    data = load_dataset(cfg)

    # Train baseline on the equalised inputs
    model = fit_baseline(data.X_train, data.y_train,
                         seed=cfg.get("random_seed", 42))

    # Stages 2–4 over rolling windows of test data
    events = []
    for window in predict_rolling(
        model, data.X_test, data.y_test, data.group_test,
        window_size=cfg["window_size"],
    ):
        # Stage 2: δ
        reading = compute_drift(window, cfg)
        # Stage 3: θ
        result = check(reading.magnitude, cfg["theta"])
        # Stage 4: κ-trigger (logs and waits; no auto-correction)
        event = dispatch(reading, result, cfg, log_path)
        events.append(event)

    breaches = [e for e in events if e["breach"]]
    summary = {
        "dataset": cfg["dataset_name"],
        "metric": cfg["metric"],
        "theta": cfg["theta"],
        "window_size": cfg["window_size"],
        "n_windows": len(events),
        "n_breaches": len(breaches),
        "breach_rate": (len(breaches) / len(events) if events else 0.0),
        "log_path": str(log_path),
        "kappa_owner": cfg["kappa_owner"],
    }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run the Equilibrium Fairness reference loop.",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (see configs/).",
    )
    args = p.parse_args(argv)

    summary = run(args.config)

    print()
    print("=" * 60)
    print("Equilibrium Fairness — run summary")
    print("=" * 60)
    print(f"  dataset       : {summary['dataset']}")
    print(f"  metric        : {summary['metric']}")
    print(f"  θ (threshold) : {summary['theta']}")
    print(f"  window size   : {summary['window_size']}")
    print(f"  windows run   : {summary['n_windows']}")
    print(f"  θ breaches    : {summary['n_breaches']}  "
          f"({100*summary['breach_rate']:.1f}% of windows)")
    print(f"  audit log     : {summary['log_path']}")
    print(f"  κ owner       : {summary['kappa_owner']}")
    print("=" * 60)
    print()
    print("Each breach is one escalation event awaiting human-governed κ.")
    print("Open the audit log to inspect the structured events:")
    print(f"  cat {summary['log_path']} | head")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
