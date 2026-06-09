"""escalate.py — κ: Correct (stage 4 of the EF architecture).

This module instantiates κ as a structured-logging stub. On a
threshold breach, it writes one event to the escalation log (JSON
Lines) and returns. It does NOT take any corrective action.

That separation is deliberate and load-bearing. The architecture's
foundational principle — monitor ≠ decide — requires that correction
be performed by a human party with standing to act. This module's
job is to surface the breach to that party in a structured,
auditable form. The corrective action itself is theirs to take.

In a production deployment, dispatch() would also:
  - notify the human party named in cfg['kappa_owner']
  - open a ticket in the operating organisation's incident system
  - block downstream decisions if the breach is severe enough

The reference implementation keeps the side-effect surface to a
single audit log so the loop is easy to read.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .monitor import DriftReading
from .threshold import ThresholdResult


def dispatch(
    reading: DriftReading,
    result: ThresholdResult,
    cfg: dict,
    log_path: Path,
) -> dict:
    """Emit one escalation event to the audit log.

    Always writes; the 'breach' field distinguishes the events that
    require κ-owner action from those that are routine. Writing all
    events (not just breaches) is a deliberate design choice — the
    audit trail must include the windows in which everything was
    fine, not only the ones where it wasn't.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["dataset_name"],
        "metric": reading.metric,
        "group_a": reading.group_a,
        "group_b": reading.group_b,
        "magnitude": reading.magnitude,
        "threshold": result.threshold,
        "breach": result.breach,
        "direction": result.direction,
        "window_size": reading.window_size,
        "window_start": reading.window_start,
        "window_end": reading.window_end,
        "n_a": reading.n_a,
        "n_b": reading.n_b,
        "kappa_owner": cfg["kappa_owner"],
        "correction_required": result.breach,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event
