# Equilibrium Fairness — Reference Implementation

A runtime governance architecture for monitoring and correcting fairness drift in high-impact AI systems.

**Paper:** Bayolo, M. (2026). *Equilibrium Fairness: A Runtime Governance Architecture for Monitoring and Correcting Fairness Drift in High-Impact AI Systems.* Working Paper v1.1.
**DOI:** [10.5281/zenodo.20547396](https://doi.org/10.5281/zenodo.20547396)
**Licence:** CC BY 4.0

---

## What this is

This repository is the companion reference implementation for the Equilibrium Fairness working paper. It demonstrates that the four-stage architecture can be executed end-to-end against real datasets and that it produces the structured artefacts (drift logs, escalation events, threshold records) on which downstream oversight depends.

The implementation is intentionally minimal: small enough to read end-to-end in twenty minutes, structured so each module maps cleanly to one stage of the architecture.

> **Fairness is not a state — it is a feedback property.**
> A system is not failing because it is imperfect — it is failing when F(t) is decreasing and no κ fires.

---

## Architecture mapping

The prototype implements the four stages of the Equilibrium Fairness loop. Each stage maps to one source module:

```
Stage 1: Initialise (E₀)     →  src/loader.py
    Strip inherited advantage from inputs.
    Protected attributes removed from features, retained as monitoring labels.
              ↓
Stage 2: Monitor (δ)          →  src/monitor.py
    Compute drift signal over rolling windows.
    Demographic parity gap or equalised odds gap (pluggable).
              ↓
Stage 3: Threshold (θ)        →  src/threshold.py
    Compare |δ| to the pre-agreed threshold.
    Breach triggers escalation.
              ↓
Stage 4: Correct (κ)          →  src/escalate.py
    Write a structured escalation event to the audit log.
    Does NOT auto-correct — correction is human-governed.
    This is the architectural realisation of: monitor ≠ decide.
              ↓
Updated System State S(t+1)   →  loop continues
```

The equation organising the architecture:

```
F(t) = E₀ − Σδᵢ(t) + Σκⱼ(t)
```

A system loses fairness through every drift event that goes uncorrected, and recovers fairness through every correction that is actually applied.

---

## Quickstart

```bash
git clone https://github.com/maphibayolo-sys/equilibrium-fairness-reference.git
cd equilibrium-fairness-reference

pip install -r requirements.txt

# Run the hiring demo
python -m src.loop --config configs/hiring_adult.yaml

# Run the lending demo
python -m src.loop --config configs/lending_german.yaml

# Generate the HTML demo report
python -m src.report

# Run tests
pytest -q
```

No network access is required. Both datasets are synthetic (deterministic, reproducible) and ship inside the loader. They are designed to produce realistic disparity patterns so the demo demonstrates non-trivial drift and breach events.

---

## Sample output

```
============================================================
Equilibrium Fairness — run summary
============================================================
  dataset       : hiring_adult
  metric        : demographic_parity_gap
  θ (threshold) : 0.05
  window size   : 500
  windows run   : 3
  θ breaches    : 3  (100.0% of windows)
  audit log     : logs/hiring_escalation.jsonl
  κ owner       : fairness-review@example.org
============================================================

Each breach is one escalation event awaiting human-governed κ.
```

Each window produces one structured JSON Lines event in the escalation log:

```json
{
    "timestamp": "2026-05-20T11:14:03+00:00",
    "dataset": "hiring_adult",
    "metric": "demographic_parity_gap",
    "group_a": "Male",
    "group_b": "Female",
    "magnitude": 0.2340,
    "threshold": 0.05,
    "breach": true,
    "direction": "above",
    "window_size": 500,
    "kappa_owner": "fairness-review@example.org",
    "correction_required": true
}
```

The HTML demo report (`python -m src.report`) opens in any browser and shows per-window drift readings, breach states, and the architecture mapping in a single page.

---

## What this implements

| Stage | Module | What it does |
|---|---|---|
| 1. Initialise (E₀) | `src/loader.py` | Loads the dataset; applies the E₀ policy (drops protected attributes from inputs, retains them as monitoring labels). |
| — | `src/baseline.py` | Trains a logistic-regression baseline on the equalised inputs. |
| 2. Monitor (δ) | `src/monitor.py` | Computes the drift signal over rolling windows. Demographic parity gap and equalised odds gap are pluggable. |
| 3. Threshold (θ) | `src/threshold.py` | Compares \|δ\| to the pre-agreed θ. Returns breach state. |
| 4. Correct (κ) | `src/escalate.py` | On breach, writes a structured event to the escalation log. **Does not auto-correct** — the architecture's load-bearing principle is *monitor ≠ decide*. |
| Orchestration | `src/loop.py` | Wires the four stages together; runs end-to-end. |
| Visual report | `src/report.py` | Reads escalation logs, generates a single-page HTML demo report. |

---

## Repository layout

```
equilibrium-fairness-reference/
├── README.md
├── LICENSE                      (MIT)
├── requirements.txt
├── .gitignore
├── configs/
│   ├── hiring_adult.yaml        demographic parity on sex
│   └── lending_german.yaml      equalised odds on age
├── src/
│   ├── __init__.py
│   ├── loader.py                E₀ — data loading and initialisation
│   ├── baseline.py              logistic-regression baseline
│   ├── monitor.py               δ — drift signal computation
│   ├── threshold.py             θ — escalation check
│   ├── escalate.py              κ — logging stub (no auto-correction)
│   ├── loop.py                  orchestrates stages 1–4
│   └── report.py                HTML demo report generator
├── notebooks/
│   ├── 01_hiring_demo.ipynb
│   ├── 02_lending_demo.ipynb
│   └── 03_cross_domain_comparison.md
├── tests/
│   ├── test_loader.py
│   ├── test_monitor.py
│   ├── test_threshold.py
│   └── test_escalate.py
└── logs/                        (created at runtime)
```

---

## Configuration

All deployment-specific choices are in the YAML configs. The architecture is the same across domains; only the configuration changes.

```yaml
# configs/hiring_adult.yaml
dataset: adult
dataset_name: hiring_adult
e0_policy:
  drop_columns: [sex, race]      # strip from inputs
  protected: sex                 # retain as monitoring label only
group_a: Male
group_b: Female
metric: demographic_parity_gap
window_size: 500
theta: 0.05                      # 5 percentage points
kappa_owner: "fairness-review@example.org"
log_path: ./logs/hiring_escalation.jsonl
```

---

## What this is *not*

- **Not a new fairness metric.** Demographic parity, equalised odds, calibration, and individual fairness remain the relevant point measures. This architecture specifies the *loop*, not the metric.
- **Not a production system.** This is a reference implementation for review. It is small enough to read end-to-end; that is the point.
- **Not an auto-corrector.** The escalation stage *logs and waits*. Correction is the responsibility of the human party named in `kappa_owner`. This is the architectural realisation of *monitor ≠ decide*.

---

## Reading order

1. `src/loop.py` — the four stages, wired together. Start here.
2. `src/monitor.py` — drift signal computation.
3. `src/threshold.py` — the θ check.
4. `src/escalate.py` — what a κ-trigger looks like in code.
5. `configs/hiring_adult.yaml` — what an operator actually configures.

---

## Regulatory alignment

The artefacts this implementation produces map directly onto the operational layer of:

- **EU AI Act** — Article 9 (risk management), Article 14 (human oversight), Article 15 (accuracy and robustness).
- **ISO/IEC 42001** — Clause 6 (planning), Clause 8 (operation), Clause 9 (evaluation), Clause 10 (improvement).
- **NIST AI RMF** — GOVERN, MAP, MEASURE, MANAGE.

See §8 of the accompanying working paper for the full mapping.

---

## Tests

```bash
pytest -q
```

28 tests covering loader, monitor, threshold, and escalation modules. 97% line coverage on `src/`.

---

## Licence

Code: MIT — see `LICENSE`.
Paper: CC BY 4.0.

## Author

Maphi Bayolo · AI Governance Practitioner · ISO/IEC 42001 Foundation
Contact: Maphi.Bayolo@gmail.com
