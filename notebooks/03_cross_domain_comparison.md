# 03 — Cross-Domain Comparison

This note compares the two configurations shipped with the reference implementation, alongside five further domains from §6 of the working paper. The comparison makes the cross-domain structural-universality claim tangible: in each case, the same four-stage architecture applies with only configuration changing.

## The two configured domains

|  | Hiring (Adult) | Lending (German Credit) |
|---|---|---|
| **E₀** | Drop `sex`, `race` from inputs; retain `sex` as monitoring label | Drop `age_binarised` from inputs; retain as monitoring label |
| **δ** | Demographic parity gap on shortlisting | Equalised odds gap on credit grant |
| **θ** | 0.05 (5 pp) | 0.10 (10 pp) |
| **Window** | 500 decisions | 100 decisions |
| **κ-owner** | `fairness-review@example.org` | `credit-fairness@example.org` |

The two configs are *interchangeable* with respect to the architecture: the same `src/loop.py` reads either YAML and runs the same four-stage loop. That is the point.

## Extending the implementation to other domains

Each of the seven domains from §6.1 of the working paper can be configured by writing a YAML and (for the loader) registering a new dataset reader. The architecture itself does not change.

### Healthcare

```yaml
dataset: <your_clinical_dataset>
dataset_name: triage
e0_policy:
  drop_columns: [insurance_status, race]
  protected: race
group_a: "White"
group_b: "Black"
metric: equalised_odds_gap
theta: 0.05
kappa_owner: "clinical-quality@hospital.org"
```

The δ here is the gap in **treatment-recommendation rates** between groups, controlling for clinical severity (i.e. the y_true labels). The κ-owner is the clinical quality lead; the corrective action on breach is a protocol review, not a model retrain.

### Platform moderation

```yaml
dataset: <your_moderation_dataset>
dataset_name: content_moderation
e0_policy:
  drop_columns: [language, locale]
  protected: language
group_a: "en"
group_b: "non_en"
metric: demographic_parity_gap
theta: 0.03
kappa_owner: "trust-and-safety@platform.org"
```

The δ here is the gap in **enforcement rates** between language groups. The κ-owner is the head of trust and safety; the corrective action on breach is a policy recalibration or moderator training intervention.

### Criminal justice — risk assessment

```yaml
dataset: <your_recidivism_dataset>
dataset_name: criminal_risk_assessment
e0_policy:
  drop_columns: [race, neighbourhood]
  protected: race
group_a: "White"
group_b: "Black"
metric: equalised_odds_gap
theta: 0.02     # very tight — high-stakes domain
kappa_owner: "judicial-review-board@court.org"
```

The δ here is the gap in **risk-score distribution** between racial groups, with particular attention to false-positive rates (the well-known finding from the ProPublica COMPAS audit). The κ-owner is a judicial review board; the corrective action on breach is a mandatory recalibration cycle. The threshold is tight because the stakes are high.

### AI model governance

```yaml
dataset: <your_model_inference_logs>
dataset_name: model_inference_governance
e0_policy:
  drop_columns: [user_segment]
  protected: user_segment
group_a: "enterprise"
group_b: "consumer"
metric: equalised_odds_gap
theta: 0.05
kappa_owner: "ai-governance@company.org"
```

Here the framework is applied recursively — monitoring an AI model's *own* performance drift across user segments. The δ is the gap in accuracy or precision between segments; the κ-owner is the AI governance function; the corrective action on breach is a retraining trigger or a deployment restriction.

### Education

```yaml
dataset: <your_student_outcomes>
dataset_name: education_outcomes
e0_policy:
  drop_columns: [school_funding_tier, socioeconomic_status]
  protected: socioeconomic_status
group_a: "low_ses"
group_b: "high_ses"
metric: demographic_parity_gap
theta: 0.05
kappa_owner: "academic-equity@institution.org"
```

The δ here is the gap in **progression rates** between socioeconomic groups. The κ-owner is the academic equity lead; the corrective action on breach is targeted student support or resource reallocation.

## What stays the same

Across all seven configurations, **the architecture is identical**: `src/loader.py`, `src/monitor.py`, `src/threshold.py`, `src/escalate.py`, `src/loop.py`. Adding a new domain requires:

1. A new dataset reader registered in `loader._load_raw`.
2. A new YAML config file in `configs/`.
3. Optionally, a new metric registered in `monitor.METRICS`.

Nothing in the runtime loop changes. That is the structural universality claim — the framework's load-bearing claim — made operational.

## What necessarily *does* change between domains

The configuration. And, in practice, the political economy: the standing of the threshold-setter, the standing of the κ-owner, the cycle time of correction, the legal constraints on what κ may do. The architecture does not pretend these are the same across domains — it provides the operational skeleton on which each domain's specific political economy is hung.
