"""report.py — generate a single-page HTML demo report.

Reads one or more escalation logs and produces a visual summary
showing the architecture's four stages, per-window drift readings,
breach states, and the mapping to the Equilibrium Fairness
architecture.

Usage:
    python -m src.report                         # all logs in ./logs/
    python -m src.report logs/hiring_escalation.jsonl
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

TEAL = "#007B8A"
SLATE = "#2C3E50"
AMBER = "#D85A30"
GREEN = "#1D9E75"


def load_events(log_path: Path) -> list[dict]:
    events = []
    for line in log_path.read_text().strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))
    return events


def render_dataset_section(events: list[dict]) -> str:
    if not events:
        return ""
    cfg = events[0]
    dataset = cfg["dataset"]
    metric = cfg["metric"]
    theta = cfg["threshold"]
    kappa_owner = cfg["kappa_owner"]
    breaches = [e for e in events if e["breach"]]

    rows = ""
    for e in events:
        mag = abs(e["magnitude"])
        is_breach = e["breach"]
        status_color = AMBER if is_breach else GREEN
        status_text = "⚠ BREACH" if is_breach else "✓ within"
        rows += f"""
        <tr>
            <td>{e['window_start']}–{e['window_end']}</td>
            <td>{e['group_a']} vs {e['group_b']}</td>
            <td><strong>{mag:.4f}</strong></td>
            <td>{theta}</td>
            <td style="color:{status_color};font-weight:600">{status_text}</td>
            <td>{e['n_a']}</td>
            <td>{e['n_b']}</td>
        </tr>"""

    breach_rate = len(breaches) / len(events) * 100 if events else 0

    return f"""
    <div class="dataset-card">
        <h2>{dataset}</h2>
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Metric (δ)</div>
                <div class="metric-val">{metric.replace('_', ' ')}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Threshold (θ)</div>
                <div class="metric-val">{theta}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Windows</div>
                <div class="metric-val">{len(events)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Breaches</div>
                <div class="metric-val" style="color:{AMBER if breaches else GREEN}">{len(breaches)} ({breach_rate:.0f}%)</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">κ owner</div>
                <div class="metric-val" style="font-size:13px">{kappa_owner}</div>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Window</th>
                    <th>Groups</th>
                    <th>|δ|</th>
                    <th>θ</th>
                    <th>Status</th>
                    <th>n(A)</th>
                    <th>n(B)</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""


def render_html(datasets: dict[str, list[dict]]) -> str:
    sections = ""
    for name, events in datasets.items():
        sections += render_dataset_section(events)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Equilibrium Fairness — Demo Report</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; color: {SLATE}; padding: 2rem; line-height: 1.5; }}
    .header {{ max-width: 900px; margin: 0 auto 2rem; }}
    .header h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
    .header .subtitle {{ color: #6c757d; font-size: 0.95rem; }}
    .arch-box {{ max-width: 900px; margin: 0 auto 2rem; background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .arch-box h3 {{ margin-bottom: 1rem; color: {TEAL}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px; }}
    .arch-stages {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .arch-stage {{ border-radius: 6px; padding: 12px; text-align: center; }}
    .arch-stage .num {{ font-size: 11px; font-weight: 600; opacity: 0.7; }}
    .arch-stage .name {{ font-size: 15px; font-weight: 600; }}
    .arch-stage .sym {{ font-size: 13px; opacity: 0.8; margin-top: 2px; }}
    .s1 {{ background: #E6F1FB; color: #0C447C; }}
    .s2 {{ background: #E1F5EE; color: #085041; }}
    .s3 {{ background: #FAEEDA; color: #633806; }}
    .s4 {{ background: #FAECE7; color: #712B13; }}
    .eq {{ text-align: center; font-size: 1.2rem; font-weight: 500; margin: 1rem 0 0.5rem; padding: 0.8rem; background: #f0f4f8; border-radius: 6px; }}
    .eq-sub {{ text-align: center; font-size: 0.85rem; color: #6c757d; margin-bottom: 0.5rem; }}
    .dataset-card {{ max-width: 900px; margin: 0 auto 2rem; background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .dataset-card h2 {{ font-size: 1.2rem; margin-bottom: 1rem; }}
    .metrics-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.2rem; }}
    .metric-box {{ background: #f8f9fa; border-radius: 6px; padding: 10px 14px; flex: 1; min-width: 120px; }}
    .metric-label {{ font-size: 11px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.3px; }}
    .metric-val {{ font-size: 16px; font-weight: 600; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #dee2e6; font-weight: 600; color: #6c757d; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.3px; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
    .footer {{ max-width: 900px; margin: 2rem auto 0; font-size: 0.8rem; color: #adb5bd; text-align: center; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Equilibrium Fairness — Demo Report</h1>
        <div class="subtitle">Reference implementation output · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>
    </div>

    <div class="arch-box">
        <h3>Architecture mapping</h3>
        <div class="arch-stages">
            <div class="arch-stage s1"><div class="num">Stage 1</div><div class="name">Initialise</div><div class="sym">E₀</div></div>
            <div class="arch-stage s2"><div class="num">Stage 2</div><div class="name">Monitor</div><div class="sym">δ</div></div>
            <div class="arch-stage s3"><div class="num">Stage 3</div><div class="name">Threshold</div><div class="sym">θ</div></div>
            <div class="arch-stage s4"><div class="num">Stage 4</div><div class="name">Correct</div><div class="sym">κ</div></div>
        </div>
        <div class="eq">F(t) = E₀ − Σδᵢ(t) + Σκⱼ(t)</div>
        <div class="eq-sub">starting equity − uncorrected drift + applied corrections</div>
    </div>

    {sections}

    <div class="footer">
        Equilibrium Fairness · DOI: 10.5281/zenodo.20547396 · CC BY 4.0 · Maphi Bayolo
    </div>
</body>
</html>"""


def main():
    log_dir = Path("./logs")
    paths = []

    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        if log_dir.exists():
            paths = sorted(log_dir.glob("*.jsonl"))

    if not paths:
        print("No escalation logs found. Run the loop first:")
        print("  python -m src.loop --config configs/hiring_adult.yaml")
        sys.exit(1)

    datasets = {}
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            events = load_events(p)
            if events:
                datasets[p.stem] = events

    if not datasets:
        print("Logs found but empty. Run the loop first.")
        sys.exit(1)

    html = render_html(datasets)
    out = Path("demo_report.html")
    out.write_text(html)
    print(f"Report written to {out}")
    print(f"  datasets: {', '.join(datasets.keys())}")
    print(f"  open {out}")


if __name__ == "__main__":
    main()
