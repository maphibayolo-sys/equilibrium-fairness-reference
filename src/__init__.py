"""Equilibrium Fairness — reference implementation.

A runtime governance architecture for fairness drift monitoring and
correction in high-impact AI systems.

Stages
------
1. Initialise (E₀)  — loader.py
2. Monitor    (δ)   — monitor.py
3. Threshold  (θ)   — threshold.py
4. Correct    (κ)   — escalate.py

Orchestration: loop.py
"""
__version__ = "1.0.0"
