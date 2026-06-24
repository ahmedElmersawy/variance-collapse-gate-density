"""gradient_gate — Gradient Gate Collapse framework extensions.

Generalizes the alpha-sigmoid-only Phase 3A profiling in run_experiments.py
(which required replacing every ReLU with a custom AlphaSigmoid to get an
analyzable gate) to REAL, unmodified architectures and a real gradient-
inversion privacy attack. See ROADMAP.md for how this fits the full project.
"""
