"""Locations owned by this repository.

Deliberately small: this module knows nothing about any particular benchmark. Per-benchmark
layout (dataset, datasubset, ground truth, results prefix) is declared in that benchmark's
`benchmark.yaml`, and the external checkout it reads from is resolved by
`agent_cost_model.experiments.external_repo`.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
RESULTS_DIR = PACKAGE_ROOT / "results"
ANALYSIS_DIR = PACKAGE_ROOT / "analysis"
