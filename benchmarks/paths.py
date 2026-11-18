"""Locations owned by this repository.

Deliberately small: this module knows nothing about any particular benchmark. Per-benchmark
layout (dataset, datasubset, ground truth, results prefix) is declared in that benchmark's
`benchmark.yaml`, and the external checkout it reads from is resolved by
`benchmarks.external_repo`.
"""

from __future__ import annotations

from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
RESULTS_DIR = REPO_ROOT / "results"
ANALYSIS_DIR = REPO_ROOT / "analysis"
