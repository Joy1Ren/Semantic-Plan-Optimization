"""Stable locations for the standalone cost-model repository."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SEMBENCH_ROOT = Path(os.environ.get("SEMBENCH_ROOT", WORKSPACE_ROOT / "SemBench"))
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
DATASET_DIR = EXPERIMENTS_DIR / "dataset"
DATASUBSET_DIR = EXPERIMENTS_DIR / "datasubset"
RESULTS_DIR = PACKAGE_ROOT / "results"
ANALYSIS_DIR = PACKAGE_ROOT / "analysis"
