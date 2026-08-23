"""Stable locations for the standalone cost-model repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _default_sembench_root() -> Path:
    """Prefer the published submodule layout while supporting this checkout."""
    for candidate in (
        WORKSPACE_ROOT / "third_party" / "SemBench",
        WORKSPACE_ROOT / "SemBench",
    ):
        if candidate.exists():
            return candidate
    return WORKSPACE_ROOT / "third_party" / "SemBench"


SEMBENCH_ROOT = Path(os.environ.get("SEMBENCH_ROOT", _default_sembench_root())).expanduser().resolve()
SEMBENCH_SRC_DIR = SEMBENCH_ROOT / "src"
SEMBENCH_FILES_DIR = SEMBENCH_ROOT / "files"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
DATASET_DIR = EXPERIMENTS_DIR / "dataset"
DATASUBSET_DIR = EXPERIMENTS_DIR / "datasubset"
RESULTS_DIR = PACKAGE_ROOT / "results"
ANALYSIS_DIR = PACKAGE_ROOT / "analysis"


def ensure_sembench_src_on_path() -> Path:
    """Make SemBench's scenario/evaluator modules importable when available."""
    src_dir = str(SEMBENCH_SRC_DIR)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return SEMBENCH_SRC_DIR
