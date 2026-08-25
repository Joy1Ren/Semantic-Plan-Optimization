"""Stable locations for the standalone cost-model repository."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def _default_sembench_root() -> Path:
    """Return the SemBench checkout that lives alongside this repository."""
    return WORKSPACE_ROOT / "SemBench"


SEMBENCH_ROOT = Path(os.environ.get("SEMBENCH_ROOT", _default_sembench_root())).expanduser().resolve()
SEMBENCH_SRC_DIR = SEMBENCH_ROOT / "src"
SEMBENCH_FILES_DIR = SEMBENCH_ROOT / "files"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
SEMBENCH_EXPERIMENT_DIR = EXPERIMENTS_DIR / "SemBench"
SEMBENCH_DATASET_DIR = SEMBENCH_EXPERIMENT_DIR / "dataset"
SEMBENCH_DATASUBSET_DIR = SEMBENCH_EXPERIMENT_DIR / "datasubset_opt"
RESULTS_DIR = PACKAGE_ROOT / "results"
ANALYSIS_DIR = PACKAGE_ROOT / "analysis"

DOCETL_ROOT = Path(os.environ.get("DOCETL_ROOT", WORKSPACE_ROOT / "docetl")).expanduser().resolve()
DOCETL_CUAD_DATA_DIR = DOCETL_ROOT / "experiments" / "reasoning" / "data" / "train"

CUAD_EXPERIMENT_DIR = EXPERIMENTS_DIR / "cuad"
CUAD_DATASET_DIR = CUAD_EXPERIMENT_DIR / "dataset"
CUAD_DATASUBSET_DIR = CUAD_EXPERIMENT_DIR / "datasubset_opt"
CUAD_GROUND_TRUTH_DIR = CUAD_EXPERIMENT_DIR / "ground_truth"

CUAD_GROUND_TRUTH_CSV = DOCETL_ROOT / "experiments" / "reasoning" / "data" / "CUAD-master_clauses.csv"
CUAD_OPTIMIZE_JSON = DOCETL_CUAD_DATA_DIR / "cuad_small_optimize.json"  # 5-doc search-time population
CUAD_FULL_JSON = DOCETL_CUAD_DATA_DIR / "cuad_small.json"  # 15-doc final-eval population
# idx -> real source filename, for re-attaching "name" at eval time only (see prepare_cuad_data.py
# and quality_evaluator.py's normalize_eval_df) -- never exposed to a plan, so it can't cheat on
# the "Document Name" extraction category by just copying the input row's filename.
CUAD_IDX_TO_NAME_JSON = CUAD_GROUND_TRUTH_DIR / "idx_to_name.json"


def ensure_sembench_src_on_path() -> Path:
    """Make SemBench's scenario/evaluator modules importable when available."""
    src_dir = str(SEMBENCH_SRC_DIR)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return SEMBENCH_SRC_DIR
