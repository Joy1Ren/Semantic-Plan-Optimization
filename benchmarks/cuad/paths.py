"""CUAD file locations, for Python callers; the runner gets its paths from benchmark.yaml.

Query 1 extracts all 41 categories; queries 2-8 each ask one question of the same 140 contracts
(queries.py). Every query is scored against its own ground_truth/Q{id}_gt.csv, written by
ground_truth/cleaning/per_query_gt.py from cuad_gt.csv (queries 2-8) or CUAD's master_clauses.csv
(query 1).
"""

from __future__ import annotations

from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent

DATASET_DIR = BENCHMARK_DIR / "dataset"
SOURCE_DIR = DATASET_DIR / "source"
SOURCE_JSON = SOURCE_DIR / "cuad.json"
MASTER_CLAUSES_CSV = SOURCE_DIR / "CUAD-master_clauses.csv"
DATASET_CSV = DATASET_DIR / "cuad_clean_140.csv"
DATASUBSET_DIR = BENCHMARK_DIR / "datasubset_opt"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"
CLEANING_DIR = GROUND_TRUTH_DIR / "cleaning"

CUAD_JSON = CLEANING_DIR / "CUADv1.json"
CATEGORY_DESCRIPTIONS_CSV = CLEANING_DIR / "category_descriptions.csv"
# The clause-level ground truth for all 510 contracts, written by cleaning/clean_cuad_gt.py.
GROUND_TRUTH_CSV = GROUND_TRUTH_DIR / "cuad_gt.csv"
ISSUES_CSV = GROUND_TRUTH_DIR / "cuad_gt_issues.csv"
# idx -> real source filename for all 510 contracts, written by prepare_cuad_data.py. The
# plans see only "idx", so they cannot copy a filename into the "Document Name" answer.
IDX_TO_NAME_JSON = GROUND_TRUTH_DIR / "idx_to_name.json"


def per_query_ground_truth(query_id: int) -> Path:
    """The ground truth for one query; benchmark.yaml names the same file through its template."""
    return GROUND_TRUTH_DIR / f"Q{query_id}_gt.csv"
