"""Where the per-category CUAD benchmarks read from.

One benchmark.yaml asks every question of the SAME contracts, one per query id, so the dataset, the
optimization subset and the ground truth are declared once. queries.py is what binds a query id to
the question it asks.

The ground truth is cuad/ground_truth/cuad_gt.csv, written by cuad/clean_cuad_gt.py -- one file
for every CUAD benchmark, so a fix to the annotations reaches all of them at once.
"""

from __future__ import annotations

from pathlib import Path

CATEGORIES_DIR = Path(__file__).resolve().parent
CUAD_DIR = CATEGORIES_DIR.parent / "cuad"

GROUND_TRUTH_CSV = CUAD_DIR / "ground_truth" / "cuad_gt.csv"
# idx -> real source filename for all 510 CUAD contracts, shared by every CUAD experiment and
# written by cuad/prepare_cuad_data.py. The plans see only "idx" so they cannot copy a filename
# into an answer.
IDX_TO_NAME_JSON = CUAD_DIR / "ground_truth" / "idx_to_name.json"

# Also declared in benchmark.yaml relative to its own directory; these are for Python callers.
DATASET_DIR = CATEGORIES_DIR / "dataset"
DATASET_CSV = DATASET_DIR / "cuad_clean_140.csv"
DATASUBSET_DIR = CATEGORIES_DIR / "datasubset_opt"

PER_QUERY_GROUND_TRUTH_DIR = CATEGORIES_DIR / "ground_truth"


def per_query_ground_truth(query_id: int) -> Path:
    """The ground truth for one query, written by cuad_categories_gt.py. benchmark.yaml names the
    same file through its own template, so this is for Python callers."""
    return PER_QUERY_GROUND_TRUTH_DIR / f"Q{query_id}_gt.csv"


# What each query id asks, and the categories it reads, live in queries.py.
