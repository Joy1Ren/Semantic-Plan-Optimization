"""CUAD-local file locations.

These live here rather than in the package-level ``paths.py`` because only this adapter reads
them: the runner gets everything it needs from ``benchmark.yaml``. The docetl checkout they
hang off is resolved from this benchmark's own ``external_repo`` block, so there is one
declaration of where docetl lives.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from agent_cost_model.experiments.external_repo import resolve_repo

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_YAML = BENCHMARK_DIR / "benchmark.yaml"


@lru_cache(maxsize=1)
def docetl_root() -> Path:
    with BENCHMARK_YAML.open() as handle:
        config = yaml.safe_load(handle)
    root = resolve_repo(config.get("external_repo"))
    if root is None:
        raise RuntimeError(f"{BENCHMARK_YAML} declares no external_repo for docetl")
    return root


def cuad_data_dir() -> Path:
    """Root of docetl's CUAD data, holding the train/ and test/ split directories. Which json
    under it is the full dataset and which is the optimization set is declared once, in
    prepare_cuad_data.py's FULL_SOURCE / OPTIMIZE_SOURCE."""
    return docetl_root() / "experiments" / "reasoning" / "data"


# -- inputs read from the docetl checkout (read-only) -------------------------
def cuad_ground_truth_csv() -> Path:
    """The real, human-annotated CUAD clause annotations. One file covers every split.

    Scoring does not read a fixed population file: it derives the document population from
    the plan's own predicted rows (see quality_evaluator.py's _evaluate)."""
    return cuad_data_dir() / "CUAD-master_clauses.csv"


# -- data prepared into this repo (see prepare_cuad_data.py) ------------------
DATASET_DIR = BENCHMARK_DIR / "dataset"
DATASUBSET_DIR = BENCHMARK_DIR / "datasubset_opt"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"

# idx -> real source filename, for re-attaching the filename at eval time only (see
# prepare_cuad_data.py and quality_evaluator.py's normalize_eval_df) -- never exposed to a
# plan, so it can't cheat on the "Document Name" extraction category by copying the input
# row's filename.
IDX_TO_NAME_JSON = GROUND_TRUTH_DIR / "idx_to_name.json"
