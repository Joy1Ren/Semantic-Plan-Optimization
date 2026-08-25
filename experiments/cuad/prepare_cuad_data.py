"""Prepare CUAD contract-clause-extraction data (from docetl's reasoning experiments) for
cost-model-agent optimization.

- cuad_small.json (15 documents) -> dataset/cuad_small.csv (full dataset)
- cuad_small_optimize.json (5 documents) -> datasubset_opt/cuad_small_optimize.csv
  (pre-made optimization subset, copied as-is -- not re-subsampled)
- CUAD-master_clauses.csv -> ground_truth/Q1.csv (copied verbatim; quality_evaluator.py's
  CuadEvaluator re-reads the real CUAD_GROUND_TRUTH_CSV path directly for scoring, so this copy
  only needs to exist -- it satisfies CostModelAgent._run_final_evaluation's ground-truth-file
  existence check and non-empty-gt_df requirement)

The source json's "name" field (the real source filename, e.g.
"...ContentLicenseAgreement.txt") is dropped from both dataset CSVs, not just renamed --
the CUAD task prompt (benchmark.yaml) asks a plan to extract "Document Name" as one of its
41 clause categories, and leaving the real filename sitting in an input column would let a
plan "cheat" on that category by just copying it over instead of actually extracting the
title from the document text. In its place, each document gets a small sequential integer
"idx" (assigned once from the 15-document full set, by name, so a document keeps the same
idx whether it appears in the full CSV or the 5-document optimize subset -- the subset is a
strict subset of the full set by name). The idx -> name mapping needed to re-attach "name"
at eval time (see quality_evaluator.py's normalize_eval_df) is written to
CUAD_IDX_TO_NAME_JSON, under ground_truth/ since that directory is already eval-only and
never fed to a plan as input.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from agent_cost_model.paths import (
    CUAD_GROUND_TRUTH_CSV,
    CUAD_IDX_TO_NAME_JSON,
    DOCETL_CUAD_DATA_DIR,
    CUAD_DATASET_DIR,
    CUAD_DATASUBSET_DIR,
    CUAD_GROUND_TRUTH_DIR,
    CUAD_FULL_JSON,
)

JSON_CONVERSIONS = [
    ("cuad_small.json", CUAD_DATASET_DIR / "cuad_small.csv"),
    ("cuad_small_optimize.json", CUAD_DATASUBSET_DIR / "cuad_small_optimize.csv"),
]


def _build_name_to_idx() -> dict[str, int]:
    """Assign every document in the 15-doc full set a small sequential idx, keyed by its
    real "name" (source filename) so the 5-doc optimize subset -- a strict subset of the full
    set by name -- looks up the same idx for a document it shares with the full set."""
    records = json.loads(CUAD_FULL_JSON.read_text())
    names = sorted({r["name"] for r in records})
    return {name: i for i, name in enumerate(names)}


def main() -> None:
    name_to_idx = _build_name_to_idx()

    for source_name, destination_path in JSON_CONVERSIONS:
        source_path = DOCETL_CUAD_DATA_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"CUAD source data not found: {source_path}")

        records = json.loads(source_path.read_text())
        missing = [r["name"] for r in records if r["name"] not in name_to_idx]
        if missing:
            raise ValueError(f"{source_name}: names not found in full dataset: {missing}")

        df = pd.DataFrame(records)
        df["idx"] = df["name"].map(name_to_idx)
        df = df[["document", "idx"]]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination_path, index=False)
        print(f"{source_name}: wrote {len(records)} rows -> {destination_path}")

    CUAD_GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    idx_to_name = {str(idx): name for name, idx in name_to_idx.items()}
    CUAD_IDX_TO_NAME_JSON.write_text(json.dumps(idx_to_name, indent=2))
    print(f"idx -> name mapping: wrote {len(idx_to_name)} entries -> {CUAD_IDX_TO_NAME_JSON}")

    if not CUAD_GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(f"CUAD ground truth not found: {CUAD_GROUND_TRUTH_CSV}")
    gt_destination = CUAD_GROUND_TRUTH_DIR / "Q1.csv"
    gt_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CUAD_GROUND_TRUTH_CSV, gt_destination)
    print(f"CUAD-master_clauses.csv: copied -> {gt_destination}")


if __name__ == "__main__":
    main()
