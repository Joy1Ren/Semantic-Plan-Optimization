"""Prepare CUAD contract-clause-extraction data (from docetl's reasoning experiments) for
cost-model-agent optimization.

Which json gets converted, and where it lands, is declared once in the four constants below.
Nothing further down names a file, so pointing this at different splits is an edit to those
four lines and nothing else.

The two sources are docetl's own train/test splits, and they are DISJOINT -- the optimize set
is a held-out set to optimize against, not a strict subset of the full set. Every document's
idx is therefore assigned from the two sources POOLED (see _build_name_to_idx), so an
optimize-set document still gets an idx even though it never appears in the full set.

The source json's "name" field (the real source filename, e.g.
"...ContentLicenseAgreement.txt") is dropped from both dataset CSVs, not just renamed --
the CUAD task prompt (benchmark.yaml) asks a plan to extract "Document Name" as one of its
41 clause categories, and leaving the real filename sitting in an input column would let a
plan "cheat" on that category by just copying it over instead of actually extracting the
title from the document text. In its place, each document gets a small sequential integer
"idx", assigned by name over the pooled sources so a document keeps one idx no matter which
CSV it appears in. The idx -> name mapping needed to re-attach "name" at eval time (see
quality_evaluator.py's normalize_eval_df) is written to IDX_TO_NAME_JSON, under ground_truth/
since that directory is already eval-only and never fed to a plan as input.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.cuad.paths import (
    DATASET_DIR,
    DATASUBSET_DIR,
    GROUND_TRUTH_DIR,
    IDX_TO_NAME_JSON,
    cuad_data_dir,
    cuad_ground_truth_csv,
)

# -- what to prepare, declared once -------------------------------------------
# Source json, relative to the docetl checkout's experiments/reasoning/data/, paired with the
# CSV it is written to in this repo. The CSV names are deliberately independent of the source
# split so repointing a source below moves nothing downstream -- but benchmark.yaml's
# source_csv / subset_path and run_single_plan.py name these same two CSVs, so renaming one
# means updating those too.
FULL_SOURCE = "test/cuad.json"       # 100 documents
FULL_CSV = DATASET_DIR / "cuad.csv"
OPTIMIZE_SOURCE = "train/cuad.json"  # 40 documents, disjoint from FULL_SOURCE
OPTIMIZE_CSV = DATASUBSET_DIR / "cuad_optimize.csv"

CONVERSIONS = [
    (FULL_SOURCE, FULL_CSV),
    (OPTIMIZE_SOURCE, OPTIMIZE_CSV),
]


def _source_path(relative_source: str) -> Path:
    path = cuad_data_dir() / relative_source
    if not path.exists():
        raise FileNotFoundError(f"CUAD source data not found: {path}")
    return path


def _load(relative_source: str) -> list[dict]:
    return json.loads(_source_path(relative_source).read_text())


def _build_name_to_idx(records_by_source: dict[str, list[dict]]) -> dict[str, int]:
    """Assign every document a small sequential idx, keyed by its real "name" (source
    filename).

    Names from every source are pooled before sorting, rather than taken from the full set
    alone: FULL_SOURCE and OPTIMIZE_SOURCE are disjoint splits, so an idx map built from the
    full set would cover none of the optimize set. Pooling also means a document appearing in
    both sources resolves to the same idx in both CSVs.
    """
    names = sorted({r["name"] for records in records_by_source.values() for r in records})
    return {name: i for i, name in enumerate(names)}


def main() -> None:
    records_by_source = {source: _load(source) for source, _ in CONVERSIONS}
    name_to_idx = _build_name_to_idx(records_by_source)

    for source, destination_path in CONVERSIONS:
        records = records_by_source[source]
        df = pd.DataFrame(records)
        df["idx"] = df["name"].map(name_to_idx)
        df = df[["document", "idx"]]
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination_path, index=False)
        print(f"{source}: wrote {len(records)} rows -> {destination_path}")

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    idx_to_name = {str(idx): name for name, idx in name_to_idx.items()}
    IDX_TO_NAME_JSON.write_text(json.dumps(idx_to_name, indent=2))
    print(f"idx -> name mapping: wrote {len(idx_to_name)} entries -> {IDX_TO_NAME_JSON}")

    if not cuad_ground_truth_csv().exists():
        raise FileNotFoundError(f"CUAD ground truth not found: {cuad_ground_truth_csv()}")
    print(f"ground truth: read in place from {cuad_ground_truth_csv()} (not copied)")


if __name__ == "__main__":
    main()
