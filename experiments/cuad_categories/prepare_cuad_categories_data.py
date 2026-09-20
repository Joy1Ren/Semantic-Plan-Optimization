"""Build the 140 contracts these queries run on.

Starts from docetl's own CUAD splits (100 test + 40 optimize) and replaces every contract whose
ground truth cannot be graded, so each of benchmark.yaml's queries can be scored on all 140:

  * anything cuad_gt.csv's `unusable` column names, in any category -- an answer with no supporting
    span, a span with no answer, two different values for one question, a renewal count the text
    does not settle (see clean_cuad_gt.py);
  * a governing law that is only "non-place" ("the state in which the breach occurs"), which leaves
    no jurisdiction to compare a plan's answer against.

Replacements are drawn at random, with SEED fixed, from the contracts that pass both tests and are
not already in the splits. Contracts keep their split, so the optimize subset stays 40.

As in cuad/prepare_cuad_data.py, the CSVs carry only `document` and `idx`: a plan never sees the
real filename, so it cannot answer a question by copying it. idx comes from the idx -> name map
cuad/prepare_cuad_data.py writes for all 510 contracts, so run that first.

    python -m agent_cost_model.experiments.cuad_categories.prepare_cuad_categories_data
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.cuad_categories.paths import (
    CUAD_DIR,
    DATASET_CSV,
    DATASUBSET_DIR,
    GROUND_TRUTH_CSV,
    IDX_TO_NAME_JSON,
)
from agent_cost_model.experiments.cuad_categories.scoring import NON_PLACE

SEED = 20260912
CUAD_JSON = CUAD_DIR / "CUADv1.json"
OPTIMIZE_CSV = DATASUBSET_DIR / "cuad_optimize.csv"
# docetl's splits, relative to its CUAD data directory; the source of the 140.
SPLITS = {"test": "test/cuad.json", "train": "train/cuad.json"}


def _gradable_everywhere(row: pd.Series) -> bool:
    if json.loads(row["unusable"]):
        return False
    law = json.loads(row["Governing Law Answer"])
    return not (law and all(place == NON_PLACE for place in law))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--docetl-data", type=Path,
                        default=CUAD_DIR.parents[2] / "docetl" / "experiments" / "reasoning" / "data",
                        help="docetl's CUAD data directory, holding the train/ and test/ splits")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_CSV)
    args = parser.parse_args()

    ground_truth = pd.read_csv(args.ground_truth)
    keep = {row["name"] for _, row in ground_truth.iterrows() if _gradable_everywhere(row)}
    documents = {doc["title"]: doc["paragraphs"][0]["context"]
                 for doc in json.loads(CUAD_JSON.read_text())["data"]}
    splits = {name: [os.path.splitext(record["name"])[0]
                     for record in json.loads((args.docetl_data / path).read_text())]
              for name, path in SPLITS.items()}

    current = {name for names in splits.values() for name in names}
    drop = sorted(name for name in current if name not in keep)
    pool = sorted(keep - current)
    random.seed(SEED)
    replacement = dict(zip(drop, random.sample(pool, len(drop))))
    print(f"{len(drop)} of the {len(current)} contracts replaced, from {len(pool)} that every query "
          f"can be scored on\n")
    for old, new in replacement.items():
        print(f"  {old[:62]}\n    -> {new[:62]}")

    splits = {name: [replacement.get(n, n) for n in names] for name, names in splits.items()}
    names = splits["test"] + splits["train"]
    if len(set(names)) != len(names) or not set(names) <= keep:
        raise RuntimeError("the rebuilt split is not 140 distinct, fully gradable contracts")

    idx_by_name = {os.path.splitext(name)[0]: int(i)
                   for i, name in json.loads(IDX_TO_NAME_JSON.read_text()).items()}
    for rows, path in ((names, DATASET_CSV), (splits["train"], OPTIMIZE_CSV)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([{"document": documents[n], "idx": idx_by_name[n]} for n in rows])
        frame.sort_values("idx").to_csv(path, index=False)
        print(f"\nwrote {len(frame):3d} rows -> {path}")
    print("\nnow rebuild the per-query ground truth: "
          "python -m agent_cost_model.experiments.cuad_categories.cuad_categories_gt")


if __name__ == "__main__":
    main()
