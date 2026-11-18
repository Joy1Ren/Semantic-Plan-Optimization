"""Build the 140 contracts every CUAD query runs on, and the idx -> name map they are keyed by.

  ground_truth/idx_to_name.json   idx -> source filename for all 510 CUAD contracts
  dataset/cuad_clean_140.csv      document, idx -- the 140 contracts the queries run on

A plan never sees a contract's real filename (e.g. "...ContentLicenseAgreement.txt"): query 1 asks
for "Document Name", and a filename in an input column would let a plan copy it instead of reading
the title from the text. Each contract gets a small integer idx instead, and the map re-attaches
the name at eval time (quality_evaluator.normalize_eval_df). It lives under ground_truth/ since
that directory is eval-only and never fed to a plan.

"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.cuad.category_scoring import NON_PLACE
from agent_cost_model.experiments.cuad.paths import (
    CUAD_JSON,
    DATASET_CSV,
    GROUND_TRUTH_CSV,
    IDX_TO_NAME_JSON,
    SOURCE_JSON,
)

SEED = 20260912


def _source_names(source_json: Path) -> list[str]:
    """docetl's 140 contracts, by source filename: the start of the 140, and of the idx numbering."""
    return [record["name"] for record in json.loads(source_json.read_text())]


def build_name_to_idx(source_json: Path) -> dict[str, int]:
    """Assign every one of CUAD's 510 contracts a small sequential idx, keyed by its real "name"
    (source filename).

    docetl's 140 come first, sorted. The rest of CUADv1 follows, sorted, so adding them never
    renumbered a document already in use (or in the embedding cache keyed by idx).
    """
    names = sorted(set(_source_names(source_json)))
    titles = [f"{doc['title']}.txt" for doc in json.loads(CUAD_JSON.read_text())["data"]]
    names += sorted(set(titles) - set(names))
    return {name: i for i, name in enumerate(names)}


def _gradable_everywhere(row: pd.Series) -> bool:
    if json.loads(row["unusable"]):
        return False
    law = json.loads(row["Governing Law Answer"])
    return not (law and all(place == NON_PLACE for place in law))


def build_dataset(source_json: Path, ground_truth: pd.DataFrame,
                  idx_by_name: dict[str, int]) -> pd.DataFrame:
    """docetl's 140 contracts, with every one not gradable everywhere replaced (see above)."""
    keep = {row["name"] for _, row in ground_truth.iterrows() if _gradable_everywhere(row)}
    documents = {doc["title"]: doc["paragraphs"][0]["context"]
                 for doc in json.loads(CUAD_JSON.read_text())["data"]}
    source = [os.path.splitext(name)[0] for name in _source_names(source_json)]

    current = set(source)
    drop = sorted(name for name in current if name not in keep)
    pool = sorted(keep - current)
    random.seed(SEED)
    replacement = dict(zip(drop, random.sample(pool, len(drop))))
    print(f"{len(drop)} of the {len(current)} contracts replaced, from {len(pool)} that every query "
          f"can be scored on\n")
    for old, new in replacement.items():
        print(f"  {old[:62]}\n    -> {new[:62]}")

    names = [replacement.get(n, n) for n in source]
    if len(set(names)) != len(names) or not set(names) <= keep:
        raise RuntimeError("the rebuilt set is not 140 distinct, fully gradable contracts")
    frame = pd.DataFrame([{"document": documents[n], "idx": idx_by_name[n]} for n in names])
    return frame.sort_values("idx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=SOURCE_JSON)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_CSV)
    args = parser.parse_args()

    name_to_idx = build_name_to_idx(args.source)
    IDX_TO_NAME_JSON.parent.mkdir(parents=True, exist_ok=True)
    IDX_TO_NAME_JSON.write_text(json.dumps({str(i): name for name, i in name_to_idx.items()}, indent=2))
    print(f"idx -> name mapping: wrote {len(name_to_idx)} entries -> {IDX_TO_NAME_JSON}\n")

    idx_by_name = {os.path.splitext(name)[0]: i for name, i in name_to_idx.items()}
    frame = build_dataset(args.source, pd.read_csv(args.ground_truth), idx_by_name)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATASET_CSV, index=False)
    print(f"\nwrote {len(frame):3d} rows -> {DATASET_CSV}")
    print("\nnow rebuild the per-query ground truth: "
          "python -m agent_cost_model.experiments.cuad.ground_truth.cleaning.per_query_gt")


if __name__ == "__main__":
    main()
