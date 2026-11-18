"""Build the 140 documents the sustainability queries run on, and keep the source labels apart.

docetl's 140 sustainability documents (its test 100 + train 40 splits, joined in
dataset/source/sustainability.json) carry each document's labels next to its text (document_category, economic_activity, company_name). A plan that sees them can copy the answer, so
the dataset CSV keeps only `document` and `idx`, and the labels go to
ground_truth/llm_grading/original_labels.csv. idx is the source dataset's own id.

    python -m agent_cost_model.experiments.sustainability.prepare_sustainability_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.sustainability.paths import (
    DATASET_CSV,
    ORIGINAL_LABELS_CSV,
    SOURCE_JSON,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=SOURCE_JSON,
                        help="docetl's 140 documents (default: dataset/source/sustainability.json)")
    args = parser.parse_args()

    records = json.loads(args.source.read_text())
    ids = [int(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two source documents share an id")

    documents = pd.DataFrame([{"document": r["tot_text_raw"], "idx": int(r["id"])} for r in records])
    labels = pd.DataFrame([{
        "idx": int(r["id"]),
        "document_category": r["document_category"],
        "economic_activity": r["economic_activity"],
        "company_name": r["company_name"],
    } for r in records])

    for frame, path in ((documents, DATASET_CSV), (labels, ORIGINAL_LABELS_CSV)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values("idx").to_csv(path, index=False)
        print(f"wrote {len(frame):3d} rows -> {path}")


if __name__ == "__main__":
    main()
