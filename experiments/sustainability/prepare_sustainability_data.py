"""Build the 140 documents the sustainability queries run on, and keep the source labels apart.

docetl's sustainability splits (train 40, test 100) carry each document's labels next to its text
(document_category, economic_activity, company_name). A plan that sees them can copy the answer, so
the dataset CSV keeps only `document` and `idx`, and the labels go to
ground_truth/original_labels.csv. idx is the source dataset's own id.

    python -m agent_cost_model.experiments.sustainability.prepare_sustainability_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.sustainability.paths import (
    DATASET_CSV,
    DOCETL_DATA_DIR,
    OPTIMIZE_CSV,
    ORIGINAL_LABELS_CSV,
)

SPLITS = {"train": "train/sustainability.json", "test": "test/sustainability.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--docetl-data", type=Path, default=DOCETL_DATA_DIR,
                        help="docetl's reasoning data directory, holding the train/ and test/ splits")
    args = parser.parse_args()

    records = [dict(record, split=split)
               for split, path in SPLITS.items()
               for record in json.loads((args.docetl_data / path).read_text())]
    ids = [int(r["id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("the train and test splits share a document id")

    documents = pd.DataFrame([{"document": r["tot_text_raw"], "idx": int(r["id"])} for r in records])
    labels = pd.DataFrame([{
        "idx": int(r["id"]),
        "split": r["split"],
        "document_category": r["document_category"],
        "economic_activity": r["economic_activity"],
        "company_name": r["company_name"],
    } for r in records])
    train = set(labels.loc[labels["split"] == "train", "idx"])

    for frame, path in ((documents, DATASET_CSV),
                        (documents[documents["idx"].isin(train)], OPTIMIZE_CSV),
                        (labels, ORIGINAL_LABELS_CSV)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values("idx").to_csv(path, index=False)
        print(f"wrote {len(frame):3d} rows -> {path}")


if __name__ == "__main__":
    main()
