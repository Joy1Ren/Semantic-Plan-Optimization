"""Write the Summary of a Haystack dataset and ground truth from the SummHay checkout.

For each use case in paths.TOPICS (one haystack each):
  dataset/{use_case}/haystack.csv           idx, document -- one row per haystack document
  ground_truth/{use_case}/Q{k}_gt.csv       one row per insight of subtopic k

Nothing is random: documents and subtopics keep the haystack file's order, so re-running
reproduces every file byte for byte.

idx is SummHay's own citation number: populate_summaries.py labels the i-th document "Article
{i+1}", and utils_eval.build_ref_insight2docids keys documents by i + 1. A summary's [n] therefore
means the same document here as in SummHay, and an optimization subset's rows keep their idx.

Usage: PYTHONPATH=. python -m agent_cost_model.experiments.summary_haystack.prepare_summary_haystack_data
"""
from __future__ import annotations

import json

import pandas as pd

from agent_cost_model.experiments.summary_haystack.paths import (
    TOPICS,
    dataset_csv,
    ground_truth_csv,
    summhay_utils_eval,
    topic_json,
)


def write(path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path}")


def main() -> None:
    build_ref_insight2docids = summhay_utils_eval().build_ref_insight2docids
    for use_case in TOPICS:
        topic = json.loads(topic_json(use_case).read_text())
        write(dataset_csv(use_case), pd.DataFrame({
            "idx": range(1, len(topic["documents"]) + 1),
            "document": [doc["document_text"] for doc in topic["documents"]],
        }))

        insight_docs = build_ref_insight2docids(topic)
        for k, subtopic in enumerate(topic["subtopics"], 1):
            write(ground_truth_csv(use_case, k), pd.DataFrame([{
                "insight_id": insight["insight_id"],
                "insight_name": insight.get("insight_name", ""),
                "insight": insight["insight"],
                "doc_idxs": json.dumps(sorted(insight_docs.get(insight["insight_id"], set()))),
            } for insight in subtopic["insights"]]))


if __name__ == "__main__":
    main()
