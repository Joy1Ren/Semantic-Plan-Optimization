"""Prepare SemBench movie data for cost-model optimization."""
from __future__ import annotations

import argparse

import pandas as pd

from agent_cost_model.experiments.SemBench.paths import DATASET_DIR, sembench_files_dir


# The only columns a plan may read. `idx` is the source table's `id` -- the MOVIE a review is
# about, not the row identifier; rows are identified by `reviewId` (see benchmark.yaml's
# movie `id_col`). The dropped columns -- scoreSentiment, originalScore, reviewState,
# isTopCritic -- are pre-computed answers to what the semantic operators must derive.
OUTPUT_COLUMNS = [
    "idx", "reviewId", "creationDate", "criticName", "publicationName", "reviewText", "reviewUrl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-factor", type=int, required=True)
    args = parser.parse_args()

    source_dir = sembench_files_dir() / "movie" / "data" / f"sf_{args.scale_factor}"
    output_dir = DATASET_DIR / "movie" / f"sf_{args.scale_factor}"
    reviews_path = source_dir / "Reviews.csv"
    if not reviews_path.exists():
        raise FileNotFoundError(f"SemBench movie data not found: {reviews_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_csv(reviews_path).rename(columns={"id": "idx"})
    output_path = output_dir / f"Reviews_{args.scale_factor}.csv"
    reviews[OUTPUT_COLUMNS].to_csv(output_path, index=False)
    print(f"Wrote {len(reviews)} rows to {output_path}")


if __name__ == "__main__":
    main()
