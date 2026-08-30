"""Prepare SemBench movie data for cost-model optimization."""
from __future__ import annotations

import argparse

import pandas as pd

from agent_cost_model.experiments.SemBench.paths import DATASET_DIR, sembench_files_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-factor", type=int, required=True)
    args = parser.parse_args()

    source_dir = sembench_files_dir() / "movie" / "data" / f"sf_{args.scale_factor}"
    output_dir = DATASET_DIR / "movie" / f"sf_{args.scale_factor}"
    # movies_path = source_dir / "Movies.csv"
    reviews_path = source_dir / "Reviews.csv"
    # if not movies_path.exists() or not reviews_path.exists():
    #     raise FileNotFoundError(f"SemBench movie data not found under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # movies = pd.read_csv(movies_path).rename(columns={"id": "idx"})
    reviews = pd.read_csv(reviews_path).rename(columns={"id": "idx"})
    # movies.drop(columns=["audienceScore", "tomatoMeter", "rating"], errors="ignore").to_csv(
    #     output_dir / f"Movies_{args.scale_factor}.csv", index=False
    # )
    reviews[["idx", "reviewId", "creationDate", "criticName", "publicationName", "reviewText", "reviewUrl"]].to_csv(
        output_dir / f"Reviews_{args.scale_factor}.csv", index=False
    )
    print(f"Wrote movie datasets to {output_dir}")


if __name__ == "__main__":
    main()
