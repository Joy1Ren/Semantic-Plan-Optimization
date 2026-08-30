"""Prepare SemBench e-commerce query datasets for cost-model optimization."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from agent_cost_model.experiments.SemBench.paths import DATASET_DIR, sembench_files_dir


def _type_name(value) -> str | None:
    return value.get("typeName") if isinstance(value, dict) else None


def _query_subset(df: pd.DataFrame, query_id: int) -> pd.DataFrame:
    """Apply only deterministic query predicates; semantic work stays with the agent."""
    if query_id == 4:
        return df[df["baseColour"].isin(["Black", "Blue", "Red", "White", "Orange", "Green"])]
    if query_id in (5, 6):
        apparel = df["masterCategory"].map(_type_name).eq("Apparel")
        excluded = df["subCategory"].map(_type_name).isin(["Saree", "Apparel Set", "Loungewear and Nightwear"])
        return df[apparel & ~excluded]
    if query_id == 7:
        return df[df["price"] <= 500]
    if query_id == 8:
        description = df["productDescriptors"].map(
            lambda value: value.get("description", {}).get("value", "") if isinstance(value, dict) else ""
        )
        return df[description.str.len() >= 3000]
    if query_id == 9:
        colors = ["Black", "Blue", "Red", "White", "Orange", "Green"]
        return df[df["baseColour"].isin(colors) & df["colour1"].fillna("").eq("") & df["colour2"].fillna("").eq("")]
    if query_id == 10:
        return df[df["baseColour"].isin(["Black", "Blue", "Red", "White"]) & (df["price"] <= 1000)]
    if query_id == 12:
        return df[df["masterCategory"].map(_type_name).isin(["Accessories", "Apparel", "Footwear"])]
    if query_id == 14:
        return df[df["price"] < 130]
    return df


def _copy_images(source_dir: Path, destination_dir: Path) -> None:
    """Copy the scale factor's shared image directory once for every query dataset."""
    if not source_dir.exists():
        return
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--queries", type=int, nargs="+", default=range(1, 15))
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    source_dir = sembench_files_dir() / "ecomm" / "data" / f"sf_{args.scale_factor}"
    source_path = source_dir / "styles_details.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"SemBench source data not found: {source_path}")
    destination_dir = DATASET_DIR / "ecomm" / f"sf_{args.scale_factor}"
    destination_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_parquet(source_path).rename(columns={"id": "idx"})
    if not args.no_images:
        _copy_images(source_dir / "images", destination_dir / "images")
    for query_id in args.queries:
        subset = _query_subset(full_df, query_id)
        subset.to_csv(destination_dir / f"styles_details_Q{query_id}.csv", index=False)
        print(f"Q{query_id}: wrote {len(subset)} rows")


if __name__ == "__main__":
    main()
