"""Prepare SemBench e-commerce query datasets for cost-model optimization."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tomllib

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
    if query_id == 8:
        # A product can carry productDescriptors with a null "description" entry, so the inner
        # get needs a fallback of its own, not just a default on the outer one.
        description = df["productDescriptors"].map(
            lambda value: (value.get("description") or {}).get("value", "") if isinstance(value, dict) else ""
        )
        return df[description.str.len() >= 3000]
    if query_id == 9:
        # q9.toml: "Restrict items to a single base color only to make inference easier."
        # Its price cap is NOT applied -- see PRICE_PREDICATE_QUERIES.
        colors = ["Black", "Blue", "Red", "White", "Orange", "Green"]
        return df[df["baseColour"].isin(colors) & df["colour1"].fillna("").eq("") & df["colour2"].fillna("").eq("")]
    if query_id == 12:
        return df[df["masterCategory"].map(_type_name).isin(["Accessories", "Apparel", "Footwear"])]
    return df


# The columns a plan may read: `idx` (the source `id`) plus the product text and price. The other
# styles_details columns (baseColour, masterCategory, ...) answer what the semantic operators must
# derive, so they are dropped once _query_subset has used them.
TEXT_COLUMNS = ["idx", "price", "productDisplayName", "productDescriptors"]
# An image-only query gets the id alone, so its answer must come from the pixels.
IMAGE_ONLY_COLUMNS = ["idx"]
# ...except q9, whose query caps the price: it gets `price` and no price predicate in
# _query_subset, so where the cheap price filter goes stays the optimizer's choice.
PRICE_PREDICATE_QUERIES = {9}


def _output_columns(query_id: int) -> list[str]:
    """Text queries get the product text; image-only queries get the id (plus price if the
    query caps it). Read from q{id}.toml's modalities, as run_opt.py does for the sampler."""
    path = sembench_files_dir() / "ecomm" / "queries" / f"q{query_id}.toml"
    with path.open("rb") as handle:
        modalities = tomllib.load(handle)["metadata"].get("modalities", [])
    if "text" in modalities:
        return TEXT_COLUMNS
    return IMAGE_ONLY_COLUMNS + (["price"] if query_id in PRICE_PREDICATE_QUERIES else [])


def _copy_images(source_dir: Path, destination_dir: Path) -> None:
    """Copy the scale factor's shared image directory once for every query dataset."""
    if not source_dir.exists():
        return
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 3, 4, 5, 6, 7, 8, 9, 12, 13])
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    source_dir = sembench_files_dir() / "ecomm" / "data" / f"sf_{args.scale_factor}"
    source_path = source_dir / "styles_details.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"SemBench source data not found: {source_path}")
    destination_dir = DATASET_DIR / "ecomm" / f"sf_{args.scale_factor}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    # Everything goes to the shared folder; config.expose_query_data copies what a run's query
    # may see into destination_dir.
    shared_dir = destination_dir.with_name(f"{destination_dir.name}_shared")
    shared_dir.mkdir(exist_ok=True)

    full_df = pd.read_parquet(source_path).rename(columns={"id": "idx"})
    if not args.no_images:
        _copy_images(source_dir / "images", shared_dir / "images")
    for query_id in args.queries:
        columns = _output_columns(query_id)
        subset = _query_subset(full_df, query_id)[columns]
        subset.to_csv(shared_dir / f"styles_details_{args.scale_factor}_Q{query_id}.csv", index=False)
        print(f"Q{query_id}: wrote {len(subset)} rows, columns={columns}")


if __name__ == "__main__":
    main()
