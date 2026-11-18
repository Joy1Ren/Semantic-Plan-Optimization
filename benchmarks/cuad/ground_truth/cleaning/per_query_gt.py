"""Split the CUAD ground truth into one file per query: ground_truth/Q{id}_gt.csv.

Query 1 (all 41 categories) is scored against CUAD's master_clauses.csv as released, so Q1_gt.csv
is that file's rows for the dataset's contracts, unchanged. Queries 2-8 read the clause-level
ground truth (ground_truth/cuad_gt.csv, written by clean_cuad_gt.py): name[, span], one row per
contract that satisfies the query's condition, i.e. the set a plan is scored against by F1. A query
that also asks why a contract qualifies (queries.py's `evidence`) adds "span": the text CUAD
highlighted for that category, which the plan's reason must match.

Contracts whose ground truth cannot settle a query (cuad_gt.csv's `unusable` column) are left out
of that query's file entirely.

    python -m benchmarks.cuad.ground_truth.cleaning.per_query_gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarks.cuad.paths import (
    DATASET_CSV,
    GROUND_TRUTH_CSV,
    GROUND_TRUTH_DIR as OUTPUT_DIR,
    IDX_TO_NAME_JSON,
    MASTER_CLAUSES_CSV,
)
from benchmarks.cuad.queries import QUERIES, Query, gradable
from benchmarks.cuad.scoring import contract_key

NAME_COL = "name"
SPAN_COL = "span"


def _contracts_in_dataset(path: Path) -> set[str]:
    """The contracts the benchmark actually runs on, by name (idx_to_name.json holds filenames)."""
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    return {mapping[str(i)].removesuffix(".txt") for i in pd.read_csv(path, usecols=["idx"])["idx"]}


def build(query: Query, ground_truth: pd.DataFrame) -> pd.DataFrame:
    rows = [row for _, row in ground_truth.iterrows() if gradable(query, row)]
    return pd.DataFrame([{NAME_COL: r[NAME_COL], **({SPAN_COL: r[query.evidence]} if query.evidence else {})}
                         for r in rows if query.holds(r)])


def build_query_1(contracts: set[str] | None) -> pd.DataFrame:
    """master_clauses.csv's rows for these contracts (every row when None), columns untouched."""
    master = pd.read_csv(MASTER_CLAUSES_CSV)
    if contracts is None:
        return master
    wanted = {contract_key(name) for name in contracts}
    rows = master[master["Filename"].map(contract_key).isin(wanted)]
    if len(rows) != len(wanted):
        raise KeyError(f"{len(wanted) - len(rows)} dataset contracts have no master_clauses.csv row")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_CSV,
                        help="clause-level ground truth to split (default: cuad_gt.csv)")
    parser.add_argument("--contracts", type=Path, default=DATASET_CSV,
                        help="dataset CSV whose idx column names the contracts to cover (default: "
                             "the 140 the benchmark runs on); pass --all-contracts to use every row")
    parser.add_argument("--all-contracts", action="store_true",
                        help="cover every contract in the ground-truth file, not just the dataset")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    ground_truth = pd.read_csv(args.ground_truth)
    wanted = None if args.all_contracts else _contracts_in_dataset(args.contracts)
    if wanted is not None:
        missing = wanted - set(ground_truth[NAME_COL])
        if missing:
            raise KeyError(f"{len(missing)} dataset contracts are not in {args.ground_truth}: "
                           f"{sorted(missing)[:3]}")
        ground_truth = ground_truth[ground_truth[NAME_COL].isin(wanted)]

    args.out.mkdir(parents=True, exist_ok=True)
    query_1 = build_query_1(wanted)
    query_1.to_csv(args.out / "Q1_gt.csv", index=False)
    print(f"Q1  cuad_clauses {len(query_1):3d} rows of {MASTER_CLAUSES_CSV.name} -> Q1_gt.csv")
    print(f"{len(ground_truth)} contracts from {args.ground_truth}\n")
    for query in QUERIES:
        table = build(query, ground_truth)
        path = args.out / f"Q{query.id}_gt.csv"
        table.to_csv(path, index=False)
        skipped = len(ground_truth) - sum(1 for _, r in ground_truth.iterrows() if gradable(query, r))
        print(f"Q{query.id:<2} {query.name:<40} {len(table):3d} rows (the contracts that qualify)"
              f"{f', {skipped} not gradable' if skipped else ''}"
              f" -> {path.name}")


if __name__ == "__main__":
    main()
