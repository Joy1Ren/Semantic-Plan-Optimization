"""Split the CUAD ground truth into one file per query: ground_truth/Q{id}_gt.csv.

Reads the clause-level ground truth (cuad/ground_truth/cuad_gt.csv, written by
cuad/clean_cuad_gt.py) and writes what each of benchmark.yaml's queries is scored against, so a
query's ground truth can be read on its own without knowing how the other eighteen work.

  extraction query   name, {category}, span -- one row per contract. {category} holds the answer;
                     an empty answer ("[]") is an answer, meaning the contract has no such clause.
                     A checker should take the one column that is neither "name" nor "span" as the
                     answer, whatever it is called. "span" holds the text CUAD highlighted, but
                     only where the answer alone cannot show the plan read the right clause -- an
                     answer of "unspecified", "redacted" or "relative", which says the contract
                     states no value. Elsewhere it reads "not needed", or "[]" where the contract
                     has no such clause.
  identification     name[, span]      -- one row per contract that satisfies the condition, i.e.
                     the set a plan is scored against by F1. Contracts that do not qualify are
                     absent rather than listed as false. A query that also asks why a contract
                     qualifies (queries.py's `evidence`) adds "span": the text CUAD highlighted for
                     that category, which the plan's reason must match.

Contracts whose ground truth cannot settle a query (cuad_gt.csv's `unusable` column) are left out
of that query's file entirely, since neither answering nor skipping them can be graded.

The answer written for each extraction query is the value the scorer compares against: the number
of parties for queries 1 and 20, and cuad_gt.csv's own JSON answer for the rest.

    python -m agent_cost_model.experiments.cuad_categories.cuad_categories_gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.cuad_categories.paths import (
    CATEGORIES_DIR,
    DATASET_CSV,
    GROUND_TRUTH_CSV,
    IDX_TO_NAME_JSON,
)
from agent_cost_model.experiments.cuad_categories.queries import QUERIES, Query, gradable
from agent_cost_model.experiments.cuad_categories.scoring import SPAN_BACKED

OUTPUT_DIR = CATEGORIES_DIR / "ground_truth"
NAME_COL = "name"
SPAN_COL = "span"
# Governing Law and Parties are judged on the answer alone (a place list, a count), so their rows
# never carry a span to check.
SPAN_CHECKED_CATEGORIES = {"Effective Date", "Expiration Date", "Renewal Term",
                           "Notice Period to Terminate Renewal"}
NOT_NEEDED = "not needed"


def _contracts_in_dataset(path: Path) -> set[str]:
    """The contracts the benchmark actually runs on, by name (idx_to_name.json holds filenames)."""
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    return {mapping[str(i)].removesuffix(".txt") for i in pd.read_csv(path, usecols=["idx"])["idx"]}


def _answer(query: Query, row: pd.Series):
    answer = json.loads(row[f"{query.category} Answer"])
    if query.category == "Parties":          # the query asks for the count, not the names
        return answer["number_parties"]
    return json.dumps(answer, ensure_ascii=False)


def _span(query: Query, row: pd.Series) -> str:
    """The text a checker needs for this row, or why it needs none."""
    answer = json.loads(row[f"{query.category} Answer"])
    if not answer:                           # the contract has no such clause
        return "[]"
    if query.category not in SPAN_CHECKED_CATEGORIES:
        return NOT_NEEDED
    items = answer if isinstance(answer, list) else [answer]
    if not any(isinstance(item, str) and item in SPAN_BACKED for item in items):
        return NOT_NEEDED                    # the answer itself settles it
    return row[query.category]               # cuad_gt.csv's own JSON list of spans


def build(query: Query, ground_truth: pd.DataFrame) -> pd.DataFrame:
    rows = [row for _, row in ground_truth.iterrows() if gradable(query, row)]
    if query.kind == "extract":
        return pd.DataFrame([{NAME_COL: r[NAME_COL], query.category: _answer(query, r),
                              SPAN_COL: _span(query, r)} for r in rows])
    return pd.DataFrame([{NAME_COL: r[NAME_COL], **({SPAN_COL: r[query.evidence]} if query.evidence else {})}
                         for r in rows if query.holds(r)])


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
    if not args.all_contracts:
        wanted = _contracts_in_dataset(args.contracts)
        missing = wanted - set(ground_truth[NAME_COL])
        if missing:
            raise KeyError(f"{len(missing)} dataset contracts are not in {args.ground_truth}: "
                           f"{sorted(missing)[:3]}")
        ground_truth = ground_truth[ground_truth[NAME_COL].isin(wanted)]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ground_truth)} contracts from {args.ground_truth}\n")
    for query in QUERIES:
        table = build(query, ground_truth)
        path = args.out / f"Q{query.id}_gt.csv"
        table.to_csv(path, index=False)
        skipped = len(ground_truth) - (len(table) if query.kind == "extract"
                                       else sum(1 for _, r in ground_truth.iterrows() if gradable(query, r)))
        print(f"Q{query.id:<2} {query.name:<40} {len(table):3d} rows"
              f"{'' if query.kind == 'extract' else ' (the contracts that qualify)'}"
              f"{f', {skipped} not gradable' if skipped else ''}"
              f" -> {path.name}")


if __name__ == "__main__":
    main()
