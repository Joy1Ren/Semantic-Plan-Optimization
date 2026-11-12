"""Score retired query 100 (old query 1), sustainability reports with company name and sector,
against ground_truth/Q100_gt.csv.

Joint F1 over the population the plan RAN on, so a report a plan dropped still counts. A row is
correct only if its document is a sustainability report, its company name matches one of the
document's aliases (scoring/aliases.py), and its sector equals the ground truth's. precision =
correct / rows returned (a second row for the same idx is a wrong row), recall = correct /
sustainability reports. Rows are matched to documents by idx.
"""
from __future__ import annotations

import json

import pandas as pd

from agent_cost_model.experiments.scoring.aliases import name_matches
from agent_cost_model.experiments.sustainability.queries import SECTORS

_TITLE_TO_SECTOR = {title: name for name, title in SECTORS.items()}


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def normalize_sector(value) -> str:
    """A plan's sector as a SECTORS name: the name itself in any case or spacing, or its title."""
    text = " ".join(_text(value).lower().split())
    return _TITLE_TO_SECTOR.get(text, text.replace(" ", "_"))


def score_report_sector(plan_output_df: pd.DataFrame, gold: pd.DataFrame, population: set[int], rows) -> dict:
    """Query 1's joint F1 and its counts."""
    gold = gold.set_index("idx")
    judgeable = population & set(gold.index[gold["unusable"].fillna("") == ""])
    reports = {i for i in judgeable if gold.loc[i, "is_sustainability_report"]}

    rows = [(idx, row) for idx, row in rows if idx in judgeable]
    first = {}
    for idx, row in rows:
        first.setdefault(idx, row)
    returned = set(first)
    passed = returned & reports
    name_ok = {i for i in passed
               if name_matches(first[i].get("company_name"), json.loads(gold.loc[i, "company_aliases"]))}
    sector_ok = {i for i in passed
                 if normalize_sector(first[i].get("economic_activity")) == gold.loc[i, "economic_activity"]}
    correct = len(name_ok & sector_ok)

    precision = correct / len(rows) if rows else 0.0
    recall = correct / len(reports) if reports else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    tp, fp = len(passed), len(returned - reports)
    fn, tn = len(reports - returned), len(judgeable - returned - reports)
    counts = {
        "filter": {"accuracy": round((tp + tn) / len(judgeable), 4) if judgeable else None,
                   "TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "company_name": f"{len(name_ok)}/{tp} correct",
        "sector": f"{len(sector_ok)}/{tp} correct",
    }
    return {"score": f1, "documents_scored": len(judgeable), "counts": counts}
