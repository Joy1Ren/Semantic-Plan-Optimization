"""The CUAD scorer both optimizers are graded by: the cost model agent through quality_evaluator.py,
MOAR through docetl/cuad_run.py.

It is docetl's own scorer (docetl/experiments/reasoning/evaluation/cuad.py, evaluate_results) with
the matching made exact. Kept as docetl has it: the 41 categories; text cleaned to letters, digits
and whitespace, a list joined into one text; per contract and category a true positive when both
sides are non-empty and their word-set Jaccard is > 0.15, a false positive when only the
prediction is, a false negative when only the truth is; precision 0 with no predictions, recall
NaN with no positives; score = avg_f1, the mean of the per-category F1s that are not NaN.

Changed:
  * Contracts are matched by idx -> contract name. docetl sorts both sides by a fuzzy filename
    match and pairs them by row position, so one duplicated or missing output row shifts every
    later contract onto another contract's ground truth.
  * Categories are matched by name, after dropping case and treating space, - and / alike, where
    docetl takes the Levenshtein-closest clause_type.
  * Every span a plan gives a category counts -- several rows for one contract, or several
    entries with one clause_type -- where docetl keeps the first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agent_cost_model.experiments.cuad.paths import IDX_TO_NAME_JSON

CATEGORIES = [
    "document_name", "parties", "agreement_date", "effective_date",
    "expiration_date", "renewal_term", "notice_period_to_terminate_renewal",
    "governing_law", "most_favored_nation", "non_compete", "exclusivity",
    "no_solicit_of_customers", "competitive_restriction_exception",
    "no_solicit_of_employees", "non_disparagement", "termination_for_convenience",
    "rofr_rofo_rofn", "change_of_control", "anti_assignment",
    "revenue_profit_sharing", "price_restrictions", "minimum_commitment",
    "volume_restriction", "ip_ownership_assignment", "joint_ip_ownership",
    "license_grant", "non_transferable_license", "affiliate_license_licensor",
    "affiliate_license_licensee", "unlimited_all_you_can_eat_license",
    "irrevocable_or_perpetual_license", "source_code_escrow",
    "post_termination_services", "audit_rights", "uncapped_liability",
    "cap_on_liability", "liquidated_damages", "warranty_duration",
    "insurance", "covenant_not_to_sue", "third_party_beneficiary",
]
JACCARD_GATE = 0.15


def category_key(name: str) -> str:
    """'Unlimited/All-You-Can-Eat-License' -> 'unlimited_all_you_can_eat_license'."""
    return re.sub(r"[ /-]", "_", str(name).strip().lower())


def contract_key(name: str) -> str:
    """One key for a contract however it is spelled: master_clauses.csv names the PDF
    ("MACY'S,INC_...AGREEMENT.PDF"), idx_to_name.json the text file ("MACY_S,INC_...AGREEMENT.txt")."""
    stem = re.sub(r"\.(pdf|txt)'?\s*$", "", str(name).strip(), flags=re.I)
    return re.sub(r"[^A-Z0-9]", "", stem.upper())


def clean_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(value)).strip()


def _jaccard(a: str, b: str) -> float:
    words_a, words_b = set(a.lower().split()), set(b.lower().split())
    union = words_a | words_b
    return len(words_a & words_b) / len(union) if union else 0.0


def load_ground_truth(path: str | Path) -> dict[str, dict[str, str]]:
    """contract key -> category -> cleaned truth, from a wide CSV keyed by "Filename": CUAD's
    master_clauses.csv, or an oracle ground truth in the same layout."""
    df = pd.read_csv(path)
    columns = {category_key(c): c for c in df.columns}
    missing = [c for c in CATEGORIES if c not in columns]
    if missing:
        raise ValueError(f"{path} has no column for {missing}")
    return {contract_key(row["Filename"]): {c: clean_text(row[columns[c]]) for c in CATEGORIES}
            for _, row in df.iterrows()}


def _idx(value) -> str:
    return str(int(float(value)))


def predictions_from_wide(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """idx -> category -> text, from the cost model agent's output: one column per category."""
    columns = {category_key(c): c for c in df.columns}
    predictions: dict[str, dict[str, list[str]]] = {}
    for _, row in df.iterrows():
        if pd.isna(row["idx"]):
            continue
        spans = predictions.setdefault(_idx(row["idx"]), {})
        for category in CATEGORIES:
            if category in columns and (text := clean_text(row[columns[category]])):
                spans.setdefault(category, []).append(text)
    return {i: {c: " ".join(t) for c, t in spans.items()} for i, spans in predictions.items()}


def predictions_from_clauses(records: list[dict]) -> dict[str, dict[str, str]]:
    """idx -> category -> text, from MOAR's output: a `clauses` list of {clause_type, text_span}."""
    predictions: dict[str, dict[str, list[str]]] = {}
    for record in records:
        if pd.isna(record.get("idx")):
            continue
        spans = predictions.setdefault(_idx(record["idx"]), {})
        clauses = record.get("clauses")
        for clause in clauses if isinstance(clauses, list) else []:
            if not isinstance(clause, dict):
                continue
            category = category_key(clause.get("clause_type", ""))
            if category in CATEGORIES and (text := clean_text(clause.get("text_span"))):
                spans.setdefault(category, []).append(text)
    return {i: {c: " ".join(t) for c, t in spans.items()} for i, spans in predictions.items()}


def score_cuad(predictions: dict[str, dict[str, str]], population_path: str | Path,
               ground_truth_path: str | Path) -> dict:
    """docetl's CUAD metrics over the contracts in population_path, the CSV the plan ran on. A
    contract the plan returned nothing for is an empty prediction in every category."""
    names = json.loads(Path(IDX_TO_NAME_JSON).read_text())
    truth = load_ground_truth(ground_truth_path)
    population = [_idx(i) for i in pd.read_csv(population_path, usecols=["idx"])["idx"]]
    no_truth = [names[i] for i in population if contract_key(names[i]) not in truth]
    if no_truth:
        raise ValueError(f"{ground_truth_path} has no row for {no_truth}")

    per_metric = {}
    for category in CATEGORIES:
        tp = fp = fn = tn = 0
        for i in population:
            gold = truth[contract_key(names[i])][category]
            pred = predictions.get(i, {}).get(category, "")
            if gold and pred:
                tp, fp = (tp + 1, fp) if _jaccard(gold, pred) > JACCARD_GATE else (tp, fp + 1)
            elif pred:
                fp += 1
            elif gold:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        if np.isnan(recall):
            f1 = float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_metric[category] = {"precision": precision, "recall": recall, "f1": f1,
                                "counts": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}}

    precisions = [m["precision"] for m in per_metric.values()]
    recalls = [m["recall"] for m in per_metric.values()]
    return {
        "avg_precision": float(np.nanmean(precisions)),
        "avg_recall": float(np.nanmean(recalls)),
        "avg_f1": float(np.nanmean([m["f1"] for m in per_metric.values()])),
        "nan_fraction": float(np.mean(np.isnan(recalls))),
        "documents_scored": len(population),
        "per_metric": per_metric,
    }


__all__ = ["CATEGORIES", "score_cuad", "predictions_from_wide", "predictions_from_clauses",
           "load_ground_truth"]
