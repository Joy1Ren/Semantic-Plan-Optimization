"""F1 over the set of documents a plan returns, for queries a plan answers by dropping the
documents that do not qualify.

A returned document that qualifies is a hit. Where the query also asks for a quote, it is a hit
only if the quote matches the ground truth's text (token Jaccard above a threshold, SPAN_THRESHOLD
unless the benchmark sets its own); one that
fails is a miss, costing recall but not precision.
"""
from __future__ import annotations

from typing import Callable

from agent_cost_model.experiments.scoring.spans import SPAN_THRESHOLD, span_jaccard


def score_returned_set(returned: dict[str, str], wanted: set[str], judgeable: set[str],
                       gold_spans: dict[str, list[str]] | None = None,
                       span_score: Callable[[list[str], str], float] = span_jaccard,
                       threshold: float = SPAN_THRESHOLD) -> dict:
    """`returned` maps each returned document key to its quote ("" when the plan gave none),
    `wanted` is the set that qualifies, and `judgeable` the documents the score is over (the
    population the plan ran on, less any the ground truth cannot settle). With `gold_spans`, a hit
    also needs `span_score(gold, quote) > threshold`."""
    returned_keys = set(returned) & judgeable
    wanted = wanted & judgeable
    found = returned_keys & wanted
    if gold_spans is not None:
        found = {key for key in found
                 if span_score(gold_spans.get(key) or [], returned[key]) > threshold}
    hits = len(found)
    wrong = len(returned_keys - wanted)
    precision = hits / (hits + wrong) if hits + wrong else 0.0
    recall = hits / len(wanted) if wanted else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"score": f1,
            "counts": identification_counts(judgeable, returned_keys, wanted, found,
                                            reason_checked=gold_spans is not None, threshold=threshold)}


def identification_counts(judgeable: set, returned: set, wanted: set, found: set,
                          reason_checked: bool, threshold: float = SPAN_THRESHOLD) -> dict:
    """TP/FP/FN/TN over the judgeable documents. A qualifying document returned with a quote that
    fails the span check is an FN."""
    missed = len(wanted - returned)
    failed_reason = len((returned & wanted) - found)
    counts = {"TP": len(found), "FP": len(returned - wanted), "FN": missed + failed_reason,
              "TN": len(judgeable - returned - wanted)}
    if reason_checked:
        counts["FN"] = {"total": counts["FN"], "not_returned": missed,
                        f"returned_but_reason_jaccard_at_most_{threshold}": failed_reason}
    return counts
