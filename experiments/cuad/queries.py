"""What queries 2-8 ask, and how their ground truth is read out of cuad_gt.csv. Query 1 (all 41
categories at once) is not here: scoring.py grades it against master_clauses.csv's rows.

Two kinds of query are supported:

  extract       pull one category's value out of every contract; scored per contract by
                category_scoring.score_document (the answer, with the quote behind the answers
                that say the contract states no value), or by the query's own metric.
  identify      return the contracts that satisfy a condition; scored by F1 over that set, since a
                plan answers it by dropping the rows that do not qualify. A query with `evidence`
                also asks why, and counts a returned contract as found only if its reason matches
                the text CUAD highlighted for that category.

An identify query names the categories it reads, so a contract whose ground truth for one of them
is not gradable (cuad_gt.csv's `unusable`) is left out of both the gold set and the plan's, rather
than counting as a miss.

The ids here are benchmark.yaml's task_prompts keys; QUERIES is the one place that says which id is
which question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

DAYS = {"days": 1, "weeks": 7, "months": 30, "years": 365}


def answer(row: pd.Series, category: str):
    return json.loads(row[f"{category} Answer"])


def _days(item) -> float | None:
    scale = DAYS.get(item.get("unit")) if isinstance(item, dict) else None
    return scale * item["length"] if scale and isinstance(item.get("length"), (int, float)) else None


def _latest_year(row: pd.Series, category: str) -> int | None:
    """The year of the date the category gives, ignoring a year the contract leaves blank."""
    years = [int(item["year"]) for item in answer(row, category)
             if isinstance(item, dict) and "year" in item and "?" not in item["year"]]
    return max(years) if years else None


def _has_clause(row: pd.Series, category: str) -> bool:
    return row[f"{category} Answer"] == "Yes"


def _has_expiration(row: pd.Series) -> bool:
    """Whether the contract says anything about the end of its initial term -- a date, a term
    length, "perpetual", or a term that ends on an event with no stated length: a non-empty
    Expiration Date answer, the same rule an Expiration Date extraction is scored by."""
    return bool(answer(row, "Expiration Date"))


@dataclass(frozen=True)
class Query:
    id: int
    name: str
    kind: str                                   # "extract" or "identify"
    task: str                                   # one line, for reports
    category: str = ""                          # extract: the category scored
    reads: tuple[str, ...] = ()                 # identify: the categories the condition reads
    conditions: tuple[tuple[str, Callable[[pd.Series], bool]], ...] = field(default=())
    evidence: str = ""                          # identify: the category a returned reason must quote
    metric: str = ""                            # benchmark.yaml's query_metrics name

    def holds(self, row: pd.Series) -> bool:
        return all(condition(row) for _, condition in self.conditions)


def _extract(query_id: int, category: str, task: str, *, name: str = "",
             metric: str = "answer_match_span_jaccard") -> Query:
    return Query(query_id, name or f"cuad_{category.lower().replace(' ', '_')}", "extract", task,
                 category=category, metric=metric)


def _identify(query_id: int, name: str, task: str, *conditions, evidence: str = "") -> Query:
    reads = tuple(dict.fromkeys(category for _, category, _ in conditions))
    return Query(query_id, name, "identify", task, reads=reads,
                 conditions=tuple((label, test) for label, _, test in conditions),
                 evidence=evidence, metric="set_f1_reason_jaccard" if evidence else "set_f1")


# Formerly cuad_categories queries 2, 10, 15, 16, 17, 21 and 22, in this order.
QUERIES = [
    _identify(2, "cuad_parties_over_two", "contracts signed by more than two parties",
              ("more than two parties", "Parties",
               lambda r: (answer(r, "Parties")["number_parties"] or 0) > 2)),
    _identify(3, "cuad_renews_indefinitely", "contracts that keep renewing with no limit",
              ("renews indefinitely", "Renewal Term",
               lambda r: any(isinstance(i, dict) and i.get("renewals") == "unlimited"
                             for i in answer(r, "Renewal Term"))), evidence="Renewal Term"),

    # More than one category: each condition is a separate read of the contract.
    _identify(4, "cuad_many_parties_new_york",
              "contracts with more than two parties that are also governed by New York law",
              ("more than two parties", "Parties",
               lambda r: (answer(r, "Parties")["number_parties"] or 0) > 2),
              ("governed by New York law", "Governing Law",
               lambda r: "New York" in answer(r, "Governing Law"))),
    _identify(5, "cuad_expires_after_2000_with_audit",
              "contracts expiring after 2000 that also grant audit rights",
              ("expires after 2000", "Expiration Date",
               lambda r: (_latest_year(r, "Expiration Date") or 0) > 2000),
              ("has an audit rights clause", "Audit Rights",
               lambda r: _has_clause(r, "Audit Rights"))),
    _identify(6, "cuad_evergreen_with_long_notice",
              "contracts that renew indefinitely and need 90 days' notice or more to stop",
              ("renews indefinitely", "Renewal Term",
               lambda r: any(isinstance(i, dict) and i.get("renewals") == "unlimited"
                             for i in answer(r, "Renewal Term"))),
              ("notice of 90 days or more", "Notice Period to Terminate Renewal",
               lambda r: any((_days(i) or 0) >= 90
                             for i in answer(r, "Notice Period to Terminate Renewal")))),

    # The expiration date, asked only of the contracts that also carry another clause, and graded
    # on the quoted expiration sentence rather than on a parsed answer.
    _identify(7, "cuad_expiration_with_volume_restriction",
              "the expiration date of contracts that also restrict volume",
              ("states its expiration terms", "Expiration Date", _has_expiration),
              ("has a volume restriction clause", "Volume Restriction",
               lambda r: _has_clause(r, "Volume Restriction")),
              evidence="Expiration Date"),
    _identify(8, "cuad_expiration_with_termination_for_convenience",
              "the expiration date of contracts that can also be terminated for convenience",
              ("states its expiration terms", "Expiration Date", _has_expiration),
              ("has a termination for convenience clause", "Termination for Convenience",
               lambda r: _has_clause(r, "Termination for Convenience")),
              evidence="Expiration Date"),
]

QUERY_BY_ID = {query.id: query for query in QUERIES}
# Extraction queries only; the category whose answers a plan's output is scored against.
CATEGORY_BY_QUERY = {q.id: q.category for q in QUERIES if q.kind == "extract"}


def gradable(query: Query, row: pd.Series) -> bool:
    """Whether this contract's ground truth can settle the query at all."""
    unusable = json.loads(row["unusable"])
    categories = (query.category,) if query.kind == "extract" else query.reads
    return not any(category in unusable for category in categories)
