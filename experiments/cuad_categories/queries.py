"""What each query id asks, and how its ground truth is read out of cuad_gt.csv.

Two kinds of query share the benchmark:

  extract       pull one category's value out of every contract; scored per contract by
                scoring.score_document (the answer, with the quote behind the answers that say the
                contract states no value), or by the query's own metric (query 20: how close the
                party count is).
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


# Single-category questions first, each extraction query followed by the question that filters on
# the same category; the questions that read more than one category come last.
QUERIES = [
    _extract(1, "Parties", "how many parties signed the contract"),
    _identify(2, "cuad_parties_over_two", "contracts signed by more than two parties",
              ("more than two parties", "Parties",
               lambda r: (answer(r, "Parties")["number_parties"] or 0) > 2)),

    _extract(3, "Agreement Date", "the date of the contract"),
    _identify(4, "cuad_agreement_before_2010", "contracts dated before 2010",
              ("dated before 2010", "Agreement Date",
               lambda r: (_latest_year(r, "Agreement Date") or 9999) < 2010)),

    _extract(5, "Effective Date", "the date the contract takes effect"),
    _identify(6, "cuad_effective_on_event", "contracts that take effect on an event, not a date",
              ("effective date is an event", "Effective Date",
               lambda r: "relative" in answer(r, "Effective Date")), evidence="Effective Date"),

    _extract(7, "Expiration Date", "when the contract's initial term expires"),
    _identify(8, "cuad_perpetual", "contracts with no expiration",
              ("runs perpetually", "Expiration Date",
               lambda r: "perpetual" in answer(r, "Expiration Date")), evidence="Expiration Date"),

    _extract(9, "Renewal Term", "the renewal term after the initial term expires"),
    _identify(10, "cuad_renews_indefinitely", "contracts that keep renewing with no limit",
              ("renews indefinitely", "Renewal Term",
               lambda r: any(isinstance(i, dict) and i.get("renewals") == "unlimited"
                             for i in answer(r, "Renewal Term"))), evidence="Renewal Term"),

    _extract(11, "Notice Period to Terminate Renewal", "the notice needed to stop a renewal"),
    _identify(12, "cuad_notice_90_days", "contracts needing 90 days' notice or more to stop a renewal",
              ("notice of 90 days or more", "Notice Period to Terminate Renewal",
               lambda r: any((_days(i) or 0) >= 90
                             for i in answer(r, "Notice Period to Terminate Renewal")))),

    _extract(13, "Governing Law", "whose law governs the contract"),
    _identify(14, "cuad_new_york_law", "contracts governed by New York law",
              ("governed by New York law", "Governing Law",
               lambda r: "New York" in answer(r, "Governing Law"))),

    # More than one category: each condition is a separate read of the contract.
    _identify(15, "cuad_many_parties_new_york",
              "contracts with more than two parties that are also governed by New York law",
              ("more than two parties", "Parties",
               lambda r: (answer(r, "Parties")["number_parties"] or 0) > 2),
              ("governed by New York law", "Governing Law",
               lambda r: "New York" in answer(r, "Governing Law"))),
    _identify(16, "cuad_expires_after_2000_with_audit",
              "contracts expiring after 2000 that also grant audit rights",
              ("expires after 2000", "Expiration Date",
               lambda r: (_latest_year(r, "Expiration Date") or 0) > 2000),
              ("has an audit rights clause", "Audit Rights",
               lambda r: _has_clause(r, "Audit Rights"))),
    _identify(17, "cuad_evergreen_with_long_notice",
              "contracts that renew indefinitely and need 90 days' notice or more to stop",
              ("renews indefinitely", "Renewal Term",
               lambda r: any(isinstance(i, dict) and i.get("renewals") == "unlimited"
                             for i in answer(r, "Renewal Term"))),
              ("notice of 90 days or more", "Notice Period to Terminate Renewal",
               lambda r: any((_days(i) or 0) >= 90
                             for i in answer(r, "Notice Period to Terminate Renewal")))),
    _identify(18, "cuad_perpetual_with_insurance",
              "contracts with no expiration that also require insurance",
              ("runs perpetually", "Expiration Date",
               lambda r: "perpetual" in answer(r, "Expiration Date")),
              ("has an insurance clause", "Insurance", lambda r: _has_clause(r, "Insurance"))),
    _identify(19, "cuad_old_with_non_compete",
              "contracts dated before 2010 that also contain a non-compete",
              ("dated before 2010", "Agreement Date",
               lambda r: (_latest_year(r, "Agreement Date") or 9999) < 2010),
              ("has a non-compete clause", "Non-Compete",
               lambda r: _has_clause(r, "Non-Compete"))),

    # Query 1's question again, scored by how far the count is off rather than exact.
    _extract(20, "Parties", "how many parties signed the contract, scored by relative error",
             name="cuad_parties_count_error", metric="party_count_relative_error"),
]

QUERY_BY_ID = {query.id: query for query in QUERIES}
# Extraction queries only; the category whose answers a plan's output is scored against.
CATEGORY_BY_QUERY = {q.id: q.category for q in QUERIES if q.kind == "extract"}


def gradable(query: Query, row: pd.Series) -> bool:
    """Whether this contract's ground truth can settle the query at all."""
    unusable = json.loads(row["unusable"])
    categories = (query.category,) if query.kind == "extract" else query.reads
    return not any(category in unusable for category in categories)
