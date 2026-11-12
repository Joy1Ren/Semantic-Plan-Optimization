"""Sustainability adapter for plan-quality evaluation, laid out like cuad_categories'.

One evaluator serves every question: the query id it is constructed with picks the question out of
queries.py. run_opt.py reaches an adapter only through its module path (benchmark.yaml's
`quality_adapter`) and passes it no benchmark config, but it does pass the query id.

Every query asks for the documents satisfying a condition, which a plan answers by dropping the
rest, so it is scored by F1 over the set of documents that come back -- the same scorer, and the
same quality_counts, as cuad_categories' identification queries. Where the query's ground-truth
file has a spans column (queries 1-3, 101), a returned document counts as found only if its reason passes quote_score (it
matches a gold span closely, or contains one); one that fails is a miss, costing recall but not
precision.

Retired query 100 (company name and sector) is scored by scoring.py's joint F1.

Every score is over the documents the plan RAN on, so a document it dropped still counts; one the
ground truth marks unusable for the query counts neither way. Each Q{id}_gt.csv has one row per
document (idx, label, [spans,] unusable, note).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.scoring.identification import score_returned_set
from agent_cost_model.experiments.scoring.spans import span_jaccard, token_set
from agent_cost_model.experiments.sustainability.paths import DATASET_CSV, per_query_ground_truth
from agent_cost_model.experiments.sustainability.queries import QUERY_BY_ID
from agent_cost_model.experiments.sustainability.scoring import score_report_sector
from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator

REASON_COL = "reason"
SPANS_COL = "spans"
# A quote passes if it sufficiently proves the fact: token Jaccard above QUOTE_THRESHOLD with a gold
# span, or it contains (CONTAINED_SHARE of the words of) a gold span of at least MIN_GOLD_WORDS words
# and is at most MAX_QUOTE_WORDS words long. The second route keeps a quote that completes the
# sentence around a short gold clause; the cap, just above the longest gold quote any frontier run
# gave (57 words), stops a pasted paragraph from passing just because it contains the gold. A bare
# fragment still needs the Jaccard.
QUOTE_THRESHOLD = 0.5
CONTAINED_SHARE = 0.9
MIN_GOLD_WORDS = 5
MAX_QUOTE_WORDS = 60
# Columns a plan's output may carry besides its answer: the input row's own fields.
_INPUT_COLS = {"idx", "document"}


def population_idxs(population_path: str | Path) -> set[int]:
    return set(pd.read_csv(population_path, usecols=["idx"])["idx"].astype(int))


def _idx(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rows(plan_output_df: pd.DataFrame) -> list[tuple[int, pd.Series]]:
    """(idx, row) for every output row whose idx parses, in output order."""
    if plan_output_df.empty or "idx" not in plan_output_df.columns:
        return []
    return [(idx, row) for _, row in plan_output_df.iterrows() if (idx := _idx(row["idx"])) is not None]


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def quote_score(gold_spans: list[str], quote: str) -> float:
    """1.0 when the quote contains a gold span (see QUOTE_THRESHOLD), else its best Jaccard."""
    words = token_set(quote)
    short_enough = len(quote.split()) <= MAX_QUOTE_WORDS
    best = 0.0
    for span in gold_spans:
        gold = token_set(span)
        if (words and short_enough and len(gold) >= MIN_GOLD_WORDS
                and len(words & gold) / len(gold) >= CONTAINED_SHARE):
            return 1.0
        best = max(best, span_jaccard([span], quote))
    return best


def _first_column(df: pd.DataFrame, preferred: str) -> str | None:
    """`preferred` if the plan used that name, else its single other column -- a plan may name the
    column something of its own, and with one reason column the mapping is clear."""
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if c not in _INPUT_COLS]
    return candidates[0] if len(candidates) == 1 else None


def score_identification(query, plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                         population_path: str | Path | None = None) -> dict:
    """F1 over the documents a plan returned, against the label column of Q{id}_gt.csv."""
    gold = pd.read_csv(ground_truth_path)
    population = population_idxs(population_path or DATASET_CSV)
    judgeable = population & set(gold.loc[gold["unusable"].fillna("") == "", "idx"])
    wanted = set(gold.loc[gold["label"], "idx"])

    returned = {}
    reason_col = _first_column(plan_output_df, REASON_COL) if SPANS_COL in gold.columns else None
    for idx, row in _rows(plan_output_df):
        returned.setdefault(idx, _text(row.get(reason_col)) if reason_col else "")

    if SPANS_COL in gold.columns:
        gold_spans = {idx: json.loads(spans) for idx, spans in zip(gold["idx"], gold[SPANS_COL])}
        result = score_returned_set(returned, wanted, judgeable, gold_spans,
                                    span_score=quote_score, threshold=QUOTE_THRESHOLD)
    else:
        result = score_returned_set(returned, wanted, judgeable)
    return {"score": result["score"], "documents_scored": len(judgeable),
            "documents_skipped": len(population) - len(judgeable), "counts": result["counts"]}


def score_classification(plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                         population_path: str | Path | None = None) -> dict:
    """Retired query 100's joint F1 (scoring.score_report_sector)."""
    population = population_idxs(population_path or DATASET_CSV)
    result = score_report_sector(plan_output_df, pd.read_csv(ground_truth_path), population, _rows(plan_output_df))
    return {**result, "documents_skipped": len(population) - result["documents_scored"]}


def score_query(query, plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                population_path: str | Path | None = None) -> dict:
    if query.kind == "classify":
        return score_classification(plan_output_df, ground_truth_path, population_path)
    return score_identification(query, plan_output_df, ground_truth_path, population_path)


class QualityEvaluator(PlanQualityEvaluator):
    """Scores one of benchmark.yaml's questions, chosen by the query id it is constructed with."""

    def __init__(self, query_id: int, ground_truth_path: str | Path | None = None,
                 oracle_client=None, oracle_model: str | None = None,
                 llm_judge_dir: str | Path | None = None, subset_path: str | Path | None = None,
                 run_dir: str | Path | None = None, oracle_reasoning_effort: str | None = None,
                 use_oracle_ground_truth: bool = False, oracle_operator_quality: bool = True,
                 **_ignored) -> None:
        try:
            self.QUERY = QUERY_BY_ID[int(query_id)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"query {query_id!r} is not one of this benchmark's questions: "
                f"{sorted(QUERY_BY_ID)} (see queries.py and benchmark.yaml's task_prompts)"
            ) from None
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=subset_path,
            llm_judge_dir=llm_judge_dir,
            run_dir=run_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
            use_oracle_ground_truth=use_oracle_ground_truth,
            oracle_operator_quality=oracle_operator_quality,
            ground_truth_path=(
                Path(ground_truth_path) if ground_truth_path
                else per_query_ground_truth(int(query_id))
            ),
        )

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        """Asks about the FILE: this benchmark scores from Q{id}_gt.csv, never from a frame."""
        if not ground_truth_path or not Path(ground_truth_path).exists():
            return False
        try:
            return not pd.read_csv(ground_truth_path, nrows=1).empty
        except Exception as error:
            print(f"[QualityEvaluator] could not read ground truth at {ground_truth_path}: {error}")
            return False

    def write_scoring_input(self, plan_output_df: pd.DataFrame, path: Path) -> None:
        plan_output_df.to_csv(path, index=False)

    def write_oracle_ground_truth(self, df: pd.DataFrame, path: Path) -> None:
        raise NotImplementedError("the sustainability benchmark scores against its own ground truth only")

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        if not ground_truth_path:
            raise ValueError("sustainability scoring needs a ground_truth_path (benchmark.yaml's "
                             "ground_truth_path)")
        result = score_query(self.QUERY, plan_output_df, ground_truth_path, population_path)
        counts = self.last_quality_counts = result["counts"]
        if self.QUERY.kind == "classify":
            print(f"[QualityEvaluator] {self.QUERY.name}: joint F1 {result['score']:.4f}, {counts}")
            return float(result["score"])
        fn = counts["FN"]["total"] if isinstance(counts["FN"], dict) else counts["FN"]
        print(f"[QualityEvaluator] {self.QUERY.name}: F1 {result['score']:.4f}, "
              f"(TP, FP, TN, FN) = ({counts['TP']}, {counts['FP']}, {counts['TN']}, {fn}), over "
              f"{result['documents_scored']} documents ({result['documents_skipped']} skipped)")
        return float(result["score"])


__all__ = ["QualityEvaluator", "quote_score", "score_classification", "score_identification", "score_query"]
