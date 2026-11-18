"""CUAD adapter for plan-quality evaluation. The query id picks the evaluator; every query reads
its own ground_truth/Q{id}_gt.csv (ground_truth/cleaning/per_query_gt.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from benchmarks.cuad.paths import (
    GROUND_TRUTH_CSV,
    IDX_TO_NAME_JSON,
    per_query_ground_truth,
)
from benchmarks.cuad.queries import QUERY_BY_ID, gradable
from benchmarks.cuad.scoring import predictions_from_wide, score_cuad
from benchmarks.scoring.identification import score_returned_set
from raptor.plan_quality_evaluator import PlanQualityEvaluator

# The document identifier, on every frame this module produces or consumes. The dataset CSVs
# carry "idx"; normalize_eval_df maps that back to the real source filename under this column.
FILENAME_COL = "filename"
# cuad_gt.csv keys rows by the CUADv1 title, which is the source filename without ".txt". Only
# that suffix may be stripped: a CUAD filename has dots inside it ("EX-10.27_Affiliate Agreement"),
# so Path(...).stem would cut it in the wrong place.
_TXT_SUFFIX = re.compile(r"\.txt$", re.I)


def normalize_eval_df(df: pd.DataFrame, *_ignored) -> pd.DataFrame:
    """Attach the real source filename, which the plan never sees, so its rows can be joined to
    the ground truth. A plan's input/output carries only 'idx', so it cannot copy a filename into
    the "Document Name" answer; the plan's own columns are kept as they are."""
    if df.empty or FILENAME_COL in df.columns or "idx" not in df.columns:
        return df
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    df = df.copy()
    df[FILENAME_COL] = df["idx"].apply(lambda i: mapping.get(str(i)))
    return df


def QualityEvaluator(query_id: int, **kwargs) -> PlanQualityEvaluator:
    """The evaluator for this query: ExtractionQualityEvaluator for query 1, else
    IdentificationQualityEvaluator."""
    cls = ExtractionQualityEvaluator if int(query_id) == 1 else IdentificationQualityEvaluator
    return cls(query_id=query_id, **kwargs)


# ── Query 1: all 41 categories ──────────────────────────────────────────────────

def _evaluate(
    plan_output_df: pd.DataFrame,
    ground_truth_path: str | Path,
    population_path: str | Path,
) -> dict:
    """score_cuad's full metrics dict for a plan's output, over population_path -- the dataset
    the plan ran over, not its output, so a contract the plan dropped is charged as an empty
    prediction rather than vanishing from the measurement. {} when the output has no idx column
    to tie a row to a contract."""
    if not plan_output_df.empty and "idx" not in plan_output_df.columns:
        return {}
    return score_cuad(predictions_from_wide(plan_output_df), population_path, ground_truth_path)


class ExtractionQualityEvaluator(PlanQualityEvaluator):
    """Query 1's evaluator: scores against CUAD's annotations (Q1_gt.csv, master_clauses.csv's
    rows)."""

    def __init__(
        self,
        query_id: int,
        ground_truth_path: str | Path,
        subset_path: str | Path | None = None,
        **_ignored,
    ) -> None:
        # `**_ignored` absorbs the keywords the engine passes to every adapter's constructor
        # that this one has no use for -- `use_case` and `scale_factor`.
        ground_truth_path = Path(ground_truth_path)
        super().__init__(
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=normalize_eval_df,
            ground_truth_path=ground_truth_path,
        )

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        """CUAD scores from the ground-truth file, so this checks the file; a header-only file
        counts as missing."""
        if not ground_truth_path or not Path(ground_truth_path).exists():
            return False
        try:
            return not pd.read_csv(ground_truth_path, nrows=1).empty
        except Exception as e:
            print(f"[QualityEvaluator] could not read ground truth at {ground_truth_path}: {e}")
            return False

    def write_scoring_input(self, plan_output_df: pd.DataFrame, path: Path) -> None:
        """The idx -> category -> text predictions score_cuad grades."""
        path.write_text(json.dumps(predictions_from_wide(plan_output_df), indent=2))

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        """Score a plan's clause extractions: score_cuad's avg_f1.

        `ground_truth_df` is unused -- score_cuad reads the ground truth from a FILE.
        """
        if not ground_truth_path:
            raise ValueError("CUAD scoring needs a ground_truth_path (benchmark.yaml's ground_truth_path)")
        metrics = _evaluate(plan_output_df, ground_truth_path, population_path)
        if not metrics:
            print("[QualityEvaluator] CUAD: no idx column in the plan's output, score nan")
            return float("nan")
        print(f"[QualityEvaluator] CUAD: avg F1 {metrics['avg_f1']:.4f} over "
              f"{metrics['documents_scored']} contracts")
        return float(metrics["avg_f1"])


# ── Queries 2-8: one question each ──────────────────────────────────────────────

NAME_COL = "name"
SPAN_COL = "span"
REASON_COL = "reason"


def _first_column(df: pd.DataFrame, preferred: str) -> str | None:
    """`preferred` if the plan used that name, else its single other column -- a plan may name its
    reason column something of its own."""
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if c not in {"idx", FILENAME_COL}]
    return candidates[0] if len(candidates) == 1 else None


def _gold_rows(ground_truth_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ground_truth_path, dtype=str).fillna("")


def score_identification(query, plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                        population_path: str | Path | None = None) -> dict:
    """F1 over the contracts a plan returned, against the set in Q{id}_gt.csv, which lists the
    contracts that qualify. A contract whose ground truth cannot settle the query (cuad_gt.csv's
    `unusable`) counts neither way: returning it is not a false positive.

    Where that file has a span column, a returned contract that qualifies is a hit only if the
    plan's reason matches the span (token Jaccard above SPAN_THRESHOLD); otherwise it is a miss,
    counted against recall only."""
    gold = _gold_rows(ground_truth_path)
    judgeable = set(_population_keys(population_path)) if population_path else set(gold[NAME_COL])
    judgeable -= _ungradable(query)

    returned = {}
    if not plan_output_df.empty and FILENAME_COL in plan_output_df.columns:
        reason_col = _first_column(plan_output_df, REASON_COL)
        for _, row in plan_output_df.iterrows():
            reason = row.get(reason_col) if reason_col else None
            returned[_TXT_SUFFIX.sub("", str(row[FILENAME_COL]))] = reason if isinstance(reason, str) else ""

    gold_spans = None
    if SPAN_COL in gold.columns:
        gold_spans = {name: _gold_span(cell) or [] for name, cell in zip(gold[NAME_COL], gold[SPAN_COL])}
    result = score_returned_set(returned, set(gold[NAME_COL]), judgeable, gold_spans)
    return {"score": result["score"], "documents_scored": len(judgeable), "documents_skipped": 0,
            "counts": result["counts"]}


def _ungradable(query) -> set[str]:
    """The contracts whose ground truth cannot settle this query."""
    ground_truth = pd.read_csv(GROUND_TRUTH_CSV, usecols=[NAME_COL, "unusable"])
    return {row[NAME_COL] for _, row in ground_truth.iterrows() if not gradable(query, row)}


def _population_keys(population_path: str | Path) -> list[str]:
    return [_TXT_SUFFIX.sub("", str(name)) for name in _population_names(population_path)]


def _gold_span(cell: str) -> list[str] | None:
    """The span column's text to check, or None when it is empty."""
    if not cell:
        return None
    try:
        spans = json.loads(cell)
    except json.JSONDecodeError:
        return [cell]
    return spans or None


def _population_names(population_path: str | Path) -> list[str]:
    df = pd.read_csv(population_path, dtype={"idx": str})
    if FILENAME_COL in df.columns:
        return df[FILENAME_COL].tolist()
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    return [mapping.get(str(i)) for i in df["idx"]]


class IdentificationQualityEvaluator(PlanQualityEvaluator):
    """Scores one of queries 2-8, chosen by the query id it is constructed with."""

    def __init__(self, query_id: int, ground_truth_path: str | Path | None = None,
                 subset_path: str | Path | None = None, **_ignored) -> None:
        try:
            self.QUERY = QUERY_BY_ID[int(query_id)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"query {query_id!r} is not one of this benchmark's questions: "
                f"{sorted(QUERY_BY_ID)} (see queries.py and benchmark.yaml's task_prompts)"
            ) from None
        super().__init__(
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=normalize_eval_df,
            # benchmark.yaml names the per-query file and the runner passes it; the default is
            # the same file, so constructing an evaluator without one (a standalone script, a
            # test) still scores against this query's own ground truth.
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

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        if not ground_truth_path:
            raise ValueError("CUAD scoring needs a ground_truth_path (benchmark.yaml's "
                             "ground_truth_path)")
        result = score_identification(self.QUERY, plan_output_df, ground_truth_path, population_path)
        counts = self.last_quality_counts = result["counts"]
        fn = counts["FN"]["total"] if isinstance(counts["FN"], dict) else counts["FN"]
        print(f"[QualityEvaluator] {self.QUERY.name}: F1 {result['score']:.4f}, "
              f"(TP, FP, TN, FN) = ({counts['TP']}, {counts['FP']}, {counts['TN']}, {fn}), over "
              f"{result['documents_scored']} contracts ({result['documents_skipped']} skipped)")
        return float(result["score"])


__all__ = ["ExtractionQualityEvaluator", "IdentificationQualityEvaluator", "QualityEvaluator",
           "normalize_eval_df", "score_identification"]
