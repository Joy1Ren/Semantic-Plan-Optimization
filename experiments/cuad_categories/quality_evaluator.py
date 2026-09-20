"""CUAD per-category adapter for plan-quality evaluation.

One evaluator serves every question: the query id it is constructed with picks the question out of
queries.py. run_opt.py reaches an adapter only through its module path (benchmark.yaml's
`quality_adapter`) and passes it no benchmark config, but it does pass the query id.

An extraction query is scored per contract by scoring.py, over the documents the plan ran on. An
identification query asks for the contracts satisfying a condition, which a plan answers by
dropping the rest, so it is scored by F1 over the set of contracts that come back. Where the query
also asks why (queries.py's `evidence`), a returned contract counts as found only if its reason's
token Jaccard with the text CUAD highlighted is above 0.5; one that fails is a miss, costing recall
but not precision. Query 20 scores the party count by how
far it is off, 1 / (1 + |gold - predicted| / gold).

Both read the query's own ground-truth file, ground_truth/Q{id}_gt.csv (cuad_categories_gt.py),
which benchmark.yaml names per query id. Its "name" column is the index and its "span" column, when
present, holds the text to check behind an answer that says the contract states no value; whatever
other column it has is the answer, whatever that column is called.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from agent_cost_model.experiments.cuad_categories.paths import IDX_TO_NAME_JSON, per_query_ground_truth
from agent_cost_model.experiments.cuad_categories.queries import QUERY_BY_ID
from agent_cost_model.experiments.cuad_categories.scoring import (
    SPAN_THRESHOLD,
    party_count_closeness,
    score_document,
    span_jaccard,
)
from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator

FILENAME_COL = "filename"
NAME_COL = "name"
SPAN_COL = "span"
NOT_NEEDED = "not needed"
REASON_COL = "reason"
# cuad_gt.csv keys rows by the CUADv1 title, which is the source filename without ".txt". Only
# that suffix may be stripped: a CUAD filename has dots inside it ("EX-10.27_Affiliate Agreement"),
# so Path(...).stem would cut it in the wrong place.
_TXT_SUFFIX = re.compile(r"\.txt$", re.I)
ANSWER_COL = "answer"
SPANS_COL = "spans"


def normalize_eval_df(df: pd.DataFrame, *_ignored) -> pd.DataFrame:
    """Attach the real source filename, which the plan never sees, so its rows can be joined to
    the ground truth (identical to the cuad adapter's own normalize_eval_df)."""
    if df.empty or FILENAME_COL in df.columns or "idx" not in df.columns:
        return df
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    df = df.copy()
    df[FILENAME_COL] = df["idx"].apply(lambda i: mapping.get(str(i)))
    return df


def _first_column(df: pd.DataFrame, preferred: str, *, avoid: tuple[str, ...]) -> str | None:
    """`preferred` if the plan used that name, else its single other column -- a plan may name the
    column something of its own, and with one answer and one spans column the mapping is clear."""
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if c not in {"idx", FILENAME_COL, *avoid}]
    return candidates[0] if len(candidates) == 1 else None


def _gold_rows(ground_truth_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ground_truth_path, dtype=str).fillna("")


def score_identification(query, plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                        population_path: str | Path | None = None) -> dict:
    """F1 over the contracts a plan returned, against the set in Q{id}_gt.csv -- which lists the
    contracts that qualify, and leaves out the ones whose ground truth cannot settle the query.

    Where that file has a span column, a returned contract that qualifies is a hit only if the
    plan's reason matches the span (token Jaccard above SPAN_THRESHOLD); otherwise it is a miss,
    counted against recall only."""
    gold = _gold_rows(ground_truth_path)
    judgeable = set(_population_keys(population_path)) if population_path else set(gold[NAME_COL])
    wanted = set(gold[NAME_COL]) & judgeable

    returned, reasons = set(), {}
    if not plan_output_df.empty and FILENAME_COL in plan_output_df.columns:
        reason_col = _first_column(plan_output_df, REASON_COL, avoid=())
        for _, row in plan_output_df.iterrows():
            name = _TXT_SUFFIX.sub("", str(row[FILENAME_COL]))
            returned.add(name)
            reason = row.get(reason_col) if reason_col else None
            reasons[name] = reason if isinstance(reason, str) else ""
    returned &= judgeable

    found = returned & wanted
    if SPAN_COL in gold.columns:
        spans = dict(zip(gold[NAME_COL], gold[SPAN_COL]))
        found = {name for name in found
                 if span_jaccard(_gold_span(spans[name]) or [], reasons[name]) > SPAN_THRESHOLD}
    hits = len(found)
    wrong = len(returned - wanted)
    precision = hits / (hits + wrong) if hits + wrong else 0.0
    recall = hits / len(wanted) if wanted else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"score": f1, "documents_scored": len(judgeable), "documents_skipped": 0,
            "returned": len(returned), "wanted": len(wanted),
            "counts": _identification_counts(judgeable, returned, wanted, found,
                                             reason_checked=SPAN_COL in gold.columns)}


def _identification_counts(judgeable: set, returned: set, wanted: set, found: set,
                           reason_checked: bool) -> dict:
    """TP/FP/FN/TN over the judgeable contracts. A qualifying contract returned with a reason that
    fails the span check is an FN."""
    missed = len(wanted - returned)
    failed_reason = len((returned & wanted) - found)
    counts = {"TP": len(found), "FP": len(returned - wanted), "FN": missed + failed_reason,
              "TN": len(judgeable - returned - wanted)}
    if reason_checked:
        counts["FN"] = {"total": counts["FN"], "not_returned": missed,
                        f"returned_but_reason_jaccard_at_most_{SPAN_THRESHOLD}": failed_reason}
    return counts


def _population_keys(population_path: str | Path) -> list[str]:
    return [_TXT_SUFFIX.sub("", str(name)) for name in _population_names(population_path)]


def score_category(category: str, plan_output_df: pd.DataFrame, ground_truth_path: str | Path,
                   population_path: str | Path | None = None, scorer=None) -> dict:
    """The category's quality over the document population: the share of contracts whose answer was
    right (see scoring.py). `scorer(gold, answer)` replaces scoring.score_document for a query that
    grades its answer its own way (query 20's party_count_closeness).

    The population is the dataset the plan RAN OVER, not its output, so a contract an operator
    dropped still counts -- scored as an empty answer, exactly as docetl's CUAD scorer treats a
    missing document.
    """
    ground_truth = _gold_rows(ground_truth_path)
    answer_col = next(c for c in ground_truth.columns if c not in (NAME_COL, SPAN_COL))
    gold = {row[NAME_COL]: row for _, row in ground_truth.iterrows()}

    predictions = {}
    if not plan_output_df.empty and FILENAME_COL in plan_output_df.columns:
        spans_col = _first_column(plan_output_df, SPANS_COL, avoid=(ANSWER_COL,))
        plan_answer_col = _first_column(plan_output_df, ANSWER_COL, avoid=(spans_col or SPANS_COL,))
        for _, row in plan_output_df.iterrows():
            predictions[_TXT_SUFFIX.sub("", str(row[FILENAME_COL]))] = (
                row.get(plan_answer_col) if plan_answer_col else None,
                row.get(spans_col) if spans_col else None)

    names = _population_keys(population_path) if population_path else list(predictions)
    scores, skipped = [], 0
    for name in names:
        row = gold.get(name)
        if row is None:                     # not gradable for this query, so not in its file
            skipped += 1
            continue
        answer, spans = predictions.get(name, (None, None))
        gold_value = _gold_value(row[answer_col])
        score = (scorer(gold_value, answer) if scorer else
                 score_document(category, gold_value, _gold_span(row.get(SPAN_COL, "")), answer, spans))
        if score is None:                   # nothing in this row to score (a law that is no place)
            skipped += 1
            continue
        scores.append(score)
    return {"score": sum(scores) / len(scores) if scores else float("nan"),
            "documents_scored": len(scores), "documents_skipped": skipped}


def _gold_value(cell: str):
    """The answer column of a per-query file: JSON for every category but Parties, whose answer is
    the number of parties."""
    try:
        return json.loads(cell) if cell else None
    except json.JSONDecodeError:
        return int(cell) if str(cell).strip().isdigit() else cell


def _gold_span(cell: str) -> list[str] | None:
    """The span column: the text to check, or None where the file says none is needed."""
    if not cell or cell == NOT_NEEDED:
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


class QualityEvaluator(PlanQualityEvaluator):
    """Scores one of benchmark.yaml's questions, chosen by the query id it is constructed with."""

    def __init__(self, oracle_client, oracle_model: str, query_id: int, llm_judge_dir: str | Path,
                 subset_path: str | Path, ground_truth_path: str | Path | None = None,
                 run_dir: str | Path | None = None, oracle_reasoning_effort: str | None = None,
                 use_oracle_ground_truth: bool = False, **_ignored) -> None:
        try:
            self.QUERY = QUERY_BY_ID[int(query_id)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"query {query_id!r} is not one of this benchmark's questions: "
                f"{sorted(QUERY_BY_ID)} (see queries.py and benchmark.yaml's task_prompts)"
            ) from None
        self.CATEGORY = self.QUERY.category
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=Path(subset_path),
            normalize_df=normalize_eval_df,
            llm_judge_dir=llm_judge_dir,
            run_dir=run_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
            use_oracle_ground_truth=use_oracle_ground_truth,
            ground_truth_path=Path(ground_truth_path or GROUND_TRUTH_CSV),
        )

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        """Asks about the FILE: this benchmark scores from cuad_gt.csv, never from a frame."""
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
        """An oracle-substituted run's output, written in a per-query file's own shape so score_plan
        reads it exactly like Q{id}_gt.csv."""
        names = [_TXT_SUFFIX.sub("", str(n)) for n in df.get(FILENAME_COL, [])]
        if self.QUERY.kind == "identify":    # the set the oracle kept, with its reasons if checked
            out = pd.DataFrame({NAME_COL: names})
            if self.QUERY.evidence:
                reason_col = _first_column(df, REASON_COL, avoid=())
                reasons = df[reason_col] if reason_col else [""] * len(df)
                out[SPAN_COL] = [json.dumps([r] if isinstance(r, str) and r.strip() else [])
                                 for r in reasons]
            out.to_csv(path, index=False)
            return
        spans_col = _first_column(df, SPANS_COL, avoid=(ANSWER_COL,)) or SPANS_COL
        answer_col = _first_column(df, ANSWER_COL, avoid=(spans_col,)) or ANSWER_COL
        spans = df.get(spans_col, pd.Series([""] * len(df)))
        pd.DataFrame({
            NAME_COL: names,
            self.CATEGORY: [json.dumps(a if isinstance(a, str) else "") for a in df.get(answer_col, [])],
            SPAN_COL: [json.dumps([s]) if isinstance(s, str) and s.strip() else NOT_NEEDED
                       for s in spans],
        }).to_csv(path, index=False)

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        if not ground_truth_path:
            raise ValueError(
                "CUAD category scoring needs a ground_truth_path: benchmark.yaml's "
                "ground_truth_path (direct mode) or the run's materialised oracle ground truth."
            )
        if self.QUERY.kind == "identify":
            result = score_identification(self.QUERY, plan_output_df, ground_truth_path, population_path)
            self.last_quality_counts = result["counts"]
            print(f"[QualityEvaluator] {self.QUERY.name}: F1 {result['score']:.4f}, returned "
                  f"{result['returned']} of {result['wanted']} wanted, over "
                  f"{result['documents_scored']} contracts ({result['documents_skipped']} skipped)")
        else:
            scorer = party_count_closeness if self.QUERY.metric == "party_count_relative_error" else None
            result = score_category(self.CATEGORY, plan_output_df, ground_truth_path, population_path,
                                    scorer=scorer)
            print(f"[QualityEvaluator] {self.CATEGORY}: score {result['score']:.4f} over "
                  f"{result['documents_scored']} contracts ({result['documents_skipped']} skipped)")
        return float(result["score"])


__all__ = ["QualityEvaluator", "normalize_eval_df", "score_category", "score_identification"]
