"""CUAD adapter for oracle/direct-ground-truth plan-quality evaluation.

Reuses docetl's own CUAD scorer (experiments.reasoning.evaluation.cuad.evaluate_results)
unmodified rather than re-implementing its clause-matching logic, so cost-model-agent quality
scores stay bug-for-bug consistent with existing docetl MOAR baseline runs -- including its
known column-name-matching gap: un-normalized hyphens in the raw ground-truth CSV headers
(e.g. "Non-Compete" -> "non-compete") mean 14 of the 41 clause-type metrics never match their
ground-truth column and always score NaN (excluded from avg_f1). This is intentional, not a
bug in this adapter -- see the raw CUAD-master_clauses.csv columns vs.
docetl/experiments/reasoning/evaluation/cuad.py's `metrics` list.

CUAD has real, human-annotated ground truth (CUAD-master_clauses.csv). use_oracle_ground_truth
is caller-controlled (run_opt.py's --use-oracle-ground-truth / --no-use-oracle-ground-truth
flag; default True, matching CostModelAgent's own default): False scores against that real
ground truth, True scores against an oracle-substituted pipeline run's own output instead.
CuadEvaluator._evaluate_single_query detects which one it received by shape -- gt_df with a
"clauses" column is oracle output (already run through normalize_eval_df by
OracleQualityEvaluator._get_oracle_context), gt_df without one is the real-ground-truth
placeholder -- and pivots oracle output into the same wide, evaluate_results-shaped CSV
CUAD-master_clauses.csv itself uses, so both modes reuse the exact same clause-matching logic
(see _write_oracle_ground_truth_csv). This also means oracle mode inherits the same 14/41
column-name-matching gap described above, since that's driven by evaluate_results' own metric
name list, not by which CSV supplied the ground truth.

The CUAD task prompt (benchmark.yaml) asks the plan to emit one wide column per clause
category (matching how the CostModelAgent's sem_map cols naturally work: one named,
described field per category, rather than a single nested list field). normalize_eval_df
reshapes that wide output into evaluate_results' expected [{clause_type, text_span}, ...]
"clauses" list before CuadEvaluator ever sees it.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent_cost_model.oracle_quality_evaluator import OracleQualityEvaluator
from agent_cost_model.paths import (
    CUAD_FULL_JSON,
    CUAD_GROUND_TRUTH_CSV,
    CUAD_IDX_TO_NAME_JSON,
    CUAD_OPTIMIZE_JSON,
    DOCETL_ROOT,
    CUAD_DATASUBSET_DIR,
)

_OPTIMIZE_DOC_COUNT = 5  # len(CUAD_OPTIMIZE_JSON) -- see _pick_population_json
# "idx" is the dataset's real identifier column (see prepare_cuad_data.py); "id" is kept here
# defensively in case a plan re-derives its own differently-named identifier column.
_ID_LIKE_COLUMNS = {"name", "idx", "id", "document", "filename"}

_idx_to_name_cache: dict[str, str] | None = None


def _idx_to_name() -> dict[str, str]:
    """idx (as str) -> real source filename, loaded once from CUAD_IDX_TO_NAME_JSON.

    The dataset CSVs only carry "idx" (see prepare_cuad_data.py -- "name" is withheld from a
    plan's input so it can't cheat on the "Document Name" extraction category by copying it).
    This mapping re-attaches the real filename at eval time only, so evaluation can still match
    a plan's predicted rows against the real, filename-keyed ground truth.
    """
    global _idx_to_name_cache
    if _idx_to_name_cache is None:
        _idx_to_name_cache = json.loads(CUAD_IDX_TO_NAME_JSON.read_text())
    return _idx_to_name_cache


def _ensure_docetl_on_path() -> None:
    """docetl's evaluation modules import as top-level `experiments.*`, matching how
    cuad_test_run.py runs with docetl/ as its working directory."""
    root = str(DOCETL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_cuad_eval_module():
    _ensure_docetl_on_path()
    return importlib.import_module("experiments.reasoning.evaluation.cuad")


def _pick_population_json(predicted_df: pd.DataFrame) -> Path:
    """Choose which document population evaluate_results should score against.

    A plan is always executed over exactly one of two fixed CSVs: the 5-document
    optimization subset (agent search, via execute_plan) or the 15-document full dataset
    (final evaluation, see CostModelAgent._run_final_evaluation). cuad_small_optimize.json is
    a strict subset of cuad_small.json, so the predicted row count alone disambiguates them --
    this assumes the plan's clause-extraction map doesn't also drop documents (e.g. via a
    filter step), which the CUAD task as posed never calls for.
    """
    return CUAD_OPTIMIZE_JSON if len(predicted_df) <= _OPTIMIZE_DOC_COUNT else CUAD_FULL_JSON


def _gt_df_name_col(gt_df: pd.DataFrame) -> str | None:
    if "name" in gt_df.columns:
        return "name"
    if "filename" in gt_df.columns:
        return "filename"
    return None


def _row_clauses_list(cell) -> list[dict]:
    """Normalize a "clauses" cell (list already, or a JSON string it round-tripped through)."""
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except (json.JSONDecodeError, TypeError):
            cell = []
    return cell if isinstance(cell, list) else []


def _write_oracle_ground_truth_csv(gt_df: pd.DataFrame, tmp_dir: Path) -> Path:
    """Pivot an oracle-substituted plan's normalized output ({"name", "clauses"} rows) into the
    wide, one-column-per-clause-type CSV evaluate_results expects as ground truth -- mirroring
    CUAD-master_clauses.csv's own layout ("Filename" + one column per clause type) -- so oracle
    mode reuses the exact same clause-matching/Jaccard-threshold scoring as real-ground-truth
    mode instead of a separate implementation.

    Every observed clause_type gets an explicit column on every row, defaulting to "[]" --
    CUAD-master_clauses.csv's own encoding of "no clause of this type" -- rather than leaving
    the cell blank. A genuinely blank CSV cell round-trips through pd.read_csv as NaN, and
    evaluate_results' clean_text() stringifies a NaN into the literal text "nan", which reads
    as a non-empty ground-truth value and manufactures false negatives against a correctly
    empty prediction.
    """
    name_col = _gt_df_name_col(gt_df)
    per_row_clauses: list[dict[str, str]] = []
    all_clause_types: set[str] = set()
    for _, row in gt_df.iterrows():
        row_map = {
            clause["clause_type"]: clause.get("text_span", "")
            for clause in _row_clauses_list(row.get("clauses"))
            if isinstance(clause, dict) and "clause_type" in clause
        }
        per_row_clauses.append(row_map)
        all_clause_types.update(row_map)

    rows = [
        {"Filename": row[name_col] if name_col else "",
         **{ct: row_map.get(ct, "[]") for ct in all_clause_types}}
        for (_, row), row_map in zip(gt_df.iterrows(), per_row_clauses)
    ]
    path = tmp_dir / "oracle_ground_truth.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_population_json(gt_df: pd.DataFrame, tmp_dir: Path) -> Path:
    """The document population evaluate_results should score against: exactly the documents the
    oracle-substituted plan produced output for (gt_df's own rows) -- always the same subset the
    predicted plan itself ran over, so no separate lookup is needed the way _pick_population_json
    needs one for the real-ground-truth path.
    """
    name_col = _gt_df_name_col(gt_df)
    names = gt_df[name_col].tolist() if name_col else []
    path = tmp_dir / "oracle_population.json"
    path.write_text(json.dumps([{"name": n} for n in names]))
    return path


@dataclass
class CuadRetrievalMetrics:
    """evaluate_results' output, reshaped to satisfy the qm_type dispatch in
    OracleQualityEvaluator.evaluate / CostModelAgent._run_final_evaluation (the type name must
    contain "Retrieval" to select the f1_score branch)."""

    f1_score: float
    avg_precision: float
    avg_recall: float
    nan_fraction: float


def _predicted_records(df: pd.DataFrame) -> list[dict]:
    """Build the [{"name", "clauses"}, ...] records evaluate_results expects.

    Assumes df has already been through normalize_eval_df: a "name"/"filename" column plus a
    "clauses" column of list[{clause_type, text_span}]. The isinstance/json.loads handling below
    is just defensive -- e.g. a "clauses" cell that round-tripped through a CSV as a JSON string.
    """
    if df.empty or "clauses" not in df.columns:
        return []
    name_col = "name" if "name" in df.columns else ("filename" if "filename" in df.columns else None)
    if name_col is None:
        return []

    return [{"name": row[name_col], "clauses": _row_clauses_list(row["clauses"])} for _, row in df.iterrows()]


class CuadEvaluator:
    """Adapter exposing docetl's evaluate_results via the _evaluate_single_query(query_id,
    predicted_df, gt_df) interface OracleQualityEvaluator / CostModelAgent._run_final_evaluation
    call.

    gt_df's shape tells us which ground truth to score against:
      - has a "clauses" column -> oracle mode (use_oracle_ground_truth=True): gt_df IS the
        oracle-substituted plan's own normalized output. Pivoted into a wide, evaluate_results-
        shaped CSV (see _write_oracle_ground_truth_csv) and scored against that.
      - otherwise -> direct mode (use_oracle_ground_truth=False, or the final full-dataset
        evaluation path, which never goes through oracle scoring at all): gt_df is just the
        non-empty placeholder from _load_direct_ground_truth / CostModelAgent._run_final_
        evaluation's own pd.read_csv(gt_path) -- ignored in favor of the real, canonical
        CUAD_GROUND_TRUTH_CSV, restricted to the right document population via
        _pick_population_json.
    """

    def _evaluate_single_query(
        self, query_id: int, predicted_df: pd.DataFrame, gt_df: pd.DataFrame
    ) -> CuadRetrievalMetrics:
        records = _predicted_records(predicted_df)
        if not records:
            return CuadRetrievalMetrics(
                f1_score=float("nan"), avg_precision=float("nan"),
                avg_recall=float("nan"), nan_fraction=1.0,
            )

        cuad_eval = _load_cuad_eval_module()
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            results_path = tmp_dir / "results.json"
            results_path.write_text(json.dumps(records, default=str))

            oracle_mode = gt_df is not None and "clauses" in gt_df.columns
            if oracle_mode:
                ground_truth_path = _write_oracle_ground_truth_csv(gt_df, tmp_dir)
                original_json_path = _write_population_json(gt_df, tmp_dir)
            else:
                ground_truth_path = CUAD_GROUND_TRUTH_CSV
                original_json_path = _pick_population_json(predicted_df)

            metrics = cuad_eval.evaluate_results(
                method_name="agent_cost_model",
                results_file=str(results_path),
                ground_truth_file=str(ground_truth_path),
                original_json_file=str(original_json_path),
            )
        return CuadRetrievalMetrics(
            f1_score=metrics["avg_f1"],
            avg_precision=metrics["avg_precision"],
            avg_recall=metrics["avg_recall"],
            nan_fraction=metrics["nan_fraction"],
        )


def load_evaluator(use_case: str, scale_factor: int | None = None) -> CuadEvaluator:
    """Construct the CUAD evaluator. use_case/scale_factor are accepted (and ignored) only for
    interface parity with SemBench's load_evaluator(use_case, scale_factor) -- CUAD has neither
    multiple use cases nor a scale-factor axis."""
    return CuadEvaluator()


def normalize_eval_df(df: pd.DataFrame, use_case: str, query_id: int) -> pd.DataFrame:
    """Reshape a plan's wide one-column-per-clause output into the
    [{"clause_type", "text_span"}, ...] list format evaluate_results expects, under a single
    "clauses" column -- the CUAD task prompt (see benchmark.yaml's task_prompts) asks the plan
    to emit one column per category, named after the category (e.g. "Document Name",
    "Third Party Beneficiary", ...). A "clauses" column already present (e.g. a plan that
    reproduces docetl's own extract_contract_info schema directly, or oracle re-normalization)
    passes through unchanged.

    A plan's input/output never carries the real "name" (source filename) -- it's withheld
    (see prepare_cuad_data.py) so a plan can't cheat on the "Document Name" clause category by
    just copying the filename instead of extracting it from the document text. So the common
    case here is an "idx" column only: it's mapped back to the real "name" via
    CUAD_IDX_TO_NAME_JSON before reshaping, purely so evaluation can match rows against the
    filename-keyed ground truth -- this mapped "name" never reaches the plan itself.

    Threaded into both the search-time (ExecutePlanTool) and final-eval-time
    (CostModelAgent._run_final_evaluation) paths via query_info["normalize_eval_df"], so
    CuadEvaluator only ever sees this canonical shape (see _predicted_records).
    """
    if df.empty or "clauses" in df.columns:
        return df

    name_col = "name" if "name" in df.columns else ("filename" if "filename" in df.columns else None)
    if name_col is None and "idx" in df.columns:
        mapping = _idx_to_name()
        df = df.copy()
        df["name"] = df["idx"].apply(lambda i: mapping.get(str(i)))
        name_col = "name"
    if name_col is None:
        return df

    clause_cols = [c for c in df.columns if c not in _ID_LIKE_COLUMNS]

    def _row_clauses(row: pd.Series) -> list[dict]:
        return [
            {"clause_type": col, "text_span": row[col]}
            for col in clause_cols
            if pd.notna(row[col]) and str(row[col]).strip()
        ]

    result = df[[name_col]].copy()
    result["clauses"] = df.apply(_row_clauses, axis=1)
    return result


def _load_direct_ground_truth() -> pd.DataFrame:
    """Non-empty placeholder gt_df so OracleQualityEvaluator.evaluate()'s "gt_df is empty"
    guard doesn't short-circuit before calling CuadEvaluator (which ignores its content)."""
    return pd.read_csv(CUAD_GROUND_TRUTH_CSV)


class QualityEvaluator(OracleQualityEvaluator):
    """CUAD QualityEvaluator, matching the SemBench QualityEvaluator constructor signature
    CostModelAgent.run() instantiates via quality_evaluator_cls(...). use_oracle_ground_truth
    is caller-controlled (see run_opt.py's --use-oracle-ground-truth flag); pass False to score
    against CUAD's real ground truth (CUAD-master_clauses.csv) instead of an oracle-substituted
    pipeline run -- the sensible choice here, since real annotations already exist."""

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        use_case: str,
        scale_factor: int | None,
        agent_dir: str,
        llm_judge_dir: str | Path,
        oracle_reasoning_effort: str | None = None,
        use_oracle_ground_truth: bool = False,
    ) -> None:
        subset_path = CUAD_DATASUBSET_DIR / "cuad_small_optimize.csv"
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=lambda df: normalize_eval_df(df, use_case, query_id),
            evaluator_factory=lambda: load_evaluator(use_case, scale_factor),
            llm_judge_dir=llm_judge_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
            # CUAD has real ground truth (CUAD-master_clauses.csv), so run_opt.py should be
            # invoked with --no-use-oracle-ground-truth -- but respect whatever value the
            # caller actually passed (see run_opt.py's --use-oracle-ground-truth /
            # --no-use-oracle-ground-truth flag, which defaults to True) rather than
            # hardcoding it here.
            use_oracle_ground_truth=use_oracle_ground_truth,
            ground_truth_loader=_load_direct_ground_truth,
        )


__all__ = ["QualityEvaluator", "CuadEvaluator", "CuadRetrievalMetrics", "load_evaluator", "normalize_eval_df"]
