"""CUAD adapter for oracle/direct-ground-truth plan-quality evaluation.

Reuses docetl's own CUAD scorer (experiments.reasoning.evaluation.cuad.evaluate_results)
unmodified rather than re-implementing its clause-matching logic, so cost-model-agent quality
scores stay bug-for-bug consistent with docetl MOAR baseline runs. Every scoring decision --
filename normalization, the 41-metric list, clause_type matching, the Jaccard > 0.15 gate,
precision/recall/avg_f1 -- happens inside that function. Nothing here re-implements it; the
code below only translates between a plan's output shape and the file paths
evaluate_results takes as arguments.

The CUAD task prompt (benchmark.yaml) asks the plan to emit one wide column per clause
category (matching how the CostModelAgent's sem_map cols naturally work: one named,
described field per category, rather than a single nested list field). Everything here stays
in that wide form -- it is also the ground truth's own layout, so normalize_eval_df only has
to attach the real filename.

evaluate_results (docetl's own CUAD scorer) takes its two data files in OPPOSITE shapes, which
is the only thing that forces a conversion anywhere (convert plan results to narrow format):

  ground_truth_file -- WIDE: one column per clause type, looked up as `metric in
    merged_df.columns`. Already what we have; write_ground_truth just adds the "Filename"
    header and the "[]" blank encoding.
  results_file -- NARROW: one "clauses" list of {clause_type, text_span} per row, read as
    `merged_df["clauses"]`. Nothing else is accepted (wide predictions raise KeyError).
"""
from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator
from agent_cost_model.experiments.external_repo import add_repo_to_syspath
from agent_cost_model.experiments.cuad.paths import (
    IDX_TO_NAME_JSON,
    docetl_root,
)

# The document identifier, on every frame this module produces or consumes. The dataset CSVs
# carry "idx" (see prepare_cuad_data.py); normalize_eval_df maps that back to the real source
# filename under this one column name, and everything downstream reads only this.
FILENAME_COL = "filename"
# Columns normalize_eval_df must NOT treat as clause categories. "idx" is the dataset's real
# identifier column
_ID_COLUMNS = {FILENAME_COL, "idx"}

def _load_cuad_eval_module():
    """docetl's evaluation modules import as top-level `experiments.*`, matching how
    cuad_test_run.py runs with docetl/ as its working directory. The repo root goes on
    sys.path via this benchmark's own external_repo declaration (see benchmark.yaml)."""
    add_repo_to_syspath(docetl_root(), ".")
    return importlib.import_module("experiments.reasoning.evaluation.cuad")


def _clause_columns(df: pd.DataFrame) -> list[str]:
    """The plan's clause-category columns: everything that is not an identifier."""
    return [c for c in df.columns if c not in _ID_COLUMNS]


def _condense_clauses_columns(df: pd.DataFrame) -> list[dict]:
    """Turn the plan's wide output into the narrow [{"filename", "clauses"}, ...] records
    evaluate_results wants in its results file -- the ONLY place the narrow form is used, and
    the only shape that argument accepts (it reads merged_df["clauses"] by name; wide
    predictions raise KeyError).

    An entry is emitted for EVERY clause category, with an empty text_span where the plan found
    nothing -- matching how docetl's own CUAD pipeline reports (its output schema is a clauses
    list, and it emits all 41 categories per document, leaving text_span empty for the ones it
    does not find).
    """
    if df.empty or FILENAME_COL not in df.columns:
        return []
    clause_cols = _clause_columns(df)
    return [
        {
            FILENAME_COL: row[FILENAME_COL],
            "clauses": [
                {
                    "clause_type": col,
                    "text_span": row[col] if pd.notna(row[col]) and str(row[col]).strip() else "",
                }
                for col in clause_cols
            ],
        }
        for _, row in df.iterrows()
    ]


def _population_filenames(population_path: str | Path) -> list[str]:
    """The documents the plan was given, read from the dataset CSV it ran over.

    This is the document POPULATION evaluate_results scores over, and docetl passes the same
    thing on its own runs: the pipeline's input file (cuad_small.json / cuad.json), never the
    results. The difference matters when a plan drops a document -- an op returning zero
    records removes it from the plan's output entirely (see physical_pipeline/base.py). Taken
    from the dataset, that document is scored as an empty prediction and counts as false
    negatives on every clause it should have had; taken from the results, it would silently
    vanish from the measurement and the plan would not be charged for losing it.

    The dataset CSVs carry only "idx" (see prepare_cuad_data.py), so it is mapped back to the
    real filename here through IDX_TO_NAME_JSON, exactly as normalize_eval_df does for a plan's
    output.
    """
    df = pd.read_csv(population_path, dtype={"idx": str})
    if FILENAME_COL in df.columns:
        return df[FILENAME_COL].tolist()
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    return [mapping.get(str(i)) for i in df["idx"]]


def _evaluate(
    plan_output_df: pd.DataFrame,
    ground_truth_path: str | Path,
    population_path: str | Path,
) -> dict:
    """Run docetl's CUAD scorer over a plan's output, returning its full metrics dict
    (per-metric precision/recall, avg_precision/avg_recall/avg_f1, nan_fraction, ...).

    evaluate_results takes four FILE PATHS, so everything here is writing those files -- no
    scoring logic is reimplemented. Returns {} when the plan produced nothing scoreable.

    `original_json_file` is the document POPULATION the metrics are computed over: the dataset
    the plan ran over, not its output (see _population_filenames). evaluate_results loops over
    that population and counts any document missing from the predictions as an empty prediction.
    """
    records = _condense_clauses_columns(plan_output_df)
    if not records:
        return {}

    cuad_eval = _load_cuad_eval_module()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        results_path = tmp_dir / "results.json"
        results_path.write_text(json.dumps(records, default=str))

        # docetl reads this file's entries as item["name"], so the key here is "name"
        # regardless of what the results file above is keyed by.
        population_file = tmp_dir / "population.json"
        population_file.write_text(
            json.dumps([{"name": n} for n in _population_filenames(population_path)])
        )

        return cuad_eval.evaluate_results(
            method_name="agent_cost_model",
            results_file=str(results_path),
            ground_truth_file=str(ground_truth_path),
            original_json_file=str(population_file),
        )


def normalize_eval_df(df: pd.DataFrame, *_ignored) -> pd.DataFrame:
    """Attach the real source filename to a plan's output. The wide one-column-per-clause shape
    the plan emits is kept as-is.

    A plan's input/output only carries 'idx' instead of the real source filename (to prevent
    cheating on the "Document Name" clause category). This mapping re-attaches the real filename.
    """
    if df.empty or FILENAME_COL in df.columns or "idx" not in df.columns:
        return df
    mapping = json.loads(IDX_TO_NAME_JSON.read_text())
    df = df.copy()
    df[FILENAME_COL] = df["idx"].apply(lambda i: mapping.get(str(i)))
    return df


class QualityEvaluator(PlanQualityEvaluator):
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
        llm_judge_dir: str | Path,
        subset_path: str | Path,
        ground_truth_path: str | Path,
        run_dir: str | Path | None = None,
        oracle_reasoning_effort: str | None = None,
        use_oracle_ground_truth: bool = False,
        op_sample_seed: int | None = None,
        **_ignored,
    ) -> None:
        # `**_ignored` absorbs the keywords the engine passes to every adapter's constructor
        # that this one has no use for -- `use_case` and `scale_factor`, which SemBench needs to
        # pick its evaluator but CUAD does not (one use case, one query, no scale factor).
        subset_path = Path(subset_path)
        ground_truth_path = Path(ground_truth_path)
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=normalize_eval_df,
            llm_judge_dir=llm_judge_dir,
            run_dir=run_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
            use_oracle_ground_truth=use_oracle_ground_truth,
            ground_truth_path=ground_truth_path,
            op_sample_seed=op_sample_seed,
        )

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        """CUAD scores from the ground-truth FILE, so this asks about the file, not the frame
        (which is always None here -- see the ground_truth_loader note above).

        `nrows=1` reads only the first chunk, so an existing-but-header-only ground truth is
        caught without materialising a frame this benchmark never scores against.
        """
        if not ground_truth_path or not Path(ground_truth_path).exists():
            return False
        try:
            return not pd.read_csv(ground_truth_path, nrows=1).empty
        except Exception as e:
            print(f"[QualityEvaluator] could not read ground truth at {ground_truth_path}: {e}")
            return False

    def write_scoring_input(self, plan_output_df: pd.DataFrame, path: Path) -> None:
        """The narrow {filename, clauses} records handed to evaluate_results as its results
        file -- byte-for-byte what _evaluate writes, since both serialize the same call."""
        path.write_text(json.dumps(_condense_clauses_columns(plan_output_df), default=str, indent=2))

    def write_oracle_ground_truth(self, df: pd.DataFrame, path: Path) -> None:
        """Write an oracle-substituted plan's output as the ground-truth CSV evaluate_results
        reads, so `score_plan` reads it exactly like CUAD-master_clauses.csv and never has to
        know which mode produced it.

        Nothing is reshaped -- the frame is already one column per clause category, the same
        layout master uses. This only writes that file's own "Filename" header and its encoding
        of an absent clause: blanks become "[]", not an empty cell, because a genuinely blank
        CSV cell round-trips through pd.read_csv as NaN and evaluate_results' clean_text()
        stringifies a NaN into the literal text "nan" -- a non-empty ground-truth value that
        manufactures false negatives against a correctly empty prediction.
        """
        out = pd.DataFrame({"Filename": df[FILENAME_COL].tolist()})
        for col in _clause_columns(df):
            out[col] = [
                value if pd.notna(value) and str(value).strip() else "[]"
                for value in df[col]
            ]
        out.to_csv(path, index=False)

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        """Score a plan's clause extractions with docetl's own CUAD scorer: `_evaluate`'s avg_f1.

        `ground_truth_df` is unused -- CUAD's scorer reads the ground truth from a FILE, and the
        path handed in already points at the right one for this run's mode (the run's
        materialised oracle ground truth, or the real annotations), so nothing here branches on
        the mode.
        """
        if not ground_truth_path:
            raise ValueError(
                "CUAD scoring needs a ground_truth_path: either benchmark.yaml's "
                "ground_truth_path (direct mode) or the run's materialised oracle ground truth "
                "(oracle mode, which needs run_dir to be set). See "
                "PlanQualityEvaluator.ground_truth_path_in_use()."
            )
        metrics = _evaluate(plan_output_df, ground_truth_path, population_path)
        return float(metrics["avg_f1"]) if metrics else float("nan")


__all__ = ["QualityEvaluator", "normalize_eval_df"]
