"""CUAD adapter for plan-quality evaluation, scored by scoring.score_cuad -- the same scorer
docetl/cuad_run.py grades MOAR with, so the two optimizers' numbers are comparable.

The CUAD task prompt (benchmark.yaml) asks the plan to emit one wide column per clause category
(matching how the CostModelAgent's sem_map cols naturally work: one named, described field per
category). That is also the ground truth's own layout, so normalize_eval_df only has to attach the
real filename, which the oracle ground truth is written under.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator
from agent_cost_model.experiments.cuad.paths import IDX_TO_NAME_JSON
from agent_cost_model.experiments.cuad.scoring import predictions_from_wide, score_cuad

# The document identifier, on every frame this module produces or consumes. The dataset CSVs
# carry "idx" (see prepare_cuad_data.py); normalize_eval_df maps that back to the real source
# filename under this one column name.
FILENAME_COL = "filename"
# Columns that are not clause categories. "idx" is the dataset's real identifier column
_ID_COLUMNS = {FILENAME_COL, "idx"}


def _clause_columns(df: pd.DataFrame) -> list[str]:
    """The plan's clause-category columns: everything that is not an identifier."""
    return [c for c in df.columns if c not in _ID_COLUMNS]


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
        query_id: int,
        ground_truth_path: str | Path,
        oracle_client=None,
        oracle_model: str | None = None,
        llm_judge_dir: str | Path | None = None,
        subset_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        oracle_reasoning_effort: str | None = None,
        use_oracle_ground_truth: bool = False,
        oracle_operator_quality: bool = True,
        **_ignored,
    ) -> None:
        # `**_ignored` absorbs the keywords the engine passes to every adapter's constructor
        # that this one has no use for -- `use_case` and `scale_factor`, which SemBench needs to
        # pick its evaluator but CUAD does not (one use case, one query, no scale factor).
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
            oracle_operator_quality=oracle_operator_quality,
            ground_truth_path=ground_truth_path,
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
        """The idx -> category -> text predictions score_cuad grades."""
        path.write_text(json.dumps(predictions_from_wide(plan_output_df), indent=2))

    def write_oracle_ground_truth(self, df: pd.DataFrame, path: Path) -> None:
        """Write an oracle-substituted plan's output as a ground-truth CSV in master_clauses.csv's
        layout ("Filename" plus one column per category, "[]" for an absent clause), so
        `score_plan` reads it exactly like the real one and never has to know which mode produced
        it."""
        out = pd.DataFrame({"Filename": df[FILENAME_COL].tolist()})
        for col in _clause_columns(df):
            out[col] = [
                value if pd.notna(value) and str(value).strip() else "[]"
                for value in df[col]
            ]
        out.to_csv(path, index=False)

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        """Score a plan's clause extractions: score_cuad's avg_f1.

        `ground_truth_df` is unused -- score_cuad reads the ground truth from a FILE, and the
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
        if not metrics:
            print("[QualityEvaluator] CUAD: no idx column in the plan's output, score nan")
            return float("nan")
        print(f"[QualityEvaluator] CUAD: avg F1 {metrics['avg_f1']:.4f} over "
              f"{metrics['documents_scored']} contracts")
        return float(metrics["avg_f1"])


__all__ = ["QualityEvaluator", "normalize_eval_df"]
