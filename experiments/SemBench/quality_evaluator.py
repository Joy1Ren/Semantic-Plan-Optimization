"""SemBench adapter for plan-quality evaluation."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator
from agent_cost_model.experiments.SemBench.normalize import normalize_eval_df
from agent_cost_model.experiments.SemBench.paths import (
    activate_sembench,
    sembench_files_dir,
    sembench_root,
    subset_ground_truth_path,
)


# ecomm: gold SQL is in q{id}.toml's [definition].ground_truth and reads
# read_parquet('styles_details.parquet') through DuckDB's file_search_path. The table is restricted
# to the subset's idx and file_search_path pointed at that copy, so the gold SQL runs unchanged.
def _regenerate_ecomm_ground_truth(
    scale_factor: int, query_id: int, subset_df: pd.DataFrame
) -> "pd.DataFrame | None":
    toml_path = sembench_files_dir() / "ecomm" / "queries" / f"q{query_id}.toml"
    parquet_path = sembench_files_dir() / "ecomm" / "data" / f"sf_{scale_factor}" / "styles_details.parquet"
    if not toml_path.exists() or not parquet_path.exists() or "idx" not in subset_df.columns:
        return None

    import tempfile

    import duckdb
    import tomli

    with open(toml_path, "rb") as f:
        ground_truth_sql = tomli.load(f)["definition"]["ground_truth"]

    # Filter and rewrite the parquet file with DuckDB itself (not a pandas/pyarrow round-trip):
    # styles_details has nested struct/list columns (e.g. articleType, colours) that pandas
    # can't reliably round-trip back into a valid parquet schema after a boolean-mask filter.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_parquet_path = Path(tmp_dir) / "styles_details.parquet"
        conn = duckdb.connect()
        conn.register("_subset_ids", pd.DataFrame({"id": subset_df["idx"].drop_duplicates()}))
        conn.execute(
            f"COPY (SELECT s.* FROM read_parquet('{parquet_path}') AS s "
            f"SEMI JOIN _subset_ids USING (id)) TO '{tmp_parquet_path}' (FORMAT PARQUET)"
        )
        conn.execute(f"set file_search_path = '{tmp_dir}'")
        result_df = conn.execute(ground_truth_sql).df()
        conn.close()
    return result_df


def _regenerate_ground_truth_from_subset(
    use_case: str, scale_factor: int, query_id: int, subset_df: pd.DataFrame
) -> "pd.DataFrame | None":
    """The gold query's ground truth over the subset's rows; None when it cannot be regenerated, so
    the caller falls back to the precomputed full-dataset CSV."""
    if use_case != "ecomm":
        return None
    try:
        return _regenerate_ecomm_ground_truth(scale_factor, query_id, subset_df)
    except Exception as e:
        print(f"[QualityEvaluator] gold-query ground-truth regeneration failed for {use_case} Q{query_id}: {e}")
        return None


def _load_ground_truth(
    use_case: str, scale_factor: int, query_id: int, subset_path: Path | None, ground_truth_path: Path
) -> pd.DataFrame:
    """Ground truth regenerated from gold SQL over the subset, else the precomputed full-dataset
    CSV. For a datasubset_opt/ subset it is also written to {subset stem}_gt.csv under
    ground_truth/ (paths.subset_ground_truth_path), for inspection only."""
    if subset_path is not None and subset_path.exists():
        subset_df = pd.read_csv(subset_path)
        regenerated = _regenerate_ground_truth_from_subset(use_case, scale_factor, query_id, subset_df)
        if regenerated is not None:
            subset_gt_path = subset_ground_truth_path(subset_path)
            if subset_gt_path is not None:
                subset_gt_path.parent.mkdir(parents=True, exist_ok=True)
                regenerated.to_csv(subset_gt_path, index=False)
            return regenerated
    return pd.read_csv(ground_truth_path)


# F1 queries whose answer is a set of single records drawn from the population, so a record
# neither returned nor wanted is a TN. Pair and tuple queries have no such population.
_RECORD_SET_QUERIES = {"ecomm": {1, 13}}


def _retrieval_counts(use_case: str, query_id: int, plan_output_df: pd.DataFrame,
                      ground_truth_df: pd.DataFrame, population_path) -> dict:
    """TP/FP/FN (and TN where defined) behind a SemBench F1, from the same sets its evaluator
    compares: the normalized `id` column."""
    returned = set(plan_output_df["id"]) if not plan_output_df.empty else set()
    wanted = set(ground_truth_df["id"]) if not ground_truth_df.empty else set()
    counts = {"TP": len(returned & wanted), "FP": len(returned - wanted),
              "FN": len(wanted - returned)}
    if population_path and query_id in _RECORD_SET_QUERIES.get(use_case, ()):
        counts["TN"] = len(pd.read_csv(population_path)) - len(returned | wanted)
    return counts


@lru_cache(maxsize=None)
def load_evaluator(use_case: str, scale_factor: int):
    """The SemBench evaluator for a use case, one per (use_case, scale_factor): it holds only
    paths and the scale factor."""
    evaluators = {
        "ecomm": ("scenario.ecomm.evaluation.evaluate", "EcommEvaluator"),
    }
    try:
        module_name, class_name = evaluators[use_case]
    except KeyError as exc:
        raise ValueError(f"No SemBench evaluator is registered for use_case={use_case!r}") from exc

    # SemBench's own modules import ``scenario``, ``evaluator``, and ``ecomm``
    # as top-level packages, so its src directory must be on sys.path.
    activate_sembench()
    try:
        evaluator_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not load the SemBench {use_case!r} evaluator from {sembench_root()}. "
            "Install SemBench's dependencies and verify SEMBENCH_ROOT."
        ) from exc
    return getattr(evaluator_module, class_name)(use_case, scale_factor)


class QualityEvaluator(PlanQualityEvaluator):
    """Configure the reusable plan-quality machinery for SemBench datasets and metrics."""

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        """SemBench scores against the DataFrame, so having a ground truth means having rows."""
        return ground_truth_df is not None and not ground_truth_df.empty

    def write_scoring_input(self, plan_output_df: pd.DataFrame, path: Path) -> None:
        """SemBench hands its evaluator the normalized frame itself, so the scoring input is
        that frame -- written as JSON records to match the artifact's .json extension."""
        plan_output_df.to_json(path, orient="records", indent=2)

    def score_plan(self, plan_output_df, ground_truth_df, _ground_truth_path=None,
                   population_path=None) -> float:
        """Score a plan's output with SemBench's own evaluator, which takes the ground truth as a
        DataFrame and returns a metric dataclass; the score is read from the field of the query's
        metric family. The population path is read only for F1's TN."""
        import dataclasses

        evaluator = self._evaluator or load_evaluator(self._use_case, self._scale_factor)
        qm = evaluator._evaluate_single_query(self._query_id, plan_output_df, ground_truth_df)
        qm_dict = dataclasses.asdict(qm)
        qm_type = type(qm).__name__
        label = f"[QualityEvaluator] {self._use_case} Q{self._query_id}"
        if "Retrieval" in qm_type:
            score = float(qm_dict.get("f1_score", float("nan")))
            counts = self.last_quality_counts = _retrieval_counts(
                self._use_case, self._query_id, plan_output_df, ground_truth_df, population_path)
            print(f"{label}: F1 {score:.4f}, (TP, FP, TN, FN) = ({counts['TP']}, {counts['FP']}, "
                  f"{counts.get('TN', 'n/a')}, {counts['FN']})")
            return score
        if "Aggregation" in qm_type:
            relative_error = float(qm_dict.get("relative_error", 1.0))
            score = 1.0 / (1.0 + relative_error)
            print(f"{label}: score {score:.4f} (relative error {relative_error:.4f}), "
                  f"true {ground_truth_df.head(5).to_dict('records')}, "
                  f"plan {plan_output_df.head(5).to_dict('records')}")
            return score
        if "Rank" in qm_type:
            score = float(qm_dict.get("spearman_correlation", float("nan")))
            print(f"{label}: spearman {score:.4f}")
            return score
        if "SingleAccuracy" in qm_type:
            score = float(qm_dict.get("accuracy", float("nan")))
            print(f"{label}: {qm_dict.get('metric_type', 'accuracy')} {score:.4f}")
            return score
        return float("nan")

    def __init__(
        self,
        query_id: int,
        use_case: str,
        scale_factor: int,
        ground_truth_path: str | Path,
        subset_path: str | Path | None = None,
        # Absorbs the keywords the engine passes every adapter that this one does not use.
        **_ignored,
    ) -> None:
        # Both come from benchmark.yaml (`subset_path`, `ground_truth_path`) via the runner, so
        # this adapter never re-derives a path the config already declares.
        subset_path = Path(subset_path) if subset_path else None
        ground_truth_path = Path(ground_truth_path)
        self._use_case = use_case
        self._scale_factor = scale_factor
        # SemBench's evaluator, reused by score_plan. Built here so a bad use_case/scale_factor
        # fails at construction rather than on the first score.
        self._evaluator = None
        try:
            self._evaluator = load_evaluator(use_case, scale_factor)
        except Exception as e:
            print(f"[QualityEvaluator] SemBench evaluator init failed: {e}")
        super().__init__(
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=lambda df: normalize_eval_df(df, use_case, query_id),
            ground_truth_loader=lambda: _load_ground_truth(
                use_case, scale_factor, query_id, subset_path, ground_truth_path
            ),
            ground_truth_path=ground_truth_path,
        )


__all__ = ["QualityEvaluator", "load_evaluator", "normalize_eval_df"]
