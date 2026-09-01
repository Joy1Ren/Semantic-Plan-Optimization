"""SemBench adapter for oracle-based plan-quality evaluation."""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from agent_cost_model.opt_agent.plan_quality_evaluator import PlanQualityEvaluator
from agent_cost_model.experiments.SemBench.paths import (
    activate_sembench,
    sembench_files_dir,
    sembench_root,
)


def _id_str(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_eval_df(df: pd.DataFrame, use_case: str, query_id: int) -> pd.DataFrame:
    """Normalize SemBench query outputs into its evaluator's expected shape."""
    if use_case != "ecomm":
        return df
    if query_id in (1, 2, 13, 14):
        if df.empty:
            return pd.DataFrame({"id": []})
        return df.rename(columns={"idx": "id"})[["id"]]
    if query_id in (3, 4, 5, 6):
        if df.empty:
            return pd.DataFrame({"id": [], "category": []})
        return df.rename(columns={"idx": "id"}).iloc[:, :2]
    if query_id in (7, 8, 9):
        if df.empty:
            return pd.DataFrame({"id": []})
        df = df.copy()
        df["id"] = df.iloc[:, :2].apply(lambda col: col.map(_id_str)).agg("-".join, axis=1)
        return df[["id"]]
    if query_id in (10, 11):
        expected_columns = 3 if query_id == 10 else 4
        if df.empty:
            return pd.DataFrame({"id": []})
        assert df.shape[1] == expected_columns, (
            f"ecomm Q{query_id} expects {expected_columns} columns, got {df.shape[1]}"
        )
        df = df.copy()
        df["id"] = df.iloc[:, :expected_columns].apply(lambda col: col.map(_id_str)).agg("-".join, axis=1)
        return df[["id"]]
    if query_id == 12:
        if df.empty:
            return pd.DataFrame({"id": []})
        assert df.shape[1] == 3, f"ecomm Q12 expects 3 columns, got {df.shape[1]}"
        df = df.copy()
        df["id"] = df.iloc[:, :3].apply(
            lambda row: json.dumps(
                {"id": int(row.iloc[0]), "brand": row.iloc[1], "category": row.iloc[2]},
                separators=(",", ":"),
            ),
            axis=1,
        )
        return df[["id"]]
    return df


# Use cases whose gold query can be regenerated on the optimization datasubset (instead of
# falling back to a precomputed full-dataset ground truth CSV, which includes rows the
# datasubset never sampled). Each entry is a `(scale_factor, query_id, subset_df) -> DataFrame |
# None` callable; return None when the gold query / domain data aren't available for that query
# (the caller falls back to the precomputed CSV in that case). Registered below per use case.
_GOLD_SQL_REGENERATORS: dict[str, Callable[[int, int, pd.DataFrame], "pd.DataFrame | None"]] = {}


def _restrict_table_to_subset_ids(table_df: pd.DataFrame, id_col: str, subset_ids) -> pd.DataFrame:
    """Restrict a full-dataset table to rows whose id_col value was sampled into the datasubset.

    If the table has no id_col column, it isn't part of what the datasubset samples at all (e.g.
    movie's Movies table has no reviewId) — return it empty (same schema, zero rows) rather than
    leaving it unrestricted, so a gold query that depends solely on that table regenerates an
    empty (but not erroring) ground truth.
    """
    if id_col not in table_df.columns:
        return table_df.iloc[0:0]
    return table_df[table_df[id_col].isin(subset_ids)]


# movie: gold SQL lives in files/movie/query/gold_sql/Q{id}.sql, run over domain tables
# registered by name with DuckDB. Each datasubset row is a sampled Review, uniquely identified by
# reviewId — with stripped-down columns, so the datasubset CSV itself can't be queried directly.
# Restrict every full domain table to the reviewId's present in the datasubset.
def _regenerate_movie_ground_truth(
    scale_factor: int, query_id: int, subset_df: pd.DataFrame
) -> "pd.DataFrame | None":
    sql_path = sembench_files_dir() / "movie" / "query" / "gold_sql" / f"Q{query_id}.sql"
    data_dir = sembench_files_dir() / "movie" / "data" / f"sf_{scale_factor}"
    if not sql_path.exists() or not data_dir.exists() or "reviewId" not in subset_df.columns:
        return None

    import duckdb

    subset_ids = subset_df["reviewId"].drop_duplicates()

    conn = duckdb.connect()
    for table_name, csv_name in {"Movies": "Movies.csv", "Reviews": "Reviews.csv"}.items():
        table_df = pd.read_csv(data_dir / csv_name)
        conn.register(table_name, _restrict_table_to_subset_ids(table_df, "reviewId", subset_ids))

    sql_query = sql_path.read_text().strip()
    return conn.execute(sql_query).fetchdf()


# ecomm: gold SQL is embedded per-query in files/ecomm/queries/q{id}.toml under
# [definition].ground_truth, and references its one domain table via
# read_parquet('styles_details.parquet') resolved against DuckDB's file_search_path (see
# EcommScenario.get_ground_truth). Each datasubset row is a sampled product, uniquely identified
# by prod_id (that table's own `id` column) — restrict it to the datasubset's products and point
# file_search_path at a temp copy, reusing the exact same gold SQL text without rewriting it.
def _regenerate_ecomm_ground_truth(
    scale_factor: int, query_id: int, subset_df: pd.DataFrame
) -> "pd.DataFrame | None":
    toml_path = sembench_files_dir() / "ecomm" / "queries" / f"q{query_id}.toml"
    parquet_path = sembench_files_dir() / "ecomm" / "data" / f"sf_{scale_factor}" / "styles_details.parquet"
    if not toml_path.exists() or not parquet_path.exists() or "prod_id" not in subset_df.columns:
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
        conn.register("_subset_ids", pd.DataFrame({"id": subset_df["prod_id"].drop_duplicates()}))
        conn.execute(
            f"COPY (SELECT s.* FROM read_parquet('{parquet_path}') AS s "
            f"SEMI JOIN _subset_ids USING (id)) TO '{tmp_parquet_path}' (FORMAT PARQUET)"
        )
        conn.execute(f"set file_search_path = '{tmp_dir}'")
        result_df = conn.execute(ground_truth_sql).df()
        conn.close()
    return result_df


_GOLD_SQL_REGENERATORS["movie"] = _regenerate_movie_ground_truth
_GOLD_SQL_REGENERATORS["ecomm"] = _regenerate_ecomm_ground_truth


def _regenerate_ground_truth_from_subset(
    use_case: str, scale_factor: int, query_id: int, subset_df: pd.DataFrame
) -> "pd.DataFrame | None":
    """Run the benchmark's gold query against domain data restricted to the optimization
    datasubset, returning None if unavailable for this use case (the caller should fall back
    to the precomputed ground truth CSV in that case)."""
    regenerate = _GOLD_SQL_REGENERATORS.get(use_case)
    if regenerate is None:
        return None
    try:
        return regenerate(scale_factor, query_id, subset_df)
    except Exception as e:
        print(f"[QualityEvaluator] gold-query ground-truth regeneration failed for {use_case} Q{query_id}: {e}")
        return None


def _load_ground_truth(
    use_case: str, scale_factor: int, query_id: int, subset_path: Path, ground_truth_path: Path
) -> pd.DataFrame:
    """Ground truth for direct (non-oracle) quality scoring: prefer regenerating it from gold
    SQL over the datasubset (accurate — includes columns the datasubset strips out, e.g. movie's
    scoreSentiment); fall back to the precomputed full-dataset ground truth CSV otherwise.

    A regenerated result is cached as Q{id}_gt.csv next to the datasubset CSV (same directory,
    keyed by the same subset — regenerating is a real DuckDB run, not free) and reused across
    plans and runs for as long as it is still newer than that subset.

    The cache is invalidated by mtime rather than by whoever rewrites the subset deleting it:
    a subset can be replaced from several places (run_opt.py --opt-subsample with either
    sampling method, LLM_Sampler's own CLI, demo.py), and any one of them forgetting to clear
    the cache would silently score a NEW subset against the OLD subset's ground truth. Ties
    count as stale, since a subset and its ground truth written in the same second are not
    ordered reliably — that only costs one extra DuckDB run.
    """
    subset_gt_path = subset_path.with_name(f"Q{query_id}_gt.csv")
    if subset_gt_path.exists() and (
        not subset_path.exists()
        or subset_gt_path.stat().st_mtime > subset_path.stat().st_mtime
    ):
        return pd.read_csv(subset_gt_path)
    if subset_path.exists():
        subset_df = pd.read_csv(subset_path)
        regenerated = _regenerate_ground_truth_from_subset(use_case, scale_factor, query_id, subset_df)
        if regenerated is not None:
            subset_gt_path.parent.mkdir(parents=True, exist_ok=True)
            regenerated.to_csv(subset_gt_path, index=False)
            return regenerated
    return pd.read_csv(ground_truth_path)


@lru_cache(maxsize=None)
def load_evaluator(use_case: str, scale_factor: int):
    """Construct the SemBench evaluator for a supported benchmark use case.

    Cached: the evaluator holds only paths and the scale factor (its _evaluate_single_query
    takes everything else as arguments), so one instance per (use_case, scale_factor) serves
    every caller. That is what keeps the full-dataset final-eval path -- which reaches the
    module-level score_plan directly, with no QualityEvaluator to carry a built one -- from
    reconstructing it on every repeated run.
    """
    evaluators = {
        "movie": ("scenario.movie.evaluation.evaluate", "MovieEvaluator"),
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

    def write_oracle_ground_truth(self, df: pd.DataFrame, path: Path) -> None:
        """Dump the frame as-is.

        SemBench's evaluators score against the ground-truth DataFrame, never a file (see
        `score_plan` below), so nothing has to read this back in a particular layout. It exists
        so `_read_persisted` can restore the run's shared ground truth for a later plan, and so
        the run directory carries an inspectable copy of what plans were scored against.
        """
        df.to_csv(path, index=False)

    def score_plan(self, plan_output_df, ground_truth_df, *_ignored) -> float:
        """Score a plan's output against ground truth using SemBench's own evaluator.

        SemBench's evaluators take the ground truth as a DataFrame -- `_evaluate_single_query` is
        declared in its evaluator/generic_evaluator.py, which is a read-only checkout -- so
        `*_ignored` absorbs the ground-truth path and the population path the engine also offers;
        neither reaches a SemBench evaluator. They return one of SemBench's metric dataclasses, and
        which field carries the score depends on the query's metric family; that mapping lives
        here rather than in the engine because the dataclasses are SemBench's own taxonomy.
        """
        import dataclasses

        evaluator = self._evaluator or load_evaluator(self._use_case, self._scale_factor)
        qm = evaluator._evaluate_single_query(self._query_id, plan_output_df, ground_truth_df)
        qm_dict = dataclasses.asdict(qm)
        qm_type = type(qm).__name__
        if "Retrieval" in qm_type:
            return float(qm_dict.get("f1_score", float("nan")))
        if "Aggregation" in qm_type:
            return 1.0 / (1.0 + float(qm_dict.get("relative_error", 1.0)))
        if "Rank" in qm_type:
            return float(qm_dict.get("spearman_correlation", float("nan")))
        if "SingleAccuracy" in qm_type:
            return float(qm_dict.get("accuracy", float("nan")))
        return float("nan")

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        use_case: str,
        scale_factor: int,
        llm_judge_dir: str | Path,
        subset_path: str | Path,
        ground_truth_path: str | Path,
        run_dir: str | Path | None = None,
        oracle_reasoning_effort: str | None = None,
        use_oracle_ground_truth: bool = True,
    ) -> None:
        # Both come from benchmark.yaml (`subset_path`, `ground_truth_path`) via the runner, so
        # this adapter never re-derives a path the config already declares.
        subset_path = Path(subset_path)
        ground_truth_path = Path(ground_truth_path)
        self._use_case = use_case
        self._scale_factor = scale_factor
        # SemBench's own evaluator, reused by score_plan. It lives here rather than in
        # PlanQualityEvaluator because only this benchmark has such an object -- CUAD calls
        # docetl's evaluate_results directly and has nothing to construct. Built eagerly so a
        # bad use_case/scale_factor surfaces at construction rather than on the first score;
        # load_evaluator is cached, so this is the same instance the final-eval path gets.
        self._evaluator = None
        try:
            self._evaluator = load_evaluator(use_case, scale_factor)
        except Exception as e:
            print(f"[QualityEvaluator] SemBench evaluator init failed: {e}")
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=lambda df: normalize_eval_df(df, use_case, query_id),
            llm_judge_dir=llm_judge_dir,
            run_dir=run_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
            use_oracle_ground_truth=use_oracle_ground_truth,
            ground_truth_loader=lambda: _load_ground_truth(
                use_case, scale_factor, query_id, subset_path, ground_truth_path
            ),
            ground_truth_path=ground_truth_path,
        )


__all__ = ["QualityEvaluator", "load_evaluator", "normalize_eval_df"]
