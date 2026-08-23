"""SemBench adapter for oracle-based plan-quality evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_cost_model.oracle_quality_evaluator import OracleQualityEvaluator, QualityResult
from agent_cost_model.paths import SEMBENCH_DATASUBSET_DIR, ensure_sembench_src_on_path

ensure_sembench_src_on_path()


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
        return df.rename(columns={"prod_id": "id"})[["id"]]
    if query_id in (3, 4, 5, 6):
        if df.empty:
            return pd.DataFrame({"id": [], "category": []})
        return df.rename(columns={"prod_id": "id"}).iloc[:, :2]
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


def load_evaluator(use_case: str, scale_factor: int):
    """Construct the SemBench evaluator for a supported benchmark use case."""
    if use_case == "movie":
        from scenario.movie.evaluation.evaluate import MovieEvaluator
        return MovieEvaluator(use_case, scale_factor)
    if use_case == "ecomm":
        from scenario.ecomm.evaluation.evaluate import EcommEvaluator
        return EcommEvaluator(use_case, scale_factor)
    raise ValueError(f"No SemBench evaluator is registered for use_case={use_case!r}")


class QualityEvaluator(OracleQualityEvaluator):
    """Configure the reusable oracle evaluator for SemBench datasets and metrics."""

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        use_case: str,
        scale_factor: int,
        agent_dir: str,
        llm_judge_dir: str | Path,
        oracle_reasoning_effort: str | None = None,
    ) -> None:
        subset_path = (
            SEMBENCH_DATASUBSET_DIR
            / use_case
            / f"sf_{scale_factor}"
            / f"Q{query_id}_subset.csv"
        )
        super().__init__(
            oracle_client=oracle_client,
            oracle_model=oracle_model,
            query_id=query_id,
            subset_path=subset_path,
            normalize_df=lambda df: normalize_eval_df(df, use_case, query_id),
            evaluator_factory=lambda: load_evaluator(use_case, scale_factor),
            llm_judge_dir=llm_judge_dir,
            oracle_reasoning_effort=oracle_reasoning_effort,
        )


_load_evaluator = load_evaluator

__all__ = ["QualityEvaluator", "QualityResult", "load_evaluator", "normalize_eval_df"]
