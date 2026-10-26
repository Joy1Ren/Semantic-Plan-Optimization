"""Reshape SemBench query outputs into its evaluator's expected shape.

Pandas only, so docetl's venv (which has no palimpzest) can import it too: the cost model agent
(quality_evaluator.py) and MOAR (docetl/experiments/sembench/ecomm_eval.py) score through this
one function.
"""

from __future__ import annotations

import json

import pandas as pd


def _id_str(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_eval_df(df: pd.DataFrame, use_case: str, query_id: int) -> pd.DataFrame:
    """Normalize SemBench query outputs into its evaluator's expected shape.

    Pair queries whose ground truth holds both orders of every pair (ecomm Q7, Q9) get the
    reverse of each returned pair added, so the prompt need not spell out pair order. Self-pairs
    are dropped where the ground truth has none (ecomm Q9, movie Q7); ecomm Q7's ground truth
    includes a,a, which its prompt still asks for since it can't be derived from the output.
    """
    if use_case == "movie":
        # SemBench's movie pair scorer already sorts each pair, so only self-pairs need removing.
        if query_id == 7 and not df.empty:
            df = df[df.iloc[:, 1].map(_id_str) != df.iloc[:, 2].map(_id_str)]
        return df
    if use_case != "ecomm":
        return df
    if query_id in (1, 2, 13):
        if df.empty:
            return pd.DataFrame({"id": []})
        return df.rename(columns={"idx": "id"})[["id"]]
    if query_id == 14:
        # The matched image's idx, whatever the plan named it (`idx_right` after a join).
        if df.empty:
            return pd.DataFrame({"id": []})
        assert df.shape[1] == 1, f"ecomm Q14 expects 1 column, got {df.shape[1]}"
        return pd.DataFrame({"id": df.iloc[:, 0].map(_id_str)})
    if query_id in (3, 4, 5, 6):
        if df.empty:
            return pd.DataFrame({"id": [], "category": []})
        return df.rename(columns={"idx": "id"}).iloc[:, :2]
    if query_id in (7, 8, 9):
        if df.empty:
            return pd.DataFrame({"id": []})
        a, b = (df.iloc[:, i].map(_id_str) for i in (0, 1))
        if query_id in (7, 9):
            a, b = pd.concat([a, b]), pd.concat([b, a])
        if query_id == 9:
            a, b = a[a != b], b[a != b]
        return pd.DataFrame({"id": (a + "-" + b).drop_duplicates().tolist()})
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


__all__ = ["normalize_eval_df"]
