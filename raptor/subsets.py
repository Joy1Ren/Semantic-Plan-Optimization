"""Optimization subsets on disk: a subset file under benchmarks/*/datasubset_opt/ stores only the
dataset's id column, so it stays small enough to track in git. load_subset rebuilds its rows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_subset(subset_path: str | Path, dataset: pd.DataFrame) -> pd.DataFrame:
    """The subset's rows, in the subset's order, taken from `dataset`, the full table it was drawn
    from (written by sampling.results.write_subset). A file holding whole rows is returned as is."""
    subset = pd.read_csv(subset_path)
    if len(subset.columns) > 1:
        return subset
    id_col = subset.columns[0]
    rows = dataset.drop_duplicates(subset=id_col, keep="first")
    rows = rows.set_index(rows[id_col].astype(str), drop=False)
    keys = subset[id_col].astype(str)
    missing = sorted(set(keys) - set(rows.index))
    if missing:
        raise ValueError(f"{subset_path}: {len(missing)} ids are not in the dataset, e.g. {missing[:3]}")
    return rows.loc[keys].reset_index(drop=True)
