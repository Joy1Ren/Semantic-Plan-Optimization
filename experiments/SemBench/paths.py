"""SemBench file locations, under the checkout benchmark.yaml's ``external_repo`` declares. The
runner reads its paths from benchmark.yaml instead."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from agent_cost_model.experiments.external_repo import add_repo_to_syspath, resolve_repo

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_YAML = BENCHMARK_DIR / "benchmark.yaml"


@lru_cache(maxsize=1)
def _external_repo_config() -> dict:
    with BENCHMARK_YAML.open() as handle:
        return yaml.safe_load(handle).get("external_repo") or {}


@lru_cache(maxsize=1)
def sembench_root() -> Path:
    """The SemBench checkout (read-only input; nothing we produce is written there)."""
    root = resolve_repo(_external_repo_config())
    if root is None:
        raise RuntimeError(f"{BENCHMARK_YAML} declares no external_repo for SemBench")
    return root


def sembench_files_dir() -> Path:
    """SemBench's own data/query/ground-truth tree."""
    return sembench_root() / "files"


def activate_sembench() -> Path:
    """Put SemBench's src/ on sys.path so `scenario.*` / `evaluator` / `runner` import.

    SemBench ships no packaging metadata and its modules import each other as top-level
    packages rooted at src/, so this is required before loading any of its evaluators.
    """
    config = _external_repo_config()
    return add_repo_to_syspath(sembench_root(), config.get("python_path", "src"))


# -- data prepared into this repo (see prepare_ecomm_data.py) ------
DATASET_DIR = BENCHMARK_DIR / "dataset"
DATASUBSET_DIR = BENCHMARK_DIR / "datasubset_opt"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"


def subset_ground_truth_path(subset_path: Path) -> Path | None:
    """Where the ground truth regenerated over a datasubset_opt/ subset is written: the same
    relative path under ground_truth/, as {subset stem}_gt.csv. None for a file outside
    datasubset_opt/ (the full dataset), whose ground truth is SemBench's own."""
    try:
        relative = Path(subset_path).resolve().relative_to(DATASUBSET_DIR.resolve())
    except ValueError:
        return None
    return GROUND_TRUTH_DIR / relative.with_name(f"{relative.stem}_gt.csv")
