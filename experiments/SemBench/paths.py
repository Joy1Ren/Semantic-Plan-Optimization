"""SemBench-local file locations.

These live here rather than in the package-level ``paths.py`` because only this adapter and
its data extractors read them: the runner gets everything it needs from ``benchmark.yaml``.
The SemBench checkout they hang off is resolved from this benchmark's own ``external_repo``
block, so there is one declaration of where SemBench lives.
"""

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


# -- data prepared into this repo (see prepare_ecomm_data.py / prepare_movie_data.py) ------
DATASET_DIR = BENCHMARK_DIR / "dataset"
DATASUBSET_DIR = BENCHMARK_DIR / "datasubset_opt"
