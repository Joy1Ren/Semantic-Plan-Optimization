"""Load a benchmark configuration and resolve its path templates.

Both the runner (which writes results) and the analysis scripts (which read them) go through
here, so a change to a `results_prefix` template in a benchmark.yaml moves the writer and the
reader together instead of leaving one behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_cost_model.paths import EXPERIMENTS_DIR, RESULTS_DIR

# Directory name under experiments/ for each benchmark, keyed by its `name:` in benchmark.yaml.
_BENCHMARK_DIRS = {"SemBench": "SemBench", "CUAD": "cuad"}


def benchmark_yaml_path(benchmark: str) -> Path:
    """Accepts either the benchmark's `name:` (SemBench, CUAD) or its directory name."""
    directory = _BENCHMARK_DIRS.get(benchmark, benchmark)
    return EXPERIMENTS_DIR / directory / "benchmark.yaml"


def load_benchmark(benchmark: str | Path) -> dict[str, Any]:
    path = Path(benchmark) if str(benchmark).endswith(".yaml") else benchmark_yaml_path(str(benchmark))
    with path.resolve().open() as handle:
        config = yaml.safe_load(handle)
    config["_path"] = str(path.resolve())
    return config


def build_context(
    config: dict[str, Any],
    *,
    runner: str = "",
    use_case: str = "",
    scale_factor: Any = "",
    query_id: Any = "",
    repo_root: str | Path = "",
) -> dict[str, Any]:
    """The substitution context every path template in a benchmark.yaml is formatted against."""
    return {
        "query_id": query_id,
        "scale_factor": scale_factor,
        "benchmark_dir": str(Path(config["_path"]).parent),
        "results_dir": str(RESULTS_DIR),
        "repo_root": str(repo_root),
        "use_case": use_case,
        "runner": runner,
    }


def format_path(template: str, context: dict[str, Any]) -> Path:
    return Path(template.format(**context))


def load_query(config: dict[str, Any], context: dict[str, Any]) -> tuple[str, str, list[str]]:
    """One query's (task prompt, evaluation metric, modalities) from a use case's config.

    Lives here rather than in the runner because the sampler CLI resolves the same query the
    same way -- a subset built from a different prompt than the one the optimizer later runs
    would be silently wrong.
    """
    import tomllib

    source = config["query_source"]
    query_id = str(context["query_id"])
    try:
        task = config["task_prompts"][query_id].strip()
        metric = config["query_metrics"][query_id]
    except KeyError as error:
        raise KeyError(
            f"benchmark.yaml is missing the task prompt or evaluation metric for query {query_id}"
        ) from error
    if source["type"] == "toml":
        path = Path(source["path"].format(**context))
        with path.open("rb") as handle:
            query = tomllib.load(handle)
        return task, metric, query["metadata"].get("modalities", [])
    if source["type"] == "text":
        return task, metric, source.get("modalities", ["text"])
    raise ValueError(f"Unsupported query source type: {source['type']!r}")


def results_prefix(
    benchmark: str | Path,
    runner: str,
    *,
    use_case: str = "",
    scale_factor: Any = "",
) -> Path:
    """Where one (benchmark, use case, scale factor, runner) writes everything it produces.

    The benchmark decides which of use_case/scale_factor appear: SemBench keys on both, CUAD
    on neither. The runner owns the layout below the returned prefix.
    """
    config = load_benchmark(benchmark)
    context = build_context(config, runner=runner, use_case=use_case, scale_factor=scale_factor)
    return format_path(config["results_prefix"], context)