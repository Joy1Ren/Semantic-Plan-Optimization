"""Load a benchmark configuration and resolve its path templates, for the runner and the analysis
scripts alike."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from benchmarks.paths import BENCHMARKS_DIR, RESULTS_DIR

# Directory name under benchmarks/ for each benchmark, keyed by its `name:` in benchmark.yaml.
_BENCHMARK_DIRS = {
    "SemBench": "SemBench",
    "CUAD": "cuad",
    "Summary of a Haystack": "summary_haystack",
    "Sustainability Reports": "sustainability",
}


def benchmark_yaml_path(benchmark: str) -> Path:
    """Accepts either the benchmark's `name:` (SemBench, CUAD) or its directory name."""
    directory = _BENCHMARK_DIRS.get(benchmark, benchmark)
    return BENCHMARKS_DIR / directory / "benchmark.yaml"


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
    """One query's (task prompt, evaluation metric, modalities) from a use case's config, shared
    by the runner and the sampler CLI.

    The prompt always comes from `task_prompts`; `query_source` only says where the modalities
    are declared: a toml file's `metadata.modalities` (type "toml") or inline (type "text")."""
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



def expose_query_data(config: dict[str, Any], context: dict[str, Any]) -> None:
    """Make a use case's dataset directory hold only what this query may see.

    Applies to a dataset with a sibling `<directory>_shared` folder, which keeps every query's
    CSV and the image folder. The dataset directory -- all the agent can see -- gets a copy of
    this query's CSV, plus the images only if the query's modalities include "image".
    """
    dataset = config["dataset"]
    data_dir = format_path(dataset["directory"], context)
    shared_dir = data_dir.with_name(f"{data_dir.name}_shared")
    if not shared_dir.exists():
        return
    for csv in data_dir.glob("*.csv"):
        csv.unlink()
    source_csv = dataset["source_csv"].format(**context)
    shutil.copy2(shared_dir / source_csv, data_dir / source_csv)
    image_subdir = dataset.get("image_subdir", "images")
    if (shared_dir / image_subdir).exists():
        shutil.rmtree(data_dir / image_subdir, ignore_errors=True)
        if "image" in load_query(config, context)[2]:
            shutil.copytree(shared_dir / image_subdir, data_dir / image_subdir)


def results_prefix(
    benchmark: str | Path,
    runner: str,
    *,
    use_case: str = "",
    scale_factor: Any = "",
    query_id: Any = "",
) -> Path:
    """Where one (benchmark, use case, scale factor, query, runner) writes everything it produces.

    The benchmark decides which of use_case/scale_factor/query_id appear: SemBench keys on all
    three, CUAD on query_id. The runner owns the layout below the returned prefix.
    """
    config = load_benchmark(benchmark)
    context = build_context(config, runner=runner, use_case=use_case, scale_factor=scale_factor,
                            query_id=query_id)
    return format_path(config["results_prefix"], context)