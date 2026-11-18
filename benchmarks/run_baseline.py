"""Run the hand-written baseline plans in each benchmark's baseline_plans.yaml on the full
dataset and score them against the benchmark's real ground truth -- the comparison point for the optimizer's results.

Each (query, scale factor) is executed and scored exactly as run_opt's final evaluation does
it: the same plan sandbox, the same adapter's normalize / score_plan. Results land under the
benchmark's results_prefix with runner "baseline" (e.g. results/CUAD/q{id}/baseline/):

    metrics.json               {query_id: {metric_type, plan_code, run1: {...}, run2: ...}}
    raw_results/Q{id}_{k}.csv  the plan's raw output, run k

Usage:
    python -m benchmarks.run_baseline \\
        --benchmark cuad --query-ids 2 3
    python -m benchmarks.run_baseline \\
        --benchmark SemBench --use-case ecomm --scale-factors 500 2000 --query-ids 1 3
    # 3 more runs, saved as runs 3-5, keeping runs 1-2
    python -m benchmarks.run_baseline \
        --benchmark cuad --runs 3 --first-run 3
    # every (scale factor, query, run) is an independent job; --workers runs that many at once
    python -m benchmarks.run_baseline \
        --benchmark SemBench --use-case ecomm --scale-factors 2000 --workers 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from benchmarks.config import (
    build_context, format_path, load_benchmark,
)
from benchmarks.external_repo import activate
from raptor import routing
from raptor.llm_client import patch_litellm_routing, set_prompt_fill
from raptor.local_python_executor import LocalPythonExecutor
from raptor.physical_pipeline import PhysicalPipeline, row_id_str

# RAPTOR's default authorized_imports, so baseline plans run in the same sandbox.
PLAN_AUTHORIZED_IMPORTS = ["json", "palimpzest", "pandas", "numpy"]


def build_pipeline(code: str, plan_name: str, data_dir: Path, source_csv: str,
                   id_col: str, image_subdir: str) -> PhysicalPipeline:
    """Execute plan code in the same minimal sandbox WritePlanTool uses, plus `source_csv`."""
    import palimpzest as pz

    def load_data(filename: str) -> pd.DataFrame:
        return pd.read_csv(data_dir / filename)

    def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
        pipeline.map(
            udf=lambda row: {col_name: os.path.join(data_dir, image_subdir, row_id_str(row[id_col]) + ".jpg")},
            cols=[{"name": col_name, "type": pz.ImageFilepath, "description": ""}],
        )
        return pipeline

    executor = LocalPythonExecutor(additional_authorized_imports=PLAN_AUTHORIZED_IMPORTS)
    executor.send_variables({
        "plan_name": plan_name,
        "source_csv": source_csv,
        "load_data": load_data,
        "add_image_data": add_image_data,
        "PhysicalPipeline": PhysicalPipeline,
        "pz": pz,
    })
    executor.send_tools({})
    pipeline = executor(code).output
    if not isinstance(pipeline, PhysicalPipeline):
        raise TypeError(f"plan code must end with a PhysicalPipeline (got {type(pipeline).__name__})")
    return pipeline


def query_paths(benchmark: dict, use_case: str, query_id: str, scale_factor, repo_root,
                runner: str = "baseline") -> dict:
    dataset_config = benchmark["use_cases"][use_case]["dataset"]
    context = build_context(benchmark, runner=runner, use_case=use_case,
                            scale_factor=scale_factor, query_id=query_id, repo_root=repo_root or "")
    data_dir = format_path(dataset_config["directory"], context)
    # Read the shared folder directly instead of expose_query_data: the exposed dir is rewritten
    # per query, so exposing would break queries running in parallel (and other runners' jobs).
    shared_dir = data_dir.with_name(f"{data_dir.name}_shared")
    out_dir = format_path(benchmark["results_prefix"], context)
    return {
        "data_dir": shared_dir if shared_dir.exists() else data_dir,
        "source_csv": dataset_config["source_csv"].format(**context),
        "gt_path": format_path(benchmark["use_cases"][use_case]["ground_truth_path"], context),
        "out_dir": out_dir,
        "metrics_path": out_dir / "metrics.json",
    }


def run_label(use_case: str, query_id: str, scale_factor, k: int) -> str:
    return (f"{use_case} Q{query_id}" + (f" sf={scale_factor}" if scale_factor is not None else "")
            + f" run {k}")


def init_worker(external_repo: dict | None) -> None:
    activate(external_repo)
    patch_litellm_routing()


def execute_run(benchmark: dict, use_case: str, query_id: str, scale_factor, code: str, k: int,
                repo_root, runner: str = "baseline", scorings: int | None = None,
                expose_insight: bool = False) -> dict:
    """Run the plan once on the full dataset, save its raw output, and score it."""
    print(f"[baseline] {run_label(use_case, query_id, scale_factor, k)} started", flush=True)
    dataset_config = benchmark["use_cases"][use_case]["dataset"]
    paths = query_paths(benchmark, use_case, query_id, scale_factor, repo_root, runner)
    data_dir, gt_path = paths["data_dir"], paths["gt_path"]
    evaluator = importlib.import_module(benchmark["quality_adapter"]).QualityEvaluator(
        query_id=int(query_id),
        use_case=use_case,
        scale_factor=scale_factor,
        ground_truth_path=gt_path,
    )
    if expose_insight:
        # The adapter shows plans one ground-truth insight in the query prompt (summary_haystack only).
        if not hasattr(evaluator, "with_exposed_insight"):
            raise ValueError(f"{benchmark['name']} has no exposed-insight support")
        code = evaluator.with_exposed_insight(code)
    gt_df = pd.read_csv(gt_path) if gt_path.exists() else None
    set_prompt_fill(evaluator.prompt_fill_values(data_dir / paths["source_csv"]))

    pipeline = build_pipeline(code, f"baseline_q{query_id}", data_dir, paths["source_csv"],
                              dataset_config.get("id_col", "idx"),
                              dataset_config.get("image_subdir", "images"))
    collection, per_op, plan_dict = pipeline.run()
    results_df = collection.to_df()
    str_cols = results_df.select_dtypes(include="object").columns
    results_df[str_cols] = results_df[str_cols].fillna("")
    results_df.to_csv(paths["out_dir"] / "raw_results" / f"Q{query_id}_{k}.csv", index=False)

    # An adapter whose metric is LLM-judged may ask for the same output to be scored several
    # times (BASELINE_SCORINGS), to measure the judge's own variance; quality is then the mean.
    scored = []
    if evaluator.has_ground_truth(gt_df, gt_path):
        for _ in range(scorings or getattr(evaluator, "BASELINE_SCORINGS", 1)):
            quality = float(evaluator.score_plan(evaluator.normalize(results_df), gt_df, gt_path,
                                                 data_dir / paths["source_csv"]))
            scored.append({"quality": quality,
                           "quality_counts": evaluator.last_quality_counts,
                           "quality_judge_cost": round(evaluator.last_quality_judge_cost_usd, 6),
                           "details": getattr(evaluator, "last_scoring_details", None)})
    qualities = [s["quality"] for s in scored]
    judge_cost = sum(s["quality_judge_cost"] for s in scored)
    if any(s["details"] for s in scored):
        details_path = paths["out_dir"] / "raw_results" / f"Q{query_id}_{k}_scorings.json"
        details_path.write_text(json.dumps([s["details"] for s in scored], indent=2, default=list))
    return {
        "latency": round(plan_dict.get("latency_s", float("nan")), 4),
        "cost": round(sum(r.get("cost_usd", 0) for r in per_op), 6),
        "quality": statistics.mean(qualities) if qualities else float("nan"),
        **({"quality_sd": statistics.stdev(qualities),
            "quality_scorings": [{k: v for k, v in s.items() if k != "details"} for s in scored]}
           if len(scored) > 1 else {"quality_counts": scored[0]["quality_counts"]} if scored else {}),
        # the scorer's own LLM spend, summed over every scoring; not part of `cost`
        **({"quality_judge_cost": round(judge_cost, 6)} if judge_cost else {}),
        **({"bullet_diff": evaluator.last_bullet_diff}
           if scored and evaluator.last_bullet_diff is not None else {}),
        "num_rows": len(results_df),
    }


def start_entry(benchmark: dict, use_case: str, query_id: str, scale_factor, code: str,
                first_run: int, repo_root, runner: str = "baseline", expose_insight: bool = False) -> dict:
    """The query's metrics.json entry before its new runs: header plus the kept earlier runs."""
    paths = query_paths(benchmark, use_case, query_id, scale_factor, repo_root, runner)
    (paths["out_dir"] / "raw_results").mkdir(parents=True, exist_ok=True)
    if not paths["gt_path"].exists():
        print(f"[baseline] no ground truth at {paths['gt_path']} -- quality will be NaN")

    metrics_path = paths["metrics_path"]
    previous = (json.loads(metrics_path.read_text()) if metrics_path.exists() else {}).get(query_id, {})
    if first_run > 1 and previous.get("plan_code") != code:
        raise ValueError(f"plan code differs from the one runs 1-{first_run - 1} executed")
    entry = {"query_id": query_id,
             "metric_type": benchmark["use_cases"][use_case]["query_metrics"][query_id],
             "plan_code": code}
    if scale_factor is not None:
        entry["scale_factor"] = scale_factor
    if expose_insight:
        entry["expose_insight"] = True     # plan_code is shown without the injected example
    entry.update({k: v for k, v in previous.items()
                  if re.fullmatch(r"run\d+", k) and int(k[3:]) < first_run})
    return entry


def write_entry(metrics_path: Path, query_id: str, entry: dict) -> None:
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[query_id] = entry
    metrics = dict(sorted(metrics.items(), key=lambda kv: int(kv[0])))
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True,
                        help="benchmark name (SemBench, CUAD, ...), directory name, or benchmark.yaml path")
    parser.add_argument("--use-case", default=None, help="required when the benchmark has several")
    parser.add_argument("--scale-factors", type=int, nargs="+", default=None,
                        help="required when the benchmark's paths key on a scale factor")
    parser.add_argument("--query-ids", nargs="+", default=None,
                        help="default: every query with a baseline plan for this use case")
    parser.add_argument("--runs", type=int, default=2, help="executions per query (default: %(default)s)")
    parser.add_argument("--first-run", type=int, default=1,
                        help="number of the first run; earlier runs in metrics.json are kept "
                             "(default: %(default)s)")
    parser.add_argument("--plans", type=Path, default=None,
                        help="default: baseline_plans.yaml beside the benchmark's benchmark.yaml")
    parser.add_argument("--runner", default="baseline",
                        help="results folder name under the query's results dir, so plans other than "
                             "the baseline (e.g. --plans test_plans/x.yaml --runner test_x) keep their "
                             "own metrics.json (default: %(default)s)")
    parser.add_argument("--scorings", type=int, default=None,
                        help="quality-judge scorings per run (default: the adapter's BASELINE_SCORINGS)")
    parser.add_argument("--workers", type=int, default=1,
                        help="runs executed at once, in separate processes (default: %(default)s)")
    args = parser.parse_args()

    # Without a key every LLM call on its route fails per record and PZ swallows the error, so a
    # plan "succeeds" with empty columns, $0 cost and quality 0.
    routing.check_keys()
    benchmark = load_benchmark(args.benchmark)
    use_cases = list(benchmark["use_cases"])
    use_case = args.use_case or (use_cases[0] if len(use_cases) == 1 else None)
    if use_case not in use_cases:
        parser.error(f"--use-case must be one of {use_cases}")
    if "{scale_factor}" in json.dumps(benchmark) and not args.scale_factors:
        parser.error(f"{benchmark['name']} keys its paths on a scale factor -- pass --scale-factors.")

    plans_path = args.plans or Path(benchmark["_path"]).parent / "baseline_plans.yaml"
    plans = yaml.safe_load(plans_path.read_text())
    # A benchmark with several use cases nests its plans under the use case; the rest key by query id.
    plans = plans.get(use_case, plans)
    query_ids = args.query_ids or list(plans)
    missing = [q for q in query_ids if q not in plans]
    if missing:
        parser.error(f"no baseline plan for {use_case} queries {missing} in {plans_path}")

    repo_root = activate(benchmark.get("external_repo"))
    # summary_haystack's benchmark.yaml shows plans the query's first insight as an example (still
    # scored on every insight); its baseline plans get the same sentence injected.
    args.expose_insight = bool(benchmark.get("expose_insight"))
    entries, jobs = {}, []
    for scale_factor in args.scale_factors or [None]:
        for query_id in query_ids:
            try:
                entries[scale_factor, query_id] = start_entry(
                    benchmark, use_case, query_id, scale_factor, plans[query_id], args.first_run,
                    repo_root, args.runner, args.expose_insight)
            except Exception as e:
                print(f"[baseline] {use_case} Q{query_id} sf={scale_factor} failed: {type(e).__name__}: {e}")
                continue
            jobs += [(scale_factor, query_id, k)
                     for k in range(args.first_run, args.first_run + args.runs)]

    # Workers only run and score; this process alone writes metrics.json, so writes never race.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(benchmark.get("external_repo"),)) as pool:
        futures = {pool.submit(execute_run, benchmark, use_case, query_id, scale_factor,
                               plans[query_id], k, repo_root, args.runner, args.scorings,
                               args.expose_insight): (scale_factor, query_id, k)
                   for scale_factor, query_id, k in jobs}
        for future in as_completed(futures):
            scale_factor, query_id, k = futures[future]
            label = run_label(use_case, query_id, scale_factor, k)
            try:
                run = future.result()
            except Exception as e:
                print(f"[baseline] {label} failed: {type(e).__name__}: {e}")
                continue
            print(f"[baseline] {label}: {run}")
            entries[scale_factor, query_id][f"run{k}"] = run
            metrics_path = query_paths(benchmark, use_case, query_id, scale_factor, repo_root,
                                       args.runner)["metrics_path"]
            write_entry(metrics_path, query_id, entries[scale_factor, query_id])


if __name__ == "__main__":
    main()
