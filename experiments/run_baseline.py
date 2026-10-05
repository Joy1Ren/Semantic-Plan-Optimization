"""Run the hand-written baseline plans in each benchmark's baseline_plans.yaml on the full
dataset and score them against the benchmark's real ground truth -- the comparison point for the optimizer's results.

Each (query, scale factor) is executed and scored exactly as run_opt's final evaluation does
it: the same plan sandbox, the same adapter's normalize / score_plan. Results land under the
benchmark's results_prefix with runner "baseline" (e.g. results/CUAD_categories/q{id}/baseline/):

    metrics.json               {query_id: {metric_type, plan_code, run1: {...}, run2: ...}}
    raw_results/Q{id}_{k}.csv  the plan's raw output, run k

Usage:
    PYTHONPATH=. python -m agent_cost_model.experiments.run_baseline \\
        --benchmark cuad_categories --query-ids 2 14
    PYTHONPATH=. python -m agent_cost_model.experiments.run_baseline \\
        --benchmark SemBench --use-case ecomm --scale-factors 500 2000 --query-ids 1 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from agent_cost_model.experiments.config import (
    build_context, expose_query_data, format_path, load_benchmark,
)
from agent_cost_model.experiments.external_repo import activate
from agent_cost_model.opt_agent.llm_client import patch_litellm_for_openrouter
from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor
from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline

# CostModelAgent's default authorized_imports, so baseline plans run in the same sandbox.
PLAN_AUTHORIZED_IMPORTS = ["json", "palimpzest", "pandas", "numpy"]


def build_pipeline(code: str, plan_name: str, data_dir: Path, source_csv: str,
                   id_col: str, image_subdir: str) -> PhysicalPipeline:
    """Execute plan code in the same minimal sandbox WritePlanTool uses, plus `source_csv`."""
    import palimpzest as pz

    def load_data(filename: str) -> pd.DataFrame:
        return pd.read_csv(data_dir / filename)

    def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
        pipeline.map(
            udf=lambda row: {col_name: os.path.join(data_dir, image_subdir, str(row[id_col]) + ".jpg")},
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


def run_query(benchmark: dict, use_case: str, query_id: str, scale_factor, code: str,
              num_runs: int, repo_root) -> None:
    use_case_config = benchmark["use_cases"][use_case]
    dataset_config = use_case_config["dataset"]
    context = build_context(benchmark, runner="baseline", use_case=use_case,
                            scale_factor=scale_factor, query_id=query_id, repo_root=repo_root or "")
    data_dir = format_path(dataset_config["directory"], context)
    source_csv = dataset_config["source_csv"].format(**context)
    expose_query_data(use_case_config, context)
    gt_path = format_path(use_case_config["ground_truth_path"], context)
    metric = use_case_config["query_metrics"][query_id]

    out_dir = format_path(benchmark["results_prefix"], context)
    raw_dir = out_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)

    evaluator = importlib.import_module(benchmark["quality_adapter"]).QualityEvaluator(
        query_id=int(query_id),
        use_case=use_case,
        scale_factor=scale_factor,
        ground_truth_path=gt_path,
    )
    gt_df = pd.read_csv(gt_path) if gt_path.exists() else None
    has_gt = evaluator.has_ground_truth(gt_df, gt_path)
    if not has_gt:
        print(f"[baseline] no ground truth at {gt_path} -- quality will be NaN")

    entry = {"query_id": query_id, "metric_type": metric, "plan_code": code}
    if scale_factor is not None:
        entry["scale_factor"] = scale_factor
    for k in range(1, num_runs + 1):
        label = f"{use_case} Q{query_id}" + (f" sf={scale_factor}" if scale_factor is not None else "")
        print(f"[baseline] {label} run {k}/{num_runs}")
        pipeline = build_pipeline(code, f"baseline_q{query_id}", data_dir, source_csv,
                                  dataset_config.get("id_col", "idx"),
                                  dataset_config.get("image_subdir", "images"))
        collection, per_op, plan_dict = pipeline.run()
        results_df = collection.to_df()
        str_cols = results_df.select_dtypes(include="object").columns
        results_df[str_cols] = results_df[str_cols].fillna("")
        results_df.to_csv(raw_dir / f"Q{query_id}_{k}.csv", index=False)

        quality = float("nan")
        if has_gt:
            quality = float(evaluator.score_plan(evaluator.normalize(results_df), gt_df, gt_path,
                                                 data_dir / source_csv))
        entry[f"run{k}"] = {
            "latency": round(plan_dict.get("latency_s", float("nan")), 4),
            "cost": round(sum(r.get("cost_usd", 0) for r in per_op), 6),
            "quality": quality,
            "num_rows": len(results_df),
        }
        print(f"[baseline] {label} run {k}: {entry[f'run{k}']}")

    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[query_id] = entry
    metrics = dict(sorted(metrics.items(), key=lambda kv: int(kv[0])))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[baseline] metrics -> {metrics_path}")


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
    parser.add_argument("--plans", type=Path, default=None,
                        help="default: baseline_plans.yaml beside the benchmark's benchmark.yaml")
    args = parser.parse_args()

    # Without it every LLM call fails per record and PZ swallows the error, so a plan "succeeds"
    # with empty columns, $0 cost and quality 0.
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is not set.")
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
    patch_litellm_for_openrouter()
    for scale_factor in args.scale_factors or [None]:
        for query_id in query_ids:
            try:
                run_query(benchmark, use_case, query_id, scale_factor, plans[query_id], args.runs, repo_root)
            except Exception as e:
                print(f"[baseline] {use_case} Q{query_id} sf={scale_factor} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
