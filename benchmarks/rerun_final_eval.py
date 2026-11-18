"""Top up an opt run's final evaluation: re-run its selected plan (final_answer/Q{qid}.json) on the
full dataset until it has NUM_FINAL_EVAL_RUNS runs, without redoing the search. New runs are saved
as the search saves its own (raw_results/Q{qid}_{opt_run}_{k}.csv, metrics/final_eval.json);
existing runs are kept.

    python -m benchmarks.rerun_final_eval
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from benchmarks.config import (
    build_context, expose_query_data, format_path, load_benchmark,
)
from benchmarks.external_repo import activate
from benchmarks.run_baseline import build_pipeline
from benchmarks.run_opt import NUM_FINAL_EVAL_RUNS, _load_adapter
from raptor.llm_client import patch_litellm_routing, set_prompt_fill
from raptor.sampling import mode_subset_path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CONFIG = Path(__file__).resolve().parent / "SemBench" / "benchmark.yaml"
USE_CASE = "ecomm"
SCALE_FACTOR = "2000"            # SemBench only; None for other benchmarks
SAMPLE_METHOD = "one-shot-1"   # subset label: "{method}-{subset id}"
RUNS = {   # query id: opt runs to top up
    6: [55]
}
# ─────────────────────────────────────────────────────────────────────────────


def top_up(benchmark: dict, repo_root, query_id: int, opt_run: int) -> None:
    use_case_config = benchmark["use_cases"][USE_CASE]
    dataset_config = use_case_config["dataset"]
    context = build_context(benchmark, runner=f"execute_optimizer_{SAMPLE_METHOD}",
                            use_case=USE_CASE, scale_factor=SCALE_FACTOR, query_id=query_id,
                            repo_root=repo_root or "")
    results_prefix = format_path(benchmark["results_prefix"], context)
    data_dir = format_path(dataset_config["directory"], context)
    source_csv = dataset_config["source_csv"].format(**context)
    dataset_path = data_dir / source_csv
    gt_path = format_path(use_case_config["ground_truth_path"], context)
    subset_path = mode_subset_path(format_path(use_case_config["subset_path"], context), SAMPLE_METHOD)
    metrics_path = results_prefix / "metrics" / "final_eval.json"
    raw_results_dir = results_prefix / "raw_results"
    key = f"{query_id}_{opt_run}"

    answer = json.loads((results_prefix / "final_answer" / f"Q{query_id}.json").read_text())[str(opt_run)]
    plan_name = answer["best_plan"]["name"]
    code = answer["plan_codes"][plan_name]
    entry = json.loads(metrics_path.read_text())[key]
    done = {int(k[3:]) for k in entry if re.fullmatch(r"run\d+", k)}
    todo = [k for k in range(1, NUM_FINAL_EVAL_RUNS + 1) if k not in done]
    print(f"\n=== Q{query_id} run {opt_run}: plan {plan_name!r}, has runs {sorted(done)}, "
          f"running {todo} ===", flush=True)

    # SemBench's dataset folder holds only the current query's CSV (and images, if it reads them),
    # as during the search; a no-op for a benchmark without a `_shared` folder.
    expose_query_data(use_case_config, context)
    evaluator = _load_adapter(benchmark["quality_adapter"])(
        query_id=query_id, use_case=USE_CASE, scale_factor=SCALE_FACTOR,
        subset_path=subset_path, ground_truth_path=gt_path,
    )
    gt_df = pd.read_csv(gt_path)
    set_prompt_fill(evaluator.prompt_fill_values(dataset_path))
    for k in todo:
        pipeline = build_pipeline(code, plan_name, data_dir, source_csv,
                                  dataset_config.get("id_col", "idx"),
                                  dataset_config.get("image_subdir", "images"))
        try:
            result_collection, op_results, plan_dict = pipeline.run()
        except Exception as e:
            print(f"[Q{query_id} run {opt_run}] run{k} FAILED: {type(e).__name__}: {e}")
            continue
        results_df = result_collection.to_df()
        str_cols = results_df.select_dtypes(include="object").columns
        results_df[str_cols] = results_df[str_cols].fillna("")
        raw_results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(raw_results_dir / f"Q{query_id}_{opt_run}_{k}.csv", index=False)
        quality = float(evaluator.score_plan(evaluator.normalize(results_df), gt_df, gt_path,
                                             dataset_path))
        op_errors = plan_dict.get("op_errors")
        judge_cost = evaluator.last_quality_judge_cost_usd
        run = {
            "latency": round(plan_dict.get("latency_s", float("nan")), 4),
            "cost": round(sum(r.get("cost_usd", 0) for r in op_results), 6),
            "quality": quality,
            # the scorer's own LLM spend, not part of `cost`
            **({"quality_judge_cost": round(judge_cost, 6)} if judge_cost else {}),
            **({"bullet_diff": evaluator.last_bullet_diff}
               if evaluator.last_bullet_diff is not None else {}),
            **({"op_errors": op_errors} if op_errors else {}),
        }
        # Re-read so a write by another process since the last run is not lost.
        metrics = json.loads(metrics_path.read_text())
        metrics[key][f"run{k}"] = run
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"[Q{query_id} run {opt_run}] run{k}: quality={quality:.4f} cost=${run['cost']:.4f} "
              f"latency={run['latency']:.1f}s", flush=True)


def main() -> None:
    benchmark = load_benchmark(CONFIG)
    repo_root = activate(benchmark.get("external_repo"))
    patch_litellm_routing()
    for query_id, opt_runs in RUNS.items():
        for opt_run in opt_runs:
            top_up(benchmark, repo_root, query_id, opt_run)


if __name__ == "__main__":
    main()
