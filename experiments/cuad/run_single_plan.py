"""Rebuild and execute a single saved CUAD plan by name, without going through the full agent
search loop -- for fast, cheap debugging of why a plan's quality score looks the way it does
(e.g. inspecting the full, untruncated extracted text, which get_op_samples truncates via
DataRecord.__repr__).

Usage:
    python3 -m agent_cost_model.experiments.cuad.run_single_plan --plan-name p1 \\
        --dataset agent_cost_model/experiments/cuad/datasubset_opt/cuad_optimize.csv
    python3 -m agent_cost_model.experiments.cuad.run_single_plan --plan-name p1 \\
        --dataset agent_cost_model/experiments/cuad/dataset/cuad.csv \\
        --final-answer path/to/other/Q1.json --runcount 1 --save-csv /tmp/p1_output.csv

Reads plan code from a CostModelAgent final_answer JSON's plan_codes (see
CostModelAgent.run()'s "plan_codes" tracking and _run_final_evaluation), rebuilds the
PhysicalPipeline exactly as WritePlanTool does (same minimal sandbox: load_data/PhysicalPipeline/
pz only, no results stores), executes it for real (this makes real LLM calls and costs money),
then scores the output the same way CuadEvaluator does -- printing the full per-metric
precision/recall breakdown, not just avg_f1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import palimpzest as pz

from agent_cost_model.opt_agent.cost_model_agent import patch_litellm_for_openrouter
from agent_cost_model.experiments.cuad.quality_evaluator import (
    _evaluate,
    normalize_eval_df,
)
from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor
from agent_cost_model.experiments.cuad.paths import DATASET_DIR, cuad_ground_truth_csv
from agent_cost_model.paths import RESULTS_DIR
from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline

DEFAULT_FINAL_ANSWER_PATH = RESULTS_DIR / "final_answer" / "cuad" / "execute_oracle_sampler" / "Q1.json"
# Matches CostModelAgent.__init__'s own default (cost_model_agent.py) -- plan code references
# pz.Model.* attribute access, which LocalPythonExecutor's sandbox blocks as "module access"
# unless the module is explicitly authorized.
PLAN_AUTHORIZED_IMPORTS = ["math", "statistics", "json", "collections", "itertools", "palimpzest", "pandas"]


def _build_pipeline(code: str, plan_name: str) -> PhysicalPipeline:
    """Rebuild a PhysicalPipeline from saved plan code, in the same minimal sandbox
    WritePlanTool uses (see cost_model_agent.py's plan_variables) -- no results stores, no
    explore_data, just what plan code itself needs."""

    def load_data(filename: str) -> pd.DataFrame:
        return pd.read_csv(DATASET_DIR / filename)

    executor = LocalPythonExecutor(additional_authorized_imports=PLAN_AUTHORIZED_IMPORTS)
    executor.send_variables({
        "plan_name": plan_name,
        "load_data": load_data,
        "PhysicalPipeline": PhysicalPipeline,
        "pz": pz,
    })
    executor.send_tools({})
    result = executor(code)
    pipeline = result.output
    if not isinstance(pipeline, PhysicalPipeline):
        raise TypeError(
            f"Plan code for {plan_name!r} did not return a PhysicalPipeline as its last "
            f"expression (got {type(pipeline).__name__})."
        )
    return pipeline


def _load_plan_code(final_answer_path: Path, plan_name: str, runcount: str | None) -> str:
    answer = json.loads(final_answer_path.read_text())
    keys = [runcount] if runcount is not None else list(reversed(list(answer.keys())))
    for key in keys:
        if key not in answer:
            continue
        plan_codes = answer[key].get("plan_codes", {})
        if plan_name in plan_codes:
            return plan_codes[plan_name]
    raise SystemExit(
        f"{plan_name!r} not found in {final_answer_path} "
        f"(searched run(s) {keys!r}); available per run: "
        f"{ {k: list(v.get('plan_codes', {})) for k, v in answer.items()} }"
    )


def _score(output_df: pd.DataFrame, dataset_path: Path) -> None:
    """Score against CUAD's real annotations with the same code path the agent uses.

    `_evaluate` is quality_evaluator.score_plan's own body; score_plan is just avg_f1 off the
    front of it, and this prints the full per-metric breakdown instead -- which is the whole
    point of this script.

    `dataset_path` is the document POPULATION the metrics are computed over: the CSV the plan
    just ran on, not its output, so a document the plan dropped is charged as an empty
    prediction rather than vanishing from the measurement (see _evaluate's own note).
    """
    normalized = normalize_eval_df(output_df, "cuad", 1)
    metrics = _evaluate(normalized, cuad_ground_truth_csv(), dataset_path)
    if not metrics:
        print("no predicted records -- normalize_eval_df found no filename + clauses "
              "shape (e.g. the plan projected away the identifier column); skipping scoring")
        return

    print(f"=== per-metric precision/recall ({len(normalized)} documents) ===")
    for metric, pr in metrics["per_metric"].items():
        print(f"  {metric:38s} precision={pr['precision']!s:<8} recall={pr['recall']!s}")
    print()
    print(
        f"avg_f1={metrics['avg_f1']}  avg_precision={metrics['avg_precision']}  "
        f"avg_recall={metrics['avg_recall']}  nan_fraction={metrics['nan_fraction']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-name", required=True, help='e.g. "p1" -- a key in the final-answer JSON\'s plan_codes')
    parser.add_argument("--final-answer", type=Path, default=DEFAULT_FINAL_ANSWER_PATH, help="final_answer JSON to read plan_codes from")
    parser.add_argument("--runcount", default=None, help="which top-level run key to read plan_codes from (default: search every run, most recent first)")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV of documents to run the plan on -- e.g. the prepared dataset or optimize-set CSV written by prepare_cuad_data.py. Also the population scoring is computed over. Makes real LLM calls, so cost scales with its row count")
    parser.add_argument("--save-csv", type=Path, default=None, help="also save the full (untruncated) output DataFrame here")
    args = parser.parse_args()

    # run_subset() writes a fresh random sample to this path when it does not exist, which would
    # silently run something other than what was asked for -- so require the file up front.
    if not args.dataset.exists():
        parser.error(f"--dataset not found: {args.dataset}. Run prepare_cuad_data.py first.")

    patch_litellm_for_openrouter()

    code = _load_plan_code(args.final_answer, args.plan_name, args.runcount)
    pipeline = _build_pipeline(code, args.plan_name)

    # run_subset() with an existing cache path just executes that file's rows verbatim -- no
    # sampling -- so it runs whichever dataset was passed, of any size.
    _per_op_list, plan_context, _plan_dict = pipeline.run_subset(
        subset_cache_path=str(args.dataset)
    )
    output_df = pd.DataFrame(plan_context.output_records)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(f"=== raw output ({len(output_df)} rows) ===")
    print(output_df.to_string())
    print()

    if args.save_csv:
        output_df.to_csv(args.save_csv, index=False)
        print(f"saved -> {args.save_csv}")
        print()

    _score(output_df, args.dataset)


if __name__ == "__main__":
    main()
