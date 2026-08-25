"""Run cost-model optimization from a benchmark configuration file.

Benchmark YAML owns query/data/evaluation details. The constants below own the
optimization policy, so the same runner can be reused for another benchmark.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd
import yaml

from agent_cost_model.opt_agent.cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore
from agent_cost_model.opt_agent.llm_sampler import LLM_Sampler
from agent_cost_model.paths import RESULTS_DIR, SEMBENCH_ROOT

# Optimization policy: intentionally kept in code rather than benchmark YAML.
# MODEL = "openai/gpt-5.4"
MODEL = "openai/gpt-5" # for matching docetl
HELPER_MODEL = "openai/gpt-5.4"
MAX_STEPS = 40
FINAL_EVAL_RUNS = 2
AGENT_TYPE = "execute_oracle_sampler"
RANDOM_SUBSET_SIZE = 10
RANDOM_SUBSET_SEED = 42


def _format(value: str, context: dict[str, Any]) -> str:
    return value.format(**context)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _load_query(config: dict[str, Any], context: dict[str, Any]) -> tuple[str, str, list[str]]:
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
        path = Path(_format(source["path"], context))
        with path.open("rb") as handle:
            query = tomllib.load(handle)
        return task, metric, query["metadata"].get("modalities", [])
    if source["type"] == "text":
        return task, metric, source.get("modalities", ["text"])
    raise ValueError(f"Unsupported query source type: {source['type']!r}")


def _load_adapter(module_name: str):
    module = importlib.import_module(module_name)
    return module.QualityEvaluator, module.normalize_eval_df, module.load_evaluator


def _write_random_subset(source_df: pd.DataFrame, subset_path: Path) -> None:
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    source_df.sample(
        n=min(RANDOM_SUBSET_SIZE, len(source_df)), random_state=RANDOM_SUBSET_SEED
    ).to_csv(subset_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="benchmark YAML file")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--runcount", type=int, default=1)
    parser.add_argument(
        "--opt-subsample",
        action="store_true",
        help="create or replace the optimization subset before running",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="embedding model required when --sample-method top-k",
    )
    parser.add_argument(
        "--sample-method",
        choices=["random", "top-k"],
        default="random",
        help="subset method used with --opt-subsample (default: random)",
    )
    parser.add_argument("--keyword", action="store_true")
    parser.add_argument("--no-image-emb", action="store_true")
    parser.add_argument(
        "--use-oracle-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "score plan quality against an oracle-substituted pipeline run (default). "
            "Pass --no-use-oracle-ground-truth to score against the benchmark's real "
            "ground truth instead (e.g. CUAD, which has real annotations and doesn't "
            "need an oracle-generated pseudo-ground-truth)."
        ),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    benchmark = _load_yaml(config_path)
    use_case_config = benchmark["use_cases"][args.use_case]
    context = {
        "query_id": args.query_id,
        "scale_factor": args.scale_factor,
        "benchmark_dir": str(config_path.parent),
        "results_dir": str(RESULTS_DIR),
        "sembench_root": str(SEMBENCH_ROOT),
        "use_case": args.use_case,
    }
    task, eval_metric, modalities = _load_query(use_case_config, context)
    dataset_dir = Path(_format(use_case_config["dataset"]["directory"], context))
    source_csv = dataset_dir / _format(use_case_config["dataset"]["source_csv"], context)
    subset_path = Path(_format(use_case_config["subset_path"], context))
    ground_truth_dir = Path(_format(use_case_config["ground_truth_dir"], context))
    final_answer_path = Path(_format(benchmark["final_answer_path"], context | {"agent_type": AGENT_TYPE}))
    if args.opt_subsample and args.sample_method == "top-k" and not args.embedding_model:
        parser.error("--embedding-model is required with --opt-subsample --sample-method top-k")
    if not args.opt_subsample and not subset_path.exists():
        parser.error(
            f"Optimization subset not found: {subset_path}. "
            "Pass --opt-subsample to create it."
        )
    if not source_csv.exists():
        raise FileNotFoundError(f"Query dataset not found: {source_csv}. Run the benchmark extractor first.")

    dataset_config = use_case_config["dataset"]
    source_df = pd.read_csv(source_csv, dtype={"idx": str})
    quality_evaluator_cls, normalize_eval_df, evaluator_loader = _load_adapter(benchmark["quality_adapter"])
    llm = OpenRouterClient(MODEL, reasoning_effort="medium")
    if args.opt_subsample and args.sample_method == "top-k":
        image_dir = dataset_dir / dataset_config.get("image_subdir", "images")
        LLM_Sampler(
            query_id=args.query_id,
            use_case=args.use_case,
            scale_factor=args.scale_factor,
            query_text=task,
            df=source_df,
            llm_client=llm,
            embedding_model=args.embedding_model,
            id_col="idx",
            image_dir=str(image_dir) if "image" in modalities else None,
            cache_dir=str(RESULTS_DIR / "sampling" / benchmark["name"] / args.use_case),
            subset_out_path=str(subset_path),
            sample_method="topk",
            keyword=args.keyword,
            no_image_emb=args.no_image_emb,
        ).sample()
    elif args.opt_subsample:
        _write_random_subset(source_df, subset_path)

    agent = CostModelAgent(
        llm, max_steps=MAX_STEPS, verbose=True,
        agent_dir=f"{AGENT_TYPE}_agent", use_case=args.use_case, helper_model=HELPER_MODEL,
        use_oracle_ground_truth=args.use_oracle_ground_truth,
    )
    answer = agent.run(
        task, plans={}, plan_results=ResultsStore([]), op_results=ResultsStore([]), mode=AGENT_TYPE,
        query_info={
            "use_case": args.use_case,
            "scale_factor": args.scale_factor,
            "query_id": args.query_id,
            "eval_metric": eval_metric,
            "final_eval_runs": FINAL_EVAL_RUNS,
            "data_dir": str(dataset_dir),
            "gt_dir": str(ground_truth_dir),
            "subset_path": str(subset_path),
            "quality_evaluator_cls": quality_evaluator_cls,
            "normalize_eval_df": normalize_eval_df,
            "evaluator_loader": evaluator_loader,
            "image_subdir": dataset_config.get("image_subdir", "images"),
            "runcount": args.runcount,
        },
    )
    final_answer_path.parent.mkdir(parents=True, exist_ok=True)
    answers = json.loads(final_answer_path.read_text()) if final_answer_path.exists() else {}
    answers[str(args.runcount or 0)] = answer
    final_answer_path.write_text(json.dumps(answers, indent=2))
    print(f"Final answer written to {final_answer_path}")


if __name__ == "__main__":
    main()
