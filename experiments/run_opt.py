"""Run cost-model optimization from a benchmark configuration file.

Benchmark YAML owns query/data/evaluation details and, via `results_prefix`, the segments
of the results path that are benchmark-specific (SemBench keys on use case and scale factor;
CUAD on neither). The constants below own the optimization policy, so the same runner can be
reused for another benchmark by pointing --config elsewhere.

The runner name (AGENT_TYPE) is the axis you compare across: it names one optimizer/variant
-- `execute_oracle_sampler`, `customCost`, ... alongside external ones like `MOAR` -- and
occupies the last segment of `results_prefix`. Everything below that is this runner's own
layout (trajectory/, metrics/, final_answer/, ...).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd

from agent_cost_model.experiments.config import build_context, format_path, load_benchmark
from agent_cost_model.experiments.external_repo import activate
from agent_cost_model.opt_agent.cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore
from agent_cost_model.opt_agent.llm_sampler import LLM_Sampler
from agent_cost_model.paths import RESULTS_DIR

# Optimization policy: intentionally kept in code rather than benchmark YAML.
# MODEL = "openai/gpt-5.4"
MODEL = "openai/gpt-5" # for matching docetl
HELPER_MODEL = "openai/gpt-5.4"
MAX_STEPS = 40
NUM_FINAL_EVAL_RUNS = 2
AGENT_TYPE = "execute_oracle_sampler"
RANDOM_SUBSET_SIZE = 10
RANDOM_SUBSET_SEED = 42
RANDOM_OP_SAMPLER_SEED = 42


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
        path = Path(source["path"].format(**context))
        with path.open("rb") as handle:
            query = tomllib.load(handle)
        return task, metric, query["metadata"].get("modalities", [])
    if source["type"] == "text":
        return task, metric, source.get("modalities", ["text"])
    raise ValueError(f"Unsupported query source type: {source['type']!r}")


def _load_adapter(module_name: str):
    """The benchmark's adapter: its QualityEvaluator subclass, plus the output-reshaping function
    the engine applies before scoring. Scoring itself is reached only through the evaluator, so
    no scorer is exported here."""
    module = importlib.import_module(module_name)
    return module.QualityEvaluator, module.normalize_eval_df


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
    benchmark = load_benchmark(config_path)
    use_case_config = benchmark["use_cases"][args.use_case]
    # Resolve the benchmark's external checkout and make its own modules importable. Read-only:
    # nothing this run produces is written there.
    repo_root = activate(benchmark.get("external_repo"))
    context = build_context(
        benchmark, runner=AGENT_TYPE, use_case=args.use_case,
        scale_factor=args.scale_factor, query_id=args.query_id,
        repo_root=repo_root or "",
    )
    task, eval_metric, modalities = _load_query(use_case_config, context)
    dataset_dir = format_path(use_case_config["dataset"]["directory"], context)
    source_csv = dataset_dir / use_case_config["dataset"]["source_csv"].format(**context)
    subset_path = format_path(use_case_config["subset_path"], context)
    ground_truth_path = format_path(use_case_config["ground_truth_path"], context)
    results_prefix = format_path(benchmark["results_prefix"], context)
    final_answer_path = results_prefix / "final_answer" / f"Q{args.query_id}.json"
    if args.opt_subsample and args.sample_method == "top-k":
        if not args.embedding_model:
            parser.error("--embedding-model is required with --opt-subsample --sample-method top-k")
        if not benchmark.get("sampling_dir"):
            parser.error(
                f"{config_path.name} declares no `sampling_dir`, so top-k sampling has nowhere "
                "to cache embeddings. Add one, or use --sample-method random."
            )
    if not args.opt_subsample and not subset_path.exists():
        parser.error(
            f"Optimization subset not found: {subset_path}. "
            "Pass --opt-subsample to create it."
        )
    if not source_csv.exists():
        raise FileNotFoundError(f"Query dataset not found: {source_csv}. Run the benchmark extractor first.")

    dataset_config = use_case_config["dataset"]
    source_df = pd.read_csv(source_csv, dtype={"idx": str})
    quality_evaluator_cls, normalize_eval_df = _load_adapter(benchmark["quality_adapter"])
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
            cache_dir=str(format_path(benchmark["sampling_dir"], context)),
            subset_out_path=str(subset_path),
            sample_method="topk",
            keyword=args.keyword,
            no_image_emb=args.no_image_emb,
        ).sample()
    elif args.opt_subsample:
        _write_random_subset(source_df, subset_path)

    agent = CostModelAgent(
        llm, max_steps=MAX_STEPS, verbose=True,
        agent_dir=AGENT_TYPE, results_prefix=results_prefix,
        use_case=args.use_case, helper_model=HELPER_MODEL,
        use_oracle_ground_truth=args.use_oracle_ground_truth,
    )
    answer = agent.run(
        task, plans={}, plan_results=ResultsStore([]), op_results=ResultsStore([]), mode=AGENT_TYPE,
        query_info={
            "use_case": args.use_case,
            "scale_factor": args.scale_factor,
            "query_id": args.query_id,
            "eval_metric": eval_metric,
            "num_final_eval_runs": NUM_FINAL_EVAL_RUNS,
            "data_dir": str(dataset_dir),
            "gt_path": str(ground_truth_path),
            "subset_path": str(subset_path),
            "dataset_path": str(source_csv),
            "quality_evaluator_cls": quality_evaluator_cls,
            "normalize_eval_df": normalize_eval_df,
            "op_sample_seed": RANDOM_OP_SAMPLER_SEED,
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
