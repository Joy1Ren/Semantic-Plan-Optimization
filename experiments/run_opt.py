"""Run cost-model optimization from a benchmark configuration file.

Benchmark YAML owns query/data/evaluation details and, via `results_prefix`, the segments
of the results path that are benchmark-specific (SemBench keys on use case and scale factor;
CUAD on neither). The constants below own the optimization policy, so the same runner can be
reused for another benchmark by pointing --config elsewhere.

The runner name (--agent_type) is the axis you compare across: it names one optimizer/variant
-- `execute_oracle_sampler`, `customCost`, ... alongside external ones like `MOAR` -- and
occupies the last segment of `results_prefix`. Everything below that is this runner's own
layout (trajectory/, metrics/, final_answer/, ...).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_cost_model.experiments.config import (
    build_context,
    format_path,
    load_benchmark,
    load_query,
)
from agent_cost_model.experiments.external_repo import activate
from agent_cost_model.opt_agent.cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore
from agent_cost_model.sampling import MODES as SAMPLER_MODES, DataSampler, mode_subset_path
from agent_cost_model.sampling.constants import (
    DEFAULT_ALPHA,
    DEFAULT_EMBED_WORKERS,
    DEFAULT_RAG_EMBEDDING_MODEL,
)

# Optimization policy: intentionally kept in code rather than benchmark YAML.
# MODEL = "openai/gpt-5.4"
MODEL = "openai/gpt-5" # for matching docetl
MAX_STEPS = 40
# How many times the chosen plan is executed on the FULL dataset once the search ends.
# Whether that happens at all is per-invocation: --final-eval/--no-final-eval.
NUM_FINAL_EVAL_RUNS = 2
USE_CHECKER = True
CHECKER_MODEL = None
CHECKER_REASONING_EFFORT = "high"
CHECKER_EVERY = 3


def _report_reused_subset(
    benchmark: dict[str, Any], context: dict[str, Any], subset_path: Path, query_id: int
) -> None:
    """Say where the pre-existing subset came from, when this run did not build one.

    Without --opt-subsample the run silently optimizes against whatever subset is already on
    disk -- possibly written weeks ago by a different sampling mode. Naming its provenance is
    the difference between a deliberate reuse (build the subset once with the sampler CLI, then
    optimize against it repeatedly) and an accidental one.
    """
    template = benchmark.get("sampling_results_path")
    if not template:
        print(f"Reusing existing subset: {subset_path}")
        return
    results_path = format_path(template, context)
    record = None
    if results_path.exists():
        try:
            entry = json.loads(results_path.read_text()).get(str(query_id))
        except Exception:
            entry = None
        if isinstance(entry, dict) and "sampled_ids" in entry:
            # A flat, pre-nesting record from the old sampler. It still describes a real run,
            # so report it rather than claiming there is nothing here.
            record = {**entry, "mode": f"{entry.get('sample_method', 'legacy')} (legacy record)"}
        elif isinstance(entry, dict):
            # {query_id: {mode: record}} -- pick the most recently written mode we can identify.
            modes = [m for m, r in entry.items() if isinstance(r, dict)]
            if modes:
                record = entry[modes[-1]]
    if record:
        cost = record.get("cost", {})
        print(
            f"Reusing existing subset: {subset_path}\n"
            f"  built by: {record.get('mode', '?')} "
            f"({record.get('n_rounds', '?')} round(s), {record.get('stop_reason', '?')}), "
            f"{len(record.get('sampled_ids', []))} rows, "
            f"${cost.get('total_usd_paid_now', 0.0):.4f} paid at sampling time"
        )
    else:
        print(
            f"Reusing existing subset: {subset_path}\n"
            f"  no sampling record for query {query_id} in {results_path} — it predates "
            "DataSampler recording every mode, random included."
        )


def _load_adapter(module_name: str):
    """The benchmark's adapter: its QualityEvaluator subclass, plus the output-reshaping function
    the engine applies before scoring. Scoring itself is reached only through the evaluator, so
    no scorer is exported here."""
    module = importlib.import_module(module_name)
    return module.QualityEvaluator, module.normalize_eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="benchmark YAML file")
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument(
        "--agent_type",
        required=True,
        help=(
            "runner name: the last segment of results_prefix, and the axis you compare across "
            "(e.g. execute_optimizer_agentic, execute_optimizer_random)"
        ),
    )
    # Only benchmarks with a scale-factor axis need this: SemBench keys its dataset, subset,
    # ground-truth, and results paths on it, while CUAD's benchmark.yaml references it nowhere.
    parser.add_argument("--scale-factor", type=int, default=None)
    parser.add_argument("--runcount", type=int, default=1)
    parser.add_argument(
        "--opt-subsample",
        action="store_true",
        help="create or replace the optimization subset before running",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="embedding model; required for every --sample-method except random",
    )
    parser.add_argument(
        "--sample-method",
        # Every mode, random included, goes through DataSampler -- there is no bypass, so every
        # subset on disk was actually built by a sampler and has a sampling_results.json record.
        choices=list(SAMPLER_MODES),
        required=True,
        help=(
            "subset method used with --opt-subsample (default: random). "
            "random = uniform draw; baseline = BM25 + embedding over the raw query, no LLM call; "
            "one-shot = one LLM call writes the retrieval spec; "
            "agentic = the same, then up to two revisions after inspecting what was retrieved"
        ),
    )
    parser.add_argument(
        "--idx-specification",
        action="store_true",
        help=(
            "guided sampling only: let the sampling agent name specific rows to include, by id, "
            "in a `preselected_idx` list. Those rows enter the subset unscored, and the strata "
            "allocate --sample-size minus their count"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "rows in the optimization subset. Required with --opt-subsample; drives every "
            "--sample-method including random, so subsets stay comparable across modes. "
            "Deliberately has no default: the subset size is an experiment parameter, and a "
            "silent one makes two runs look comparable when they are not."
        ),
    )
    parser.add_argument(
        "--sample-alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="weight on the sparse (BM25 + keyphrase) half of the sampling score, 0.0-1.0",
    )
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=DEFAULT_EMBED_WORKERS,
        help="concurrent embedding requests during sampling (default: %(default)s)",
    )
    parser.add_argument("--no-image-emb", action="store_true")
    parser.add_argument(
        "--final-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after the search, execute the selected plan on the full dataset and score it "
            "against the benchmark's real ground truth (default). Pass "
            "--no-final-eval to stop after the search — the metrics entry is still written, "
            "with no run1/run2 full-dataset numbers."
        ),
    )
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
    # --scale-factor is optional because only some benchmarks have that axis. A benchmark whose
    # path templates DO reference it still needs one -- without this its paths would silently
    # resolve to a literal "sf_None" directory instead of failing here.
    if args.scale_factor is None and "{scale_factor}" in json.dumps(benchmark):
        parser.error(f"{config_path.name} keys its paths on a scale factor — pass --scale-factor.")
    context = build_context(
        benchmark, runner=args.agent_type, use_case=args.use_case,
        scale_factor=args.scale_factor, query_id=args.query_id,
        repo_root=repo_root or "",
    )
    task, eval_metric, modalities = load_query(use_case_config, context)
    dataset_dir = format_path(use_case_config["dataset"]["directory"], context)
    source_csv = dataset_dir / use_case_config["dataset"]["source_csv"].format(**context)
    # One subset file per sampling method, so comparing methods on a query does not have each run
    # overwrite the last one's subset. `--sample-method` therefore also picks WHICH subset a
    # reusing run (no --opt-subsample) optimizes against.
    subset_path = mode_subset_path(
        format_path(use_case_config["subset_path"], context), args.sample_method
    )
    ground_truth_path = format_path(use_case_config["ground_truth_path"], context)
    results_prefix = format_path(benchmark["results_prefix"], context)
    final_answer_path = results_prefix / "final_answer" / f"Q{args.query_id}.json"
    if args.opt_subsample:
        # random needs no embedding model -- DataSampler's random mode never scores -- but every
        # mode, random included, now runs through DataSampler, so all of them need somewhere to
        # cache embeddings and record what they selected.
        if args.sample_method != "random" and not args.embedding_model:
            parser.error(
                f"--embedding-model is required with --opt-subsample --sample-method {args.sample_method}"
            )
        missing = [k for k in ("sampling_dir", "sampling_results_path") if not benchmark.get(k)]
        if missing:
            parser.error(
                f"{config_path.name} declares no {' or '.join(missing)}, so sampling has "
                "nowhere to cache embeddings or record what it selected."
            )
    if args.opt_subsample and args.sample_size is None:
        parser.error("--opt-subsample needs --sample-size: how many rows the subset should hold.")
    if not args.opt_subsample and not subset_path.exists():
        parser.error(
            f"Optimization subset not found: {subset_path}. "
            "Pass --opt-subsample to create it."
        )
    if not source_csv.exists():
        raise FileNotFoundError(f"Query dataset not found: {source_csv}. Run the benchmark extractor first.")

    dataset_config = use_case_config["dataset"]
    # Which column identifies a row is the benchmark's to declare, not this runner's: SemBench's
    # movie table already spends `idx` on the movie a review is about, so its rows are keyed by
    # `reviewId` instead. Read as str so ids never round-trip through float.
    id_col = dataset_config.get("id_col", "idx")
    source_df = pd.read_csv(source_csv, dtype={id_col: str})
    quality_evaluator_cls, normalize_eval_df = _load_adapter(benchmark["quality_adapter"])
    llm = OpenRouterClient(MODEL, reasoning_effort="medium")
    if args.opt_subsample:
        image_dir = dataset_dir / dataset_config.get("image_subdir", "images")
        # baseline needs no chat model, only embeddings; random needs neither; the two LLM
        # modes plan with `llm`.
        no_llm = args.sample_method in ("baseline", "random")
        DataSampler(
            query_id=args.query_id,
            use_case=args.use_case,
            query_text=task,
            df=source_df,
            mode=args.sample_method,
            llm_client=None if no_llm else llm,
            embedding_model=args.embedding_model or DEFAULT_RAG_EMBEDDING_MODEL,
            id_col=id_col,
            image_dir=str(image_dir) if "image" in modalities else None,
            # An image-only query's table carries no scorable text, so BM25 and the keyphrase
            # matcher have nothing to read and the dense score rides on the image alone.
            text_cols=None if "text" in modalities else [],
            cache_dir=str(format_path(benchmark["sampling_dir"], context)),
            subset_out_path=str(subset_path),
            results_path=str(format_path(benchmark["sampling_results_path"], context)),
            sample_size=args.sample_size,
            alpha=args.sample_alpha,
            llm_model=None if no_llm else MODEL,
            embed_workers=args.embed_workers,
            no_image_emb=args.no_image_emb,
            idx_specification=args.idx_specification,
        ).sample()
    else:
        _report_reused_subset(benchmark, context, subset_path, args.query_id)

    agent = CostModelAgent(
        llm, max_steps=MAX_STEPS, verbose=True,
        agent_dir=args.agent_type, results_prefix=results_prefix,
        use_case=args.use_case,
        use_oracle_ground_truth=args.use_oracle_ground_truth,
        use_checker=USE_CHECKER, checker_model=CHECKER_MODEL,
        checker_reasoning_effort=CHECKER_REASONING_EFFORT, checker_every=CHECKER_EVERY,
    )
    answer = agent.run(
        task, plans={}, plan_results=ResultsStore([]), op_results=ResultsStore([]), mode=args.agent_type,
        query_info={
            "use_case": args.use_case,
            "scale_factor": args.scale_factor,
            "query_id": args.query_id,
            "eval_metric": eval_metric,
            "run_final_eval": args.final_eval,
            "num_final_eval_runs": NUM_FINAL_EVAL_RUNS,
            "data_dir": str(dataset_dir),
            "gt_path": str(ground_truth_path),
            "subset_path": str(subset_path),
            "dataset_path": str(source_csv),
            "id_col": id_col,
            "quality_evaluator_cls": quality_evaluator_cls,
            "normalize_eval_df": normalize_eval_df,
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
