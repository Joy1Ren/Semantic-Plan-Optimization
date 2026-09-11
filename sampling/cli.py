"""Build the optimization subset for one query, without running the optimizer.

Two ways to say where things go:

  --config (recommended) resolves every path from a benchmark.yaml exactly as
  experiments/run_opt.py does -- same dataset, same subset_path, same embedding cache, same
  task prompt. A later `run_opt.py` run **without** `--opt-subsample` then picks that subset up
  as-is, so sampling and optimization can be paid for separately:

      python -m agent_cost_model.sampling.cli \\
        --config agent_cost_model/experiments/SemBench/benchmark.yaml \\
        --use-case ecomm --scale-factor 500 --query-id 11 \\
        --mode agentic --embedding-model qwen/qwen3-embedding-8b

      python -m agent_cost_model.experiments.run_opt \\
        --config agent_cost_model/experiments/SemBench/benchmark.yaml \\
        --use-case ecomm --scale-factor 500 --query-id 11        # reuses the subset above

  Explicit paths (--csv/--subset-out/--results/--cache-dir) for a table that is not part of a
  benchmark. Nothing downstream will find that subset on its own.

Either way the run writes the subset CSV, the per-row scores CSV, the cost record in
sampling_results.json, and (for one-shot/agentic) the prompt trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent_cost_model.sampling.constants import (
    DEFAULT_ALPHA,
    DEFAULT_EMBED_WORKERS,
    DEFAULT_RAG_EMBEDDING_MODEL,
    MODES,
)
from agent_cost_model.sampling.data_sampler import DataSampler
from agent_cost_model.sampling.results import mode_subset_path


def _from_benchmark(args, parser) -> dict:
    """Resolve dataset, paths, and task prompt the way run_opt.py does."""
    from agent_cost_model.experiments.config import (
        build_context,
        format_path,
        load_benchmark,
        load_query,
    )
    from agent_cost_model.experiments.external_repo import activate

    config_path = Path(args.config).resolve()
    benchmark = load_benchmark(config_path)
    if args.use_case not in benchmark["use_cases"]:
        parser.error(
            f"{config_path.name} has no use case {args.use_case!r} "
            f"(has: {sorted(benchmark['use_cases'])})"
        )
    use_case_config = benchmark["use_cases"][args.use_case]
    repo_root = activate(benchmark.get("external_repo"))
    if args.scale_factor is None and "{scale_factor}" in json.dumps(benchmark):
        parser.error(f"{config_path.name} keys its paths on a scale factor — pass --scale-factor.")

    missing = [k for k in ("sampling_dir", "sampling_results_path") if not benchmark.get(k)]
    if missing:
        parser.error(
            f"{config_path.name} declares no {' or '.join(missing)}, so there is nowhere to "
            "cache embeddings or record what was selected."
        )

    # runner="" — sampling produces a shared input to every runner, not one runner's output,
    # so it must not land under a runner-specific prefix.
    context = build_context(
        benchmark, runner="", use_case=args.use_case, scale_factor=args.scale_factor,
        query_id=args.query_id, repo_root=repo_root or "",
    )
    task, _metric, modalities = load_query(use_case_config, context)
    dataset_dir = format_path(use_case_config["dataset"]["directory"], context)
    source_csv = dataset_dir / use_case_config["dataset"]["source_csv"].format(**context)
    if not source_csv.exists():
        parser.error(f"query dataset not found: {source_csv}. Run the benchmark extractor first.")

    image_subdir = use_case_config["dataset"].get("image_subdir", "images")
    return {
        "query_text": task,
        "source_csv": source_csv,
        "image_dir": str(dataset_dir / image_subdir) if "image" in modalities else None,
        "text_cols": None if "text" in modalities else [],
        "cache_dir": str(format_path(benchmark["sampling_dir"], context)),
        "subset_out_path": str(format_path(use_case_config["subset_path"], context)),
        "results_path": str(format_path(benchmark["sampling_results_path"], context)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="benchmark YAML; resolves every path below")
    ap.add_argument("--use-case", default="adhoc")
    ap.add_argument("--scale-factor", type=int, default=None)
    ap.add_argument("--query-id", default="0")

    ap.add_argument("--query", default=None, help="query text (only without --config)")
    ap.add_argument("--csv", default=None, help="source table (only without --config)")
    ap.add_argument("--cache-dir", default=None, help="embedding cache dir (only without --config)")
    ap.add_argument("--subset-out", default=None, help="subset CSV path (only without --config)")
    ap.add_argument("--results", default=None, help="sampling_results.json (only without --config)")
    ap.add_argument("--images", default=None, help="directory of {id}.jpg (only without --config)")

    ap.add_argument("--mode", choices=list(MODES), default="agentic")
    ap.add_argument("--id-col", default="idx")
    ap.add_argument("--model", default="openai/gpt-5", help="planning LLM (OpenRouter id)")
    ap.add_argument("--embedding-model", default=DEFAULT_RAG_EMBEDDING_MODEL)
    ap.add_argument("--sample-size", type=int, required=True,
                    help="rows to select into the optimization subset")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--embed-workers", type=int, default=DEFAULT_EMBED_WORKERS,
                    help="concurrent embedding requests (default: %(default)s)")
    ap.add_argument("--no-image-emb", action="store_true")
    ap.add_argument(
        "--idx-specification", action="store_true",
        help=(
            "let the agent name specific rows to include, by id, in a `preselected_idx` list. "
            "Those rows enter the subset unscored and the strata allocate k minus their count"
        ),
    )
    args = ap.parse_args()

    if args.config:
        conflicting = [
            name for name in ("query", "csv", "cache_dir", "subset_out", "results", "images")
            if getattr(args, name)
        ]
        if conflicting:
            ap.error(
                f"--config resolves these itself; drop {', '.join('--' + c.replace('_', '-') for c in conflicting)}"
            )
        resolved = _from_benchmark(args, ap)
    else:
        required = {"query": args.query, "csv": args.csv, "cache-dir": args.cache_dir,
                    "subset-out": args.subset_out, "results": args.results}
        missing = [f"--{k}" for k, v in required.items() if not v]
        if missing:
            ap.error(f"without --config these are required: {', '.join(missing)}")
        resolved = {
            "query_text": args.query,
            "source_csv": Path(args.csv),
            "image_dir": args.images,
            "text_cols": None,
            "cache_dir": args.cache_dir,
            "subset_out_path": args.subset_out,
            "results_path": args.results,
        }

    # Mode-suffixed here rather than in the resolution branches above, so the config path and the
    # hand-passed --subset-out get the same treatment.
    resolved["subset_out_path"] = str(mode_subset_path(resolved["subset_out_path"], args.mode))

    llm = None
    if args.mode not in ("baseline", "random"):
        from agent_cost_model.opt_agent.llm_client import OpenRouterClient

        llm = OpenRouterClient(args.model, reasoning_effort="medium")

    df = pd.read_csv(resolved["source_csv"], dtype={args.id_col: str})
    result = DataSampler(
        query_id=args.query_id,
        use_case=args.use_case,
        query_text=resolved["query_text"],
        df=df,
        mode=args.mode,
        llm_client=llm,
        embedding_model=args.embedding_model,
        id_col=args.id_col,
        image_dir=resolved["image_dir"],
        text_cols=resolved["text_cols"],
        cache_dir=resolved["cache_dir"],
        subset_out_path=resolved["subset_out_path"],
        results_path=resolved["results_path"],
        sample_size=args.sample_size,
        alpha=args.alpha,
        no_image_emb=args.no_image_emb,
        idx_specification=args.idx_specification,
        embed_workers=args.embed_workers,
        llm_model=None if llm is None else args.model,
    ).sample()

    print(f"\nselected {len(result.sampled_ids)} rows ({result.stop_reason}): {result.sampled_ids}")
    for record in result.rounds:
        # A round with no strata is the agent declining to target anything, which is a real
        # outcome rather than a missing spec -- print the decision instead of nothing.
        if not record.strata:
            print(f"  round {record.round}: no scorer requested — sampled uniformly")
            continue
        for i, st in enumerate(record.strata, 1):
            # The recorded spec carries only the components that were actually scored -- an
            # image-only query has no BM25 -- so print what is there rather than assuming.
            spec = st.spec
            cols = spec.get("cols", {})
            where = f"  round {record.round}"
            if len(record.strata) > 1:
                name = f" {st.label}" if st.label else ""
                where += f" stratum {i}{name} ({st.m} row(s))"
            head = f"{where}: alpha={spec.get('alpha')}"
            if spec.get("filter"):
                head += f" filter={spec['filter']!r}"
            lines = [head]
            if "bm25_query" in spec:
                lines.append(f"    bm25={spec['bm25_query']!r} over {cols.get('bm25', [])}")
            if spec.get("keyphrases"):
                lines.append(
                    f"    keyphrases={spec['keyphrases']} over {cols.get('keyphrase', [])}"
                )
            if "dense_query" in spec:
                lines.append(f"    dense={spec['dense_query']!r} over {cols.get('dense', [])}")
            print("\n".join(lines))
        if record.fill_ids:
            print(f"    (+{len(record.fill_ids)} row(s) drawn uniformly to reach the subset size)")
    print(f"\nsubset  -> {result.subset_path}")
    print(f"scores  -> {result.scores_path}")
    print(f"results -> {result.results_path}")
    print(f"cost    -> {json.dumps(result.cost.to_json(result.mode), indent=2)}")
    print(f"latency -> {json.dumps(result.cost.latency_json(result.mode), indent=2)}")
    if args.config:
        print(
            "\nRe-run run_opt.py for this query WITHOUT --opt-subsample to optimize against "
            "this subset."
        )


if __name__ == "__main__":
    main()
