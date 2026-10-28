"""The main optimization agent: a bounded tool-loop agent that writes, executes, and
compares physical query plans in a sandboxed environment, converging on the best
cost/latency-quality trade-off.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time
from dataclasses import dataclass
from typing import Any

from agent_cost_model.opt_agent import run_log
from agent_cost_model.opt_agent.errors import ParseError, StepFailed
from agent_cost_model.opt_agent.llm_client import LLMClient, OpenRouterClient, patch_litellm_for_openrouter
from agent_cost_model.opt_agent.cost_model_types import (
    ResultsStore,
    get_op_type,
    iter_operators,
    run_key,
)
from agent_cost_model.opt_agent.step_parsing import _Step, _parse_step
from agent_cost_model.opt_agent.prompts import (
    _AVAILABLE_MODELS_TEXT,
    _COST_AWARENESS,
    _PHYSICAL_NONSEMANTIC_OPERATORS,
    _PHYSICAL_SEMANTIC_OPERATORS,
    _QUALITY_METRIC_DOCS,
    _SYSTEM_TEMPLATE,
    _lower_variance_text,
    _per_op_info_text,
    _per_op_quality_text,
    _quality_metric_reminder,
    _quality_metric_section,
)
from agent_cost_model.opt_agent.exploration_checker_agent import ExplorationCheckerAgent
from agent_cost_model.opt_agent.tools import (
    ExecutePlanTool,
    ExploreImagesTool,
    ExploreSampleTool,
    ExploreSchemaT,
    GetOpSamplesTool,
    ListFilesTool,
    Tool,
    WritePlanTool,
)


@dataclass
class _RunContext:
    """Everything `CostModelAgent._setup_run` wires up for the main loop to consume."""
    executor: Any
    system: str
    checker: "ExplorationCheckerAgent | None"
    quality_evaluator: Any
    plan_codes: dict
    plan_results_agg: ResultsStore


class CostModelAgent:
    """A bounded tool-loop agent that authors, applies, and refines semantic plans."""
    briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds.

        Your goal is to write physical plans, observe cost/latency/quality on a optimization SUBSET
        of the dataset, and iterate to converge on the best cost-quality trade-off that will generalize well
        onto the full dataset. Do not overfit optimizations on to the subset and do not hard code the plan.


        Suggested workflow:
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table's
           columns and format. You ALSO have direct access to the full CSVs via `explore_data(filename)`,
           which returns the whole table as a DataFrame. Use it to understand the structure of the file(s)
           and compute AGGREGATE statistics — e.g. how many rows contain a keyword/keyphrase, or
           `value_counts()` on a column to gauge how common an item or attribute is. This informs the
           selectivity of your filters, the size of the tables feeding a join, and the class distribution of
           a classification/search task (a rare target vs. common groups). Let it GUIDE how you design plans --
           do not rely on exploration to solve the query and do not hardcode row IDs.
           If the dataset has an `images/` folder, you can inspect a few product images with
           `explore_images(ids, question=None)` (up to 5, selected by the dataset's image id column) — useful to
           judge what a vision operator would work with. Pass `question` to ask the describer a natural-language
           question it must address for every image, on top of its default description. Images are shown once
           and are costly, so inspect just a couple.
        2. Write a plan with `write_plan(code, name, description, optimizations)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - ALWAYS pass `optimizations`, recording which optimizations this plan uses and how far
             each one is pushed (see the write_plan tool description). This is how the record of
             what you have and have not tried stays legible across plans.
           - NEVER hardcode row IDs to identify specific records (e.g. `filter(lambda row: row["id"].isin([3, 17, 42]))`
             is cheating and invalid). What you learn from exploration may only shape HOW you build a
             general plan, not which records it targets.
           - DO use semantic operators for conditions that need semantic judgment, and prefer
             deterministic, free Python/regex functions in plain `filter` / `map` when a condition
             can be expressed that way — e.g. `filter(lambda row: row["score"] >= 4)`,
             `filter(lambda row: row["description"].isin(["settee", "sofa", "couch"]))`, or
             `map(lambda row: {"red_freq_count": row["description"].str.count("red")}, ...)`.
             These are general and save on cost and latency.
           - DO start with baseline plans with an inexpensive model (e.g., gpt-6-luna) that have no or minimal cost/quality optimizations to understand
             the range of achievable quality/cost.
           - `plans[name]["plan"]` is populated immediately with the newly written PhysicalPipeline instance.
        3. Execute with `execute_plan(name)`:
           <no_lower_variance>
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           </no_lower_variance>
           <lower_variance>
           - Executes the plan {n_executions} times, in parallel, on the same reproducible sample
             of records (same records across all plans).
           - Appends one row per execution to `plan_results` (`execution` = 1..{n_executions}), and
             one row per plan to `plan_results_agg` with the mean and std over the executions
             (`quality_avg`, `quality_std`, `cost_avg`, `cost_std`, `latency_wallclock_avg`,
             `latency_wallclock_std`) and `n_unstable_records`, the number of records whose
             output differed between executions.
           </lower_variance>
           <per_op_info>
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           </per_op_info>
           - After execution, `plans[name]["plan"]` is updated with the executed PhysicalPipeline.
        <per_op_info>
        4. Evaluate with `plan_results.df` and `op_results.df`:
        </per_op_info>
        <no_per_op_info>
        4. Evaluate with `plan_results.df`:
        </no_per_op_info>
           - `quality`: 0–1 overall plan quality.
           - `quality_counts`: the TP/FP/FN/TN counts behind `quality`, for metrics built from counts.
           <lower_variance>
           - Compare plans on `plan_results_agg.df`; use `plan_results.df` to see how the
             individual executions differ.
           </lower_variance>
           <per_op_quality>
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck.
           </per_op_quality>
           <per_op_info>
           - Use `get_op_samples(plan_name, op_name)` — one operator per call — to inspect
             input/output pairs.
           </per_op_info>
        <per_op_info>
        5. Based on `plan_results.df` and `op_results.df`, write ONE new plan and execute it,
        </per_op_info>
        <no_per_op_info>
        5. Based on `plan_results.df`, write ONE new plan and execute it,
        </no_per_op_info>
           then repeat from step 2. Each new plan should embody a NEW optimization idea
           or a COMBINATION of ideas that worked. Make sure to write baseline plans and explore different
           optimization strategies. Levers for lowering cost and latency include:
           swapping to a cheaper/faster model, narrowing `depends_on` to fewer columns, truncating long text with a `map`
           before a semantic op, reordering to push selective filters earlier, removing images,
           or changing the logical structure. Levers for increasing quality include: swapping to a more capable model,
           being less aggressive with pre-filtering or truncation, or changing logical structure.
           - Don't just hunt for cost cuts that hold quality steady — also spend some plans actively
             trying to raise quality, then weigh whether the resulting cost/latency is worth it.
           - Do NOT overfit by fine-tuning a single knob (e.g. the exact truncation length):
             we run on a small subsample and micro-tuning is inefficient and prone to overfitting.
           - Do NOT chase one or two records: spending excessive time and cost on optimizations that only effect one or two records
            may be wasteful. Make changes that may generalize.
           - To prevent falling in a local minimum, make sure to also explore a wider range of the
            optimization space: try different models and different logical structures.
           - Stop and give your final answer when the gains over your current best plan are
             marginal, or you run low on steps.
        <lower_variance>

        ## Execution variance
        The same plan can score differently each time it is executed, so every plan is executed
        {n_executions} times.
        - Judge plans by `quality_avg`, and weigh the gap between two plans' averages against
          their `quality_std`: a gap is meaningful only if it is large relative to the std.
        - Lower variance is part of the goal. When `quality_avg` and cost are similar, prefer the
          plan with the lower `quality_std`; when that is similar too, prefer the simpler plan.
          Choose your final plan the same way.
        - Start simple. A simple plan is often already stable: measure it and tune its settings
          (wording, model, context) before adding operators. Keep an added operator (a verifier, a
          pre-filter, an ensemble) only if it beats the plan without it by more than the noise.
        - A small `n_unstable_records` means a few borderline records drive the variance:
          compare an operator's samples across executions with `get_op_samples(...,
          execution=k)` and make the wording decide them.
        - Ways to reduce variance, simplest first: state the condition or column description more
          precisely (define the borderline cases; give a short example of what counts and what
          does not); use a stronger model; or, if those fall short, ensemble -- get several
          independent LLM judgments (the same or different models and wordings) and consolidate
          them. Each judgment can also return a short reason column: an aggregator that reads the
          reasons can weigh the judgments, not just count them.
            # map: each judgment adds answer + reason columns; one final sem_map reads only those
            p.sem_map(cols=[{"name": "ans_1", ...}, {"name": "why_1", "type": str, ...}], model=...)  # likewise ans_2/why_2
            p.sem_map(cols=[{"name": "ans", ...}], model=..., depends_on=["ans_1", "why_1", "ans_2", "why_2"])
            # filter: sem_filters in a row AND together, so vote with bool sem_map columns instead
            p.sem_map(cols=[{"name": "vote_1", "type": bool, ...}, {"name": "why_1", "type": str, ...}], model=...)  # likewise vote_2/why_2
            p.filter(lambda r: bool(r["vote_1"]) or bool(r["vote_2"]))             # both no: dropped
            p.sem_filter("...", model=..., depends_on=[..., "why_1", "why_2"],    # decides only the splits
                         skip_if=lambda r: bool(r["vote_1"]) and bool(r["vote_2"]))  # both yes: kept
          An ensemble multiplies that operator's cost; weigh it like any other cost.
        </lower_variance>
        """)

    final_answer_doc = textwrap.dedent("""\
        Emit a JSON object summarizing the best plan:
          {
            "best_plan": {
              "name": "...",
              "rationale": "why this plan wins (cost/quality trade-off)"
            }
          }
        Use plain JSON literals only (no python expressions, no trailing commas).""")

    def __init__(
        self,
        llm: LLMClient,
        *,
        data_dir: str = "dataset/use_case",
        results_prefix: str | pathlib.Path | None = None,
        agent_dir: str = "no_name",
        use_case: str = "use_case",
        max_steps: int = 40,
        max_recover_retries: int = 1,
        context_budget_chars: int = 200_000,
        authorized_imports: list[str] | None = None,
        verbose: bool = True,
        oracle_model: str = "openai/o4-mini",
        oracle_reasoning_effort: str | None = "high",
        use_oracle_ground_truth: bool = True,
        oracle_operator_quality: bool = True,
        per_operator_info: bool = True,
        use_checker: bool = True,
        checker_model: str | None = None,
        checker_reasoning_effort: str | None = "high",
        checker_every: int = 3,
        max_checker_vetoes: int = 2,
        cost_aware_opt: bool = False,
        lower_variance: bool = False,
        executions_per_plan: int = 3,
        image_describer_model: str = "google/gemini-2.5-flash-lite",
    ) -> None:
        self.llm = llm
        self.image_describer_model = image_describer_model
        self.agent_dir = agent_dir
        self.use_case = use_case
        self.data_dir = data_dir
        self.results_prefix = pathlib.Path(results_prefix)
        self.oracle_model = oracle_model
        self.oracle_reasoning_effort = oracle_reasoning_effort
        # When False, quality is scored against the benchmark's real ground truth instead of
        # an oracle-substituted plan. Per-op oracle judging is a separate switch, below.
        self.use_oracle_ground_truth = use_oracle_ground_truth
        # When False, the oracle does not judge per-operator quality: no `per_sem_op_quality`
        # is computed or shown to the agent or the exploration checker.
        self.oracle_operator_quality = oracle_operator_quality
        # When False, the agent is shown plan-level results only: no per-operator stats
        # (`op_summary`, `op_results`) and no per-operator samples (`get_op_samples`).
        if oracle_operator_quality and not per_operator_info:
            raise ValueError(
                "per_operator_info=False requires oracle_operator_quality=False: per-operator "
                "quality is per-operator information."
            )
        self.per_operator_info = per_operator_info
        # Lower-variance search: every plan is executed `executions_per_plan` times in parallel,
        # the agent and the exploration checker are shown the spread, and the oracle's
        # per-operator judging is turned off to pay for the extra executions.
        self.lower_variance = lower_variance
        self.n_executions = executions_per_plan if lower_variance else 1
        if lower_variance:
            self.oracle_operator_quality = False
        # The exploration checker reviews how well the optimization space has been explored:
        # after every `checker_every` executions, and once when the agent tries to finish. Its
        # model defaults to the main agent's. `max_checker_vetoes` bounds how many times it may
        # send a final answer back, so a checker and an agent that disagree cannot deadlock.
        self.use_checker = use_checker
        self.checker_model = checker_model
        self.checker_reasoning_effort = checker_reasoning_effort
        self.checker_every = checker_every
        self.max_checker_vetoes = max_checker_vetoes
        # Cost-aware search: the agent and the exploration checker are shown what the search is
        # spending and the two dataset sizes, and are told to weigh the cost of ONE more plan
        # against the full-dataset saving it would buy. Off by default so runs stay comparable.
        self.cost_aware_opt = cost_aware_opt
        self.max_steps = max_steps
        self.max_recover_retries = max_recover_retries
        self.context_budget_chars = context_budget_chars
        # Only what BASE_BUILTIN_MODULES (local_python_executor) does not already grant --
        # the executor unions the two, so re/random/datetime/time/collections/itertools/math/
        # statistics are available to plan code without being listed here.
        self.authorized_imports = authorized_imports or [
            "json", "palimpzest", "pandas", "numpy"
        ]
        self.verbose = verbose
        # Rebuilt per run(); kept on the instance so callers can read it after.
        self.messages: list[dict] = []
        self.reasoning_steps: list[str | None] = []
        self.trajectory_steps: list[dict] = []
        self._pending_images: list[dict] = []  # base64 images staged by explore_images
        # Per-run bookkeeping the teardown needs when the search dies mid-loop (see
        # _save_unfinished_run). `_ctx` is what `_setup_run` wired up; `_run_saved` records
        # whether the teardown has already run, so it happens exactly once either way.
        self._ctx: "_RunContext | None" = None
        self._run_saved: bool = False
        self._current_step: int = 0
        self._optimization_start: float = 0.0
        self.agent_cost_usd: float = 0.0
        self.opt_latency: float = 0.0
        self.execution_cost_usd: float = 0.0
        self.oracle_cost_usd: float = 0.0
        self.checker_cost_usd: float = 0.0
        self.checker_checks: list[dict] = []
        self.checker_vetoes: int = 0
        # Total opt cost as of the end of the previous step, so each step's own cost is a delta.
        self._prev_opt_cost: float = 0.0
        # What each plan cost the search to write and execute (see `_attribute_plan_costs`).
        self.plan_search_cost: dict[str, float] = {}
        self._prev_plan_cost_base: float = 0.0
        self._step_plans: dict = {}
        self._step_plans_before: dict = {}
        self._step_rows_before: int = 0
        self._subset_n: int = 0
        self._dataset_n: int = 0

    # -- logging -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _run_artifact(self, subdir: str, query_info: dict, suffix: str) -> pathlib.Path:
        """`{results_prefix}/{subdir}/Q{id}_{runcount}{suffix}`, with the directory created.

        Every per-run artifact is one of these, so the layout is stated once rather than
        re-derived by each writer."""
        out_dir = self.results_prefix / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{run_key(query_info['query_id'], query_info.get('runcount'))}{suffix}"

    def _save_trajectory_df(self, query_info: dict) -> None:
        """Dump per-step reasoning and assistant responses to CSV."""
        if not self.trajectory_steps:
            return
        import pandas as pd
        out = self._run_artifact("trajectory", query_info, ".csv")
        pd.DataFrame(self.trajectory_steps).to_csv(out, index=False)
        self._log(f"[run] trajectory → {out}")

    def _merge_sub_agent_trajectory(
        self, agent: Any, label: str, step: int, consumed: int
    ) -> int:
        """Interleave a sub-agent's new trajectory rows into the MAIN trajectory so every agent
        in a run is saved together in one file. Rows produced while handling main step `step`
        are labeled `"{step}[{label}]-{k}"` (k = 1..N) so the ordering reads naturally.
        Returns the updated `consumed` index into `agent.trajectory_steps`."""
        if agent is None:
            return consumed
        new_rows = agent.trajectory_steps[consumed:]
        for k, row in enumerate(new_rows, 1):
            merged = dict(row)
            merged["step"] = f"{step}[{label}]-{k}"
            self.trajectory_steps.append(merged)
        return len(agent.trajectory_steps)

    def _save_exploration_checks(self, query_info: dict) -> None:
        """Dump every exploration-checker verdict of this run to
        {results_prefix}/exploration_check/Q{id}_{rc}.json, so what the checker saw and what
        it asked for can be read back without parsing the trajectory."""
        if not self.checker_checks:
            return
        out = self._run_artifact("exploration_check", query_info, ".json")
        out.write_text(json.dumps(self.checker_checks, indent=2, default=str))
        self._log(f"[run] exploration checks → {out}")

    def _save_results_df(
        self, results: ResultsStore, query_info: dict, final_answer: dict | None = None,
        results_agg: ResultsStore | None = None,
    ) -> None:
        """Dump the accumulated plan-level results table to CSV, plus the per-plan aggregate
        table when plans were executed several times."""
        if not results.rows:
            return
        out = self._run_artifact("metrics", query_info, "_results.csv")
        df = results.df
        df["use_case"] = query_info["use_case"]
        df["query_id"] = query_info["query_id"]
        best_name = (final_answer or {}).get("best_plan", {}).get("name")
        df["final_selected"] = df["plan_name"] == best_name if best_name else False
        if results_agg is not None and results_agg.rows:
            agg = results_agg.df
            agg["use_case"] = query_info["use_case"]
            agg["query_id"] = query_info["query_id"]
            agg["final_selected"] = agg["plan_name"] == best_name if best_name else False
            agg_out = self._run_artifact("metrics", query_info, "_results_agg.csv")
            agg.to_csv(agg_out, index=False)
            self._log(f"[run] aggregate results table → {agg_out}")
        # re-attach large blob columns (stripped by results.df) and move them to the end
        for col in ["plan_str", "op_samples"]:
            if col not in df.columns:
                df[col] = [r.get(col) for r in results.rows]
        df = df[[col for col in df.columns if col not in ["plan_str", "op_samples"]] + ["plan_str", "op_samples"]]
        df.to_csv(out, index=False)
        self._log(f"[run] results table → {out}")

    # -- prompt assembly ---------------------------------------------------
    def _system_prompt(self, tools: list[Tool], eval_metric: str | None = None) -> str:
        return _lower_variance_text(_per_op_info_text(_per_op_quality_text(_SYSTEM_TEMPLATE.format(
            briefing=self.briefing,
            quality_metric=_quality_metric_section(eval_metric),
            tools_doc="\n\n".join(t.doc for t in tools),
            physical_sem_ops="\n".join(f"- {d}" for d in _PHYSICAL_SEMANTIC_OPERATORS.values()),
            physical_nonsem_ops="\n".join(f"- {d}" for d in _PHYSICAL_NONSEMANTIC_OPERATORS.values()),
            available_models=_AVAILABLE_MODELS_TEXT,
            max_steps=self.max_steps,
            cost_awareness=_COST_AWARENESS if self.cost_aware_opt else "",
            final_answer_doc=self.final_answer_doc,
        ), self.oracle_operator_quality), self.per_operator_info), self.lower_variance, self.n_executions)

    def _opening_message(self, task: str, plans: dict, results: ResultsStore) -> str:
        plan_lines = []
        for name, entry in plans.items():
            plan = entry["plan"]
            desc = entry.get("description", "")
            ops = iter_operators(plan)
            op_str = "("+ ",".join(get_op_type(o) for o in ops) +")"
            label = f" — {desc}" if desc else ""
            plan_lines.append(f"  - {name}: [{op_str}]{label}")
        if plan_lines:
            plans_section = "Available plans (`plans` dict):\n" + "\n".join(plan_lines)
        else:
            plans_section = "No plans yet — use write_plan to create the first one."
        summary = results.summary()
        if not self.per_operator_info:
            del summary["by_op_type"]
        return (
            f"{task}\n\n"
            f"{plans_section}\n\n"
            f"Observed-results store summary:\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            f"Begin. Output exactly ONE ```python``` block for your first step — "
            f"a single tool call (e.g. list_files()). Do not write multiple blocks, "
            f"plan ahead in prose, or produce a final answer yet."
        )

    def _trim(self, messages: list[dict]) -> list[dict]:
        budget = self.context_budget_chars
        total = sum(len(m["content"]) for m in messages)
        if total <= budget:
            return messages
        head = [messages[0], {"role": "user", "content": "...(earlier steps truncated)..."}]
        remaining = budget - sum(len(m["content"]) for m in head)
        tail: list[dict] = []
        for m in reversed(messages[1:]):
            if remaining - len(m["content"]) < 0:
                break
            tail.append(m)
            remaining -= len(m["content"])
        dropped = len(messages) - 1 - len(tail)  # originals dropped (excludes kept task message)
        print(
            f"[warning] context budget exceeded ({total:,} > {budget:,} chars); "
            f"trimmed {dropped} older message(s) from the middle of the trajectory "
            f"(task + {len(tail)} most-recent messages kept)."
        )
        return head + tail[::-1]

    # -- setup / teardown --------------------------------------------------
    def _setup_run(
        self,
        task: str,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        query_info: dict,
    ) -> _RunContext:
        """Wire up everything the loop needs and reset per-run state.

        Patches litellm→OpenRouter, builds the oracle/quality evaluator, seeds the sandbox
        variables + tools, constructs the exploration checker, builds the system prompt, and
        resets `self.messages`/cost accumulators. Returns the handful of objects `run`'s loop
        consumes.
        """
        patch_litellm_for_openrouter()

        from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor

        plan_codes: dict = {}  # populated by WritePlanTool; shared with ExecutePlanTool
        data_dir = query_info["data_dir"]
        # The dataset's row-identifier column, declared by the benchmark (see run_opt.py).
        # Image files always follow <data_dir>/<image_subdir>/<id_col>.jpg.
        id_col = query_info.get("id_col", "idx")
        image_subdir = query_info.get("image_subdir", "images")

        # Build oracle client and quality evaluator (oracle runs inside QualityEvaluator)
        oracle_client = OpenRouterClient(self.oracle_model, reasoning_effort=self.oracle_reasoning_effort)
        # Scoped per query: the judge cache is keyed by plan, so two runs on different queries
        # in the same use case/scale factor would otherwise wipe each other's cache mid-flight.
        llm_judge_dir = self.results_prefix / "llm_judge" / f"Q{query_info['query_id']}"
        import shutil
        if llm_judge_dir.exists():
            shutil.rmtree(llm_judge_dir)
        llm_judge_dir.mkdir(parents=True, exist_ok=True)

        # Both subset-adjacent ground-truth artifacts, named after the subset they came from:
        # _oracle_result.csv in oracle mode, _gt.csv when the benchmark regenerates its gold
        # query over the subset. Each run rewrites the one it uses; clearing both here keeps a
        # previous run's file from sitting next to the subset looking like this run's.
        _subset_path = query_info.get("subset_path")
        if _subset_path:
            _subset_path = pathlib.Path(_subset_path)
            for _suffix in ("_oracle_result.csv", "_gt.csv"):
                _subset_path.with_name(f"{_subset_path.stem}{_suffix}").unlink(missing_ok=True)

        # This run's opt_results directory. The evaluator writes the run's single shared ground
        # truth here; ExecutePlanTool writes one subdirectory of debug artifacts per plan.
        opt_run_dir = self.results_prefix / "opt_results" / run_key(
            query_info["query_id"], query_info.get("runcount")
        )
        # Start each run without a ground truth, for the same reason llm_judge is wiped above:
        # the evaluator persists it on the first plan and reads it back for every later plan, so
        # a leftover file from a previous run of this same runcount would be reused even if the
        # datasubset has since been resampled -- scoring the new subset against the old subset's
        # ground truth.
        (opt_run_dir / "oracle_ground_truth.csv").unlink(missing_ok=True)

        quality_evaluator = None
        quality_evaluator_cls = query_info.get("quality_evaluator_cls")
        if quality_evaluator_cls is not None:
            # Fatal on purpose. Without an evaluator every plan scores NaN and the full-dataset
            # evaluation skips itself, so the run would spend its whole budget and produce no
            # quality number anywhere -- better to stop here than to find that out afterwards.
            try:
                quality_evaluator = quality_evaluator_cls(
                    oracle_client=oracle_client,
                    oracle_model=self.oracle_model,
                    query_id=query_info["query_id"],
                    use_case=query_info["use_case"],
                    # A benchmark without a scale-factor axis passes None; its adapter absorbs
                    # the keyword it has no use for (see CUAD's QualityEvaluator.__init__).
                    scale_factor=query_info.get("scale_factor"),
                    llm_judge_dir=llm_judge_dir,
                    subset_path=query_info["subset_path"],
                    ground_truth_path=query_info["gt_path"],
                    run_dir=opt_run_dir,
                    oracle_reasoning_effort=self.oracle_reasoning_effort,
                    use_oracle_ground_truth=self.use_oracle_ground_truth,
                    oracle_operator_quality=self.oracle_operator_quality,
                )
            except Exception as e:
                raise RuntimeError(
                    f"{quality_evaluator_cls.__module__}.{quality_evaluator_cls.__name__} could "
                    f"not be constructed for query {query_info['query_id']}: "
                    f"{type(e).__name__}: {e}"
                ) from e

        import pandas as pd
        from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline, row_id_str
        try:
            import palimpzest as pz
        except ImportError as exc:
            raise ImportError("palimpzest is required to build plans") from exc

        def load_data(filename: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(data_dir, filename))

        def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
            # `col_name` is the NEW image column being added; the on-disk path is built from the
            # fixed naming convention (<image_subdir>/<row[id_col]>.jpg).
            # `col_name` MUST NOT collide with an existing column (e.g. the id column). PZ's convert
            # only generates fields not already present on the record, so a colliding name produces
            # an empty field_answers and raises `max() iterable argument is empty` on every row.
            pipeline.map(
                udf=lambda row: {col_name: os.path.join(data_dir, image_subdir, row_id_str(row[id_col]) + ".jpg")},
                cols=[{"name": col_name, "type": pz.ImageFilepath, "description": ""}],
            )
            return pipeline

        def explore_data(filename: str) -> pd.DataFrame:
            # Direct full-CSV read for the MAIN loop's data exploration ONLY. Returns the whole
            # table (unlike the row-capped explore_sample tool) so the agent can compute aggregate
            # statistics — keyword/keyphrase frequencies, value distributions, class balance — to
            # judge filter selectivity, join input sizes, and how rare a search target is. It is
            # deliberately NOT exposed to plan code (see plan_variables / plan_executor below), so
            # query plans can never depend on directly scanning the dataset.
            return pd.read_csv(os.path.join(data_dir, filename))

        # Main-loop sandbox variables: inspection stores + operator introspection + direct CSV
        # access for data exploration (explore_data). Intentionally EXCLUDES the plan-building
        # helpers (load_data/add_image_data/PhysicalPipeline/pz) — those live only in the minimal
        # plan-construction sandbox below.
        variables = {
            "explore_data": explore_data,
            "plans": plans,
            "plan_codes": plan_codes,
            "plan_results": plan_results,
            "iter_operators": iter_operators,
            "get_op_type": get_op_type,
        }
        if self.per_operator_info:
            variables["op_results"] = op_results
        # One row per execute_plan call, aggregated over that call's executions. Filled (and
        # shown to the agent) only in a lower-variance run.
        plan_results_agg = ResultsStore([])
        if self.lower_variance:
            variables["plan_results_agg"] = plan_results_agg

        # Minimal sandbox variables for WritePlanTool: only what plan CODE needs to construct a
        # PhysicalPipeline — the source-table reader (required by PhysicalPipeline.__init__ for
        # schema inference), the image helper, the pipeline class, and palimpzest (pz.Model /
        # pz.ImageFilepath). No results stores, no operator introspection, no explore_data.
        plan_variables = {
            "load_data": load_data,
            "add_image_data": add_image_data,
            "PhysicalPipeline": PhysicalPipeline,
            "pz": pz,
        }

        # Buffer of base64 image parts staged by explore_images; drained by the run loop, which
        # describes them (via a cheap vision model) into this step's textual observation. Images are
        # NOT forwarded to the main agent as pixels — explore_images is a one-step describe-only tool.
        self._pending_images: list[dict] = []
        # Dedicated cheap vision model for image descriptions, kept separate from the (possibly
        # pricey) main-agent model so exploring images stays cheap.
        self._image_describer = OpenRouterClient(self.image_describer_model, reasoning_effort="low")

        base_tools = [
            ListFilesTool(data_dir),
            ExploreSchemaT(data_dir),
            ExploreSampleTool(data_dir),
            ExploreImagesTool(
                data_dir, self._pending_images,
                subdir=image_subdir, id_col=id_col,
            ),
            *([GetOpSamplesTool(plan_results)] if self.per_operator_info else []),
            ExecutePlanTool(
                plan_codes=plan_codes,
                plans=plans,
                plan_results=plan_results,
                op_results=op_results,
                use_case=query_info["use_case"],
                query_id=query_info["query_id"],
                data_dir=data_dir,
                agent_dir=self.agent_dir,
                quality_evaluator=quality_evaluator,
                scale_factor=query_info.get("scale_factor"),
                results_prefix=self.results_prefix,
                runcount=query_info.get("runcount"),
                eval_metric=query_info.get("eval_metric"),
                subset_path=query_info.get("subset_path"),
                oracle_operator_quality=self.oracle_operator_quality,
                per_operator_info=self.per_operator_info,
                n_executions=self.n_executions,
                plan_results_agg=plan_results_agg,
                # Repeated executions run on fresh copies of the plan, built from its code the
                # same way the final full-dataset runs rebuild it.
                rebuild_pipeline=lambda name: self._fresh_pipeline(name, plan_codes),
                id_col=id_col,
            ),
        ]

        # Main-loop sandbox: the agent's step-by-step python blocks run here.
        executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        executor.send_variables(variables)

        # Separate, minimal sandbox for plan CONSTRUCTION. WritePlanTool executes plan code here,
        # so plan code sees only the pipeline-building helpers — never the results stores or
        # explore_data. This keeps query plans from depending on direct dataset access.
        plan_executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        plan_executor.send_variables(plan_variables)
        # Install the base Python builtins (float/int/str/len/... from BASE_PYTHON_TOOLS) so plan
        # UDFs can type-cast, e.g. `map(lambda row: {"p": float(row["price"])}, ...)`. Passing {}
        # means NO agent tools leak into plan code — send_tools sets static_tools to
        # {} | BASE_PYTHON_TOOLS | additional_functions.
        plan_executor.send_tools({})
        write_plan_tool = WritePlanTool(plan_codes, plans=plans, executor=plan_executor)
        # Kept so _run_final_evaluation can re-instantiate a fresh pipeline from a plan's source code
        # (rebuilding for each repeated final run keeps runs independent).
        self._plan_executor = plan_executor
        # Kept for the same reason: the full-dataset evaluation scores through this same object,
        # so search-time and final quality go through one flow (the adapter's own score_plan and
        # has_ground_truth) instead of the engine reassembling the steps itself.
        self._quality_evaluator = quality_evaluator

        # The exploration checker reviews the SEARCH -- which optimizations have been tried --
        # not any individual plan.
        checker: ExplorationCheckerAgent | None = None
        if self.use_checker:
            checker_model = self.checker_model or getattr(self.llm, "model", None)
            if not checker_model:
                raise ValueError(
                    "use_checker=True requires a checker LLM model id: pass checker_model=... to "
                    "CostModelAgent, or use an OpenRouterClient main llm exposing `.model`."
                )
            checker = ExplorationCheckerAgent(
                OpenRouterClient(checker_model, reasoning_effort=self.checker_reasoning_effort),
                plans=plans,
                plan_results=plan_results,
                task=task,
                eval_metric=query_info.get("eval_metric"),
                verbose=self.verbose,
                cost_aware=self.cost_aware_opt,
                per_op_quality=self.oracle_operator_quality,
                lower_variance=self.lower_variance,
                n_executions=self.n_executions,
                plan_results_agg=plan_results_agg,
            )

        tools = base_tools + [write_plan_tool]
        executor.send_tools({t.name: t for t in tools})

        system = self._system_prompt(tools, eval_metric=query_info.get("eval_metric"))
        opening = self._opening_message(task, plans, plan_results)
        self.messages = [{"role": "user", "content": opening}]
        self.reasoning_steps = []
        self.trajectory_steps = []
        self._run_saved = False
        self._current_step = 0
        self.agent_cost_usd = 0.0
        self.opt_latency = 0.0
        self.execution_cost_usd = 0.0
        self.oracle_cost_usd = 0.0
        self.checker_cost_usd = 0.0
        self.checker_checks = []
        self.checker_vetoes = 0
        self._prev_opt_cost = 0.0
        self.plan_search_cost = {}
        self._prev_plan_cost_base = 0.0
        self._step_plans, self._step_plans_before, self._step_rows_before = {}, {}, 0
        # Row counts, not line counts: CUAD contract text carries embedded newlines. Both feed
        # the per-step budget footer, which is what lets the agent scale a subset saving to the
        # full dataset it will finally run on.
        self._subset_n = len(pd.read_csv(query_info["subset_path"]))
        self._dataset_n = len(pd.read_csv(query_info["dataset_path"]))
        self._log(f"\n=== system prompt ({len(system)} chars) ===\n{system}\n")
        self._log(f"=== task ===\n{opening}\n")

        self._ctx = _RunContext(
            executor=executor, system=system, checker=checker,
            quality_evaluator=quality_evaluator, plan_codes=plan_codes,
            plan_results_agg=plan_results_agg,
        )
        return self._ctx

    def _account_costs(
        self,
        plan_results: ResultsStore,
        quality_evaluator: Any,
        checker: "ExplorationCheckerAgent | None" = None,
    ) -> None:
        """Roll up the per-run cost totals (subset execution, oracle, checker) onto the
        instance."""
        self.execution_cost_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in plan_results.rows)
        self.oracle_cost_usd = (
            float(getattr(quality_evaluator, "total_oracle_cost_usd", 0.0) or 0.0)
            if quality_evaluator is not None else 0.0
        )
        if checker is not None:
            self.checker_cost_usd = checker.cost_usd
            self.checker_checks = list(checker.checks)

    def _run_config(self) -> dict:
        """Per-LLM-actor settings and spend, grouped by WHO spent it.

        One block per actor that runs a model during the search -- the core agent doing the
        plan search, the exploration checker reviewing it, and the oracle scoring plan quality
        -- each carrying that actor's model, how it was configured, and what it cost. Keeping
        the dollars next to the model that spent them is what makes a run's cost readable
        after the fact.

        The one search cost that is NOT here is `subset_execution_cost`: executing candidate
        plans on the subset is not an actor, it is the work being measured. `_total_opt_cost`
        adds it back.

        Recorded identically in the final answer and in metrics/final_eval.json, so the two
        can never disagree about how a run was set up or what it spent.
        """
        return {
            "core_agent": {
                "model": getattr(self.llm, "model", None),
                "reasoning_effort": getattr(self.llm, "reasoning_effort", None),
                "cost_aware_opt": self.cost_aware_opt,
                "per_operator_info": self.per_operator_info,
                "lower_variance": self.lower_variance,
                "executions_per_plan": self.n_executions,
                "cost": round(self.agent_cost_usd, 6),
            },
            "optimization_checker": {
                "use_checker": self.use_checker,
                "model": self.checker_model or getattr(self.llm, "model", None),
                "reasoning_effort": self.checker_reasoning_effort,
                "check_every": self.checker_every,
                "cost": round(self.checker_cost_usd, 6),
                "n_checks": len(self.checker_checks),
                "n_vetoes": self.checker_vetoes,
            },
            "oracle": {
                "oracle_ground_truth": self.use_oracle_ground_truth,
                "oracle_operator_quality": self.oracle_operator_quality,
                "oracle_model": self.oracle_model,
                "oracle_reasoning_effort": self.oracle_reasoning_effort,
                "cost": round(self.oracle_cost_usd, 6),
            },
        }

    def _total_opt_cost(self) -> float:
        """Every dollar the SEARCH spent: each actor's LLM spend (core agent, exploration
        checker, oracle) plus executing candidate plans on the subset.

        Excludes running the CHOSEN plan on the full dataset at the end -- that is the cost of
        using the result, not of finding it, and is reported per run under runK."""
        return round(
            self.agent_cost_usd
            + self.checker_cost_usd
            + self.oracle_cost_usd
            + self.execution_cost_usd,
            6,
        )

    def _cost_snapshot(
        self, ctx: "_RunContext", plan_results: ResultsStore, *, commit: bool = True
    ) -> dict:
        """What the search has spent as of right now, and what the step just finished cost.

        `_account_costs` assigns rather than accumulates, so calling it every step is just a
        re-read of the same live counters the teardown reads -- there is no second accumulator
        to keep in sync, and the total here is the same `_total_opt_cost` that lands in
        metrics/final_eval.json.

        `commit=False` reads without closing the step, for a sub-agent that is about to run and
        so cannot be shown its own cost: the step stays open and whatever that sub-agent spends
        still lands in this step's delta when the step is finally committed."""
        self._account_costs(plan_results, ctx.quality_evaluator, ctx.checker)
        self._attribute_plan_costs(plan_results)
        total = self._total_opt_cost()
        step_cost = total - self._prev_opt_cost
        if commit:
            self._prev_opt_cost = total
            # Recorded whether or not the run is cost-aware: this is what makes the two arms of
            # a cost-aware/not comparison readable step by step, after the fact.
            self.trajectory_steps[-1]["step_cost_usd"] = round(step_cost, 6)
            self.trajectory_steps[-1]["total_opt_cost_usd"] = total
        return {
            "total_usd": total, "step_usd": step_cost,
            "subset_n": self._subset_n, "dataset_n": self._dataset_n,
            "plan_search_cost": dict(self.plan_search_cost),
        }

    def _start_step_attribution(self, plans: dict, plan_results: ResultsStore) -> None:
        """Remember which plans exist and how many executions there are as this step begins, so
        `_attribute_plan_costs` can tell which plans the step wrote or executed."""
        self._step_plans = plans
        self._step_plans_before = dict(plans)
        self._step_rows_before = len(plan_results.rows)

    def _attribute_plan_costs(self, plan_results: ResultsStore) -> None:
        """Bill the core agent, oracle and subset-execution spend since the last call to the plans
        this step wrote (write_plan stores a new entry) or executed, split evenly. A step that
        touched no plan bills nothing; checker spend is never billed to a plan."""
        base = self.agent_cost_usd + self.oracle_cost_usd + self.execution_cost_usd
        delta = base - self._prev_plan_cost_base
        self._prev_plan_cost_base = base
        touched = {n for n, e in self._step_plans.items() if self._step_plans_before.get(n) is not e}
        touched |= {
            r["plan_name"] for r in plan_results.rows[self._step_rows_before:] if r.get("plan_name")
        }
        for name in touched:
            self.plan_search_cost[name] = self.plan_search_cost.get(name, 0.0) + delta / len(touched)

    def _cost_footer(self, budget: dict) -> str:
        """The budget line appended to every observation in a cost-aware run.

        The two sizes appear without their ratio on purpose: the subset is relevance-sampled, so
        (dataset_n / subset_n) is an UPPER BOUND on the real scale factor, and printing it would
        invite the very multiplication the system prompt tells the agent not to do."""
        if not self.cost_aware_opt:
            return ""
        return (
            f"[optimization budget] this step: ${budget['step_usd']:.4f} | "
            f"total search spend so far: ${budget['total_usd']:.4f} (sunk) | "
            f"optimization subset: {budget['subset_n']} records | "
            f"full dataset: {budget['dataset_n']} records"
        )

    def _observe(self, obs: str, log_prefix: str = "") -> None:
        """Hand one observation to the agent: message history, trajectory row, and the log.

        The console gets the stamped header instead of the observation's own
        "Observation (step N):" first line; what the agent sees is untouched.
        """
        self.messages.append({"role": "user", "content": obs})
        self.trajectory_steps[-1]["observation"] = obs
        marker = f"Observation (step {self._current_step}):"
        shown = obs[len(marker):].lstrip() if obs.startswith(marker) else obs
        self._log(
            run_log.header("observation", f"step {self._current_step}") + f"{log_prefix}{shown}"
        )

    def _budget_turn(self, ctx: "_RunContext", plan_results: ResultsStore) -> None:
        """Close the step with the budget line, as the LAST thing the agent sees before its next
        turn.

        Last, not folded into the observation, because the exploration checker runs AFTER the
        observation is handed over: closing the step here is what bills its spend to the step
        that actually triggered it, instead of to the next one. The step is committed either
        way -- when the run is not cost-aware, only the trajectory columns are written and the
        agent is shown nothing."""
        budget = self._cost_snapshot(ctx, plan_results)
        footer = self._cost_footer(budget)
        if not footer:
            return
        self.messages.append({"role": "user", "content": footer})
        self.trajectory_steps[-1]["observation"] = (
            (self.trajectory_steps[-1]["observation"] or "") + "\n\n" + footer
        )
        self._log(footer)

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        task: str,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        *,
        query_info: dict | None = None,
    ) -> Any:
        """Search for the best plan and return the JSON final answer.

        `query_info` carries everything benchmark-specific the run needs; run_opt.py assembles it.
        The keys read here:
            use_case, query_id, scale_factor  identify the query
            runcount                          which repeat of it this is; keys every artifact
            eval_metric                       e.g. "f1-score"; explains the metric in the briefing
            data_dir, dataset_path, subset_path, gt_path, id_col, image_subdir
            quality_evaluator_cls             the benchmark's QualityEvaluator subclass
            run_final_eval                    execute the chosen plan on the FULL dataset at the
                                              end (default True); False = search only
            num_final_eval_runs               how many times, fresh pipeline each

        An ERROR inside the search is re-raised, but only after what the run already did is
        written to disk: a run that spent money on plans should leave a record of it either way.
        An INTERRUPT is not -- Ctrl-C (KeyboardInterrupt) and SystemExit derive from
        BaseException rather than Exception, so the `except` below does not see them and they
        propagate with nothing written. Stopping a run by hand means abandoning it, and a
        half-run left in results/ next to real ones is worse than no record at all.
        """
        query_info = query_info if query_info is not None else {}
        try:
            return self._run_search(
                task, plans, plan_results, op_results, query_info=query_info
            )
        except Exception as exc:
            run_log.log_error("fatal", f"{type(exc).__name__}: {exc}", exc=exc)
            self._save_unfinished_run(exc, plans, plan_results, query_info)
            raise

    def _save_unfinished_run(
        self, exc: Exception, plans: dict, plan_results: ResultsStore, query_info: dict
    ) -> None:
        """Run the normal teardown for a search that failed part-way through.

        Reached only from `run`'s except branch, and so only for a real error -- never for a
        Ctrl-C. Both ordinary exits return through `_finish_run` themselves, and `_run_saved`
        keeps a teardown that raised from being retried here. Nothing in here may raise -- the
        original exception is what the caller needs to see.
        """
        ctx = self._ctx
        if ctx is None or self._run_saved:
            return
        try:
            if not self.opt_latency:
                self.opt_latency = time.time() - self._optimization_start
            self._finish_run(
                {"error": f"{type(exc).__name__}: {exc}"},
                ctx, plans, plan_results, query_info, self._current_step,
            )
            self._log(f"[run] search failed at step {self._current_step}; partial results saved")
        except Exception as save_error:
            print(f"[run] could not save the unfinished run: {type(save_error).__name__}: {save_error}")
            run_log.log_error(
                "teardown", f"{type(save_error).__name__}: {save_error}", exc=save_error
            )

    def _run_search(
        self,
        task: str,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        *,
        query_info: dict,
    ) -> Any:
        """Bounded tool loop over `plans`/`plan_results`/`op_results`, returning the JSON final
        answer. `_setup_run` does all the wiring; this method is just the step loop."""
        # Opened before setup so a setup failure is logged too. `start_run` is also what sets
        # the elapsed-time origin every agent's step headers are stamped against.
        optimization_start = run_log.start_run(
            self._run_artifact("error_logs", query_info, ".txt"),
            f"Q{query_info.get('query_id')} rc={query_info.get('runcount')} "
            f"{self.use_case} {self.agent_dir}",
        )
        self._optimization_start = optimization_start
        # Cleared before setup, not after: a run whose setup fails has nothing to save, and a
        # previous run's context must never be mistaken for this one's.
        self._ctx, self._run_saved, self._current_step = None, False, 0

        ctx = self._setup_run(task, plans, plan_results, op_results, query_info)
        # Only what the loop itself touches is unpacked; everything the teardown needs
        # (quality_evaluator, plan_codes) is read off `ctx` in `_finish_run`.
        executor, system, checker = ctx.executor, ctx.system, ctx.checker

        checker_consumed = 0  # exploration-checker trajectory rows already merged into the main one
        # One entry per execute_plan call: a lower-variance run appends several plan_results
        # rows per call, but exploring the space means running plans, not repeating them.
        plan_calls = ctx.plan_results_agg if self.lower_variance else plan_results
        checked_plan_rows = len(plan_calls.rows)  # plans executed as of the last exploration check
        step = 0
        while step < self.max_steps:
            step += 1
            self._current_step = step
            run_log.set_step(step)
            self._start_step_attribution(plans, plan_results)
            try:
                raw = self._llm_step(system)
            except Exception as e:  # LLM transport failure
                self._log(f"[step {step}] LLM error: {e}")
                run_log.log_error("llm", f"{type(e).__name__}: {e}", exc=e)
                raise
            self.messages.append({"role": "assistant", "content": raw})
            reasoning = self.reasoning_steps[-1]
            self.trajectory_steps.append(
                {"step": step, "reasoning": reasoning, "assistant": raw, "observation": None}
            )
            if reasoning:
                self._log(run_log.header("reasoning", f"step {step}") + f"{reasoning}\n")
            self._log(run_log.header("assistant", f"step {step}") + f"{raw}\n")

            # parse (with a couple of format-error retries)
            try:
                parsed = self._parse_with_retries(system, raw)
            except ParseError as e:
                run_log.log_error("parse", e.detail, content=e.raw)
                obs = self._clip_observation(f"Observation (step {step}): {e.detail}")
                self._observe(obs, "[parse error] ")
                self._budget_turn(ctx, plan_results)
                continue

            if parsed.code is None:  # final answer
                # Before accepting it, let the exploration checker look at what was actually
                # tried. A veto sends the agent back to keep searching -- but only while there
                # are steps left to act on the suggestion, and only up to `max_checker_vetoes`
                # times, so a checker and an agent that disagree can never deadlock.
                steps_left = self.max_steps - step
                if (
                    checker is not None
                    and self.checker_vetoes < self.max_checker_vetoes
                    and steps_left >= 3
                ):
                    check = checker.check(
                        reason="final-answer",
                        budget=self._cost_snapshot(ctx, plan_results, commit=False)
                        if self.cost_aware_opt else None,
                    )
                    self.checker_checks = list(checker.checks)
                    if check.underexplored:
                        self.checker_vetoes += 1
                        veto = (
                            "Hold off on finalizing — a review of the optimizations you have "
                            f"explored so far found gaps:\n\n{check.suggestion}\n\n"
                            "Write and execute at least one more plan addressing this, then "
                            f"re-emit your final answer. ({steps_left} steps remain.)"
                        )
                        self.messages.append({"role": "user", "content": veto})
                        self.trajectory_steps[-1]["observation"] = (
                            f"[final answer withheld — exploration check {self.checker_vetoes}/"
                            f"{self.max_checker_vetoes}]\n{veto}"
                        )
                        self._log(veto)
                        # Being sent back to search is exactly when the price of one more plan
                        # matters, so a vetoed step closes with the budget line like any other.
                        self._budget_turn(ctx, plan_results)
                        # Merge only AFTER this step's own row has been written: merging appends
                        # the checker's rows, which would otherwise occupy the [-1] slot.
                        checker_consumed = self._merge_sub_agent_trajectory(
                            checker, "exploration_checker", step, checker_consumed
                        )
                        continue

                self.opt_latency = time.time()-optimization_start
                self._cost_snapshot(ctx, plan_results)  # close the step for the record
                self._log(f"[final answer] {parsed.result}")
                self.trajectory_steps[-1]["observation"] = f"[final answer] {parsed.result}"
                return self._finish_run(
                    parsed.result, ctx, plans, plan_results, query_info, step,
                    checker_consumed=checker_consumed,
                )

            # execute the python tool-call block
            try:
                out = executor(parsed.code)
            except Exception as e:
                # The observation the agent sees is clipped; the log keeps the whole thing,
                # plus the traceback and the code block, neither of which survives otherwise.
                run_log.log_error(
                    "exec", f"{type(e).__name__}: {e}", content=parsed.code, exc=e
                )
                obs = self._clip_observation(
                    f"Observation (step {step}): exec failed — {type(e).__name__}: {e}"
                )
                self._observe(obs)
                self._budget_turn(ctx, plan_results)
                continue

            obs = self._format_observation(step, out)
            if self._pending_images:
                # explore_images staged images this step. Describe them with the cheap vision model
                # and fold the description into THIS step's textual observation. The pixels are NOT
                # forwarded to the main agent — this is a one-step, describe-only exploration.
                shown_ids = [img["id"] for img in self._pending_images]
                description = self._describe_images(self._pending_images)
                self._pending_images.clear()
                obs = f"{obs}\n\nImage descriptions (auto-generated) for ids {shown_ids}:\n{description}"
            self._observe(obs)

            # Every `checker_every` executions, review how well the optimization space has been
            # explored and pass any gap back to the agent as ordinary feedback. Counted in
            # EXECUTIONS, not steps: exploring the space means running plans, not writing them.
            if checker is not None and len(plan_calls.rows) - checked_plan_rows >= self.checker_every:
                checked_plan_rows = len(plan_calls.rows)
                check = checker.check(
                    reason="periodic",
                    budget=self._cost_snapshot(ctx, plan_results, commit=False)
                    if self.cost_aware_opt else None,
                )
                self.checker_checks = list(checker.checks)
                if check.underexplored:
                    check_obs = (
                        "[review of the optimizations explored so far]\n"
                        f"{check.suggestion}"
                    )
                    self.messages.append({"role": "user", "content": check_obs})
                    self.trajectory_steps[-1]["observation"] = (
                        (self.trajectory_steps[-1]["observation"] or "") + "\n\n" + check_obs
                    )
                    self._log(check_obs)

            # Close the step only now: the exploration check above spends, and was triggered by
            # THIS step.
            self._budget_turn(ctx, plan_results)

            # interleave any exploration-checker rows generated during this step into the shared
            # trajectory
            checker_consumed = self._merge_sub_agent_trajectory(
                checker, "exploration_checker", step, checker_consumed
            )

        # out of steps — one forced terminal turn. The exploration checker is deliberately NOT
        # consulted here: there are no steps left to act on anything it might ask for.
        result = self._terminal_turn(system)
        self.opt_latency = time.time() - optimization_start
        return self._finish_run(
            result, ctx, plans, plan_results, query_info, step,
            checker_consumed=checker_consumed,
        )

    def _finish_run(
        self,
        result: Any,
        ctx: _RunContext,
        plans: dict,
        plan_results: ResultsStore,
        query_info: dict,
        step: int,
        *,
        checker_consumed: int = 0,
    ) -> Any:
        """Roll up costs, annotate the final answer, and save everything the run produced.

        Both ways out of the loop -- an accepted final answer, or the forced terminal turn after
        the step budget runs out -- end here, so a run that ran out of steps is recorded exactly
        like one that finished on its own. That includes `final_selected` in the results table,
        which needs the final answer and so was previously missing from every out-of-steps run.

        `self.opt_latency` is set by the caller: it ends at the accepted answer, not here.
        """
        self._run_saved = True
        self._account_costs(plan_results, ctx.quality_evaluator, ctx.checker)
        if isinstance(result, dict):
            result["plan_codes"] = ctx.plan_codes
            result["plan_descriptions"] = {name: entry.get("description", "") for name, entry in plans.items()}
            result["plan_optimizations"] = {name: entry.get("optimizations") for name, entry in plans.items()}
            # Model settings and spend for every actor, including the core agent's own.
            result.update(self._run_config())
        self._save_results_df(
            plan_results, query_info, final_answer=result, results_agg=ctx.plan_results_agg
        )
        self._merge_sub_agent_trajectory(ctx.checker, "exploration_checker", step, checker_consumed)
        self._save_trajectory_df(query_info)
        self._save_exploration_checks(query_info)
        self._run_final_evaluation(result, plans, query_info, ctx.plan_codes, plan_results)
        return result

    # -- helpers -----------------------------------------------------------
    def _describe_images(self, images: list[dict]) -> str:
        """Generate a textual description of the staged explore_images pictures.

        Uses a dedicated cheap vision model (`_image_describer`, `image_describer_model`) so image
        exploration stays inexpensive regardless of the main-agent model. The description is shown in
        THIS step's observation and persisted in the trajectory; the pixels themselves are not
        forwarded to the main agent. Cost is billed to agent_cost_usd, matching normal LLM steps.

        The describer's system prompt is `sampling.prompts.image_describe_system` — the same one the
        data sampler describes images with, so an image reads the same way to both."""
        from agent_cost_model.sampling.prompts import image_describe_system

        content: list[dict] = [{
            "type": "text",
            "text": f"Describe each of these {len(images)} image(s), one short line each, prefixed with its id:",
        }]
        for img in images:
            content.append({"type": "text", "text": f"id {img['id']}:"})
            content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        question = next((img.get("question") for img in images if img.get("question")), None)
        system = image_describe_system(question)
        try:
            result = self._image_describer.generate(system, [{"role": "user", "content": content}])
        except Exception as e:
            return f"[image description unavailable: {type(e).__name__}: {e}]"
        text, meta = result, {}
        if isinstance(result, tuple):
            text = result[0] if result else ""
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.agent_cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
        return (str(text) or "").strip() or "[no description returned]"

    def _llm_step(self, system: str, extra: list[dict] | None = None) -> str:
        msgs = self._trim(self.messages)
        if extra:
            msgs = msgs + extra
        result = self.llm.generate(system, msgs)
        content = result
        reasoning = None
        meta: dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) >= 1:
                content = result[0]
            if len(result) >= 2:
                reasoning = result[1]
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.agent_cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
        self.reasoning_steps.append(reasoning)
        return content

    def _parse_with_retries(self, system: str, raw: str) -> _Step:
        attempt = 0
        text = raw
        while True:
            try:
                return _parse_step(text)
            except ParseError as e:
                if attempt >= self.max_recover_retries:
                    raise
                attempt += 1
                fix = (
                    f"Your previous reply could not be parsed: {e.detail}\n"
                    "Re-send exactly ONE ```python``` or ```json``` fenced block."
                )
                # transient repair exchange (kept in history so the model sees it)
                self.messages.append({"role": "user", "content": fix})
                text = self._llm_step(system)
                self.messages.append({"role": "assistant", "content": text})

    _OBS_CHAR_LIMIT = 22_000

    def _format_observation(self, step: int, out: Any) -> str:
        parts = [f"Observation (step {step}):"]
        logs = (getattr(out, "logs", "") or "").strip()
        if logs:
            parts.append(f"[stdout]\n{logs}")
        result = out.output if hasattr(out, "output") else out
        result_s = "" if result is None else str(result).strip()
        if result_s and result_s != logs and result_s not in logs:
            parts.append(f"[result]\n{result_s}")
        if len(parts) == 1:
            parts.append("[no output]")
        return self._clip_observation("\n\n".join(parts))

    def _clip_observation(self, obs: str) -> str:
        """Cap any observation — successful output OR an error message — at the context budget.
        Error text interpolates offending values, so it can be just as large as a result."""
        limit = self._OBS_CHAR_LIMIT
        if len(obs) <= limit:
            return obs
        hint = (
            " Use get_op_samples(plan_name, op_name) to inspect specific input/output pairs."
            if self.per_operator_info else ""
        )
        return obs[:limit] + f"\n\n[output truncated — {len(obs) - limit} chars omitted.{hint}]"

    def _fresh_pipeline(self, plan_name: str, plan_codes: dict | None):
        """Re-instantiate a FRESH PhysicalPipeline from a plan's source code via the plan-construction
        executor, so each repeated final run is fully independent (no join accumulation or other
        operator state carried across runs). Returns None if the code or executor is unavailable."""
        code = (plan_codes or {}).get(plan_name)
        executor = getattr(self, "_plan_executor", None)
        if code is None or executor is None:
            return None
        try:
            executor.send_variables({"plan_name": plan_name})
            result = executor(code)
        except Exception as e:
            self._log(f"[final_eval] could not rebuild {plan_name!r} from code: {type(e).__name__}: {e}")
            run_log.log_error(
                "final-eval-rebuild", f"{plan_name}: {type(e).__name__}: {e}", content=code, exc=e
            )
            return None
        return getattr(result, "output", None)

    def _run_final_evaluation(
        self,
        final_answer: Any,
        plans: dict,
        query_info: dict,
        plan_codes: dict | None = None,
        plan_results: "ResultsStore | None" = None,
    ) -> None:
        """Run the agent-selected plan on the full dataset vs. real ground truth; append to metrics JSON.

        The chosen plan is re-run `num_final_eval_runs` times (each a fresh pipeline) to average out LLM
        stochasticity. The metrics entry keeps the agent-search-level fields once, with per-run
        full-dataset execution metrics nested under run1/run2/…; raw output of run k is saved as
        Q{query_id}_{runcount}_{k}.csv.

        `query_info["run_final_eval"]` (default True) gates only the full-dataset EXECUTION — the
        expensive half. With it False the search still records its metrics entry (agent cost/latency,
        plans written/executed, run config), just with no runK blocks.

        Everything that can go wrong here concerns the full-dataset runs only, so none of it
        returns early: the metrics entry is written either way, carrying the search-level fields
        with no runK blocks. The search is what cost the money, and losing its record because the
        agent named a plan that does not exist -- or because a ground-truth file is missing --
        leaves no trace of a run that really happened."""
        query_id = query_info["query_id"]
        scale_factor = query_info.get("scale_factor")
        runcount = query_info.get("runcount")
        run_suffix = run_key(query_id, runcount)

        runs, metric_type, n_runs = {}, "unknown", 0
        best_name = final_answer.get("best_plan", {}).get("name") if isinstance(final_answer, dict) else None
        if not best_name or best_name not in plans:
            self._log(
                f"[final_eval] plan {best_name!r} not in plans — recording the search, "
                "with no full-dataset evaluation"
            )
        elif not query_info.get("run_final_eval", True):
            self._log("[final_eval] run_final_eval=False — skipping the full-dataset evaluation")
        else:
            outcome = self._full_dataset_runs(best_name, plans, query_info, plan_codes, run_suffix)
            if outcome is not None:
                runs, metric_type, n_runs = outcome

        self._write_final_eval_metrics(
            query_id=query_id, runcount=runcount, scale_factor=scale_factor,
            runs=runs, n_runs=n_runs, metric_type=metric_type,
            plan_codes=plan_codes, plan_results=plan_results,
        )

    def _full_dataset_runs(
        self,
        best_name: str,
        plans: dict,
        query_info: dict,
        plan_codes: dict | None,
        run_suffix: str,
    ) -> "tuple[dict[str, dict], str, int] | None":
        """Execute the chosen plan on the FULL dataset `num_final_eval_runs` times (fresh pipeline
        each), scored against the benchmark's real ground truth.

        Returns (per-run metrics keyed run1/run2/…, metric type, number of runs attempted), or None
        when the evaluation cannot be set up at all (no gt_path, no ground-truth file, no quality
        evaluator). The caller still records the run either way — only the runK blocks are lost."""
        gt_path_str = query_info.get("gt_path")
        if not gt_path_str:
            self._log("[final_eval] no gt_path in query_info — skipping")
            return

        import pathlib

        import pandas as pd

        # The benchmark declares the ground-truth FILE (benchmark.yaml's ground_truth_path),
        # rather than the engine assuming a {dir}/Q{id}.csv naming convention -- CUAD's is a
        # single file covering every query, under a different name entirely.
        gt_path = pathlib.Path(gt_path_str)
        if not gt_path.exists():
            self._log(f"[final_eval] ground truth not found at {gt_path} — skipping")
            return

        raw_results_dir = self.results_prefix / "raw_results"
        raw_results_dir.mkdir(parents=True, exist_ok=True)

        # Score through the run's QualityEvaluator -- the same object, and the same adapter
        # methods, the search loop scores with. Only the ground truth differs: the full-dataset
        # evaluation always scores against the benchmark's REAL ground truth (gt_path), never an
        # oracle-substituted run, so `ground_truth_path_in_use()` is deliberately not consulted.
        evaluator = getattr(self, "_quality_evaluator", None)
        if evaluator is None:
            self._log("[final_eval] no quality evaluator — skipping")
            return
        # These runs are on the FULL dataset, not the optimization subset, so a benchmark whose
        # scorer measures over a document population must score over the full dataset's
        # documents -- otherwise a document the plan dropped would not be counted against it.
        population_path = query_info.get("dataset_path")
        # Loaded once and reused across every repeated final run. A benchmark that scores from
        # the ground-truth FILE ignores this frame; has_ground_truth below decides whether the
        # form it does consume is actually there.
        try:
            gt_df = pd.read_csv(gt_path)
        except Exception as e:
            self._log(f"[final_eval] ground-truth load failed: {type(e).__name__}: {e}")
            gt_df = None

        n_runs = max(1, int(query_info.get("num_final_eval_runs", 1) or 1))
        metric_type = "unknown"
        runs: dict[str, dict] = {}
        for final_run in range(1, n_runs + 1):
            # Rebuild a fresh pipeline from the plan code so this run is independent of prior runs.
            pipeline = self._fresh_pipeline(best_name, plan_codes)
            if pipeline is None:
                # Could not rebuild (missing code/executor): fall back to the stored pipeline. Fine
                # for a single run; repeated runs of a join plan could carry over accumulated state.
                pipeline = plans[best_name]["plan"]
            self._log(f"[final_eval] running {best_name!r} on full dataset (run {final_run}/{n_runs})...")
            try:
                result_collection, op_results_full, plan_dict = pipeline.run()
            except Exception as e:
                self._log(f"[final_eval] pipeline.run() failed (run {final_run}): {type(e).__name__}: {e}")
                continue

            results_df = result_collection.to_df()
            # Replace NaN in string columns so evaluators (e.g. adjusted-rand-index) don't crash.
            str_cols = results_df.select_dtypes(include="object").columns
            results_df[str_cols] = results_df[str_cols].fillna("")
            raw_name = f"{run_suffix}_{final_run}"
            results_df.to_csv(raw_results_dir / f"{raw_name}.csv", index=False)
            self._log(f"[final_eval] raw results (run {final_run}) → {raw_results_dir / f'{raw_name}.csv'}")
            eval_df = evaluator.normalize(results_df)

            # Wall-clock latency of the full run. plan_dict["latency_s"] is time.time()-based
            # (see PhysicalPipeline.run), so it reflects real elapsed time with the join/convert
            # parallelism. Do NOT fall back to sum(op_results_full["latency_s"]): each op's latency_s
            # is a SUM of per-record times, which ignores the ~20-way parallelism and overcounts
            # wall-clock by roughly the parallelism factor (this is what made Q7 look like ~8179s vs
            # PZ's ~300s). If the wall-clock time is somehow absent, record NaN rather than that
            # misleading overcount.
            total_latency = plan_dict.get("latency_s", float("nan"))
            total_cost = sum(r.get("cost_usd", 0) for r in op_results_full)

            quality = float("nan")
            if evaluator.has_ground_truth(gt_df, gt_path):
                try:
                    quality = float(evaluator.score_plan(
                        eval_df, gt_df, gt_path, population_path
                    ))
                    metric_type = query_info.get("eval_metric", metric_type)
                except Exception as e:
                    self._log(f"[final_eval] evaluation failed (run {final_run}): {type(e).__name__}: {e}")
            else:
                self._log(f"[final_eval] no ground truth at {gt_path} — quality is N/A")

            op_errors = plan_dict.get("op_errors")
            if op_errors:
                self._log(f"[final_eval] operator errors (run {final_run}): {op_errors}")
            runs[f"run{final_run}"] = {
                "latency": round(total_latency, 4),
                "cost": round(total_cost, 6),
                "quality": quality,
                **({"op_errors": op_errors} if op_errors else {}),
            }

        return runs, metric_type, n_runs

    def _write_final_eval_metrics(
        self,
        *,
        query_id: Any,
        runcount: Any,
        scale_factor: Any,
        runs: dict,
        n_runs: int,
        metric_type: str,
        plan_codes: dict | None,
        plan_results: "ResultsStore | None",
    ) -> None:
        """Append this run's entry to metrics/final_eval.json: the agent-search-level fields once,
        then whatever full-dataset runs were performed nested under run1/run2/… (none when the
        full-dataset evaluation was turned off)."""
        metrics_path = self.results_prefix / "metrics" / "final_eval.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict = {}
        if metrics_path.exists():
            try:
                entry = json.loads(metrics_path.read_text())
            except Exception:
                entry = {}
        plans_written = len(plan_codes) if plan_codes is not None else 0
        _executed_rows = [r.get("plan_name") for r in (plan_results.rows if plan_results is not None else []) if r.get("plan_name")]
        # Rows come in complete groups of n_executions (execute_plan appends all or none).
        plans_executed = len(_executed_rows) // self.n_executions
        unique_plans_executed = len(set(_executed_rows))
        metrics_key = run_key(query_id, runcount).removeprefix("Q")
        # Agent-search-level fields once, then per-run full-dataset execution metrics nested under runK.
        entry[metrics_key] = {
            "query_id": str(query_id),
            # Only benchmarks that actually have a scale-factor axis record one (SemBench does,
            # CUAD does not) -- recording a placeholder would make it look like a real dimension.
            **({"scale_factor": scale_factor} if scale_factor is not None else {}),
            "metric_type": metric_type,
            # Which ground truth search-time plan quality was scored against -- "oracle" (an
            # oracle-substituted pipeline run, the default) or "ground_truth" (the benchmark's
            # real ground truth via `ground_truth_loader`). Doesn't describe the full-dataset
            # runK entries below, which always score against real ground truth regardless of
            # this setting -- it's specifically so a reader (or another runner being compared
            # against this one, e.g. docetl MOAR) can filter to runs comparable on that axis.
            "ground_truth_mode": "oracle" if self.use_oracle_ground_truth else "ground_truth",
            # Model settings and spend, grouped by the actor that spent it.
            **self._run_config(),
            # Executing candidate plans on the subset: the work the search was measuring,
            # rather than any one actor's LLM spend.
            "subset_execution_cost": round(self.execution_cost_usd, 6),
            "total_opt_cost": self._total_opt_cost(),
            # Wall-clock seconds from the start of the search to the accepted final answer --
            # every LLM step, subset execution, oracle scoring and exploration check included.
            # The full-dataset runs below are NOT in it; they happen after the plan is chosen.
            "total_opt_latency": round(self.opt_latency, 4),
            "plans_written": plans_written,
            "plans_executed": plans_executed,
            "unique_plans_executed": unique_plans_executed,
            **runs,
        }

        # Sort entries by increasing query_id, then by runcount. Keys are "{query_id}_{runcount}"
        # (or just "{query_id}" when runcount is None → treated as runcount -1 so it sorts first);
        # any non-numeric legacy keys sort last.
        def _entry_sort_key(k: str) -> tuple[float, int]:
            qid_str, _, rc_str = k.partition("_")
            try:
                qid = float(int(qid_str))
            except ValueError:
                return (float("inf"), -1)
            try:
                rc = int(rc_str) if rc_str else -1
            except ValueError:
                rc = -1
            return (qid, rc)

        entry = {k: entry[k] for k in sorted(entry, key=_entry_sort_key)}
        metrics_path.write_text(json.dumps(entry, indent=2))
        detail = (
            f"({len(runs)}/{n_runs} full-dataset run(s) succeeded)" if n_runs
            else "(search only — no full-dataset run)"
        )
        self._log(f"[final_eval] metrics {detail} → {metrics_path}")

    _TERMINAL_PROMPT = (
        "You are out of steps. Do NOT call any tool — emit exactly ONE ```json``` block: "
        "either your best final answer in the required format, or "
        '{"error": "<2-4 sentence note on what you tried and what blocked you>"}.'
    )

    def _terminal_turn(self, system: str) -> Any:
        try:
            text = self._llm_step(system, extra=[{"role": "user", "content": self._TERMINAL_PROMPT}])
            parsed = _parse_step(text)
            if parsed.code is None:
                return parsed.result
        except Exception as e:
            raise StepFailed("max steps without accepted final answer", diagnostic=str(e)) from e
        raise StepFailed("max steps without accepted final answer", diagnostic=text)
