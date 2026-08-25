"""The main optimization agent: a bounded tool-loop agent that writes, executes, and
compares physical query plans in a sandboxed environment, converging on the best
cost/latency-quality trade-off.

Supporting pieces used to live in this file directly; they are now split out so this
module can stay focused on `CostModelAgent`'s own setup/loop logic:
  - `errors.py`            — ParseError, StepFailed
  - `llm_client.py`        — LLMClient, OpenRouterClient, patch_litellm_for_openrouter
  - `cost_model_types.py`  — PlanCostEstimate/CostModel, ResultsStore, CostModelRegistry,
                              and the duck-typed operator-introspection helpers
  - `step_parsing.py`      — the "one fenced block per turn" step parser
  - `prompts.py`           — the physical-operator catalog, model catalog, and
                              quality-metric briefing text
  - `cost_helper_agent.py` — CostHelperAgent, which authors/refits cost models on
                              this agent's behalf in "customCost" mode
  - `tools/`                — the Tool subclasses exposed to the sandbox

Everything that used to be importable as `agent_cost_model.opt_agent.cost_model_agent.X` is
re-exported below, so existing `from agent_cost_model.opt_agent.cost_model_agent import X`
call sites keep working unchanged.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Guarded SemBench evaluator import.
# Add SemBench's src/ to sys.path when the sibling checkout is present.
# ---------------------------------------------------------------------------
from agent_cost_model.paths import (
    RESULTS_DIR,
    SEMBENCH_DATASUBSET_DIR,
    SEMBENCH_FILES_DIR,
    ensure_sembench_src_on_path,
)

ensure_sembench_src_on_path()

try:
    from agent_cost_model.experiments.SemBench.quality_evaluator import (
        QualityEvaluator as _QualityEvaluator,
    )
except ImportError:
    try:
        from experiments.SemBench.quality_evaluator import (  # type: ignore
            QualityEvaluator as _QualityEvaluator,
        )
    except ImportError:
        _QualityEvaluator = None  # type: ignore

# ---------------------------------------------------------------------------
# Re-exports (see module docstring) — used both by this module's own code below
# and by external callers importing them from here.
# ---------------------------------------------------------------------------
from agent_cost_model.opt_agent.errors import ParseError, StepFailed
from agent_cost_model.opt_agent.llm_client import LLMClient, OpenRouterClient, patch_litellm_for_openrouter
from agent_cost_model.opt_agent.cost_model_types import (
    HAVE_PALIMPZEST,
    OperatorCostEstimates,
    PhysicalOperator,
    PhysicalPlan,
    PlanCost,
    CostModel,
    CostModelRegistry,
    PlanCostEstimate,
    ResultsStore,
    _normalize_plan_df,
    describe_operator,
    get_op_id,
    get_op_model,
    get_op_type,
    iter_operators,
    make_observed_op_stats,
)
from agent_cost_model.opt_agent.step_parsing import _Step, _parse_step
from agent_cost_model.opt_agent.prompts import (
    _AVAILABLE_MODELS_TEXT,
    _PHYSICAL_NONSEMANTIC_OPERATORS,
    _PHYSICAL_SEMANTIC_OPERATORS,
    _QUALITY_METRIC_DOCS,
    _SYSTEM_TEMPLATE,
    _quality_metric_reminder,
    _quality_metric_section,
)
from agent_cost_model.opt_agent.cost_helper_agent import CostHelperAgent
from agent_cost_model.opt_agent.tools import (
    ComparePlanCostsTool,
    EstimatePlanCostTool,
    ExecutePlanTool,
    ExploreImagesTool,
    ExploreSampleTool,
    ExploreSchemaT,
    GetOpSamplesTool,
    ListFilesTool,
    ReviewPlansTool,
    Tool,
    UpdateCostModelTool,
    WritePlanTool,
)


@dataclass
class _RunContext:
    """Everything `CostModelAgent._setup_run` wires up for the main loop to consume."""
    executor: Any
    system: str
    registry: CostModelRegistry
    cost_helper: "CostHelperAgent | None"
    quality_evaluator: Any
    plan_codes: dict


class CostModelAgent:
    """A bounded tool-loop agent that authors, applies, and refines cost models."""
    execute_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds.

        Your goal is to write physical plans, observe cost/latency/quality,
        and converge on the best cost-quality trade-off. Do not hard code the plan.

        Suggested workflow:
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table's
           columns and format. You ALSO have direct access to the full CSVs via `explore_data(filename)`,
           which returns the whole table as a DataFrame. Use it to compute AGGREGATE statistics —
           e.g. how many rows contain a keyword/keyphrase, or `value_counts()` on a column — to gauge
           how common an item or attribute is (if "plastic" appears in 60% of rows but "marble" in
           2%, plastic items are far more common). This informs the selectivity of your filters, the
           size of the tables feeding a join, and the class distribution of a classification/search
           task (a rare target vs. common groups) — let it GUIDE how you design plans. Keep
           exploration lightweight: do NOT run excessive regex/brute-force scans, and never use the
           data to solve the query and reverse-engineer a plan; the plan must remain a general solution.
           If the dataset has an `images/` folder, you can SEE a few product images with
           `explore_images(ids)` (up to 5, selected by the dataset's image id column) — useful to judge what a vision
           operator would work with. Images are shown once and are costly, so inspect just a couple.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - NEVER hardcode row IDs, indexes, or values copied from `explore_sample`/`explore_data`
             output to identify specific records (e.g. `filter(lambda row: row["id"] in [3, 17, 42])`
             is cheating and invalid). What you learn from exploration may only shape HOW you build a
             general plan, not which records it targets.
           - DO use `sem_filter` / `sem_map` for conditions that need semantic judgment, and prefer
             deterministic, free Python/regex functions in plain `filter` / `map` when a condition
             can be expressed that way — e.g. `filter(lambda row: row["score"] >= 4)`,
             `filter(lambda row: row["description"].isin(["settee", "sofa", "couch"]))`, or
             `map(lambda row: {"red_freq_count": row["description"].str.count("red")}, ...)`.
             These are general (they don't depend on which specific rows exist) and cost nothing to
             run, unlike a semantic op.
           - `plans[name]["plan"]` is populated immediately with the newly written PhysicalPipeline instance.
        3. Execute with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]["plan"]` is updated with the executed PhysicalPipeline.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck. Use `get_op_samples(plan_name)` to inspect
             input/output pairs for an operator.
        5. Based on `plan_results.df` and `op_results.df`, write ONE new plan and execute it,
           then repeat from step 2. Each new plan should embody a DIFFERENT optimization idea
           (or a COMBINATION of ideas that worked). Levers for lowering cost and latency include:
           swapping to a cheaper/faster model, narrowing `depends_on` to fewer columns, truncating long text with a `map`
           before a semantic op, reordering to push selective filters earlier, removing images,
           or changing the logical structure. Levers for increasing quality include: swapping to a more capable model,
           being less aggressive with pre-filtering or truncation, or changing logical structure.
           - Don't just hunt for cost cuts that hold quality steady — also spend some plans actively
             trying to raise quality, then weigh whether the resulting cost/latency is worth it. Some
             queries are hard and won't get close to 1.0; that's fine, but you should still have
             pushed on quality before settling.
           - Do NOT rabbit-hole on fine-tuning a single knob (e.g. the exact truncation length):
             we run on a small subsample and micro-tuning is inefficient and prone to overfitting.
           - To prevent falling in a local minimum, make sure to also explore a wider range of the
            optimization space: try different models and different logical structures.
           - Stop and give your final answer when the gains over your current best plan are
             marginal, or you run low on steps.
        """)

    sampleCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds. A sample-based cost model (SampleBasedCostModel) is
        pre-installed and re-fits automatically on every `estimate_plan_cost` call —
        use it to compare plan variants before deciding which ones to execute.

        Your goal is to write physical plans, observe cost/latency/quality,
        and converge on the best cost-quality trade-off. Do not hard code the plan.

        Suggested workflow:
        To get initial data and performance traces:
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table's
           columns and format. You ALSO have direct access to the full CSVs via `explore_data(filename)`,
           which returns the whole table as a DataFrame. Use it to compute AGGREGATE statistics —
           e.g. how many rows contain a keyword/keyphrase, or `value_counts()` on a column — to gauge
           how common an item or attribute is (if "plastic" appears in 60% of rows but "marble" in
           2%, plastic items are far more common). This informs the selectivity of your filters, the
           size of the tables feeding a join, and the class distribution of a classification/search
           task (a rare target vs. common groups) — let it GUIDE how you design plans. Keep
           exploration lightweight: do NOT run excessive regex/brute-force scans, and never use the
           data to solve the query and reverse-engineer a plan; the plan must remain a general solution.
           If the dataset has an `images/` folder, you can SEE a few product images with
           `explore_images(ids)` (up to 5, selected by the dataset's image id column) — useful to judge what a vision
           operator would work with. Images are shown once and are costly, so inspect just a couple.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - NEVER hardcode row IDs, indexes, or values copied from `explore_sample`/`explore_data`
             output to identify specific records (e.g. `filter(lambda row: row["id"] in [3, 17, 42])`
             is cheating and invalid). What you learn from exploration may only shape HOW you build a
             general plan, not which records it targets.
           - DO use `sem_filter` / `sem_map` for conditions that need semantic judgment, and prefer
             deterministic, free Python/regex functions in plain `filter` / `map` when a condition
             can be expressed that way — e.g. `filter(lambda row: row["score"] >= 4)`,
             `filter(lambda row: row["description"].isin(["settee", "sofa", "couch"]))`, or
             `map(lambda row: {"red_freq_count": row["description"].str.count("red")}, ...)`.
             These are general (they don't depend on which specific rows exist) and cost nothing to
             run, unlike a semantic op.
           - `plans[name]["plan"]` is populated immediately — call `estimate_plan_cost(plans[name]["plan"])`
             right after `write_plan` to get a pre-execution cost estimate.
        3. Execute with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]["plan"]` is updated with the executed PhysicalPipeline.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck.

        Then, iteratively write new plans, estimate their cost, and execute the most promising ones:
        - Use `plan_results.df` and `op_results.df` to identify quality and cost/latency trade-offs
            -- `plan_results.df` includes operator descriptions; use `get_op_samples(plan_name)` to view input/output pairs
            -- `op_results.df` includes per-operator performances, operators are named by `name_op1`, `name_op2`, ...
        - Always use `estimate_plan_cost(plans[name]["plan"])` to get cost/latency estimates BEFORE executing promising plans.
            -- the cost model averages past execution to get per-operator cost/latency estimates
            -- thus, only changing a few operators for each plan writing will produce more comparable estimations
        """)

    customCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds.

        Your goal is to search for the physical plan with the best cost/latency-quality trade-off.
        Do not hard code the plan.

        You do NOT build or run a cost model. A separate cost helper maintains it for you:
        after each `execute_plan` it refreshes cost/latency ESTIMATES for all your plans and shows
        them to you automatically; you can also request them anytime with `review_plans()`. Treat
        estimates as RELATIVE signals for ranking candidate plans — absolute values may be inaccurate.
        Focus entirely on writing and choosing good plans.

        === Bootstrap ===
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table's
           columns and format. You ALSO have direct access to the full CSVs via `explore_data(filename)`,
           which returns the whole table as a DataFrame. Use it to compute AGGREGATE statistics —
           e.g. how many rows contain a keyword/keyphrase, or `value_counts()` on a column — to gauge
           how common an item or attribute is (if "plastic" appears in 60% of rows but "marble" in
           2%, plastic items are far more common). This informs the selectivity of your filters, the
           size of the tables feeding a join, and the class distribution of a classification/search
           task (a rare target vs. common groups) — let it GUIDE how you design plans. Keep
           exploration lightweight: do NOT run excessive regex/brute-force scans, and never use the
           data to solve the query and reverse-engineer a plan; the plan must remain a general solution.
           If the dataset has an `images/` folder, you can SEE a few product images with
           `explore_images(ids)` (up to 5, selected by the dataset's image id column) — useful to judge what a vision
           operator would work with. Images are shown once and are costly, so inspect just a couple.
        2. Write 1–2 baseline plans with `write_plan(code, name, description)` that capture
           meaningfully different design approaches (e.g., one with a strong early filter, one
           without; one using a capable model, one using a cheaper model). Give each a short
           `description` of its approach/optimizations — it appears in the estimate tables.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a NEW unique string identifier (e.g. "p1", "p2", ...) — never reuse a name.
           - NEVER hardcode row IDs, indexes, or values copied from `explore_sample`/`explore_data`
             output to identify specific records (e.g. `filter(lambda row: row["id"] in [3, 17, 42])`
             is cheating and invalid). What you learn from exploration may only shape HOW you build a
             general plan, not which records it targets.
           - DO use `sem_filter` / `sem_map` for conditions that need semantic judgment, and prefer
             deterministic, free Python/regex functions in plain `filter` / `map` when a condition
             can be expressed that way — e.g. `filter(lambda row: row["score"] >= 4)`,
             `filter(lambda row: row["description"].isin(["settee", "sofa", "couch"]))`, or
             `map(lambda row: {"red_freq_count": row["description"].str.count("red")}, ...)`.
             These are general (they don't depend on which specific rows exist) and cost nothing to
             run, unlike a semantic op.
        3. Execute the baseline plans with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` and per-operator stats to `op_results`.
           - After the first execution, the cost helper builds a cost model and an estimate table
             appears — use it to guide the rest of your search.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck. Use `get_op_samples(plan_name)` to inspect
             input/output pairs for an operator.

        === Iterate (flexible) ===
        Work in whatever order the estimates suggest — there is no fixed cycle. In general:
        - Write candidate plans that embody DIFFERENT optimization ideas AND their COMBINATIONS,
          each with a clear `description`. Levers include: swapping to a cheaper/faster model,
          narrowing `depends_on` to fewer columns, truncating long text with a `map` before a
          semantic op, reordering to push selective filters earlier, removing images, or changing
          the logical structure.
        - Use `review_plans()` (or the auto-refreshed table) to compare candidates against each
          other and against your executed plans, then execute the 1–2 most promising. You may
          execute any earlier written-but-unexecuted plan at any time when the estimates now
          favor it (keep a backlog of ideas rather than discarding them).
        - Do NOT rabbit-hole on fine-tuning a single knob (e.g. the exact truncation length):
          we run on a small subsample and micro-tuning is inefficient and prone to overfitting.
        - When a candidate uses an operator/model with no observed data yet, its estimate is
          marked `extrap?=yes` — lean toward executing such a plan to gather real data rather
          than fully trusting the estimate.
        - Stop and give your final answer when the estimated gains over your current best plan
          are marginal, or you run low on steps.
    """)

    execute_final_answer_doc = textwrap.dedent("""\
        Emit a JSON object summarizing the best plan:
          {
            "best_plan": {
              "name": "...",
              "rationale": "why this plan wins (cost/quality trade-off)"
            }
          }
        Use plain JSON literals only (no python expressions, no trailing commas).""")

    sampleCost_final_answer_doc = textwrap.dedent("""\
        Emit a JSON object with the best plan and a cost-model explanation:
          {
            "best_plan": {
              "name": "p1",
              "rationale": "why this plan wins (cost/quality trade-off)"
            },
            "cost_model_explanation": "1-4 sentences on how the cost model guided your plan design. Which operators or models were cheapest/fastest?"
          }
        Use plain JSON literals only (no python expressions, no trailing commas).""")

    customCost_final_answer_doc = sampleCost_final_answer_doc

    def __init__(
        self,
        llm: LLMClient,
        *,
        data_dir: str = "dataset/use_case",
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
        helper_model: str | None = None,
        helper_reasoning_effort: str | None = "medium",
    ) -> None:
        self.llm = llm
        self.agent_dir = agent_dir
        self.use_case = use_case
        self.data_dir = data_dir
        self.oracle_model = oracle_model
        self.oracle_reasoning_effort = oracle_reasoning_effort
        # When False, quality is scored against the benchmark's real ground truth instead of
        # an oracle-substituted plan — skips all oracle LLM calls (and per-op oracle quality,
        # which depends on running that oracle-substituted plan).
        self.use_oracle_ground_truth = use_oracle_ground_truth
        # customCost mode offloads cost-model authoring to a CostHelperAgent that
        # ALWAYS runs on its own OpenRouterClient (never the main llm). Defaults to
        # the main model id if none is given; run() errors if neither is resolvable.
        self.helper_model = helper_model
        self.helper_reasoning_effort = helper_reasoning_effort
        self.max_steps = max_steps
        self.max_recover_retries = max_recover_retries
        self.context_budget_chars = context_budget_chars
        self.authorized_imports = authorized_imports or [
            "math", "statistics", "json", "collections", "itertools",
            "palimpzest", "pandas"
        ]
        self.verbose = verbose
        # Rebuilt per run(); kept on the instance so callers can read it after.
        self.messages: list[dict] = []
        self.reasoning_steps: list[str | None] = []
        self.trajectory_steps: list[dict] = []
        self._pending_images: list[dict] = []  # base64 images staged by explore_images
        self.agent_cost_usd: float = 0.0
        self.opt_latency: float = 0.0
        self.execution_cost_usd: float = 0.0
        self.oracle_cost_usd: float = 0.0
        self.helper_cost_usd: float = 0.0

    # -- logging -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _save_trajectory_df(self, query_info: dict) -> None:
        """Dump per-step reasoning and assistant responses to CSV."""
        if not self.trajectory_steps:
            return
        import pandas as pd
        import pathlib
        metrics_dir = RESULTS_DIR / "trajectory" / self.use_case / f"sf_{query_info['scale_factor']}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = metrics_dir / f"{qkey}_{self.agent_dir}_trajectory.csv"
        pd.DataFrame(self.trajectory_steps).to_csv(out, index=False)
        self._log(f"[run] trajectory → {out}")

    def _merge_helper_trajectory(
        self, cost_helper: "CostHelperAgent | None", step: int, consumed: int
    ) -> int:
        """Interleave the cost helper's new trajectory rows into the MAIN trajectory so both
        agents are saved together in one file. Rows produced while handling main step `step`
        are labeled `"{step}[cost_helper]-{k}"` (k = 1..N) so the ordering reads naturally.
        Returns the updated `consumed` index into `cost_helper.trajectory_steps`."""
        if cost_helper is None:
            return consumed
        new_rows = cost_helper.trajectory_steps[consumed:]
        for k, row in enumerate(new_rows, 1):
            merged = dict(row)
            merged["step"] = f"{step}[cost_helper]-{k}"
            self.trajectory_steps.append(merged)
        return len(cost_helper.trajectory_steps)

    def _save_cost_model_codes(self, codes: dict, query_info: dict) -> None:
        """Save all versioned cost model code strings to
        results/costModel/{use_case}/{agent_dir}/Q{id}_{rc}.json."""
        if not codes:
            return
        import json
        import pathlib
        out_dir = pathlib.Path(__file__).resolve().parent / "results" / "costModel" / self.use_case / self.agent_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = out_dir / f"{qkey}.json"
        with open(out, "w") as f:
            json.dump(codes, f, indent=2)
        self._log(f"[run] cost model codes → {out}")

    def _save_results_df(
        self, results: ResultsStore, query_info: dict, final_answer: dict | None = None
    ) -> None:
        """Dump the accumulated plan-level results table to CSV."""
        if not results.rows:
            return
        import pathlib
        metrics_dir = RESULTS_DIR / "metrics" / self.use_case / f"sf_{query_info['scale_factor']}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = metrics_dir / f"{qkey}_{self.agent_dir}_results.csv"
        df = results.df
        df["use_case"] = query_info["use_case"]
        df["query_id"] = query_info["query_id"]
        best_name = (final_answer or {}).get("best_plan", {}).get("name")
        df["final_selected"] = df["plan_name"] == best_name if best_name else False
        # re-attach large blob columns (stripped by results.df) and move them to the end
        for col in ["plan_str", "op_samples"]:
            if col not in df.columns:
                df[col] = [r.get(col) for r in results.rows]
        df = df[[col for col in df.columns if col not in ["plan_str", "op_samples"]] + ["plan_str", "op_samples"]]
        df.to_csv(out, index=False)
        self._log(f"[run] results table → {out}")

    # -- prompt assembly ---------------------------------------------------
    def _system_prompt(
        self,
        tools: list[Tool],
        briefing: str | None = None,
        final_answer_doc: str | None = None,
        mode: str = "",
        eval_metric: str | None = None,
    ) -> str:
        if "customCost" in mode:
            estimate_rule = (
                "### Cost estimates\n"
                "You do NOT build or run the cost model — a cost helper does. Call `review_plans()` to get\n"
                "estimated cost/latency for your candidate plans (with actuals for executed ones); the same\n"
                "table is also refreshed automatically after each `execute_plan`. Use these as RELATIVE\n"
                "signals to decide which plans are worth executing — check them before executing."
            )
        elif "sampleCost" in mode:
            estimate_rule = (
                "### Estimate before executing\n"
                "Once a cost model is installed, you MUST call `compare_plan_costs` on all new candidate plans\n"
                "before executing any of them. Executing a plan without first consulting cost estimates wastes\n"
                "budget steps and defeats the purpose of the cost model."
            )
        else:
            estimate_rule = ""
        return _SYSTEM_TEMPLATE.format(
            briefing=briefing if briefing is not None else self.briefing,
            quality_metric=_quality_metric_section(eval_metric),
            tools_doc="\n\n".join(t.doc for t in tools),
            physical_sem_ops="\n".join(f"- {d}" for d in _PHYSICAL_SEMANTIC_OPERATORS.values()),
            physical_nonsem_ops="\n".join(f"- {d}" for d in _PHYSICAL_NONSEMANTIC_OPERATORS.values()),
            available_models=_AVAILABLE_MODELS_TEXT,
            max_steps=self.max_steps,
            final_answer_doc=final_answer_doc if final_answer_doc is not None else self.final_answer_doc,
            estimate_rule=estimate_rule,
        )

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
        return (
            f"{task}\n\n"
            f"{plans_section}\n\n"
            f"Observed-results store summary:\n"
            f"{json.dumps(results.summary(), indent=2)}\n\n"
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
        mode: str,
        query_info: dict,
    ) -> _RunContext:
        """Wire up everything the loop needs and reset per-run state.

        Patches litellm→OpenRouter, builds the oracle/quality evaluator, seeds the sandbox
        variables + mode-specific tools, constructs the cost helper (customCost only), builds
        the system prompt, and resets `self.messages`/cost accumulators. Returns the handful of
        objects `run`'s loop consumes.
        """
        patch_litellm_for_openrouter()

        from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor

        registry = CostModelRegistry()
        plan_codes: dict = {}  # populated by WritePlanTool; shared with ExecutePlanTool
        data_dir = query_info["data_dir"]
        # Image files always follow <data_dir>/<image_subdir>/<idx>.jpg.
        image_subdir = query_info.get("image_subdir", "images")

        # Build oracle client and quality evaluator (oracle runs inside QualityEvaluator)
        oracle_client = OpenRouterClient(self.oracle_model, reasoning_effort=self.oracle_reasoning_effort)
        llm_judge_dir = RESULTS_DIR / "llm_judge" / query_info["use_case"]
        import shutil
        _llm_judge_path = pathlib.Path(llm_judge_dir)
        if _llm_judge_path.exists():
            shutil.rmtree(_llm_judge_path)
        _llm_judge_path.mkdir(parents=True, exist_ok=True)

        _oracle_result_path = (
            SEMBENCH_DATASUBSET_DIR
            / query_info["use_case"]
            / f"sf_{query_info['scale_factor']}"
            / f"Q{query_info['query_id']}_oracle_result.csv"
        )
        _oracle_result_path.unlink(missing_ok=True)

        quality_evaluator = None
        quality_evaluator_cls = query_info.get("quality_evaluator_cls", _QualityEvaluator)
        if quality_evaluator_cls is not None:
            try:
                quality_evaluator = quality_evaluator_cls(
                    oracle_client=oracle_client,
                    oracle_model=self.oracle_model,
                    query_id=query_info["query_id"],
                    use_case=query_info["use_case"],
                    scale_factor=query_info["scale_factor"],
                    agent_dir=self.agent_dir,
                    llm_judge_dir=llm_judge_dir,
                    oracle_reasoning_effort=self.oracle_reasoning_effort,
                    use_oracle_ground_truth=self.use_oracle_ground_truth,
                )
            except Exception as e:
                print(f"[run] QualityEvaluator init failed: {e}")

        import pandas as pd
        from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline
        try:
            import palimpzest as pz
        except ImportError as exc:
            raise ImportError("palimpzest is required to build plans") from exc

        def load_data(filename: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(data_dir, filename))

        def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
            # `col_name` is the NEW image column being added; the on-disk path is built from the
            # fixed naming convention (<image_subdir>/<row["idx"]>.jpg).
            # `col_name` MUST NOT collide with an existing column (e.g. the id column). PZ's convert
            # only generates fields not already present on the record, so a colliding name produces
            # an empty field_answers and raises `max() iterable argument is empty` on every row.
            pipeline.map(
                udf=lambda row: {col_name: os.path.join(data_dir, image_subdir, str(row["idx"]) + ".jpg")},
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

        # Main-loop sandbox variables: inspection stores + operator/cost introspection + direct
        # CSV access for data exploration (explore_data). Intentionally EXCLUDES the plan-building
        # helpers (load_data/add_image_data/PhysicalPipeline/pz) — those live only in the minimal
        # plan-construction sandbox below.
        variables = {
            "explore_data": explore_data,
            "plans": plans,
            "plan_codes": plan_codes,
            "plan_results": plan_results,
            "op_results": op_results,
            "iter_operators": iter_operators,
            "get_op_type": get_op_type,
            "get_op_id": get_op_id,
            "get_op_model": get_op_model,
            "describe_operator": describe_operator,
            "observed_op_stats": make_observed_op_stats(op_results),
        }

        # Minimal sandbox variables for WritePlanTool: only what plan CODE needs to construct a
        # PhysicalPipeline — the source-table reader (required by PhysicalPipeline.__init__ for
        # schema inference), the image helper, the pipeline class, and palimpzest (pz.Model /
        # pz.ImageFilepath). No results stores, no cost-model introspection, no explore_data.
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
        self._image_describer = OpenRouterClient("google/gemini-2.5-flash-lite")

        base_tools = [
            ListFilesTool(data_dir),
            ExploreSchemaT(data_dir),
            ExploreSampleTool(data_dir),
            ExploreImagesTool(
                data_dir, self._pending_images,
                subdir=image_subdir,
            ),
            GetOpSamplesTool(plan_results),
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
                scale_factor=query_info["scale_factor"],
                eval_metric=query_info.get("eval_metric"),
                subset_path=query_info.get("subset_path"),
                normalize_eval_df=query_info.get("normalize_eval_df"),
            ),
        ]

        cost_helper: CostHelperAgent | None = None
        if "customCost" in mode:
            # Cost-model authoring + estimation is fully offloaded to the helper.
            # The main agent gets NO estimate/compare/update tools and no registry.
            helper_model = self.helper_model or getattr(self.llm, "model", None)
            if not helper_model:
                raise ValueError(
                    "customCost mode requires a helper LLM model id: pass helper_model=... "
                    "to CostModelAgent, or use an OpenRouterClient main llm exposing `.model`."
                )
            helper_llm = OpenRouterClient(helper_model, reasoning_effort=self.helper_reasoning_effort)
            cost_helper = CostHelperAgent(
                helper_llm,
                registry=registry,
                plans=plans,
                plan_results=plan_results,
                op_results=op_results,
                task=task,
                use_case=query_info["use_case"],
                agent_dir=self.agent_dir,
                verbose=self.verbose,
            )
            base_tools += [ReviewPlansTool(cost_helper)]
            briefing = self.customCost_briefing
            final_answer_doc = self.customCost_final_answer_doc
        elif "execute" in mode:
            briefing = self.execute_briefing
            final_answer_doc = self.execute_final_answer_doc
        elif "sampleCost" in mode:
            try:
                from agent_cost_model.sample_based_cost_model import SampleBasedCostModel
            except ImportError:
                from agent_cost_model.sample_based_cost_model import SampleBasedCostModel  # type: ignore
            registry.install(SampleBasedCostModel(op_results), notes="v0: SampleBasedCostModel pre-installed")
            base_tools += [EstimatePlanCostTool(registry)]
            variables.update({
                "PlanCostEstimate": PlanCostEstimate,
                "SampleBasedCostModel": SampleBasedCostModel,
            })
            briefing = self.sampleCost_briefing
            final_answer_doc = self.sampleCost_final_answer_doc

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

        tools = base_tools + [write_plan_tool]
        executor.send_tools({t.name: t for t in tools})

        system = self._system_prompt(
            tools, briefing, final_answer_doc, mode=mode, eval_metric=query_info.get("eval_metric")
        )
        opening = self._opening_message(task, plans, op_results)
        self.messages = [{"role": "user", "content": opening}]
        self.reasoning_steps = []
        self.trajectory_steps = []
        self.agent_cost_usd = 0.0
        self.opt_latency = 0.0
        self.execution_cost_usd = 0.0
        self.oracle_cost_usd = 0.0
        self.helper_cost_usd = 0.0
        self._log(f"\n=== system prompt ({len(system)} chars) ===\n{system}\n")
        self._log(f"=== task ===\n{opening}\n")

        return _RunContext(
            executor=executor, system=system, registry=registry,
            cost_helper=cost_helper, quality_evaluator=quality_evaluator, plan_codes=plan_codes,
        )

    def _account_costs(
        self, plan_results: ResultsStore, quality_evaluator: Any, cost_helper: "CostHelperAgent | None"
    ) -> None:
        """Roll up the per-run cost totals (subset execution, oracle, helper) onto the instance."""
        self.execution_cost_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in plan_results.rows)
        self.oracle_cost_usd = (
            float(getattr(quality_evaluator, "total_oracle_cost_usd", 0.0) or 0.0)
            if quality_evaluator is not None else 0.0
        )
        if cost_helper is not None:
            self.helper_cost_usd = cost_helper.cost_usd

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        task: str,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        *,
        mode: str,
        query_info: dict = {
            "use_case": "use_case",
            "scale_factor": 0,
            "query_id": 0,
            "eval_metric": None,  # e.g. "f1-score" / "adjusted-rand-index"; explains the quality metric in the briefing
            "final_eval_runs": 1,  # times to re-run the chosen plan on the full dataset (fresh pipeline each)
            "data_dir": "experiments/dataset/use_case",
            "gt_dir": "files/use_case/raw_results/ground_truth/sf_0",
            "image_subdir": "images",
        },
    ) -> Any:
        """Bounded tool loop over `plans`/`plan_results`/`op_results`, returning the JSON final
        answer. `_setup_run` does all the wiring; this method is just the step loop."""
        optimization_start = time.time()

        ctx = self._setup_run(task, plans, plan_results, op_results, mode, query_info)
        executor, system, registry = ctx.executor, ctx.system, ctx.registry
        cost_helper, quality_evaluator, plan_codes = ctx.cost_helper, ctx.quality_evaluator, ctx.plan_codes

        prev_plan_rows = len(plan_results.rows)  # detect execution batches to trigger the helper
        helper_consumed = 0  # cost-helper trajectory rows already merged into the main trajectory
        step = 0
        while step < self.max_steps:
            step += 1
            try:
                raw = self._llm_step(system)
            except Exception as e:  # LLM transport failure
                self._log(f"[step {step}] LLM error: {e}")
                raise
            self.messages.append({"role": "assistant", "content": raw})
            reasoning = self.reasoning_steps[-1]
            self.trajectory_steps.append(
                {"step": step, "reasoning": reasoning, "assistant": raw, "observation": None}
            )
            if reasoning:
                self._log(f"\n--- reasoning (step {step}) ---\n{reasoning}\n")
            self._log(f"\n--- assistant (step {step}) ---\n{raw}\n")

            # parse (with a couple of format-error retries)
            try:
                parsed = self._parse_with_retries(system, raw)
            except ParseError as e:
                obs = f"Observation (step {step}): {e.detail}"
                self.messages.append({"role": "user", "content": obs})
                self.trajectory_steps[-1]["observation"] = obs
                self._log(f"[parse error] {obs}")
                continue

            if parsed.code is None:  # final answer
                self.opt_latency = time.time()-optimization_start
                self._log(f"[final answer] {parsed.result}")
                self.trajectory_steps[-1]["observation"] = f"[final answer] {parsed.result}"
                if isinstance(parsed.result, dict):
                    parsed.result["plan_codes"] = plan_codes
                    parsed.result["plan_descriptions"] = {name: entry.get("description", "") for name, entry in plans.items()}
                    parsed.result["agent_model"] = getattr(self.llm, "model", None)
                    parsed.result["oracle_model"] = self.oracle_model
                    parsed.result["helper_model"] = self.helper_model or getattr(self.llm, "model", None)
                self._save_results_df(plan_results, query_info, final_answer=parsed.result)
                helper_consumed = self._merge_helper_trajectory(cost_helper, step, helper_consumed)
                self._save_trajectory_df(query_info)
                if cost_helper is not None:
                    self._save_cost_model_codes(cost_helper.model_versions, query_info)
                self._account_costs(plan_results, quality_evaluator, cost_helper)
                self._run_final_evaluation(parsed.result, plans, query_info, plan_codes, plan_results)
                return parsed.result

            # execute the python tool-call block
            try:
                out = executor(parsed.code)
            except Exception as e:
                obs = f"Observation (step {step}): exec failed — {type(e).__name__}: {e}"
                self.messages.append({"role": "user", "content": obs})
                self.trajectory_steps[-1]["observation"] = obs
                self._log(obs)
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
            self.messages.append({"role": "user", "content": obs})
            self.trajectory_steps[-1]["observation"] = obs
            self._log(obs)

            # After an execution batch (plan_results grew — only ExecutePlanTool appends
            # there), auto-invoke the cost helper: it decides whether to (re)fit the cost
            # model and returns fresh estimate-vs-actual signal for the main agent.
            if cost_helper is not None and len(plan_results.rows) > prev_plan_rows:
                prev_plan_rows = len(plan_results.rows)
                reason = "bootstrap" if registry.version == 0 else "post-batch"
                try:
                    table = cost_helper.review(reason=reason)
                except Exception as e:
                    table = f"[cost-helper error: {type(e).__name__}: {e}]"
                helper_obs = f"[cost estimates refreshed after execution]\n{table}"
                self.messages.append({"role": "user", "content": helper_obs})
                self.trajectory_steps[-1]["observation"] = (
                    (self.trajectory_steps[-1]["observation"] or "") + "\n\n" + helper_obs
                )
                self._log(helper_obs)

            # interleave any cost-helper rows generated during this step (via review_plans or
            # the post-execution auto-invoke) into the shared trajectory, labeled per this step
            helper_consumed = self._merge_helper_trajectory(cost_helper, step, helper_consumed)

        # out of steps — one forced terminal turn
        result = self._terminal_turn(system)
        self.opt_latency = time.time() - optimization_start
        self._save_results_df(plan_results, query_info)
        helper_consumed = self._merge_helper_trajectory(cost_helper, step, helper_consumed)
        self._save_trajectory_df(query_info)
        if cost_helper is not None:
            self._save_cost_model_codes(cost_helper.model_versions, query_info)
        if isinstance(result, dict):
            result["plan_codes"] = plan_codes
            result["agent_model"] = getattr(self.llm, "model", None)
            result["oracle_model"] = self.oracle_model
            result["helper_model"] = self.helper_model or getattr(self.llm, "model", None)
        self._account_costs(plan_results, quality_evaluator, cost_helper)
        self._run_final_evaluation(result, plans, query_info, plan_codes, plan_results)
        return result

    # -- helpers -----------------------------------------------------------
    _IMAGE_DESCRIBE_SYSTEM = (
        "You are a vision assistant helping a query-planning agent explore a dataset's images. "
        "For each attached image, write ONE concise, factual line describing what it shows — main "
        "object/subject, dominant colors, and any attributes useful for filtering (product type, "
        "style, visible text). Prefix each line with the image's id. No preamble, no summary."
    )

    def _describe_images(self, images: list[dict]) -> str:
        """Generate a textual description of the staged explore_images pictures.

        Uses a dedicated cheap vision model (`_image_describer`, gemini-2.5-flash-lite) so image
        exploration stays inexpensive regardless of the main-agent model. The description is shown in
        THIS step's observation and persisted in the trajectory; the pixels themselves are not
        forwarded to the main agent. Cost is billed to agent_cost_usd, matching normal LLM steps."""
        content: list[dict] = [{
            "type": "text",
            "text": f"Describe each of these {len(images)} image(s), one short line each, prefixed with its id:",
        }]
        for img in images:
            content.append({"type": "text", "text": f"id {img['id']}:"})
            content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        try:
            result = self._image_describer.generate(self._IMAGE_DESCRIBE_SYSTEM, [{"role": "user", "content": content}])
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

    _OBS_CHAR_LIMIT = 10_000

    @staticmethod
    def _format_observation(step: int, out: Any) -> str:
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
        obs = "\n\n".join(parts)
        limit = CostModelAgent._OBS_CHAR_LIMIT
        if len(obs) > limit:
            obs = obs[:limit] + (
                f"\n\n[output truncated — {len(obs) - limit} chars omitted. "
                "Use get_op_samples(plan_name, op_name) to inspect specific input/output pairs.]"
            )
        return obs

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

        The chosen plan is re-run `final_eval_runs` times (each a fresh pipeline) to average out LLM
        stochasticity. The metrics entry keeps the agent-search-level fields once, with per-run
        full-dataset execution metrics nested under run1/run2/…; raw output of run k is saved as
        Q{query_id}_{runcount}_{k}.csv."""
        if not isinstance(final_answer, dict):
            return
        best_name = final_answer.get("best_plan", {}).get("name")
        if not best_name or best_name not in plans:
            self._log(f"[final_eval] plan {best_name!r} not in plans — skipping final evaluation")
            return

        query_id = query_info["query_id"]
        use_case = query_info["use_case"]
        scale_factor = query_info["scale_factor"]
        runcount = query_info.get("runcount")
        run_suffix = f"Q{query_id}_{runcount}" if runcount is not None else f"Q{query_id}"
        gt_dir = query_info.get("gt_dir", SEMBENCH_FILES_DIR / use_case / "raw_results" / "ground_truth" / f"sf_{scale_factor}")

        import dataclasses
        import pathlib

        import pandas as pd

        gt_path = pathlib.Path(gt_dir) / f"Q{query_id}.csv"
        if not gt_path.exists():
            self._log(f"[final_eval] ground truth not found at {gt_path} — skipping")
            return

        raw_results_dir = RESULTS_DIR / "raw_results" / use_case / self.agent_dir / f"sf_{scale_factor}"
        raw_results_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluator + ground truth once; reused across every repeated final run.
        try:
            evaluator_loader = query_info.get("evaluator_loader")
            if evaluator_loader is None:
                from agent_cost_model.experiments.SemBench.quality_evaluator import load_evaluator
                evaluator_loader = load_evaluator
            evaluator = evaluator_loader(use_case, scale_factor)
            gt_df = pd.read_csv(gt_path)
        except Exception as e:
            self._log(f"[final_eval] evaluator/ground-truth load failed: {type(e).__name__}: {e}")
            evaluator, gt_df = None, None

        n_runs = max(1, int(query_info.get("final_eval_runs", 1) or 1))
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
            normalize_eval_df = query_info.get("normalize_eval_df", _normalize_plan_df)
            eval_df = normalize_eval_df(results_df, use_case, query_id)

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
            if evaluator is not None and gt_df is not None:
                try:
                    qm = evaluator._evaluate_single_query(query_id, eval_df, gt_df)
                    qm_dict = dataclasses.asdict(qm)
                    qm_type = type(qm).__name__
                    if "Retrieval" in qm_type:
                        metric_type = "f1_score"
                        quality = float(qm_dict.get("f1_score", float("nan")))
                    elif "Aggregation" in qm_type:
                        metric_type = "relative_error"
                        quality = 1.0 / (1.0 + float(qm_dict.get("relative_error", 1.0)))
                    elif "Rank" in qm_type:
                        metric_type = "spearman_correlation"
                        quality = float(qm_dict.get("spearman_correlation", float("nan")))
                    elif "SingleAccuracy" in qm_type:
                        metric_type = "accuracy"
                        quality = float(qm_dict.get("accuracy", float("nan")))
                except Exception as e:
                    self._log(f"[final_eval] evaluation failed (run {final_run}): {type(e).__name__}: {e}")

            runs[f"run{final_run}"] = {
                "latency": round(total_latency, 4),
                "cost": round(total_cost, 6),
                "quality": quality,
            }

        metrics_path = RESULTS_DIR / "metrics" / use_case / f"sf_{scale_factor}" / f"{self.agent_dir}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict = {}
        if metrics_path.exists():
            try:
                entry = json.loads(metrics_path.read_text())
            except Exception:
                entry = {}
        plans_written = len(plan_codes) if plan_codes is not None else 0
        _executed_rows = [r.get("plan_name") for r in (plan_results.rows if plan_results is not None else []) if r.get("plan_name")]
        plans_executed = len(_executed_rows)
        unique_plans_executed = len(set(_executed_rows))
        metrics_key = f"{query_id}_{runcount}" if runcount is not None else str(query_id)
        # Agent-search-level fields once, then per-run full-dataset execution metrics nested under runK.
        entry[metrics_key] = {
            "query_id": str(query_id),
            "scale_factor": scale_factor,
            "agent_model": getattr(self.llm, "model", None),
            "oracle_model": self.oracle_model,
            "helper_model": self.helper_model or getattr(self.llm, "model", None),
            "agent_cost": round(self.agent_cost_usd, 6),
            "agent_latency": round(self.opt_latency, 4),
            "helper_cost": round(self.helper_cost_usd, 6),
            "subset_execution_cost": round(self.execution_cost_usd, 6),
            "oracle_cost": round(self.oracle_cost_usd, 6),
            "metric_type": metric_type,
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
        self._log(f"[final_eval] metrics ({len(runs)}/{n_runs} run(s) succeeded) → {metrics_path}")

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
