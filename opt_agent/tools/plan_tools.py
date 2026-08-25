"""Tools for building, storing, executing, and inspecting physical query plans."""

from __future__ import annotations

import pathlib
from typing import Any

from agent_cost_model.opt_agent.cost_model_types import ResultsStore, _dump_opt_debug_artifacts, _normalize_plan_df
from agent_cost_model.paths import SEMBENCH_DATASUBSET_DIR
from agent_cost_model.opt_agent.prompts import _quality_metric_reminder

from .base import Tool


class GetOpSamplesTool(Tool):
    name = "get_op_samples"
    doc = """\
### get_op_samples(plan_name, op_name=None, n=3)
Retrieve sample (input, output) pairs for an executed plan from `plan_results`.
Optionally scope to a single operator with `op_name` (e.g. "p1_op1").
Returns at most `n` samples per operator.

```python
get_op_samples("p1")                  # all operators, 3 samples each
get_op_samples("p1", "p1_op2", n=5)  # just op2, up to 5 samples
```"""

    def __init__(self, plan_results: ResultsStore) -> None:
        self._plan_results = plan_results

    def __call__(self, plan_name: str, op_name: str | None = None, n: int = 3) -> str:
        row = next((r for r in self._plan_results.rows if r.get("plan_name") == plan_name), None)
        if row is None:
            available = [r.get("plan_name") for r in self._plan_results.rows]
            return f"No executed plan named {plan_name!r}. Available: {available}"
        samples = row.get("op_samples", {})
        if not samples:
            return f"No op_samples recorded for plan {plan_name!r}."
        if op_name is not None:
            if op_name not in samples:
                return f"No operator {op_name!r} in plan {plan_name!r}. Available: {list(samples)}"
            samples = {op_name: samples[op_name]}
        lines = []
        for op, pairs in samples.items():
            lines.append(f"{op} ({min(len(pairs), n)}/{len(pairs)} samples shown):")
            for i, pair in enumerate(pairs[:n]):
                lines.append(f"  [{i}] input:  {pair['input']}")
                lines.append(f"       output: {pair['output']}")
        return "\n".join(lines)


class WritePlanTool(Tool):
    name = "write_plan"
    doc = """\
### write_plan(code, name, description="")
Build and store a physical query plan WITHOUT executing it. `code` is a Python
string that constructs a PhysicalPipeline and returns it as its last expression —
do NOT call `.run()` in the plan code; `execute_plan` handles execution.
`name` is the plan identifier you choose (e.g. "p1"). `description` is a short
high-level label of the plan and the optimizations it embodies
(e.g. "cheap sem_filter on truncated text, then sem_filter on image") — it is
shown back to you in cost/estimate tables and helps you compare optimization ideas.

After this call, `plans[name]["plan"]` holds the built pipeline and
`plans[name]["description"]` holds your label.

Use `load_data(filename)` to read the relavent CSV and seed the pipeline's source table.
Use `add_image_data(pipeline: PhysicalPipeline, image_col: str)` to attach images: it adds a NEW
column named `image_col` (type `pz.ImageFilepath`) holding each row's image-file path. `image_col`
MUST be a fresh column name — do NOT reuse an existing column such as the id column. Reusing an
existing name generates nothing (the map produces no new field), so every row errors and the plan
returns 0 output. Then pass images to a model via `depends_on=["<image_col>"]` on the semantic op.


```python
write_plan(\"\"\"
email = PhysicalPipeline(plan_name, "emails", load_data("Emails.csv"))
email.sem_filter("this email quotes someone outside of the the sender's company", model=pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE)
email.project(["emailId"])
email.limit(5)
email
\"\"\", "p1", description="baseline: single cheap sem_filter on full text")
# `plan_name` is automatically set to the name you pass (here "p1")
# plans["p1"]["plan"] now holds the built `email`pipeline instance.
```"""

    def __init__(self, plan_codes: dict, plans: dict, executor: Any) -> None:
        self._plan_codes = plan_codes
        self._plans = plans
        self._executor = executor

    def __call__(self, code: str, plan_name: str, description: str = "") -> dict:
        try:
            from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline
        except ImportError:
            from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline

        self._executor.send_variables({"plan_name": plan_name})
        exec_result = self._executor(code)
        pipeline = exec_result.output
        if not isinstance(pipeline, PhysicalPipeline):
            raise TypeError(
                f"Plan code must return a PhysicalPipeline as its last expression "
                f"(got {type(pipeline).__name__}). Do not call .run() in the plan code."
            )

        self._plan_codes[plan_name] = code
        self._plans[plan_name] = {"plan": pipeline, "description": description}
        return {"plan_name": plan_name, "description": description, "total_plans": len(self._plan_codes)}


class ExecutePlanTool(Tool):
    name = "execute_plan"
    doc = """\
### execute_plan(name)
Execute the stored plan `name` on a reproducible sample of records. Plan-level and
operator-level quality, latency, cost, and token usage are appended to `plan_results`
and `op_results`, respectively. After execution, `plans[name]["plan"]` holds the PhysicalPipeline.

Returns a compact summary dict with plan stats and per-operator stats. Key fields:
- `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
  Treat oracle quality scores as ground truth.
- `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
  which operator is the bottleneck.
- `cost_usd`, `latency_s`, `input_tokens`, `output_tokens`: aggregated over all ops.
To inspect accumulated results use `plan_results.df` and `op_results.df`.
To view sample input/output pairs use `get_op_samples(plan_name)`.
```python
execute_plan("p1")
```"""

    def __init__(
        self,
        plan_codes: dict,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        use_case: str,
        query_id: int,
        data_dir: str,
        agent_dir: str,
        quality_evaluator: Any,
        scale_factor: int,
        eval_metric: str | None = None,
        subset_path: str | pathlib.Path | None = None,
        normalize_eval_df: Any = None,
    ) -> None:
        import pathlib

        self._plan_codes = plan_codes
        self._plans = plans
        self._op_results = op_results
        self._plan_results = plan_results
        self._use_case = use_case
        self._query_id = query_id
        self._data_dir = pathlib.Path(data_dir)
        self._agent_dir = agent_dir
        self._quality_evaluator = quality_evaluator
        self._scale_factor = scale_factor
        self._eval_metric = eval_metric
        self._subset_path = pathlib.Path(subset_path) if subset_path else None
        self._normalize_eval_df = normalize_eval_df or _normalize_plan_df

    def __call__(self, plan_name: str) -> dict:
        import pandas as pd

        entry = self._plans.get(plan_name)
        if entry is None:
            raise KeyError(
                f"No plan named {plan_name!r}. Call write_plan first. "
                f"Available: {list(self._plans)}"
            )
        pipeline = entry["plan"]

        subset_path = self._subset_path or (
            SEMBENCH_DATASUBSET_DIR / self._use_case / f"sf_{self._scale_factor}" / f"Q{self._query_id}_subset.csv"
        )
        plan_exec_error: Exception | None = None
        per_op_list, plan_context, plan_dict = [], None, {}
        try:
            # The plan is (re)executed on the subset every time — never cached — so that
            # plan_dict["latency_s"] is a real WALL-CLOCK measurement of this run. (Only the
            # oracle, which is used purely for quality scoring, is cached across plans.)
            per_op_list, plan_context, plan_dict = pipeline.run_subset(subset_cache_path=str(subset_path))
        except Exception as e:
            plan_exec_error = e
            print(f"[execute_plan] plan execution failed for {plan_name}: {type(e).__name__}: {e}")
        entry["plan"] = pipeline

        raw_output_df = pd.DataFrame(
            plan_context.output_records if plan_context is not None else []
        )
        plan_output_df = self._normalize_eval_df(raw_output_df, self._use_case, self._query_id)

        quality_result = None
        if self._quality_evaluator is not None:
            try:
                # Pass an empty SubsetExecutionContext when the plan failed so the oracle
                # still runs (populating _canonical_oracle_df for later plans).
                if plan_context is None:
                    from agent_cost_model.opt_agent.physical_pipeline import SubsetExecutionContext
                    plan_context = SubsetExecutionContext(
                        sampled_records=[], output_records=[],
                        per_sem_op_info={}, has_join=False, right_sampled_records=None,
                    )
                quality_result = self._quality_evaluator.evaluate(
                    pipeline, plan_name, plan_context, plan_output_df
                )
            except Exception as e:
                print(f"[execute_plan] quality evaluation failed for {plan_name}: {type(e).__name__}: {e}")

        try:
            _dump_opt_debug_artifacts(
                self._query_id, plan_name, raw_output_df, plan_output_df, plan_context, quality_result
            )
        except Exception as e:
            print(f"[execute_plan] opt_results debug dump failed for {plan_name}: {type(e).__name__}: {e}")

        if plan_exec_error is not None:
            raise RuntimeError(
                f"Plan {plan_name!r} execution failed: {type(plan_exec_error).__name__}: {plan_exec_error}"
            )

        # Build op_samples for GetOpSamplesTool compatibility
        op_samples: dict[str, list] = {}
        for _stage_idx, info in plan_context.per_sem_op_info.items():
            op_n = info["op_name"]
            raw_samples = info.get("samples", [])
            op_samples[op_n] = [
                {"input": str(inp), "output": str(out) if out is not None else None}
                for inp, out in raw_samples[:5]
            ]

        total_cost = sum(e.get("cost_usd", 0) for e in per_op_list)
        # latency_s: SUM of per-op per-record latencies (serial-equivalent). Kept as the plan's
        # "actual latency" for the cost-model est-vs-act comparison, whose prediction (est.time)
        # is also a per-op sum. wall_latency_s: real WALL-CLOCK time of this subset execution
        # (from plan_dict), which reflects the join/convert parallelism.
        total_latency = sum(e.get("latency_s", 0) for e in per_op_list)
        wall_latency_s = float(plan_dict.get("latency_s", 0.0) or 0.0)
        total_in_tok = sum(e.get("input_tokens", 0) for e in per_op_list)
        total_out_tok = sum(e.get("output_tokens", 0) for e in per_op_list)

        plan_row = {
            "plan_name": plan_name,
            "description": entry.get("description", ""),
            "plan_str": str(pipeline),
            "cost_usd": total_cost,
            "latency_s": total_latency,
            "wall_latency_s": wall_latency_s,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
            "quality": quality_result.quality if quality_result is not None else float("nan"),
            "per_sem_op_quality": (
                quality_result.per_sem_op_quality if quality_result is not None else {}
            ),
            "op_samples": op_samples,
        }
        self._plan_results.append([plan_row])
        self._op_results.append(per_op_list)

        plan_summary = {k: v for k, v in plan_row.items() if k not in ("op_samples", "plan_str")}
        # When quality is N/A because the oracle returned no rows, explain it in the observation so
        # the agent doesn't read the bare NaN as a failed plan.
        if quality_result is not None and getattr(quality_result, "quality_note", None):
            plan_summary["quality_note"] = quality_result.quality_note
        # Remind the agent, on every execution, what `quality` means and how it differs from
        # per_sem_op_quality (context for why the two can diverge).
        return {
            "plan_summary": plan_summary,
            "op_summary": per_op_list,
            "quality_metric": _quality_metric_reminder(self._eval_metric),
        }
