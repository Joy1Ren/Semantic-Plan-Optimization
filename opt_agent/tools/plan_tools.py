"""Tools for building, storing, executing, and inspecting physical query plans."""

from __future__ import annotations

import pathlib
from typing import Any

from agent_cost_model.opt_agent.cost_model_types import ResultsStore, _dump_opt_debug_artifacts, _normalize_plan_df
from agent_cost_model.opt_agent.prompts import _quality_metric_reminder, _sem_op_quality_docs

from .base import Tool

# ---------------------------------------------------------------------------
# Operator input/output samples
#
# A sample is stored per FIELD ({field_name: value}) rather than as one flat
# `str(DataRecord)`, which went through palimpzest's DataRecord.__str__ and cut EVERY field to 15
# characters (palimpzest/core/elements/records.py). Per-field storage is what lets the display
# below truncate, budget, and de-duplicate values field by field.
# ---------------------------------------------------------------------------
SAMPLE_FIELD_CHARS = 100    # per-field cap applied when a sample is STORED
OP_DISPLAY_BUDGET = 4000    # per-operator cap on value chars applied when samples are DISPLAYED
_ELLIPSIS = "…"


def _cap(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + _ELLIPSIS


def _record_fields(dr: Any, limit: int = SAMPLE_FIELD_CHARS) -> dict[str, str] | None:
    """{field_name: value} for one DataRecord, each value stringified and capped at `limit` chars.

    None (the record did not pass the operator) maps to None."""
    if dr is None:
        return None
    try:
        schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
        field_names = list(schema_cls.model_fields)
    except Exception:
        return {"<record>": _cap(str(dr), limit)}
    return {k: _cap(str(getattr(dr, k, None)), limit) for k in field_names}


def _allocate(lengths: list[int], budget: int) -> list[int]:
    """Max-min fair split of `budget` characters over values of the given lengths.

    Every value that fits in its equal share (budget / #values) prints in full, and the share it
    does not use is redistributed over the values that are still too long — repeatedly, shortest
    first — so the budget is spent evenly without cutting short values needlessly."""
    alloc = [0] * len(lengths)
    remaining_budget, remaining_n = budget, len(lengths)
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        share = remaining_budget // remaining_n if remaining_n > 0 else 0
        alloc[i] = min(lengths[i], max(share, 0))
        remaining_budget -= alloc[i]
        remaining_n -= 1
    return alloc


def _unpack_op_samples(entry: Any) -> tuple[str, list]:
    """(op_type, pairs) from a stored op_samples entry. Accepts the current
    {"op_type", "samples"} shape and the older bare list of {"input", "output"} strings."""
    if isinstance(entry, dict):
        return str(entry.get("op_type") or ""), list(entry.get("samples") or [])
    return "", list(entry or [])


class GetOpSamplesTool(Tool):
    name = "get_op_samples"
    doc = """\
### get_op_samples(plan_name, op_name, n=3)
Retrieve sample (input, output) pairs for an executed plan from `plan_results`.
You must scope to a single operator with `op_name`, which is `{plan}_op{idx}_{op_type}`
(e.g. "p1_op1_rag_map") — the same names `plan_str` and `op_results` use.
Returns at most `n` samples per operator.

```python
get_op_samples("p1", "p1_op2_sem_map", n=5)   # up to 5 samples from this operator
```"""

    def __init__(self, plan_results: ResultsStore) -> None:
        self._plan_results = plan_results

    def __call__(self, plan_name: str, op_name: str, n: int = 3) -> str:
        if not op_name:
            return "You must specify an operator name (e.g. 'p1_op2_sem_map') to retrieve samples."
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

        # De-duplication is a whole-plan concern: it collapses values that a downstream operator
        # merely carries through from an upstream one. Asking for a single operator prints it
        # in full (subject to the char budget). Keyed by (sample index, field name) so that two
        # different records of the same operator never collapse into each other.
        dedup = op_name is None
        seen: dict[tuple[int, str], str] = {}

        lines = [
            f"[fields capped at {SAMPLE_FIELD_CHARS} chars when sampled, "
            f"{OP_DISPLAY_BUDGET} chars of values per operator when shown"
            + ("; `field=…` = unchanged from where it was first shown above]" if dedup else "]")
        ]
        for op, entry in samples.items():
            op_type, pairs = _unpack_op_samples(entry)
            shown = pairs[:n]
            # Filters and joins do not rewrite fields: the sampled output IS the input record when
            # the record passes, and None when it does not (see PhysicalPipeline._execute_core).
            # Printing the verdict says everything repeating every field would.
            gate = "filter" if op_type.endswith("filter") else "join condition" if op_type.endswith("join") else None

            # Pass 1: lay out every cell, marking the ones de-duplication collapses.
            rendered: list[tuple[int, str, Any]] = []   # (sample idx, side, str | list[cell])
            budgeted: list[dict] = []                   # cells that still need a char allowance
            for i, pair in enumerate(shown):
                for side in ("input", "output"):
                    fields = pair.get(side) if isinstance(pair, dict) else None
                    if side == "output" and gate is not None:
                        rendered.append((i, side, f"passed {gate}" if fields is not None else f"did not pass {gate}"))
                        continue
                    if fields is None:
                        rendered.append((i, side, "∅"))
                        continue
                    if not isinstance(fields, dict):    # legacy flat-string sample
                        rendered.append((i, side, str(fields)))
                        continue
                    cells = []
                    for k, v in fields.items():
                        collapsed = dedup and seen.get((i, k)) == v
                        cell = {"field": k, "value": v, "collapsed": collapsed, "alloc": 0}
                        cells.append(cell)
                        if not collapsed:
                            seen[(i, k)] = v
                            budgeted.append(cell)
                    rendered.append((i, side, cells))

            # Pass 2: spend this operator's char budget over the cells that are actually printed.
            for cell, alloc in zip(budgeted, _allocate([len(c["value"]) for c in budgeted], OP_DISPLAY_BUDGET)):
                cell["alloc"] = alloc

            lines.append(f"{op} ({len(shown)}/{len(pairs)} samples shown):")
            for i, side, payload in rendered:
                label = f"  [{i}] input:  " if side == "input" else "       output: "
                body = payload if isinstance(payload, str) else _render_cells(payload)
                lines.append(f"{label}{body}")
        return "\n".join(lines)


def _render_cells(cells: list[dict]) -> str:
    # Very common once de-duplication is on (a filter's input, or any operator that only reads
    # what an upstream operator produced): naming every carried-through field one by one is pure
    # noise, so say it once.
    if cells and all(c["collapsed"] for c in cells):
        return f"{{{_ELLIPSIS} all {len(cells)} fields unchanged}}"
    parts = []
    for cell in cells:
        value, alloc = cell["value"], cell["alloc"]
        if cell["collapsed"] or alloc <= 0:
            parts.append(f"{cell['field']}={_ELLIPSIS}")
        elif alloc < len(value):
            parts.append(f"{cell['field']}={value[:alloc]!r}{_ELLIPSIS}")
        else:
            parts.append(f"{cell['field']}={value!r}")
    return "{" + ", ".join(parts) + "}"


class WritePlanTool(Tool):
    name = "write_plan"
    doc = """\
### write_plan(code, plan_name, description="", optimizations=None)
Build and store a physical query plan WITHOUT executing it. `code` is a Python
string that constructs a PhysicalPipeline and returns it as its last expression —
do NOT call `.run()` in the plan code; `execute_plan` handles execution.
`plan_name` is the plan identifier you choose (e.g. "p1"). `description` is a short
high-level label of the plan and the optimizations it embodies
(e.g. "cheap sem_filter on truncated text, then sem_filter on image") — it is
shown back to you in cost/estimate tables and helps you compare optimization ideas.

`optimizations` records WHICH optimizations this plan uses and HOW FAR each one is
pushed. Pass a dict with the top-level keys "improve quality" and/or "reduce cost",
each mapping an optimization name you choose to the extent you took it:

```python
optimizations={
    "improve quality": {"model": "medium",
                        "logical structure": "divide single sem_map into four separate sem_map"},
    "reduce cost": {"input truncation": "embedding-based RAG before map"},
}
```

The structure is free-form — name the optimizations in your own words, and include
only the ones this plan ACTUALLY uses (not every plan reduces cost, not every plan
improves quality). The *extent* is the part that matters: write "medium" / "top-3
chunks" / "4-way split", not just "model" / "truncation", so that how far a knob has
been turned is readable from the record.

After this call, `plans[name]["plan"]` holds the built pipeline,
`plans[name]["description"]` holds your label, and `plans[name]["optimizations"]`
holds what you recorded here.

EVERY column name a `map`/`sem_map`/`rag_map` declares must be NEW — an operator can only ADD
columns, it can never overwrite one that an earlier operator produced. Re-declaring a name is
rejected here.
```python
pipe.rag_map(cols=[{"name": "Parties_rag", ...}], ...)      # NOT "Parties"
pipe.sem_map(cols=[{"name": "Parties_llm", ...}], ...)
pipe.map(lambda r: {"Parties": r["Parties_llm"] or r["Parties_rag"]},
         cols=[{"name": "Parties", ...}])                   # first use of "Parties"
```

Use `load_data(filename)` to read the relavent CSV and seed the pipeline's source table.
Use `add_image_data(pipeline: PhysicalPipeline, image_col: str)` to attach images: it adds a NEW
column named `image_col` (type `pz.ImageFilepath`) holding each row's image-file path. `image_col`
is subject to the same rule — it MUST be a fresh column name, not an existing one such as the id
column. Then pass images to a model via `depends_on=["<image_col>"]` on the semantic op.


```python
write_plan(\"\"\"
email = PhysicalPipeline(plan_name, "emails", load_data("Emails.csv"))
email.sem_filter("this email quotes someone outside of the the sender's company", model=pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE)
email.project(["emailId"])
email.limit(5)
email
\"\"\", "p1", description="baseline: single cheap sem_filter on full text",
   optimizations={"reduce cost": {"model": "weak"}})
# `plan_name` is automatically set to the name you pass (here "p1")
# plans["p1"]["plan"] now holds the built `email`pipeline instance.
```"""

    def __init__(self, plan_codes: dict, plans: dict, executor: Any) -> None:
        self._plan_codes = plan_codes
        self._plans = plans
        self._executor = executor

    def __call__(
        self, code: str, plan_name: str, description: str = "", optimizations: Any = None
    ) -> dict:
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
        # `optimizations` is stored EXACTLY as given and never validated: its structure is a
        # contract between the plan-writing agent and the exploration checker, both LLMs. No
        # code here (or anywhere else) reads its keys.
        self._plans[plan_name] = {
            "plan": pipeline, "description": description, "optimizations": optimizations,
        }
        result = {
            "plan_name": plan_name,
            "description": description,
            "optimizations": optimizations,
            "total_plans": len(self._plan_codes),
        }
        if not optimizations:
            result["warning"] = (
                "No `optimizations` recorded for this plan. Pass optimizations={...} naming which "
                "optimizations it uses and how far each is pushed, so the optimizations you have "
                "and haven't tried stay legible across plans."
            )
        return result


class ExecutePlanTool(Tool):
    name = "execute_plan"
    doc = """\
### execute_plan(plan_name)
Execute the stored plan `plan_name` on a reproducible sample of records. Plan-level and
operator-level quality, latency, cost, and token usage are appended to `plan_results`
and `op_results`, respectively. After execution, `plans[plan_name]["plan"]` holds the PhysicalPipeline.

Returns a compact summary dict with plan stats and per-operator stats. Key fields:
- `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
  Treat oracle quality scores as ground truth.
- `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
  which operator is the bottleneck. `per_sem_op_quality_metric` states how that
  score is computed for each operator type in this plan — read it before acting on
  a low score, since the definition differs by operator type.
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
        results_prefix: str | pathlib.Path,
        eval_metric: str | None = None,
        subset_path: str | pathlib.Path | None = None,
        normalize_eval_df: Any = None,
        runcount: Any = None,
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
        self._results_prefix = pathlib.Path(results_prefix)
        self._runcount = runcount
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

        if self._subset_path is None:
            raise ValueError(
                "execute_plan needs a datasubset to run on, but no subset_path was supplied. "
                "It comes from the benchmark's `subset_path` (see benchmark.yaml), threaded "
                "through query_info by the runner."
            )
        subset_path = self._subset_path
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

        total_cost = sum(e.get("cost_usd", 0) for e in per_op_list)
        # latency_s: SUM of per-op per-record latencies (serial-equivalent). Kept as the plan's
        # "actual latency" for the cost-model est-vs-act comparison, whose prediction (est.time)
        # is also a per-op sum. wall_latency_s: real WALL-CLOCK time of this subset execution
        # (from plan_dict), which reflects the join/convert parallelism.
        total_latency = sum(e.get("latency_s", 0) for e in per_op_list)
        wall_latency_s = float(plan_dict.get("latency_s", 0.0) or 0.0)
        total_in_tok = sum(e.get("input_tokens", 0) for e in per_op_list)
        total_out_tok = sum(e.get("output_tokens", 0) for e in per_op_list)

        try:
            _dump_opt_debug_artifacts(
                self._results_prefix, self._query_id, plan_name,
                raw_output_df, plan_output_df, plan_context, quality_result,
                plan_metrics={
                    "cost_usd": total_cost,
                    "latency_s": total_latency,
                    "wall_latency_s": wall_latency_s,
                },
                runcount=self._runcount,
                # Set by PlanQualityEvaluator.evaluate on the call just above; absent when
                # the plan has no evaluator or evaluation raised. The ground truth is not passed
                # here -- the evaluator owns it and writes one shared copy at the run level.
                oracle_df=getattr(self._quality_evaluator, "last_oracle_df", None),
                evaluator=self._quality_evaluator,
            )
        except Exception as e:
            print(f"[execute_plan] opt_results debug dump failed for {plan_name}: {type(e).__name__}: {e}")

        if plan_exec_error is not None:
            raise RuntimeError(
                f"Plan {plan_name!r} execution failed: {type(plan_exec_error).__name__}: {plan_exec_error}"
            )

        # Build op_samples for GetOpSamplesTool. Records are stored field by field (each value
        # capped at SAMPLE_FIELD_CHARS) so the display can budget and de-duplicate them; op_type
        # rides along so the display knows e.g. that a filter's output is just a verdict.
        op_samples: dict[str, dict] = {}
        for _stage_idx, info in plan_context.per_sem_op_info.items():
            op_n = info["op_name"]
            raw_samples = info.get("samples", [])
            op_samples[op_n] = {
                "op_type": info.get("op_type", ""),
                "samples": [
                    {"input": _record_fields(inp), "output": _record_fields(out)}
                    for inp, out in raw_samples[:5]
                ],
            }

        plan_row = {
            "plan_name": plan_name,
            "description": entry.get("description", ""),
            "optimizations": entry.get("optimizations"),
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
        # per_sem_op_quality (context for why the two can diverge), plus how each per-op score in
        # THIS plan is computed -- the definition differs by operator type, and a rag_* score in
        # particular measures only the LLM step, not retrieval.
        result = {
            "plan_summary": plan_summary,
            "op_summary": per_op_list,
            "quality_metric": _quality_metric_reminder(self._eval_metric),
        }
        sem_op_docs = _sem_op_quality_docs(o.get("op_type") for o in per_op_list)
        if sem_op_docs:
            result["per_sem_op_quality_metric"] = sem_op_docs
        return result
