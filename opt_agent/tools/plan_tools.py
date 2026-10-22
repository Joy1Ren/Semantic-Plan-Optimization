"""Tools for building, storing, executing, and inspecting physical query plans."""

from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from agent_cost_model.opt_agent import run_log
from agent_cost_model.opt_agent.cost_model_types import ResultsStore, _dump_opt_debug_artifacts
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
`op_name` is required and names the operator(s) you want — `{plan}_op{idx}_{op_type}`
(e.g. "p1_op1_rag_map"), the same names `plan_str` and `op_results` use. Pass a list to see
several at once. Ask only for the operators you actually want to look at: samples are long,
and a plan's operators often repeat each other's fields.
Returns at most `n` samples per operator.
<lower_variance>
Pass `execution=k` (1..{n_executions}) for a specific execution's samples; the default is the last one.
</lower_variance>

```python
get_op_samples("p1", "p1_op2_sem_map", n=5)                  # up to 5 samples from one operator
get_op_samples("p1", ["p1_op1_sem_filter", "p1_op2_sem_map"])  # two operators side by side
```"""

    def __init__(self, plan_results: ResultsStore) -> None:
        self._plan_results = plan_results

    def __call__(
        self, plan_name: str, op_name: str | list[str], n: int = 3, execution: int | None = None
    ) -> str:
        if not op_name:
            return "You must specify an operator name (e.g. 'p1_op2_sem_map') to retrieve samples."
        requested = [op_name] if isinstance(op_name, str) else list(op_name)
        # The LAST row for this plan (and execution, when one is named): re-executing a plan
        # appends another row, and the samples have to belong to the same execution as the
        # numbers the agent is reading. (The exploration checker resolves the same duplication
        # the same way.)
        row = next(
            (r for r in reversed(self._plan_results.rows) if r.get("plan_name") == plan_name
             and (execution is None or r.get("execution") == execution)),
            None,
        )
        if row is None:
            if execution is not None:
                return f"No execution {execution} of plan {plan_name!r}."
            available = [r.get("plan_name") for r in self._plan_results.rows]
            return f"No executed plan named {plan_name!r}. Available: {available}"
        samples = row.get("op_samples", {})
        if not samples:
            return f"No op_samples recorded for plan {plan_name!r}."
        missing = [name for name in requested if name not in samples]
        if missing:
            return f"No operator {missing} in plan {plan_name!r}. Available: {list(samples)}"
        samples = {name: samples[name] for name in requested}

        # De-duplication collapses values that a downstream operator merely carries through from
        # an upstream one, so it only means anything when more than one operator is shown; a
        # single operator prints in full (subject to the char budget). Keyed by (sample index,
        # field name) so that two different records of the same operator never collapse into
        # each other.
        dedup = len(requested) > 1
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
<per_op_info>
Execute the stored plan `plan_name` on a reproducible sample of records. Plan-level and
operator-level quality, latency, cost, and token usage are appended to `plan_results`
and `op_results`, respectively. After execution, `plans[plan_name]["plan"]` holds the PhysicalPipeline.

Returns a compact summary dict with plan stats and per-operator stats. Key fields:
</per_op_info>
<no_per_op_info>
Execute the stored plan `plan_name` on a reproducible sample of records. Plan-level quality,
latency, cost, and token usage are appended to `plan_results`. After execution,
`plans[plan_name]["plan"]` holds the PhysicalPipeline.

Returns a compact summary dict with plan stats. Key fields:
</no_per_op_info>
<lower_variance>
The plan is executed {n_executions} times in parallel on the same sample. The result holds
`executions` (one summary per execution, with the fields below) followed by `aggregate`, the
plan's new row in `plan_results_agg`.
</lower_variance>
- `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
  Treat oracle quality scores as ground truth.
<per_op_quality>
- `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
  which operator is the bottleneck. `per_sem_op_quality_metric` states how that
  score is computed for each operator type in this plan — read it before acting on
  a low score, since the definition differs by operator type.
</per_op_quality>
- `quality_counts` (only for metrics built from counts): the TP/FP/FN/TN behind `quality`,
  split by reason where a Jaccard gate applies. Use it to tell missed records from wrong ones.
- `cost_usd`, `latency_s`, `input_tokens`, `output_tokens`: aggregated over all ops.
- `op_errors` (only when something went wrong): per operator, the errors it hit, as
  "<count>x <message>". A record whose operator raised (e.g. a Python lambda that failed) was
  dropped from the run; a record whose LLM call failed went on with a failed filter verdict or
  None fields. Either way `quality` reflects the bug, not the plan's idea: fix it and re-run.
<per_op_info>
To inspect accumulated results use `plan_results.df` and `op_results.df`.
To view sample input/output pairs use `get_op_samples(plan_name, op_name)`, where `op_name` is one
operator name or a list of them.
</per_op_info>
<no_per_op_info>
To inspect accumulated results use `plan_results.df`.
</no_per_op_info>
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
        runcount: Any = None,
        oracle_operator_quality: bool = True,
        per_operator_info: bool = True,
        n_executions: int = 1,
        plan_results_agg: ResultsStore | None = None,
        rebuild_pipeline: Callable[[str], Any] | None = None,
        id_col: str = "idx",
    ) -> None:
        import pathlib

        self._oracle_operator_quality = oracle_operator_quality
        self._per_operator_info = per_operator_info
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
        # How many times one execute_plan call executes the plan (run_opt's --lower-variance).
        # Above 1, each extra execution runs on a fresh copy rebuilt from the plan's code, and
        # every call adds one aggregate row to plan_results_agg.
        self._n_executions = n_executions
        self._plan_results_agg = plan_results_agg
        self._rebuild_pipeline = rebuild_pipeline
        self._id_col = id_col

    def __call__(self, plan_name: str) -> dict:
        entry = self._plans.get(plan_name)
        if entry is None:
            raise KeyError(
                f"No plan named {plan_name!r}. Call write_plan first. "
                f"Available: {list(self._plans)}"
            )
        if self._subset_path is None:
            raise ValueError(
                "execute_plan needs a datasubset to run on, but no subset_path was supplied. "
                "It comes from the benchmark's `subset_path` (see benchmark.yaml), threaded "
                "through query_info by the runner."
            )

        # One pipeline per execution: a pipeline carries per-run operator state (join
        # accumulators, RAG retrieval contexts), so concurrent executions cannot share one. The
        # copies are built one at a time because the plan sandbox is not thread-safe.
        pipelines = [entry["plan"]]
        for _ in range(self._n_executions - 1):
            pipeline = self._rebuild_pipeline(plan_name)
            if pipeline is None:
                raise RuntimeError(f"Could not rebuild plan {plan_name!r} for a repeated execution.")
            pipelines.append(pipeline)
        if len(pipelines) == 1:
            runs = [self._run_once(plan_name, pipelines[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(pipelines)) as pool:
                runs = list(pool.map(lambda p: self._run_once(plan_name, p), pipelines))
        entry["plan"] = pipelines[0]

        # Scored one at a time: the evaluator keeps per-call state (last_quality_counts, and in
        # oracle mode a per-plan oracle cache).
        repeated = self._n_executions > 1
        executions = [k if repeated else None for k in range(1, len(pipelines) + 1)]
        scored = [
            self._score(plan_name, pipeline, run, execution)
            for pipeline, run, execution in zip(pipelines, runs, executions)
        ]
        error = next((run[3] for run in runs if run[3] is not None), None)
        if error is not None:
            raise RuntimeError(
                f"Plan {plan_name!r} execution failed: {type(error).__name__}: {error}"
            )

        plan_rows = [
            self._plan_row(plan_name, entry, pipeline, run, quality_result, execution)
            for pipeline, run, (quality_result, _), execution in zip(pipelines, runs, scored, executions)
        ]
        self._plan_results.append(plan_rows)
        if self._per_operator_info:
            for (per_op_list, _, _, _), execution in zip(runs, executions):
                self._op_results.append(
                    [{**o, "execution": execution} for o in per_op_list] if repeated else per_op_list
                )

        summaries = [
            {
                **({"execution": execution} if repeated else {}),
                "plan_summary": self._plan_summary(row, quality_result, repeated),
                **({"op_summary": run[0]} if self._per_operator_info else {}),
            }
            for row, run, (quality_result, _), execution in zip(plan_rows, runs, scored, executions)
        ]
        quality_metric = _quality_metric_reminder(self._eval_metric, self._oracle_operator_quality)
        if repeated:
            agg_row = self._aggregate(plan_name, entry, plan_rows, [out for _, out in scored])
            self._plan_results_agg.append([agg_row])
            return {"executions": summaries, "quality_metric": quality_metric, "aggregate": agg_row}

        # Remind the agent, on every execution, what `quality` means and how it differs from
        # per_sem_op_quality (context for why the two can diverge), plus how each per-op score in
        # THIS plan is computed -- the definition differs by operator type, and a rag_* score in
        # particular measures only the LLM step, not retrieval.
        result = {**summaries[0], "quality_metric": quality_metric}
        sem_op_docs = (
            _sem_op_quality_docs(o.get("op_type") for o in runs[0][0])
            if self._oracle_operator_quality else {}
        )
        if sem_op_docs:
            result["per_sem_op_quality_metric"] = sem_op_docs
        return result

    def _run_once(self, plan_name: str, pipeline: Any) -> tuple[list, Any, dict, Exception | None]:
        """Execute one pipeline on the subset: (per_op_list, plan_context, plan_dict, error)."""
        try:
            # The plan is (re)executed on the subset every time — never cached — so that
            # plan_dict["latency_s"] is a real WALL-CLOCK measurement of this run. (Only the
            # oracle, which is used purely for quality scoring, is cached across plans.)
            per_op_list, plan_context, plan_dict = pipeline.run_subset(
                subset_cache_path=str(self._subset_path)
            )
            return per_op_list, plan_context, plan_dict, None
        except Exception as e:
            print(f"[execute_plan] plan execution failed for {plan_name}: {type(e).__name__}: {e}")
            run_log.log_error(
                "plan-exec", f"{type(e).__name__}: {e}", content=f"plan={plan_name}", exc=e
            )
            return [], None, {}, e

    def _score(
        self, plan_name: str, pipeline: Any, run: tuple, execution: int | None
    ) -> tuple[Any, "pd.DataFrame"]:
        """Score one execution and write its debug artifacts: (quality_result, raw output)."""
        import pandas as pd

        per_op_list, plan_context, plan_dict, _ = run
        raw_output_df = pd.DataFrame(
            plan_context.output_records if plan_context is not None else []
        )
        # The evaluator owns the reshaping: it is the thing that knows what its scorer reads.
        plan_output_df = (
            self._quality_evaluator.normalize(raw_output_df)
            if self._quality_evaluator is not None else raw_output_df
        )

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
                run_log.log_error(
                    "quality-eval", f"{type(e).__name__}: {e}", content=f"plan={plan_name}", exc=e
                )

        try:
            _dump_opt_debug_artifacts(
                self._results_prefix, self._query_id, plan_name,
                raw_output_df, plan_output_df, plan_context, quality_result,
                plan_metrics={
                    "cost_usd": sum(e.get("cost_usd", 0) for e in per_op_list),
                    "latency_s": sum(e.get("latency_s", 0) for e in per_op_list),
                    "wall_latency_s": float(plan_dict.get("latency_s", 0.0) or 0.0),
                    "op_summary": per_op_list,
                },
                runcount=self._runcount,
                # Set by PlanQualityEvaluator.evaluate on the call just above; absent when
                # the plan has no evaluator or evaluation raised. The ground truth is not passed
                # here -- the evaluator owns it and writes one shared copy at the run level.
                oracle_df=getattr(self._quality_evaluator, "last_oracle_df", None),
                evaluator=self._quality_evaluator,
                execution=execution,
            )
        except Exception as e:
            print(f"[execute_plan] opt_results debug dump failed for {plan_name}: {type(e).__name__}: {e}")
            run_log.log_error(
                "debug-dump", f"{type(e).__name__}: {e}", content=f"plan={plan_name}", exc=e
            )
        return quality_result, raw_output_df

    def _plan_row(
        self, plan_name: str, entry: dict, pipeline: Any, run: tuple, quality_result: Any,
        execution: int | None,
    ) -> dict:
        per_op_list, plan_context, plan_dict, _ = run
        # Build op_samples for GetOpSamplesTool. Records are stored field by field (each value
        # capped at SAMPLE_FIELD_CHARS) so the display can budget and de-duplicate them; op_type
        # rides along so the display knows e.g. that a filter's output is just a verdict.
        op_samples: dict[str, dict] = {}
        for _stage_idx, info in plan_context.per_sem_op_info.items():
            raw_samples = info.get("samples", [])
            op_samples[info["op_name"]] = {
                "op_type": info.get("op_type", ""),
                "samples": [
                    {"input": _record_fields(inp), "output": _record_fields(out)}
                    for inp, out in raw_samples[:5]
                ],
            }
        # latency_s: SUM of per-op per-record latencies (serial-equivalent). Kept as the plan's
        # "actual latency" for the cost-model est-vs-act comparison, whose prediction (est.time)
        # is also a per-op sum. wall_latency_s: real WALL-CLOCK time of this subset execution
        # (from plan_dict), which reflects the join/convert parallelism.
        return {
            "plan_name": plan_name,
            **({"execution": execution} if execution is not None else {}),
            "description": entry.get("description", ""),
            "optimizations": entry.get("optimizations"),
            "plan_str": str(pipeline),
            "cost_usd": sum(e.get("cost_usd", 0) for e in per_op_list),
            "latency_s": sum(e.get("latency_s", 0) for e in per_op_list),
            "wall_latency_s": float(plan_dict.get("latency_s", 0.0) or 0.0),
            "input_tokens": sum(e.get("input_tokens", 0) for e in per_op_list),
            "output_tokens": sum(e.get("output_tokens", 0) for e in per_op_list),
            "quality": quality_result.quality if quality_result is not None else float("nan"),
            **({"per_sem_op_quality": (
                quality_result.per_sem_op_quality if quality_result is not None else {}
            )} if self._oracle_operator_quality else {}),
            "quality_counts": quality_result.quality_counts if quality_result is not None else None,
            **({"op_samples": op_samples} if self._per_operator_info else {}),
            # Shown whatever the agent's mode: errors explain a score, they are not extra info.
            **({"op_errors": plan_dict["op_errors"]} if plan_dict.get("op_errors") else {}),
        }

    @staticmethod
    def _plan_summary(plan_row: dict, quality_result: Any, repeated: bool) -> dict:
        # A repeated execution's description and optimizations are the aggregate's; repeating
        # them in every execution's summary would only lengthen the observation.
        dropped = ("op_samples", "plan_str", "execution") + (("description", "optimizations") if repeated else ())
        summary = {k: v for k, v in plan_row.items() if k not in dropped}
        if summary["quality_counts"] is None:
            del summary["quality_counts"]
        # When quality is N/A because the oracle returned no rows, explain it in the observation so
        # the agent doesn't read the bare NaN as a failed plan.
        if quality_result is not None and getattr(quality_result, "quality_note", None):
            summary["quality_note"] = quality_result.quality_note
        return summary

    def _aggregate(
        self, plan_name: str, entry: dict, plan_rows: list[dict], outputs: list["pd.DataFrame"]
    ) -> dict:
        import pandas as pd

        df = pd.DataFrame(plan_rows)
        return {
            "plan_name": plan_name,
            "description": entry.get("description", ""),
            "quality_avg": df["quality"].mean(),
            "quality_std": df["quality"].std(),
            "cost_avg": df["cost_usd"].mean(),
            "cost_std": df["cost_usd"].std(),
            "latency_wallclock_avg": df["wall_latency_s"].mean(),
            "latency_wallclock_std": df["wall_latency_s"].std(),
            "n_unstable_records": _n_unstable_records(outputs, self._id_col),
        }


def _n_unstable_records(outputs: list["pd.DataFrame"], id_col: str) -> int | None:
    """How many records' output differs between executions; a record that one execution did not
    return counts as differing. None when the output carries no id column (e.g. an aggregate)."""
    if any(not df.empty and id_col not in df.columns for df in outputs):
        return None

    def view(df) -> dict:
        rows: dict[str, list] = {}
        for r in df.astype(str).to_dict("records"):
            rows.setdefault(r.pop(id_col), []).append(tuple(sorted(r.items())))
        return {k: sorted(v) for k, v in rows.items()}

    views = [view(df) for df in outputs]
    return sum(len({repr(v.get(i)) for v in views}) > 1 for i in set().union(*views))
