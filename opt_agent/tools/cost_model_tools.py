"""Tools for estimating, comparing, and installing cost models.

`EstimatePlanCostTool` / `ComparePlanCostsTool` / `UpdateCostModelTool` are used in
"sampleCost" mode, where the main agent authors its own cost model directly.
`ReviewPlansTool` is used in "customCost" mode, where cost-model authoring is
delegated to a `CostHelperAgent` and the main agent just asks it for a table.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from agent_cost_model.opt_agent.cost_model_types import CostModelRegistry, PlanCostEstimate, ResultsStore

from .base import Tool

if TYPE_CHECKING:
    from agent_cost_model.opt_agent.cost_helper_agent import CostHelperAgent


class EstimatePlanCostTool(Tool):
    name = "estimate_plan_cost"
    doc = """\
### estimate_plan_cost(plan)
Apply the CURRENTLY INSTALLED cost model to a physical (sub)plan and return its
`PlanCostEstimate` (dollar cost, latency seconds). Errors if
you have not installed a cost model yet (do that with `update_cost_model`).
Note: estimates may not accurately reflect absolute execution costs, but provide
a good relative signal for comparing candidate plans. Use this before executing
to prioritize which plans are worth running.

```python
estimate_plan_cost(plans["p1"]["plan"])
```"""

    def __init__(self, registry: CostModelRegistry):
        self._registry = registry

    def __call__(self, plan: Any) -> PlanCostEstimate:
        model = self._registry.current()
        est = model.estimate_plan(plan)
        if not isinstance(est, PlanCostEstimate):
            # Be lenient: accept a dict the agent returned and coerce it.
            if isinstance(est, dict):
                est = PlanCostEstimate(**est)
            else:
                raise TypeError(
                    "estimate_plan must return a PlanCostEstimate (or a dict with "
                    f"cost/time/quality), got {type(est).__name__}"
                )
        return est


class ComparePlanCostsTool(Tool):
    name = "compare_plan_costs"
    doc = """\
### compare_plan_costs(plan_names)
Estimate cost/latency for a list of NEW candidate plans using the CURRENTLY INSTALLED
cost model. ALL previously written plans (executed or not) are automatically included
for comparison. For executed plans, the actual observed cost/latency/quality are shown
alongside the estimate. Returns a table sorted by estimated cost.
Use this to rank candidate plans before deciding which to execute.
Errors if no cost model is installed yet.

```python
compare_plan_costs(["p3", "p4", "p5"])
# All previously written plans (e.g., p1, p2) are automatically included in the output.
```"""

    def __init__(self, registry: "CostModelRegistry", plans: dict, plan_results: "ResultsStore") -> None:
        self._registry = registry
        self._plans = plans
        self._plan_results = plan_results

    def __call__(self, plan_names: list) -> str:
        model = self._registry.current()

        actuals: dict[str, dict] = {}
        for row in self._plan_results.rows:
            name = row.get("plan_name")
            if name:
                actuals[name] = row

        # Combine: new candidates + all written plans (executed or not), deduplicated
        all_names = list(plan_names)
        for written_name in self._plans:
            if written_name not in all_names:
                all_names.append(written_name)

        rows = []
        errors = []
        for name in all_names:
            entry = self._plans.get(name)
            if entry is None:
                errors.append(f"  {name}: not found in plans dict (skipped)")
                continue
            plan = entry["plan"]
            try:
                est = model.estimate_plan(plan)
                if isinstance(est, dict):
                    est = PlanCostEstimate(**est)
            except Exception as e:
                errors.append(f"  {name}: estimation failed — {e}")
                continue
            actual = actuals.get(name, {})
            is_new = name in plan_names
            # Executed but NaN quality (`q != q`) => the plan produced no
            # evaluable output; its ~0 actual cost/latency are meaningless.
            q = actual.get("quality")
            failed = bool(actual) and (q is None or q != q)
            rows.append({
                "plan_name": name,
                "is_new": is_new,
                "est_cost": est.cost,
                "est_latency": est.time,
                "actual_cost": actual.get("cost_usd"),
                "actual_latency": actual.get("latency_s"),
                "actual_quality": actual.get("quality"),
                "failed": failed,
            })

        rows.sort(key=lambda r: (r["est_cost"] is None, r["est_cost"] or 0.0))

        def fmt(v, fmt_str=".6f"):
            return "—" if v is None else format(float(v), fmt_str)

        header = f"{'plan':<12} {'new?':<6} {'est_cost':>12} {'est_latency':>12} {'actual_cost':>12} {'actual_latency':>14} {'actual_quality':>14}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            new_marker = "yes" if r["is_new"] else "no"
            if r["failed"]:
                act_cost, act_lat, act_qual = "—", "—", "FAILED"
            else:
                act_cost = fmt(r["actual_cost"])
                act_lat = fmt(r["actual_latency"])
                act_qual = fmt(r["actual_quality"], ".3f")
            lines.append(
                f"{r['plan_name']:<12} {new_marker:<6} {fmt(r['est_cost']):>12} {fmt(r['est_latency']):>12} "
                f"{act_cost:>12} {act_lat:>14} {act_qual:>14}"
            )
        if errors:
            lines.append("\nErrors:")
            lines.extend(errors)
        lines.append("\n(Sorted by estimated cost. All written plans included automatically. Absolute values may be inaccurate; use for relative comparison. "
                     "actual_quality=FAILED means the plan ran but produced no evaluable output — its actual cost/latency are meaningless; rewrite it, don't select it.)")
        return "\n".join(lines)


class UpdateCostModelTool(Tool):
    name = "update_cost_model"
    doc = """\
### update_cost_model(cost_model, notes="")
Install a cost model. Pass EITHER a class you just defined (it will be
instantiated for you) OR an instance you constructed. The model must define
`estimate_plan(self, plan) -> PlanCostEstimate`. Returns the new version number.

If your __init__ takes `op_results` and/or `plan_results`, the observed-results
STORES are passed in automatically. Exact shapes (so you don't need defensive plumbing):
  - `op_results` : ResultsStore. `op_results.rows` -> list[dict], ONE per executed
    plan-operator, keys: op_name, op_type, op_id, latency_s, cost_usd, input_tokens,
    output_tokens, num_records (inputs to the op), num_passed (rows it kept).
    `op_results.df` -> the same as a pandas DataFrame. `op_results.summary()` -> per-op_type means.
  - `plan_results` : ResultsStore. `plan_results.df` / `.rows` -> ONE per executed plan,
    keys: plan_name, description, cost_usd, latency_s, input_tokens, output_tokens, quality.

Inside `estimate_plan(self, plan)`, iterate operators with `iter_operators(plan)` (topological
order) and inspect each with these accessors (no attribute-guessing needed):
  - `get_op_type(op)`  -> e.g. 'sem_filter', 'sem_map', 'rag_filter', 'rag_map', 'filter', 'map', 'project'
  - `get_op_model(op)` -> e.g. 'google/gemini-2.5-flash-lite' (None for non-LLM ops)
  - `get_op_id(op)`    -> stable hash of op_type + params (model, cols, depends_on)
  - `describe_operator(op)` -> {'op_id','op_type','attrs'} where attrs holds depends_on, cols, ...
  - `observed_op_stats(op)` -> PREFER THIS. Empirical per-input-record stats for this op,
    matched most-specific-first (exact op_id -> same op_type -> global). Returns:
    {found, match_level, n_ops, cost_per_rec, latency_per_rec, in_tok_per_rec,
     out_tok_per_rec, selectivity}. Multiply *_per_rec by the op's estimated input rows.

START SIMPLE: use `observed_op_stats(op)` for cost/latency/selectivity, with a small hardcoded
fallback only when `found` is False. Add size effects (image, truncation, depends_on column
count) later, and only if the calibration view shows a real mis-ranking.

```python
class MyCostModel:
    def estimate_plan(self, plan):
        cost = time = 0.0
        rows = 10.0  # subset size; refine from data if you like
        for op in iter_operators(plan):
            s = observed_op_stats(op)
            if s["found"]:
                cost += s["cost_per_rec"] * rows
                time += s["latency_per_rec"] * rows
                rows *= (s["selectivity"] if get_op_type(op) in ("filter", "sem_filter", "rag_filter") else 1.0)
            else:
                cost += 0.0     # tiny hardcoded prior for an unseen op
                time += 0.01
        return PlanCostEstimate(cost=cost, time=time, quality=None)

update_cost_model(MyCostModel, notes="v1: empirical per-op means via observed_op_stats + fallback")
```"""

    def __init__(self, registry: CostModelRegistry, *, op_results: ResultsStore, plan_results: ResultsStore):
        self._registry = registry
        self._op_results = op_results
        self._plan_results = plan_results

    def __call__(self, cost_model: Any, notes: str = "") -> dict:
        # A class → instantiate it. Inspect __init__ to decide whether to pass
        # the results store (so the agent can fit from observations).
        if isinstance(cost_model, type):
            model = self._instantiate(cost_model)
        else:
            model = cost_model
        if not callable(getattr(model, "estimate_plan", None)):
            raise TypeError(
                "cost model must define estimate_plan(self, plan) -> PlanCostEstimate"
            )
        version = self._registry.install(model, notes=notes)
        return {"installed_version": version, "type": type(model).__name__, "notes": notes}

    def _instantiate(self, cls: type) -> Any:
        try:
            params = inspect.signature(cls).parameters
            kwargs = {}
            if "op_results" in params: kwargs["op_results"] = self._op_results
            if "plan_results" in params: kwargs["plan_results"] = self._plan_results
            return cls(**kwargs)
        except (ValueError, TypeError):
            pass  # builtins / sandbox funcs may not introspect cleanly
        return cls()


class ReviewPlansTool(Tool):
    name = "review_plans"
    doc = """\
### review_plans()
Ask the cost helper for a cost/latency estimate of ALL your written plans (using the
latest cost model), shown next to the ACTUAL execution results for plans you've already
run. Returns a table sorted by estimated cost. Estimates are RELATIVE signals for ranking
candidate plans — absolute values may be inaccurate. You do NOT build or run the cost model
yourself; the helper maintains it and refreshes it automatically after each `execute_plan`
(so you may not need to call this every step). Available once at least one plan has executed.

```python
review_plans()
```"""

    def __init__(self, cost_helper: "CostHelperAgent") -> None:
        self._cost_helper = cost_helper

    def __call__(self) -> str:
        return self._cost_helper.review(reason="main-requested")
