"""Cost-model data types, the contract the agent's CostModel must satisfy, duck-typed
plan/operator introspection helpers, and the observed-execution results store.

These are shared by `cost_model_agent.py`, `cost_helper_agent.py`, and the tool
modules under `tools/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Guarded palimpzest import.
#
# We import the real types when palimpzest is installed (for nicer reprs and so
# the cost model the agent writes can `isinstance`-check if it wants), but the
# harness never *requires* them: every code path degrades to duck typing. See
# `get_op_type` / `get_op_id` / `iter_operators` below.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    from palimpzest.query.optimizer.plan import PhysicalPlan  # type: ignore
    from palimpzest.query.operators.physical import PhysicalOperator  # type: ignore
    from palimpzest.core.models import OperatorCostEstimates, PlanCost  # type: ignore

    HAVE_PALIMPZEST = True
except Exception:  # ImportError, or a partial install
    PhysicalPlan = PhysicalOperator = OperatorCostEstimates = PlanCost = None  # type: ignore
    HAVE_PALIMPZEST = False


# ===========================================================================
# Cost-model data types + the contract the agent's CostModel must satisfy
# ===========================================================================
@dataclass
class PlanCostEstimate:
    """The estimate a cost model produces for a whole (sub)plan.

    `cost` is dollars, `time` is wall-clock seconds (latency), `quality` is an
    optional [0, 1] score. `details` is free-form (e.g. per-operator breakdown)
    so the agent can show its work."""

    cost: float
    time: float
    quality: float | None = None
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        q = "None" if self.quality is None else f"{self.quality:.3f}"
        return f"PlanCostEstimate(cost=${self.cost:.4f}, time={self.time:.3f}s, quality={q})"


class CostModel:
    """Optional base class for the cost model the *agent* writes.

    The only hard requirement enforced by `update_cost_model` is a callable
    `estimate_plan(self, plan) -> PlanCostEstimate`. Subclassing this is not
    required (a plain class with that method works), but it documents the
    contract and gives a default __repr__.
    """

    def estimate_plan(self, plan: Any) -> PlanCostEstimate:  # pragma: no cover
        raise NotImplementedError(
            "Write estimate_plan(self, plan) -> PlanCostEstimate in your subclass."
        )


# ---------------------------------------------------------------------------
# Duck-typed plan/operator helpers — work for real palimpzest objects AND for
# the stand-ins in demo.py. These are injected into the sandbox so the agent's
# CostModel can introspect operators uniformly.
# ---------------------------------------------------------------------------
def iter_operators(plan: Any) -> list[Any]:
    """All operators in a plan. Real `PhysicalPipeline` is iterable (topological order over
    the operator tree). Falls back to a `.operators` / `.ops` attribute, else treats `plan`
    as already-a-list."""
    try:
        return list(plan)
    except TypeError:
        # for attr in ("operators", "ops"):
        #     if hasattr(plan, attr):
        #         return list(getattr(plan, attr))
        raise TypeError(f"don't know how to iterate operators of {plan!r}")


def get_op_type(op: Any) -> str:
    """Operator type label, robust across palimpzest ops and demo ops."""
    return getattr(op, "op_type", None) or type(op).__name__


def get_op_id(op: Any) -> str:
    """Return PhysicalPipeline's `params_id`"""
    try:
        return op.params_id
    except Exception as e:
        raise Exception(f"failed to get op_id from {op!r}") from e

def get_op_model(op: Any) -> str | None:
    """The LLM model an operator uses, if any (None for non-LLM ops)."""
    m = getattr(op, "model", None)
    if m is None:
        return None
    # palimpzest models are often enums with a `.value`; fall back to str.
    return getattr(m, "value", None) or str(m)


def describe_operator(op: Any) -> dict:
    """A compact, JSON-friendly view of one operator: id, type, model, and a
    handful of likely-relevant numeric/string attributes (best effort)."""
    return {
        "op_id": get_op_id(op),
        "op_type": get_op_type(op),
        "attrs": op.attributes if hasattr(op, "attributes") else {},
    }


def _observed_op_stats(op_results: Any, op: Any) -> dict:
    """Empirical per-record stats for `op`, drawn from observed `op_results` rows.

    Uses the most specific match available and reports which one it used:
        exact op_id  ->  same op_type  ->  all ops (global fallback)
    `op_id` is a stable hash of the operator's type + params (model, cols,
    depends_on, ...), so an exact match means "this exact operator ran before"
    and its numbers can be used directly; a type/global match is a coarse prior.

    Returns a dict with per-input-record `cost_per_rec`, `latency_per_rec`,
    `in_tok_per_rec`, `out_tok_per_rec`, and `selectivity` (num_passed/num_records,
    fraction of rows kept), plus `found` (bool), `match_level`, and `n_ops`
    (how many observed operator rows were averaged). Multiply the per-record
    values by the operator's estimated input cardinality to get its cost/latency.
    When there is no data at all, `found` is False and the numeric fields None."""
    rows = list(getattr(op_results, "rows", None) or [])
    miss = {"found": False, "match_level": None, "n_ops": 0,
            "cost_per_rec": None, "latency_per_rec": None,
            "in_tok_per_rec": None, "out_tok_per_rec": None, "selectivity": None}
    if not rows:
        return miss

    try:
        op_id = get_op_id(op)
    except Exception:
        op_id = None
    try:
        op_type = get_op_type(op)
    except Exception:
        op_type = None

    def _f(r: dict, k: str) -> float:
        try:
            return float(r.get(k, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _agg(matched: list[dict]) -> dict | None:
        recs = sum(_f(r, "num_records") for r in matched)
        if recs <= 0:
            return None
        passed = sum(_f(r, "num_passed") for r in matched)
        return {
            "found": True,
            "n_ops": len(matched),
            "cost_per_rec": sum(_f(r, "cost_usd") for r in matched) / recs,
            "latency_per_rec": sum(_f(r, "latency_s") for r in matched) / recs,
            "in_tok_per_rec": sum(_f(r, "input_tokens") for r in matched) / recs,
            "out_tok_per_rec": sum(_f(r, "output_tokens") for r in matched) / recs,
            "selectivity": passed / recs,
        }

    for level, matched in (
        ("op_id", [r for r in rows if op_id is not None and r.get("op_id") == op_id]),
        ("op_type", [r for r in rows if op_type is not None and r.get("op_type") == op_type]),
        ("global", rows),
    ):
        stats = _agg(matched)
        if stats is not None:
            stats["match_level"] = level
            return stats
    return miss


def make_observed_op_stats(op_results: Any):
    """Bind `_observed_op_stats` to a live results store so cost models can call
    `observed_op_stats(op)` in the sandbox (reads the store fresh on each call)."""
    def observed_op_stats(op: Any) -> dict:
        return _observed_op_stats(op_results, op)
    return observed_op_stats


def _dump_opt_debug_artifacts(
    results_prefix: Any,
    query_id: int,
    plan_name: str,
    raw_output_df: Any,
    normalized_output_df: Any,
    plan_context: Any,
    quality_result: Any,
    runcount: Any = None,
    oracle_df: Any = None,
    evaluator: Any = None,
) -> None:
    """Write execute_plan's intermediate state under the run's opt_results directory, for
    debugging quality/per_sem_op_quality mysteries (e.g. comparing against a standalone re-run
    via experiments/cuad/run_single_plan.py).

        opt_results/Q1_2/                  <- one directory per (query, runcount)
            oracle_ground_truth.csv        <- the run's oracle ground truth, ONE copy shared by
                                              every plan (written and read by
                                              PlanQualityEvaluator, in whatever shape the
                                              benchmark's scorer reads back). Oracle mode only:
                                              in direct mode the benchmark's own ground-truth
                                              file is scored against in place, never copied.
            p1/
                raw_output.csv             the plan's own output, before normalization
                normalized_output.json     exactly what the benchmark's scorer was handed
                                           (written by evaluator.write_scoring_input)
                oracle_result.csv          THIS plan's own oracle-substituted output
                quality_result.json        the resulting scores
                per_sem_op_info.json       per-semantic-operator input/output samples
            p2/ ...

    The run directory carries `runcount` so a second run of the same query keeps its own copies
    instead of overwriting the first run's (matching how trajectory/ and metrics/ filenames are
    keyed). Within one run a plan's directory is overwritten on re-execution, so it reflects
    that plan's most recent run -- scratch debug output, not accumulated history.

    `oracle_result.csv` is per-plan and is NOT the ground truth: in oracle mode the first plan's
    oracle output becomes the canonical ground truth for the whole run, so every later plan's
    own oracle result diverges from it. Comparing the two is often what explains a surprising
    quality score.
    """
    import json as _json
    import pathlib as _pathlib

    run_key = f"Q{query_id}_{runcount}" if runcount is not None else f"Q{query_id}"
    out_dir = _pathlib.Path(results_prefix) / "opt_results" / run_key / str(plan_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_output_df.to_csv(out_dir / "raw_output.csv", index=False)
    # The scorer's actual input, not a CSV rendering of it: for CUAD the two differ (docetl is
    # handed narrow {filename, clauses} records that no frame in the engine holds), and the
    # whole point of this artifact is to read a surprising score back to what produced it.
    if evaluator is not None:
        evaluator.write_scoring_input(normalized_output_df, out_dir / "normalized_output.json")
    if oracle_df is not None:
        oracle_df.to_csv(out_dir / "oracle_result.csv", index=False)

    quality_summary = None
    if quality_result is not None:
        quality_summary = {
            "quality": quality_result.quality,
            "per_sem_op_quality": quality_result.per_sem_op_quality,
            "quality_note": quality_result.quality_note,
        }
    (out_dir / "quality_result.json").write_text(_json.dumps(quality_summary, indent=2, default=str))

    per_sem_op_info: dict[str, Any] = {}
    for stage_idx, info in (plan_context.per_sem_op_info if plan_context is not None else {}).items():
        raw_samples = info.get("samples", [])
        per_sem_op_info[str(stage_idx)] = {
            "op_name": info.get("op_name"),
            "op_type": info.get("op_type"),
            "attributes": info.get("attributes"),
            "num_samples": len(raw_samples),
            "samples_preview": [
                {"input": str(inp), "output": str(out) if out is not None else None}
                for inp, out in raw_samples[:3]
            ],
        }
    (out_dir / "per_sem_op_info.json").write_text(_json.dumps(per_sem_op_info, indent=2, default=str))


def _normalize_plan_df(
    df: "pd.DataFrame",
    use_case: str,
    query_id: int,
) -> "pd.DataFrame":
    """Identity fallback when a benchmark supplies no `normalize_eval_df`.

    Reshaping a plan's output into the shape a benchmark's evaluator expects is inherently
    benchmark-specific, so the real implementation lives in each adapter
    (experiments/*/quality_evaluator.py) and reaches the engine via
    query_info["normalize_eval_df"]. A benchmark whose evaluator already accepts the plan's
    natural output needs no override.
    """
    return df


# ===========================================================================
# Observed-execution results store
# ===========================================================================
@dataclass
class ResultsStore:
    """Append-only log of observed per-operator-invocation stats.

    One row per (operator, single input) execution. The canonical columns:

        op_id          str    -- which operator instance produced this row
        op_type        str    -- e.g. "SemFilterOp", "SemMapOp"
        model          str   -- LLM model used (None for non-LLM ops)
        input_id       str    -- id of / pointer to the input record
        output         any    -- the operator's decision/output for this input
                                  (e.g. True/False for a filter)
        cost_usd       float  -- dollar cost of this invocation
        latency_s      float  -- wall-clock latency of this invocation
        input_tokens   int
        output_tokens  int

    Extra columns are fine; the cost model can use whatever it finds. Access the
    data as `results.rows` (list[dict]) or `results.df` (pandas, if installed).
    """

    rows: list[dict] = field(default_factory=list)

    def append(self, new_rows: list[dict]) -> int:
        self.rows.extend(new_rows)
        return len(new_rows)

    _LARGE_COLS = frozenset({"op_samples", "plan_str"})

    @property
    def df(self):
        """pandas view (requires pandas). Use `.rows` if you don't have pandas.
        Large blob columns (op_samples, plan_str) are excluded — use get_op_samples() instead."""
        import pandas as pd

        rows = [{k: v for k, v in r.items() if k not in self._LARGE_COLS} for r in self.rows]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Counts + mean cost/latency per op_type — a quick orientation aid."""
        by_type: dict[str, dict[str, float]] = {}
        for r in self.rows:
            t = r.get("op_type", "?")
            agg = by_type.setdefault(t, {"n": 0, "cost_usd": 0.0, "latency_s": 0.0})
            agg["n"] += 1
            agg["cost_usd"] += float(r.get("cost_usd", 0.0) or 0.0)
            agg["latency_s"] += float(r.get("latency_s", 0.0) or 0.0)
        for t, agg in by_type.items():
            n = max(agg["n"], 1)
            agg["mean_cost_usd"] = round(agg["cost_usd"] / n, 6)
            agg["mean_latency_s"] = round(agg["latency_s"] / n, 4)
        return {"total_rows": len(self.rows), "by_op_type": by_type}


# ===========================================================================
# Cost-model registry (what `estimate_plan_cost` applies; what
# `update_cost_model` swaps in). Versioned so the agent can see its history.
# ===========================================================================
@dataclass
class _Version:
    version: int
    model: Any
    notes: str


class CostModelRegistry:
    def __init__(self) -> None:
        self._history: list[_Version] = []

    @property
    def version(self) -> int:
        return len(self._history)

    def current(self) -> Any:
        if not self._history:
            raise RuntimeError(
                "No cost model installed yet. Author a class with "
                "estimate_plan(self, plan) -> PlanCostEstimate and call "
                "update_cost_model(YourClass)."
            )
        return self._history[-1].model

    def install(self, model: Any, notes: str = "") -> int:
        self._history.append(_Version(self.version + 1, model, notes))
        return self.version

    def describe(self) -> str:
        if not self._history:
            return "(no cost model installed)"
        lines = []
        for v in self._history:
            lines.append(f"  v{v.version}: {type(v.model).__name__} — {v.notes or '(no notes)'}")
        return "\n".join(lines)
