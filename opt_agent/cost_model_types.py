"""Duck-typed plan/operator introspection helpers, per-run artifact naming, and the
observed-execution results store.

These are shared by `cost_model_agent.py`, `exploration_checker_agent.py`, and the tool
modules under `tools/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Duck-typed plan/operator helpers.
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
    """Operator type label: `op.op_type`, else the class name."""
    return getattr(op, "op_type", None) or type(op).__name__


def run_key(query_id: Any, runcount: Any = None) -> str:
    """`Q{id}_{runcount}`, or `Q{id}` when there is no runcount -- the stem every per-run
    artifact is named by."""
    return f"Q{query_id}_{runcount}" if runcount is not None else f"Q{query_id}"


def _dump_opt_debug_artifacts(
    results_prefix: Any,
    query_id: int,
    plan_name: str,
    raw_output_df: Any,
    normalized_output_df: Any,
    plan_context: Any,
    quality_result: Any,
    plan_metrics: dict | None = None,
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
                plan_result.json           the plan's quality (overall and per semantic
                                           operator), cost, and latency (both the wall-clock
                                           time of the subset run and the sum of per-op
                                           per-record latencies)
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

    out_dir = _pathlib.Path(results_prefix) / "opt_results" / run_key(query_id, runcount) / str(plan_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_output_df.to_csv(out_dir / "raw_output.csv", index=False)
    # The scorer's actual input, not a CSV rendering of it: for CUAD the two differ (docetl is
    # handed narrow {filename, clauses} records that no frame in the engine holds), and the
    # whole point of this artifact is to read a surprising score back to what produced it.
    if evaluator is not None:
        evaluator.write_scoring_input(normalized_output_df, out_dir / "normalized_output.json")
    if oracle_df is not None:
        oracle_df.to_csv(out_dir / "oracle_result.csv", index=False)

    # One record per plan: what the plan scored AND what it cost/took, so a plan directory can be
    # read on its own without cross-referencing the run's metrics CSVs.
    plan_summary: dict[str, Any] = {"plan_name": str(plan_name)}
    plan_summary.update(plan_metrics or {})
    plan_summary.update({
        "quality": quality_result.quality if quality_result is not None else None,
        "per_sem_op_quality": (
            quality_result.per_sem_op_quality if quality_result is not None else None
        ),
        "quality_note": quality_result.quality_note if quality_result is not None else None,
        "quality_counts": quality_result.quality_counts if quality_result is not None else None,
    })
    (out_dir / "plan_result.json").write_text(_json.dumps(plan_summary, indent=2, default=str))

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

    Extra columns are fine. Access the data as `results.rows` (list[dict]) or
    `results.df` (pandas, if installed).
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
