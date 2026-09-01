"""The exploration checker: a single-call LLM reviewer that judges whether the main
`CostModelAgent` has explored the optimization space far enough.

The main agent self-reports, per plan, an `optimizations` record naming which optimizations
that plan uses and how far each is pushed (see `WritePlanTool`). This reviewer reads those
records next to the plans' MEASURED cost/quality/latency and answers one question: is any
optimization the agent itself opened still underexplored, and what should it try next?

Deliberately kept ignorant of our own catalog of optimizations: the system prompt below names
NO specific optimization (no models, no truncation, no reordering). It must infer the
dimensions from what the agent actually reported, so its verdict is not just an echo of the
levers we already listed in the main agent's briefing.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any

from agent_cost_model.opt_agent.cost_model_types import ResultsStore, get_op_type, iter_operators
from agent_cost_model.opt_agent.errors import ParseError
from agent_cost_model.opt_agent.llm_client import LLMClient
from agent_cost_model.opt_agent.prompts import _quality_metric_section
from agent_cost_model.opt_agent.step_parsing import _parse_step


@dataclass
class ExplorationCheck:
    """One verdict. `suggestion` is natural language for the main agent to act on;
    it is meaningful only when `underexplored` is True."""

    underexplored: bool
    suggestion: str = ""


class ExplorationCheckerAgent:
    """A one-call reviewer of how well the optimization space has been explored.

    STATELESS across calls: every invocation builds a fresh message list from the current
    snapshot of executed plans, and the checker keeps no memory of its own prior suggestions.
    A suggestion the main agent acted on stops being suggested on its own, because the plan
    that acted on it shows up as executed in the next snapshot.

    Only EXECUTED plans are shown. A plan that was written but never run has not explored
    anything — including it would let the agent claim coverage it never measured.
    """

    _SYSTEM = textwrap.dedent("""\
        You review how thoroughly a query-optimization search has explored its options.

        THE SETTING. A *semantic plan* answers a natural-language query over a table of records.
        It is a chain of operators — filters, maps, joins, aggregations, projections — applied in
        order. Some operators are executed by an LLM, so they cost real dollars and real seconds;
        the rest are ordinary code and are free. Plan quality is scored 0-1 against ground truth.

        Another agent is searching for the plan with the best cost/quality trade-off for one
        query. It writes many plans, each embodying different optimizations, executes them, and
        observes what each one cost and how well it scored. For every plan it also records an
        `optimizations` field of its own writing: free-form, in its own vocabulary, naming each
        optimization the plan uses and HOW FAR that optimization is pushed. There is no fixed
        catalog of optimizations — read the ones it reported and reason about those.

        YOUR JOB. Looking at all executed plans together, decide whether any optimization the
        agent has opened is still UNDEREXPLORED, and if so say — in natural language — what to
        try next. You are judging coverage of the search, not the correctness of any one plan.

        HOW TO JUDGE (guidance, not rules — weigh the whole picture):
        - Follow the gradient. If pushing an optimization further kept paying off — quality
          rising, or cost falling with quality intact — it is underexplored. Name the concrete
          next setting along that same dimension.
        - Respect diminishing returns. If an optimization has been pushed two steps with no
          further gain, or pushing it started costing quality, it is done. Do not ask for more.
        - One setting is not exploration. An optimization tried at exactly a single extent is
          untested: one more point, in the direction the evidence favors, settles it.
        - Check both directions. If every plan has been chasing cost and none has genuinely
          tried to raise quality (or the reverse), say so — the trade-off has only been mapped
          from one side.
        - Do not ask for micro-tuning. Plans are measured on a small subsample, so nudging a
          numeric knob by a little overfits and teaches nothing. Suggest moves that are big
          enough to change the answer.
        - Be concrete and be brief: at most 2-3 suggestions, most valuable first. Point at the
          specific optimization and the specific next extent to take it to.

        Concluding the search is a real and useful verdict. If the optimizations the agent has
        reported have each been pushed to the point of diminishing returns, say so rather than
        inventing work.

        RESPOND with EXACTLY ONE fenced ```json``` block and nothing else:
          {"underexplored": true,  "suggestion": "<what to try next, and why the evidence supports it>"}
          {"underexplored": false, "suggestion": ""}
        """)

    def __init__(
        self,
        llm: LLMClient,
        *,
        plans: dict,
        plan_results: ResultsStore,
        task: str,
        eval_metric: str | None = None,
        verbose: bool = True,
        context_budget_chars: int = 80_000,
    ) -> None:
        self.llm = llm
        self.plans = plans
        self.plan_results = plan_results
        self.task = task
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.context_budget_chars = context_budget_chars

        self.cost_usd: float = 0.0
        # Every verdict this run produced — saved to exploration_check/Q{id}_{rc}.json so a run
        # can be read back as "what did the checker see, and what did the agent do next".
        self.checks: list[dict] = []
        # Printed for easy following and interleaved into the MAIN trajectory CSV, but NEVER
        # added to the main agent's LLM context (only the suggestion text reaches it).
        self.trajectory_steps: list[dict] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- public entry point ------------------------------------------------
    def check(self, *, reason: str) -> ExplorationCheck:
        """Ask the reviewer for one verdict. Never raises: any transport, parse, or shape
        problem is logged and returns "not underexplored", so a checker failure can never
        break the plan search that depends on it."""
        n_executed = len(self._executed_rows())
        try:
            result = self._check(reason=reason)
        except Exception as e:
            self._log(f"[explore-check] failed ({reason}): {type(e).__name__}: {e}")
            self.trajectory_steps.append({
                "agent": "exploration_checker", "event": "check_error", "check_reason": reason,
                "observation": f"{type(e).__name__}: {e}",
            })
            result = ExplorationCheck(False, "")
        self.checks.append({
            "reason": reason,
            "n_executed": n_executed,
            "underexplored": result.underexplored,
            "suggestion": result.suggestion,
        })
        return result

    def _check(self, *, reason: str) -> ExplorationCheck:
        if not self._executed_rows():
            # Nothing has been measured yet, so there is nothing to judge coverage of.
            self._log(f"[explore-check] skipped ({reason}): no executed plans yet")
            return ExplorationCheck(False, "")

        msgs: list[dict] = [{"role": "user", "content": self._build_context(reason)}]
        # One reprompt: the contract is a single json block, and a model that missed it once
        # usually gets it on being told. Beyond that, treat the check as inconclusive.
        for attempt in (1, 2):
            raw = self._llm_step(msgs)
            msgs.append({"role": "assistant", "content": raw})
            self._log(f"\n--- [explore-check] reviewer (attempt {attempt}, reason={reason}) ---\n{raw}\n")
            try:
                parsed = _parse_step(raw)
            except ParseError as e:
                detail = e.detail
            else:
                if parsed.code is not None or not isinstance(parsed.result, dict):
                    detail = "expected a ```json``` block holding an object"
                else:
                    result = ExplorationCheck(
                        underexplored=bool(parsed.result.get("underexplored")),
                        suggestion=str(parsed.result.get("suggestion") or "").strip(),
                    )
                    # A True verdict with nothing to act on is unusable — treat it as "done"
                    # rather than sending the main agent back with no instruction.
                    if result.underexplored and not result.suggestion:
                        result = ExplorationCheck(False, "")
                    self._emit(reason, attempt, raw, result)
                    return result
            self._log(f"[explore-check] attempt {attempt}: parse problem — {detail}")
            self.trajectory_steps.append({
                "agent": "exploration_checker", "event": "parse_error", "check_reason": reason,
                "checker_step": attempt, "assistant": raw, "observation": detail,
            })
            msgs.append({"role": "user", "content": (
                f"Could not parse your reply: {detail}. Re-send EXACTLY ONE ```json``` block "
                '{"underexplored": <true|false>, "suggestion": "<text>"}.'
            )})
        return ExplorationCheck(False, "")

    def _emit(self, reason: str, attempt: int, raw: str, result: ExplorationCheck) -> None:
        verdict = "UNDEREXPLORED" if result.underexplored else "sufficiently explored"
        self._log(f"[explore-check] {reason}: {verdict}")
        self.trajectory_steps.append({
            "agent": "exploration_checker", "event": "check", "check_reason": reason,
            "checker_step": attempt, "assistant": raw,
            "underexplored": result.underexplored,
            "observation": f"{verdict}: {result.suggestion}" if result.suggestion else verdict,
        })

    # -- context -----------------------------------------------------------
    def _executed_rows(self) -> list[dict]:
        """One row per executed plan, most recent execution of a name winning. Written-but-never
        -executed plans are deliberately absent: they measured nothing."""
        by_name: dict[str, dict] = {}
        for row in self.plan_results.rows:
            name = row.get("plan_name")
            if name:
                by_name[name] = row
        return list(by_name.values())

    def _op_chain(self, plan_name: str) -> str:
        entry = self.plans.get(plan_name)
        if entry is None:
            return "(unavailable)"
        try:
            return ",".join(get_op_type(o) for o in iter_operators(entry["plan"]))
        except Exception:
            return "(unavailable)"

    @staticmethod
    def _fmt_optimizations(value: Any) -> str:
        if not value:
            return "(the agent recorded none for this plan)"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _fmt_num(value: Any, spec: str = ".4f") -> str:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if f != f else format(f, spec)  # f != f is True only for NaN

    def _plan_block(self, row: dict) -> str:
        name = row.get("plan_name", "?")
        per_op = row.get("per_sem_op_quality") or {}
        try:
            per_op_s = json.dumps(per_op, default=str)
        except Exception:
            per_op_s = str(per_op)
        return "\n".join([
            f"--- plan {name} ---",
            f"description: {row.get('description', '') or '(none)'}",
            f"operators: ({self._op_chain(name)})",
            f"optimizations (recorded by the plan-writing agent):\n{self._fmt_optimizations(row.get('optimizations'))}",
            f"quality: {self._fmt_num(row.get('quality'), '.3f')}",
            f"per-operator quality: {per_op_s}",
            f"cost_usd: {self._fmt_num(row.get('cost_usd'), '.6f')}",
            f"latency_s (sum of per-operator times): {self._fmt_num(row.get('latency_s'), '.3f')}",
            f"wall_latency_s (wall-clock time of the run): {self._fmt_num(row.get('wall_latency_s'), '.3f')}",
        ])

    def _build_context(self, reason: str) -> str:
        rows = self._executed_rows()
        parts = [
            f"=== Exploration review (trigger: {reason}) ===",
            f"The query being answered:\n{self.task}",
            _quality_metric_section(self.eval_metric),
            f"Executed plans so far ({len(rows)}), in execution order:",
            "\n\n".join(self._plan_block(r) for r in rows),
            "Judge whether any optimization these plans have opened is still underexplored. "
            "Reply with exactly one ```json``` block.",
        ]
        context = "\n\n".join(parts)
        if len(context) > self.context_budget_chars:
            # Keep the head (task + metric) and drop the OLDEST plan blocks: the recent plans
            # carry the gradient the reviewer is being asked to read.
            keep = self.context_budget_chars - len(parts[0]) - len(parts[1]) - len(parts[2]) - 500
            blocks = [self._plan_block(r) for r in rows]
            kept: list[str] = []
            for block in reversed(blocks):
                if keep - len(block) < 0:
                    break
                kept.append(block)
                keep -= len(block)
            parts[3] = (f"Executed plans ({len(rows)} total; the {len(kept)} most recent shown, "
                        "earlier ones omitted for length):")
            parts[4] = "\n\n".join(reversed(kept))
            context = "\n\n".join(parts)
        return context

    # -- llm ---------------------------------------------------------------
    def _llm_step(self, msgs: list[dict]) -> str:
        result = self.llm.generate(self._SYSTEM, msgs)
        content, meta = result, {}
        if isinstance(result, tuple):
            content = result[0] if result else ""
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
        return content
