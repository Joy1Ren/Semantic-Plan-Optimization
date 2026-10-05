"""The exploration checker: a single-call LLM reviewer that judges whether the main
`CostModelAgent` has explored the optimization space far enough.

The main agent self-reports, per plan, an `optimizations` record naming which optimizations
that plan uses and how far each is pushed (see `WritePlanTool`). This reviewer reads those
records next to the plans' MEASURED cost/quality/latency and answers one question: is any
optimization the agent itself opened still underexplored, and what should it try next?

The reviewer IS shown the space that exists -- the operator/knob catalog and the model catalog
the main agent already has in its own briefing. It is shown no catalog of OPTIMIZATIONS (no
"try truncation", no "try reordering"): those it must still infer from what the agent reported,
so its verdict is not an echo of a lever list. The distinction matters, and the failure mode it
fixes is real: a reviewer that knows only what the agent reported cannot tell an untried model
family from a nonexistent one, so it re-suggests the axis already in play (observed on CUAD Q1
run 2, where the agent sampled only the 4o family and the reviewer kept proposing moves within
it) and it cannot see a knob that every plan pinned to the same value. Since the main agent
already holds both catalogs, showing them here adds no information to the system -- it only
stops the reviewer from being the one actor that cannot distinguish "untried" from "impossible".
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any

from agent_cost_model.opt_agent.cost_model_types import ResultsStore, get_op_type, iter_operators
from agent_cost_model.opt_agent import run_log
from agent_cost_model.opt_agent.errors import ParseError
from agent_cost_model.opt_agent.llm_client import LLMClient
from agent_cost_model.opt_agent.prompts import (
    _AVAILABLE_MODELS_TEXT,
    _OPERATOR_CATALOG_BRIEF,
    _per_op_quality_text,
    _quality_metric_section,
    _sem_op_quality_docs,
)
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

    _COST_BUDGET = textwrap.dedent("""\

        THE COST OF CONTINUING. Searching is not free: every extra plan the agent writes and executes
        spends real dollars, and asking for one is asking to spend them. You are shown what the search
        has spent so far, what each executed plan cost the search to write and execute, the average
        of those per-plan costs, and the size of the optimization subset next to the full dataset the
        chosen plan will finally be run on.

        A plan's search cost is far larger than its own cost_usd: writing and executing a plan also
        pays for the other optimization machinery that runs alongside it. The average per-plan search
        cost is the price of one more plan — never price it at a plan's cost_usd.

        Weigh only the NEXT plan. Money already spent is sunk — it is a reason neither to continue nor
        to stop. Compare the price of one more plan against the saving your suggestion would
        plausibly produce ON THE FULL DATASET.

        Do not scale that saving by (full dataset size / subset size). The subset was chosen to be
        RELEVANT to this query, so its rows reach the paid operators far more often than an average
        full-dataset row does, and on the full dataset a deterministic filter discards many rows for
        free. The size ratio is an upper bound on the real scale factor, usually a loose one; the agent
        can measure the true one against the full dataset, you cannot. Treat a saving that only looks
        worthwhile at the upper bound as not established.

        This weighing applies to suggestions that chase COST. A suggestion with a real chance of
        raising QUALITY is worth making even when it costs more than it saves. And early in a search,
        measuring the cheap and expensive extremes is worth its price even if neither plan wins — that
        is what establishes the range every later judgment is made against.
        """)

    _SYSTEM_TEMPLATE = textwrap.dedent("""\
        You review how thoroughly a query-optimization search has explored its options.

        THE SETTING. A *semantic plan* answers a natural-language query over a table of records.
        It is a chain of operators — filters, maps, joins, aggregations, projections — applied in
        order. Some operators are executed by an LLM, so they cost real dollars and real seconds;
        the rest are ordinary code and are free. Plan quality is scored 0-1, higher is better.

        Another agent is searching for the plan with the best cost/quality trade-off for one
        query. It writes many plans, each embodying different optimizations, executes them, and
        observes what each one cost and how well it scored. For every plan it also records an
        `optimizations` field of its own writing: free-form, in its own vocabulary, naming each
        optimization the plan uses and HOW FAR that optimization is pushed. There is no fixed
        catalog of optimizations — read the ones it reported and reason about those.

        YOUR JOB. Looking at all executed plans together, decide whether the search should keep
        going, and if it should, name the 1-2 optimizations the agent should try next. You are
        judging coverage of the search, not the correctness of any one plan.


        "MEANINGFULLY" IS RELATIVE TO THIS RUN. There is no fixed threshold and you must not
        invent one — the right size of an improvement depends entirely on this query and this
        subsample. Use the scale the run itself gives you: you are shown the quality spread over
        the executed plans and the gaps between neighbouring frontier points. A move whose
        plausible effect is smaller than the differences already visible between plans is inside
        the noise of a small subsample, and would teach the agent nothing. Do not continue pushing
        for improved quality or reduced cost if the expected improvement is low (not meaningful
        compared to noise from the subsample performance).

        THE BAR RISES AS THE SEARCH MATURES. Early on — few plans, and the cheap and expensive
        extremes not yet measured — breadth is genuinely valuable and a coarse gap is worth
        naming. Once the frontier has several points and recent plans have stopped advancing it,
        the search is near done, and only a clearly promising move justifies another plan.
        "Exhausted" is the EXPECTED verdict for a mature search, not a failure to find something.
        The burden of proof is on continuing, not on stopping.
        {cost_budget}
        THE SPACE THAT EXISTS. Below are the operators and models the agent is allowed to use.
        This is NOT a list of optimizations to recommend, and the agent has it already — it is
        here only so you can tell which choices are POSSIBLE, so that you never propose something
        that does not exist, and never dismiss something that was in fact available. An option no
        plan has touched is a CANDIDATE worth weighing, not automatically a gap: it still has to
        clear THE BAR above.

        {operator_catalog}
        {model_catalog}

        HOW TO READ THE EVIDENCE:
        - HAS THE FRONTIER MOVED LATELY? This is the strongest signal about whether the search is
          done. Several plans in a row that landed inside the frontier means the agent has
          stopped learning; unless there exists an optimization choice that has potential to improve
          the frontier, stop.
        - A CONSTANT MAY BE A DIMENSION. Settings identical in every plan — one model family, one
          retrieval method, one context budget, one logical shape — have no evidence behind them,
          however many plans there are, and will never show up in the agent's `optimizations`
          records, which describe only what it did do. That makes them the best CANDIDATES, but
          they still have to clear the bar: say what you expect moving one to do, and to whose
          number.
        - AN UNSCORED PLAN EXPLORED NOTHING. A plan whose quality is n/a produced no measurement,
          whatever it cost. Never treat it as evidence for or against a direction, and do not ask
          for a more elaborate version of it. If that direction still looks worth testing, ask
          for the SIMPLEST plan that would actually score.
        <per_op_quality>
        - PER-OPERATOR QUALITY IS NOT PLAN QUALITY. It measures one operator against the input it
          actually received, so it can be high while the plan scores badly (and the reverse).
          Plan `quality` is the objective. Never recommend a direction on per-operator scores
          alone when plan quality points the other way.
        </per_op_quality>

        HOW TO JUDGE (guidance, not rules — weigh the whole picture):
        - Make sure to also execute BASELINE plans to understand the range of possible quality/cost.
          For example, run a plan with no cost optimizations to see what quality is achievable.
          Similarly, run a plan with more significant cost optimizations to see what quality a cheap
          plan can get.
        - Follow the gradient. If pushing an optimization further kept paying off — quality
          rising, or cost falling with quality intact — it is underexplored. Name the concrete
          next setting along that same dimension.
        - Respect diminishing returns. If an optimization has been pushed two steps with no
          further gain, or pushing it started costing quality, it is done. Do not ask for more.
        - One setting is not exploration. An optimization tried at exactly a single extent is
          untested: one more point, in the direction the evidence favors, settles it.
        - Make sure to try different optimization strategies instead of chasing one lever. For example,
          consider different logical structures, different model choices, and different context reduction strategies.
        - Check both directions. If every plan has been chasing cost and none has genuinely
          tried to raise quality (or the reverse), say so — the trade-off has only been mapped
          from one side.
        - Do not ask for micro-tuning. Plans are measured on a small subsample, so nudging a
          numeric knob by a little overfits and teaches nothing. Suggest moves that are big
          enough to change the answer.
        - Prefer the cheapest plan that would answer the open question. A dimension can usually
          be tested on a cheap model; do not ask for an expensive plan when a cheap one settles
          the same thing.
        - Be concrete and be brief: at most 1-2 suggestions, most valuable first. Point at the
          specific optimization and the specific next extent to take it to. Each suggestion must
          describe ONE plan the agent could write next, in a couple of sentences — not a program
          of work, and never a long enumeration of per-field or per-record specifics.

        Concluding the search is a real and useful verdict. If the optimizations the agent has
        reported have each been pushed to the point of diminishing returns, say so rather than
        inventing work. Do not suggest plans that have a low likelihood of improving the current
        cost-quality frontier.

        RESPOND with EXACTLY ONE fenced ```json``` block and nothing else:
          {{"underexplored": true,  "suggestion": "<what to try next, and why the evidence supports it>"}}
          {{"underexplored": false, "suggestion": ""}}
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
        cost_aware: bool = False,
        per_op_quality: bool = True,
    ) -> None:
        self.llm = llm
        self.plans = plans
        self.plan_results = plan_results
        self.task = task
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.context_budget_chars = context_budget_chars
        # Formatted per instance, in ONE pass, because the cost-budget section is per-run while
        # the two catalogs are not. A second .format() pass would eat the `{{...}}` escapes that
        # hold the JSON output contract.
        self.cost_aware = cost_aware
        self.per_op_quality = per_op_quality
        self._system = _per_op_quality_text(self._SYSTEM_TEMPLATE.format(
            operator_catalog=_OPERATOR_CATALOG_BRIEF.rstrip(),
            model_catalog=_AVAILABLE_MODELS_TEXT.strip(),
            cost_budget=self._COST_BUDGET if cost_aware else "",
        ), per_op_quality)

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
    def check(self, *, reason: str, budget: dict | None = None) -> ExplorationCheck:
        """Ask the reviewer for one verdict. Never raises: any transport, parse, or shape
        problem is logged and returns "not underexplored", so a checker failure can never
        break the plan search that depends on it.

        `budget` is the caller's spend snapshot as of the step that triggered this check
        (see `_budget_section`); passed only in cost-aware runs."""
        n_executed = len(self._executed_rows())
        try:
            result = self._check(reason=reason, budget=budget)
        except Exception as e:
            self._log(f"[explore-check] failed ({reason}): {type(e).__name__}: {e}")
            run_log.log_error(
                "explore-check", f"{type(e).__name__}: {e}", content=f"reason={reason}", exc=e
            )
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

    def _check(self, *, reason: str, budget: dict | None = None) -> ExplorationCheck:
        if not self._executed_rows():
            # Nothing has been measured yet, so there is nothing to judge coverage of.
            self._log(f"[explore-check] skipped ({reason}): no executed plans yet")
            return ExplorationCheck(False, "")

        msgs: list[dict] = [{"role": "user", "content": self._build_context(reason, budget)}]
        # One reprompt: the contract is a single json block, and a model that missed it once
        # usually gets it on being told. Beyond that, treat the check as inconclusive.
        for attempt in (1, 2):
            raw, reasoning = self._llm_step(msgs)
            msgs.append({"role": "assistant", "content": raw})
            if reasoning:
                self._log(
                    run_log.header("[explore-check] reasoning", f"attempt {attempt}, reason={reason}")
                    + f"{reasoning}\n"
                )
            self._log(
                run_log.header("[explore-check] reviewer", f"attempt {attempt}, reason={reason}")
                + f"{raw}\n"
            )
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
                    self._emit(reason, attempt, raw, reasoning, result)
                    return result
            self._log(f"[explore-check] attempt {attempt}: parse problem — {detail}")
            self.trajectory_steps.append({
                "agent": "exploration_checker", "event": "parse_error", "check_reason": reason,
                "checker_step": attempt, "reasoning": reasoning, "assistant": raw,
                "observation": detail,
            })
            msgs.append({"role": "user", "content": (
                f"Could not parse your reply: {detail}. Re-send EXACTLY ONE ```json``` block "
                '{"underexplored": <true|false>, "suggestion": "<text>"}.'
            )})
        return ExplorationCheck(False, "")

    def _emit(
        self, reason: str, attempt: int, raw: str, reasoning: str | None, result: ExplorationCheck
    ) -> None:
        verdict = "UNDEREXPLORED" if result.underexplored else "sufficiently explored"
        self._log(f"[explore-check] {reason}: {verdict}")
        self.trajectory_steps.append({
            "agent": "exploration_checker", "event": "check", "check_reason": reason,
            "checker_step": attempt, "reasoning": reasoning, "assistant": raw,
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

    def _plan_block(self, row: dict, search_cost: float | None = None) -> str:
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
            *([f"quality counts: {json.dumps(row['quality_counts'], default=str)}"]
              if row.get("quality_counts") else []),
            *([f"per-operator quality: {per_op_s}"] if self.per_op_quality else []),
            f"cost_usd: {self._fmt_num(row.get('cost_usd'), '.6f')}",
            f"latency_s (sum of per-operator times): {self._fmt_num(row.get('latency_s'), '.3f')}",
            f"wall_latency_s (wall-clock time of the run): {self._fmt_num(row.get('wall_latency_s'), '.3f')}",
            *([f"search cost to write + execute this plan: ${search_cost:.4f} "
               "(includes its cost_usd plus the other optimization costs of writing and executing it)"]
              if search_cost is not None else []),
        ])

    def _per_op_quality_legend(self, rows: list[dict]) -> str:
        """What the `per-operator quality` numbers in each plan block actually mean, for the
        operator types that appear in these plans.

        The main agent gets this on every execute_plan result (see plan_tools); without it the
        reviewer is reading bare numbers whose definition it can only guess at -- and guessing
        wrong is consequential, because a rag_* per-op score deliberately excludes retrieval
        misses, so a high one is not evidence that the plan's retrieval is working."""
        if not self.per_op_quality:
            return ""
        op_types: list[str] = []
        for row in rows:
            entry = self.plans.get(row.get("plan_name"))
            if entry is None:
                continue
            try:
                op_types.extend(get_op_type(op) for op in iter_operators(entry["plan"]))
            except Exception:
                continue
        docs = _sem_op_quality_docs(op_types)
        if not docs:
            return ""
        lines = "\n".join(f"- {op_type}: {doc}" for op_type, doc in docs.items())
        return (
            "How each `per-operator quality` score below is computed (it is a per-operator "
            f"diagnostic, NOT the plan's objective):\n{lines}"
        )

    @staticmethod
    def _budget_section(budget: dict | None, plan_costs: list[float]) -> str:
        """What the search has spent as of the step that triggered this check.

        `plan_costs` are the search costs of the executed plans shown below; their average is the
        price of one more plan. The two sizes are given without their ratio on purpose: the subset
        is relevance-sampled, so that ratio is an upper bound on the real scale factor rather than
        the scale factor, and the system prompt says so."""
        if not budget:
            return ""
        return "\n".join([
            "=== Search spend ===",
            f"spent on the search so far: ${budget['total_usd']:.4f} "
            "(sunk — a reason neither to continue nor to stop)",
            *([f"average search cost of one plan (write + execute; per-plan figures are in each "
               f"plan block below): ${sum(plan_costs) / len(plan_costs):.4f} "
               "(what one more plan will cost)"] if plan_costs else []),
            f"optimization subset: {budget['subset_n']} records; "
            f"full dataset: {budget['dataset_n']} records",
        ])

    def _build_context(self, reason: str, budget: dict | None = None) -> str:
        rows = self._executed_rows()
        costs = (budget or {}).get("plan_search_cost") or {}
        block = lambda r: self._plan_block(r, costs.get(r.get("plan_name")))
        parts = [
            f"=== Exploration review (trigger: {reason}) ===",
            f"The query being answered:\n{self.task}",
            "\n\n".join(
                s for s in (_quality_metric_section(self.eval_metric),
                            self._per_op_quality_legend(rows),
                            self._budget_section(
                                budget, [costs[r["plan_name"]] for r in rows if r.get("plan_name") in costs]
                            )) if s
            ),
            f"Executed plans so far ({len(rows)}), in execution order:",
            "\n\n".join(block(r) for r in rows),
            "Judge whether any optimization these plans have opened is still underexplored. "
            "Reply with exactly one ```json``` block.",
        ]
        context = "\n\n".join(parts)
        if len(context) > self.context_budget_chars:
            # Keep the head (task + metric) and drop the OLDEST plan blocks: the recent plans
            # carry the gradient the reviewer is being asked to read.
            keep = self.context_budget_chars - len(parts[0]) - len(parts[1]) - len(parts[2]) - 500
            blocks = [block(r) for r in rows]
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
    def _llm_step(self, msgs: list[dict]) -> tuple[str, str | None]:
        result = self.llm.generate(self._system, msgs)
        content, reasoning, meta = result, None, {}
        if isinstance(result, tuple):
            content = result[0] if result else ""
            if len(result) >= 2:
                reasoning = result[1] or None
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
        return content, reasoning
