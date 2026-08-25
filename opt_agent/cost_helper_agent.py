"""The cost-helper agent: a lightweight agent that owns cost-model authoring/refitting
so the main `CostModelAgent` (in `cost_model_agent.py`) can focus purely on plan search."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from agent_cost_model.opt_agent.cost_model_types import (
    CostModel,
    CostModelRegistry,
    PlanCostEstimate,
    ResultsStore,
    describe_operator,
    get_op_id,
    get_op_model,
    get_op_type,
    iter_operators,
    make_observed_op_stats,
)
from agent_cost_model.opt_agent.errors import ParseError
from agent_cost_model.opt_agent.llm_client import LLMClient
from agent_cost_model.opt_agent.prompts import _AVAILABLE_MODELS_TEXT
from agent_cost_model.opt_agent.step_parsing import _parse_step
from agent_cost_model.opt_agent.tools.cost_model_tools import UpdateCostModelTool


class CostHelperAgent:
    """A lightweight agent that OWNS the cost model.

    The main `CostModelAgent` focuses purely on plan search; it never authors or
    runs cost models. Instead it delegates to this helper, which:
      - authors / refits a cost model whose ONLY job is to rank candidate plans
        correctly *relative to each other* (absolute $ / s need not be accurate);
      - returns a consolidated table of estimated cost/latency (latest model) next
        to actual execution data for plans already run.

    It is invoked automatically after each execution batch and on demand via the
    main agent's `review_plans()` tool. Its memory across a run is deliberately
    COMPACT to avoid duplicating information: a one-line `version_log` per model
    decision, the code of each installed version (`model_versions`), and per-version
    estimate snapshots (`version_estimates`). It does NOT keep a growing transcript —
    each refit is built fresh with a single current snapshot of plan_results/op_results,
    so the full data tables are never re-sent across refits. The helper alone decides
    whether a refit is worthwhile — the main loop carries no cost-model decision burden.
    """

    _HELPER_SYSTEM = textwrap.dedent("""\
        You maintain a cost model for a physical-query-plan optimizer. Its ONLY purpose is
        to rank candidate plans correctly *relative to each other* — absolute dollar/second
        values do NOT need to be accurate. Perfect ranking is NOT expected: treat the Kendall
        rank-correlation (tau) you are shown as a SIGNAL, and refit only when the current
        model is *meaningfully* mis-ranking the executed plans (low/negative tau). A few
        mis-ordered pairs is fine, so prefer KEEPING the current model. Be efficient — define
        one class and install it in a single step.

        On each invocation you are given: the task, the plans written so far (with
        descriptions), the observed `plan_results` (real per-plan cost/latency/quality) and the
        FULL `op_results` table (one row per executed plan-operator), and — once a model exists —
        the current model's code plus a relative-calibration view (its estimates vs actuals on
        executed plans, with the Kendall rank-correlation tau; tau=1.0 means perfect ranking).

        Make your decision in ONE step. Respond with EXACTLY ONE fenced block:
          - a ```python``` block that defines a cost-model class and calls
            `update_cost_model(YourClass, notes="vN: ...")` to install it (build/refit); or
          - a ```json``` block `{"decision": "keep", "reason": "..."}` to keep the current
            model unchanged (not allowed until a model exists — you MUST install v1 first).
        If the model you install RAISES while estimating any plan, you will be shown the exact
        error and asked to fix it — return a corrected model. So always return a PlanCostEstimate
        with numeric cost/time, and fall back to a prior for operators with no observations.

        BUILD THE SIMPLEST MODEL THAT RANKS. A cost model is a class with
        `estimate_plan(self, plan) -> PlanCostEstimate`
        (`PlanCostEstimate(cost=<dollars>, time=<seconds>, quality=None, details={...})`).

        Use empirical observations as the foundation, and add impact on finer-grain effects when needed.
        1. starting point: observed means. For each op in `iter_operators(plan)`, call `observed_op_stats(op)`
           and use its per-record `cost_per_rec`/`latency_per_rec` x the op's input rows, and
           `selectivity` to propagate cardinality through filters. This grounds every number in
           real data — do NOT hard-code token counts or model prices when observations exist.
           `observed_op_stats` matches most-specific-first (exact op_id -> op_type -> global) and
           tells you which via `match_level`; when `found` is False, use a small hardcoded prior
           (and record it in `details["unseen_ops"]`).
        2. refining: consider the impact from second-order effects such as image inputs (much larger input_tokens), text
           truncation, and `depends_on` column count. Read `input_tokens/num_records` from the
           op_results rows as a per-record size proxy when you need to extrapolate to an unseen
           operator configuration.

        Data shapes (no defensive plumbing needed): `op_results.rows` is list[dict] with keys
        op_name, op_type, op_id, cost_usd, latency_s, input_tokens, output_tokens, num_records,
        num_passed; `plan_results.df` has plan_name, description, cost_usd, latency_s, quality.
        Inspect operators with `get_op_type(op)`, `get_op_model(op)`, `get_op_id(op)`,
        `describe_operator(op)` (attrs include depends_on/cols), and `observed_op_stats(op)`.
        """)

    def __init__(
        self,
        llm: LLMClient,
        *,
        registry: CostModelRegistry,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        task: str,
        use_case: str = "use_case",
        agent_dir: str = "no_name",
        verbose: bool = True,
        max_repair_retries: int = 3,
        context_budget_chars: int = 120_000,
        refit_tau_threshold: float = 0.9,
        authorized_imports: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.plans = plans
        self.plan_results = plan_results
        self.op_results = op_results
        self.task = task
        self.use_case = use_case
        self.agent_dir = agent_dir
        self.verbose = verbose
        # The helper decides (keep or a new model) in ONE step; if the newly installed model
        # raises while estimating plans, it is re-prompted with the error up to this many times.
        self.max_repair_retries = max_repair_retries
        self.context_budget_chars = context_budget_chars
        # Only wake the (LLM) refit when the current model's relative ranking of executed
        # plans degrades below this Kendall-tau — otherwise just refresh estimates (no LLM).
        self.refit_tau_threshold = refit_tau_threshold

        # full system prompt = base contract + the available-models catalog, so the
        # helper can reason about per-model cost/latency when authoring the cost model.
        self.system_prompt = (
            self._HELPER_SYSTEM
            + "\n## Available Models (same catalog the plan-writing agent uses; cheaper "
            "options listed first within a tier):\n"
            + _AVAILABLE_MODELS_TEXT
        )

        # persistent memory across a run — deliberately COMPACT (no raw-data transcript).
        # Each refit gets one fresh plan_results/op_results snapshot; what carries across
        # refits is only: a one-line log per version decision, the code of each version, and
        # per-version estimate snapshots.
        self.version_log: list[str] = []                  # ["v1: INSTALLED at bootstrap", ...]
        self.model_versions: dict[str, str] = {}          # {"v1": code, ...}
        self.version_estimates: dict[int, dict[str, tuple]] = {}  # version -> {plan: (cost, time)}
        self.cost_usd: float = 0.0
        self._last_reviewed_plan_rows: int = 0
        # Full record of the helper's own reasoning/decisions — printed for easy following and
        # interleaved into the MAIN trajectory CSV (rows labeled "{main_step}[cost_helper]-{k}"),
        # but NEVER added to the main agent's LLM context.
        self.trajectory_steps: list[dict] = []
        self._last_reasoning: str | None = None

        # own sandbox: only the update tool + introspection (never share the main
        # agent's executor — WritePlanTool mutates that sandbox's state).
        from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor
        default_imports = ["math", "statistics", "json", "collections", "itertools"]
        if __import__("importlib").util.find_spec("palimpzest") is not None:
            default_imports.append("palimpzest")
        self.executor = LocalPythonExecutor(
            additional_authorized_imports=authorized_imports or default_imports
        )
        self._update_tool = UpdateCostModelTool(
            registry, op_results=op_results, plan_results=plan_results
        )
        self.executor.send_variables({
            "PlanCostEstimate": PlanCostEstimate,
            "CostModel": CostModel,
            "op_results": op_results,
            "plan_results": plan_results,
            "iter_operators": iter_operators,
            "get_op_type": get_op_type,
            "get_op_id": get_op_id,
            "get_op_model": get_op_model,
            "describe_operator": describe_operator,
            "observed_op_stats": make_observed_op_stats(op_results),
        })
        self.executor.send_tools({self._update_tool.name: self._update_tool})

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- public entry point ------------------------------------------------
    def review(self, *, reason: str) -> str:
        """Refresh the estimate table (cheap, no LLM) and, only when warranted, wake the
        helper LLM to (re)fit the cost model.

        The LLM refit is gated so it does NOT run after every execution:
          - if no model exists yet -> build v1 (LLM);
          - else, only on NEW execution data AND when the current model's relative ranking of
            executed plans has degraded (min Kendall tau over cost/latency below the threshold).
        When ranking is still good we skip the LLM entirely and just return fresh estimates.
        """
        n_exec = len(self.plan_results.rows)
        new_data = n_exec > self._last_reviewed_plan_rows
        self._last_reviewed_plan_rows = n_exec
        n_plans = len(self._executed_actuals())  # distinct executed plans

        need_refit = False
        tau_cost = tau_lat = None
        gate = "no-new-data"
        if n_plans < 2:
            # A cost model is only meaningful for RELATIVE comparison — need >=2 executed plans
            # before building/refitting one (covers the bootstrap and review_plans-too-early cases).
            gate = f"skip (need >=2 executed plans to build/compare a cost model; have {n_plans})"
        elif self.registry.version == 0:
            need_refit = True  # no model yet — author v1 from the bootstrap executions
            gate = "bootstrap (no model yet)"
        elif new_data:
            # Gate the LLM refit on the COST ranking only (latency tau is shown for context
            # but does not trigger a refit on its own).
            tau_cost, tau_lat = self._rank_taus()
            if tau_cost is None:
                # ranking not assessable (e.g. all-tied estimates); don't spend an LLM call
                gate = "skip (cost ranking not assessable)"
            elif tau_cost < self.refit_tau_threshold:
                need_refit = True
                gate = f"refit (cost tau={tau_cost} < {self.refit_tau_threshold})"
            else:
                gate = f"skip (cost ranking still good: cost tau={tau_cost} >= {self.refit_tau_threshold})"
        self._log(f"[cost-helper] review (reason={reason}, executed={n_exec}, "
                  f"cost tau={tau_cost}, lat tau={tau_lat}) → {gate}")
        # Record the gate decision in the helper's own trajectory (printed + saved,
        # but NOT added to the main agent's LLM context).
        self.trajectory_steps.append({
            "agent": "cost_helper", "event": "review", "refit_reason": reason,
            "n_executed": n_exec, "tau_cost": tau_cost, "tau_lat": tau_lat,
            "gate_decision": gate, "reasoning": None, "assistant": None, "observation": gate,
        })

        if need_refit:
            try:
                # reuse the cost tau already computed by the gate (None on bootstrap)
                self._refit(reason=reason, pre_tau=tau_cost)
            except Exception as e:  # never let cost-model trouble break plan search
                self._log(f"[cost-helper] refit failed: {type(e).__name__}: {e}")
                self.trajectory_steps.append({
                    "agent": "cost_helper", "event": "refit_error", "refit_reason": reason,
                    "observation": f"{type(e).__name__}: {e}",
                })
        return self.estimate_table()

    def _estimate_recs(self) -> list[dict]:
        """Per-executed-plan estimated vs actual (cost, latency), for calibration.

        Only plans that ran AND were successfully quality-evaluated (non-NaN
        quality) are included. A plan that failed to execute or produced no
        evaluable output gets NaN quality and cost/latency ~0; keeping those in
        the calibration set injects spurious near-zero-cost outliers that create
        rank inversions and needlessly thrash the refit gate (see ecomm Q12
        run 2, where the two broken baselines pinned Kendall tau negative)."""
        actuals = self._executed_actuals()
        recs = []
        for name in actuals:
            entry = self.plans.get(name)
            if entry is None:
                continue
            # Skip plans that didn't succeed: NaN (or missing) quality. `q != q`
            # is True only for NaN floats (same idiom used for tau below).
            q = actuals[name].get("quality")
            if q is None or q != q:
                continue
            try:
                est = self._estimate_plan(entry["plan"])
            except Exception:
                continue
            recs.append({
                "name": name,
                "est_cost": est.cost, "est_lat": est.time,
                "act_cost": actuals[name].get("cost_usd"),
                "act_lat": actuals[name].get("latency_s"),
            })
        return recs

    # Pairs whose ACTUAL values differ by less than this (relative to the
    # smaller value) are treated as ties when scoring the ranking. The cheap
    # plans in a query often cluster within a few percent of each other; without
    # a tolerance, standard Kendall tau penalizes the model for failing to order
    # those indistinguishable plans, which needlessly trips the refit gate and
    # burns LLM calls chasing an unachievable ranking (see ecomm Q12 run 2).
    RANK_TIE_EPS: float = 0.10

    @classmethod
    def _tau_stats(cls, recs: list[dict], k_est: str, k_act: str) -> tuple[float | None, int, int]:
        """(tau, concordant, discordant) for epsilon-tolerant Kendall ranking.

        Only pairs whose ACTUAL values are *meaningfully* different (relative
        gap >= RANK_TIE_EPS) are scored; near-tie pairs are ignored so the model
        isn't penalized for failing to order plans that are indistinguishable in
        reality. tau = (C - D) / (C + D); None when <2 comparable plans or no
        separable, est-untied pairs exist."""
        rs = [r for r in recs if r[k_act] is not None]
        if len(rs) < 2:
            return None, 0, 0
        C = D = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                ai, aj = float(rs[i][k_act]), float(rs[j][k_act])
                denom = min(abs(ai), abs(aj))
                # Skip near-ties: actual gap within the epsilon band (both ~0 too).
                if denom == 0:
                    if ai == aj:
                        continue
                elif abs(ai - aj) / denom < cls.RANK_TIE_EPS:
                    continue
                de = float(rs[i][k_est]) - float(rs[j][k_est])
                if de == 0:
                    continue  # model can't separate this pair — no signal, skip
                concordant = (de > 0) == (ai - aj > 0)
                C += concordant
                D += not concordant
        if C + D == 0:
            return None, C, D
        return (C - D) / (C + D), C, D

    @classmethod
    def _tau(cls, recs: list[dict], k_est: str, k_act: str) -> float | None:
        """Epsilon-tolerant Kendall tau (see `_tau_stats`)."""
        return cls._tau_stats(recs, k_est, k_act)[0]

    def _rank_taus(self) -> tuple[float | None, float | None]:
        """(tau_cost, tau_lat) for the CURRENT model over executed plans."""
        if self.registry.version == 0:
            return None, None
        recs = self._estimate_recs()
        return self._tau(recs, "est_cost", "act_cost"), self._tau(recs, "est_lat", "act_lat")

    # -- refit -------------------------------------------------------------
    def _validate_current_model(self) -> str | None:
        """Run the currently installed model over ALL plans (executed and candidate) to make
        sure it actually produces numeric estimates. Returns an error string on the first plan
        that raises (or produces a non-numeric estimate), else None."""
        try:
            model = self.registry.current()
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        for name, entry in self.plans.items():
            try:
                est = model.estimate_plan(entry["plan"])
                if isinstance(est, dict):
                    est = PlanCostEstimate(**est)
                float(est.cost)
                float(est.time)
            except Exception as e:
                return f"plan {name!r}: {type(e).__name__}: {e}"
        return None

    def _refit(self, *, reason: str, pre_tau: float | None = None) -> int | None:
        """Wake the helper LLM to produce its decision (keep or a new cost model) in ONE step.
        The loop only continues to REPAIR: if the emitted model fails to run, or installs but
        raises while estimating any plan, the specific error is fed back and the helper is asked
        to fix it — up to `max_repair_retries` times — until we have a working model.

        `pre_tau` is the cost Kendall-tau that triggered this refit (passed in from `review`'s
        gate, so it is not recomputed here; None on bootstrap).

        Uses a FRESH, self-contained message list each call so the full plan_results/op_results
        snapshot is sent exactly once (never accumulated); durable cross-refit memory is the
        compact `version_log` + `model_versions`, folded in by `_build_context`."""
        first_ever = self.registry.version == 0
        msgs: list[dict] = [{"role": "user", "content": self._build_context(reason)}]
        installed_version: int | None = None
        attempt = 0
        max_attempts = 1 + self.max_repair_retries  # 1 decision + N repair iterations
        while attempt < max_attempts:
            attempt += 1
            raw = self._llm_step(msgs)
            reasoning = self._last_reasoning
            msgs.append({"role": "assistant", "content": raw})

            # print the helper's own reasoning/output for easy following (not shown to the main agent)
            if reasoning:
                self._log(f"\n--- [cost-helper] reasoning (attempt {attempt}, reason={reason}) ---\n{reasoning}\n")
            self._log(f"\n--- [cost-helper] assistant (attempt {attempt}) ---\n{raw}\n")

            def emit(observation: str) -> None:
                """Record + print one helper attempt (saved to the helper trajectory)."""
                self._log(f"[cost-helper] attempt {attempt}: {observation}")
                self.trajectory_steps.append({
                    "agent": "cost_helper", "event": "refit_step", "refit_reason": reason,
                    "helper_step": attempt, "tau_cost": pre_tau, "reasoning": reasoning,
                    "assistant": raw, "observation": observation,
                })

            try:
                parsed = _parse_step(raw)
            except ParseError as e:
                emit(f"parse error: {e.detail}")
                msgs.append({"role": "user", "content": (
                    f"Could not parse: {e.detail}. Emit ONE ```python``` block that installs a "
                    "model, or ```json``` {\"decision\": \"keep\"}."
                )})
                continue

            if parsed.code is None:  # json → keep decision
                if first_ever:
                    emit("keep requested but no model installed yet — asking for v1")
                    msgs.append({"role": "user", "content": (
                        "No model is installed yet — you MUST install v1. Emit a ```python``` "
                        "block that defines a class and calls update_cost_model(...)."
                    )})
                    continue
                # Keeping still requires the current model to actually work on all plans.
                err = self._validate_current_model()
                if err is not None:
                    emit(f"keep requested but current model fails to estimate: {err}")
                    msgs.append({"role": "user", "content": (
                        f"You chose to keep the current model, but it RAISES when estimating a "
                        f"plan: {err}. It must produce numeric estimates for ALL plans. Write a "
                        "fixed model and install it with update_cost_model(...)."
                    )})
                    continue
                reason_txt = parsed.result.get("reason", "") if isinstance(parsed.result, dict) else ""
                emit(f"KEEP current model v{self.registry.version} — {reason_txt}")
                self.version_log.append(
                    f"v{self.registry.version}: KEPT at {reason} (cost tau={pre_tau}) — {reason_txt}"
                )
                break

            prev_version = self.registry.version
            try:
                out = self.executor(parsed.code)
            except Exception as e:
                emit(f"model code failed to run — {type(e).__name__}: {e}")
                msgs.append({"role": "user", "content": (
                    f"Execution failed — {type(e).__name__}: {e}. Fix and re-send, or keep via "
                    "```json``` {\"decision\": \"keep\"}."
                )})
                continue

            if self.registry.version > prev_version:
                # installed — VALIDATE it produces estimates before accepting
                new_version = self.registry.version
                err = self._validate_current_model()
                if err is not None:
                    emit(f"installed v{new_version} but estimation failed: {err} — asking to fix")
                    msgs.append({"role": "user", "content": (
                        f"The model installed but RAISES when estimating a plan: {err}. Fix the "
                        "estimate_plan logic (handle every operator type; return a PlanCostEstimate "
                        "with numeric cost/time) and re-install with update_cost_model(...)."
                    )})
                    continue
                installed_version = new_version
                self.model_versions[f"v{installed_version}"] = parsed.code
                self._snapshot_estimates(installed_version)
                emit(f"INSTALLED + validated cost model v{installed_version}")
                self.version_log.append(
                    f"v{installed_version}: INSTALLED at {reason}"
                    + ("" if first_ever else f" (prev model cost tau={pre_tau})")
                )
                break

            # ran code but did not install → nudge
            result_s = str(getattr(out, "output", out) or "").strip()[:2000]
            emit("ran code but did not call update_cost_model — nudging to install")
            msgs.append({"role": "user", "content": (
                (f"Ran, no model installed. Output:\n{result_s}\n\n" if result_s else "Ran, no model installed. ")
                + "Now call update_cost_model(YourClass, notes=...) to install."
            )})

        if installed_version is None and self.registry.version > 0:
            # exhausted repairs without a validated model; current model may be broken
            self._log(f"[cost-helper] WARNING: refit exhausted {max_attempts} attempts without a "
                      f"validated model; estimates may error until the next refit.")
        return installed_version

    def _snapshot_estimates(self, version: int) -> None:
        snap: dict[str, tuple] = {}
        for name, entry in self.plans.items():
            try:
                est = self._estimate_plan(entry["plan"])
                snap[name] = (est.cost, est.time)
            except Exception:
                pass
        self.version_estimates[version] = snap

    # -- estimation / tables ----------------------------------------------
    def _estimate_plan(self, pipeline: Any) -> PlanCostEstimate:
        est = self.registry.current().estimate_plan(pipeline)
        if isinstance(est, dict):
            est = PlanCostEstimate(**est)
        return est

    def _executed_actuals(self) -> dict[str, dict]:
        actuals: dict[str, dict] = {}
        for row in self.plan_results.rows:
            name = row.get("plan_name")
            if name:
                actuals[name] = row  # last execution of a name wins
        return actuals

    def estimate_table(self) -> str:
        if self.registry.version == 0:
            return ("No cost model yet — execute a few baseline plans first; estimates will then be "
                    "available (they are refreshed automatically after each execution).")
        actuals = self._executed_actuals()
        rows, errors = [], []
        for name, entry in self.plans.items():
            try:
                est = self._estimate_plan(entry["plan"])
            except Exception as e:
                errors.append(f"  {name}: estimation failed — {e}")
                continue
            a = actuals.get(name, {})
            details = getattr(est, "details", {}) or {}
            # A plan that executed but has NaN quality (`q != q`) failed to
            # produce evaluable output — its near-zero actual cost/latency are
            # meaningless and must not be read as a cheap plan.
            q = a.get("quality")
            failed = bool(a) and (q is None or q != q)
            rows.append({
                "plan_name": name,
                "description": (entry.get("description", "") or "")[:40],
                "est_cost": est.cost,
                "est_latency": est.time,
                "actual_cost": a.get("cost_usd"),
                "actual_latency": a.get("latency_s"),
                "actual_quality": a.get("quality"),
                "extrapolated": bool(details.get("unseen_ops")),
                "failed": failed,
            })
        rows.sort(key=lambda r: (r["est_cost"] is None, r["est_cost"] or 0.0))

        def fmt(v, spec=".6f"):
            return "—" if v is None else format(float(v), spec)

        header = (f"{'plan':<10} {'description':<40} {'est_cost':>11} {'est_lat':>9} "
                  f"{'act_cost':>11} {'act_lat':>9} {'act_qual':>9} {'extrap?':>8}")
        lines = [header, "-" * len(header)]
        for r in rows:
            if r["failed"]:
                act_cost, act_lat, act_qual = "—", "—", "FAILED"
            else:
                act_cost = fmt(r["actual_cost"])
                act_lat = fmt(r["actual_latency"], ".3f")
                act_qual = fmt(r["actual_quality"], ".3f")
            lines.append(
                f"{r['plan_name']:<10} {r['description']:<40} {fmt(r['est_cost']):>11} "
                f"{fmt(r['est_latency'], '.3f'):>9} {act_cost:>11} "
                f"{act_lat:>9} {act_qual:>9} "
                f"{('yes' if r['extrapolated'] else 'no'):>8}"
            )
        if errors:
            lines.append("\nErrors:")
            lines.extend(errors)
        lines.append("\n(Cost model v%d. Sorted by estimated cost. RELATIVE comparison only — "
                     "absolute values may be inaccurate. 'extrap?'=yes means the plan uses "
                     "operators/models with no observed data yet, so consider executing to learn. "
                     "act_qual=FAILED means the plan ran but produced no evaluable output — its "
                     "actual cost/latency are meaningless; rewrite it, don't select it.)"
                     % self.registry.version)
        return "\n".join(lines)

    # -- context / calibration --------------------------------------------
    def _memory_summary(self) -> str:
        """Compact durable memory carried across refits (NOT the raw data tables)."""
        if not self.version_log:
            return "Cost model history: none yet — you will author v1."
        return ("Cost model history (your memory across this run — prior versions and how they "
                "calibrated):\n" + "\n".join(f"  - {line}" for line in self.version_log))

    def _build_context(self, reason: str) -> str:
        first_ever = self.registry.version == 0
        parts = [
            f"=== Cost-helper invocation (reason: {reason}) ===",
            f"Task:\n{self.task}",
            self._memory_summary(),
        ]
        desc_lines = []
        for n, e in self.plans.items():
            ops = ",".join(get_op_type(o) for o in iter_operators(e["plan"]))
            desc_lines.append(f"  - {n}: {e.get('description', '') or '(no description)'} | ops=({ops})")
        parts.append("Plans written so far:\n" + ("\n".join(desc_lines) if desc_lines else "  (none)"))
        try:
            parts.append("Observed plan_results (real execution):\n"
                         + self.plan_results.df.to_string(index=False))
        except Exception:
            parts.append(f"Observed plan_results rows: {len(self.plan_results.rows)}")
        try:
            parts.append(
                "Observed op_results (one row per executed plan-operator — the FULL table "
                "your model receives as `op_results`; columns: op_name, op_type, op_id, "
                "latency_s, cost_usd, input_tokens, output_tokens, num_records, num_passed):\n"
                + self.op_results.df.to_string(index=False)
            )
        except Exception:
            parts.append("op_results summary (per op_type):\n"
                         + json.dumps(self.op_results.summary(), indent=2))
        if first_ever:
            parts.append(
                "No cost model is installed yet. Design v1: a class with "
                "estimate_plan(self, plan) -> PlanCostEstimate that fits per-operator "
                "cost/latency and cardinality from the observed data, and INSTALL it with "
                "update_cost_model(YourClass, notes=\"v1: ...\")."
            )
        else:
            cur_v = self.registry.version
            prev_code = self.model_versions.get(f"v{cur_v - 1}")
            if prev_code:
                parts.append(f"Previous cost model (v{cur_v - 1}) code — for diffing against the "
                             f"current one so you can see what your last change did:\n```python\n"
                             + prev_code + "\n```")
            parts.append(f"Current cost model (v{cur_v}) code — this is the one you are deciding "
                         f"whether to keep or replace:\n```python\n"
                         + self.model_versions.get(f"v{cur_v}", "(unavailable)")
                         + "\n```")
            parts.append(self._calibration_view())
            parts.append(
                "Decide: if the current model already ranks the executed plans acceptably "
                "(perfect ranking is NOT expected — a few mis-ordered pairs / high tau is fine), "
                "keep it via ```json\n{\"decision\": \"keep\", \"reason\": \"...\"}\n```. Otherwise "
                "emit ONE ```python``` block that defines an improved model (targeting the "
                "mis-ordered pairs) and calls update_cost_model(...)."
            )
        return "\n\n".join(parts)

    def _calibration_view(self) -> str:
        recs = self._estimate_recs()

        def rank_signal(k_est: str, k_act: str) -> str:
            """Epsilon-tolerant Kendall's tau between estimated and actual ordering,
            with the count of discordant (mis-ordered) pairs among plans whose actual
            values differ by more than the tie tolerance. tau=1.0 -> perfect ranking."""
            tau, C, D = self._tau_stats(recs, k_est, k_act)
            if tau is None:
                return f"n/a (need >=2 executed plans with actual gaps >{self.RANK_TIE_EPS:.0%}; near-ties ignored)"
            return f"tau={tau:+.3f}; {D}/{C + D} separable pairs mis-ordered (pairs within {self.RANK_TIE_EPS:.0%} treated as ties)"

        lines = ["Relative calibration on executed plans (estimated vs actual):"]
        h = f"{'plan':<10}{'est_cost':>12}{'act_cost':>12}{'est_lat':>10}{'act_lat':>10}"
        lines.append(h)
        for r in sorted(recs, key=lambda r: r["act_cost"] if r["act_cost"] is not None else 0.0):
            lines.append(
                f"{r['name']:<10}{r['est_cost']:>12.6f}{(r['act_cost'] or 0.0):>12.6f}"
                f"{r['est_lat']:>10.3f}{(r['act_lat'] or 0.0):>10.3f}"
            )
        lines.append(
            "Kendall rank-correlation, est vs actual (SIGNAL only, 1.0=perfect ranking; "
            "do not chase perfect ranking) — "
            f"cost: {rank_signal('est_cost', 'act_cost')}; "
            f"latency: {rank_signal('est_lat', 'act_lat')}."
        )
        return "\n".join(lines)

    # -- llm ---------------------------------------------------------------
    def _trim(self, messages: list[dict]) -> list[dict]:
        budget = self.context_budget_chars
        if sum(len(m["content"]) for m in messages) <= budget:
            return messages
        tail: list[dict] = []
        remaining = budget
        for m in reversed(messages):
            if remaining - len(m["content"]) < 0:
                break
            tail.append(m)
            remaining -= len(m["content"])
        return tail[::-1]

    def _llm_step(self, msgs: list[dict]) -> str:
        result = self.llm.generate(self.system_prompt, self._trim(msgs))
        content, reasoning, meta = result, None, {}
        if isinstance(result, tuple):
            content = result[0] if result else ""
            if len(result) >= 2:
                reasoning = result[1]
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
        self._last_reasoning = reasoning
        return content
