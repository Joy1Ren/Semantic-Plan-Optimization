"""Plan-quality evaluation for the cost model agent.

`PlanQualityEvaluator` owns everything that is the same for every benchmark: running the
oracle-substituted plan, memoizing oracle calls across plans, per-operator oracle judging,
resolving and materialising ground truth, and cost accounting. Three hooks, all abstract here
and all implemented in experiments/*/quality_evaluator.py, carry everything benchmark-specific:

  `score_plan(plan_output_df, ground_truth_df, ground_truth_path)` -- turn a plan's output plus
    the ground truth into a 0-1 score. The benchmarks' scorers take genuinely different inputs:
    SemBench's take a DataFrame and return its own metric dataclasses, while docetl's CUAD scorer
    reads a ground-truth FILE and returns a dict. Both forms are handed over on every call and
    each adapter uses only what its scorer needs.
  `has_ground_truth(ground_truth_df, ground_truth_path)` -- is there anything to score against?
    Asked of whichever form that benchmark actually consumes.
  `write_oracle_ground_truth(df, path)` -- materialise the run's oracle ground truth in the shape
    that benchmark's scorer reads back (CUAD writes the header evaluate_results looks for).

Neither hook is told which ground-truth mode is in effect: `ground_truth_path_in_use()` already
resolves to the right file for the configured mode, so no adapter branches on it.

Per-operator quality (identical regardless of use_oracle_ground_truth — only needs the plan's own
execution samples, never a separately-run oracle-substituted pipeline):
  sem_filter / sem_join / rag_filter: the oracle directly judges each of the plan's own inputs
    against the operator's condition; per-op quality is the agreement rate between the oracle's
    decisions and the plan's own pass/fail decisions.
  sem_map / rag_map: the oracle judges each output field against the field's own description;
    fields the operator MISSED entirely (absent/None, as opposed to a deliberate "") score 0
    without being sent. Per-op quality is the mean over all (record, field) verdicts.
  Every judge call carries exactly ONE record, and an operator is scored from at most
    MAX_JUDGE_RECORDS records, issued concurrently. When a response leaves some fields unscored,
    only those fields are re-asked (the verdicts already returned are kept); a record still
    missing verdicts after _JUDGE_ATTEMPTS is dropped rather than scored on a partial field set.
  RAG operators are judged on their POST-RETRIEVAL input (the retrieved chunks), not the full
    source field, so these scores measure the LLM step and not the retrieval step.
  Any other op_type (rag_join doesn't exist; project, filter, groupby, join, map, ...) is
  non-semantic and never scored here — an empty per_sem_op_quality for a plan built only from
  those isn't a bug, just nothing semantic to score.

Plan quality: plan output vs. ground truth, using the original quality metrics (f1,
relative_error, spearman_correlation, accuracy). The ground truth is either:
  - oracle-generated (use_oracle_ground_truth=True, the default): the plan rebuilt with every
    semantic operator replaced by the oracle model, run once and cached per plan_name in
    llm_judge_dir. Oracle LLM calls are also memoized per-operator across plans (see
    _MemoizingGenerator): a semantic operator shared by two plans is judged by the oracle only
    once. Only the run(s) up to and including the one that yields a non-empty ground truth are
    billed to total_oracle_cost_usd; later plans still run the oracle (it is how oracle
    consistency is checked) but their cost lands in oracle_consistency_cost_usd instead.
  - directly supplied (use_oracle_ground_truth=False, via ground_truth_loader): skips building
    the oracle-substituted plan entirely, avoiding the cost of running the whole plan through the
    oracle.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import mimetypes
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from agent_cost_model.opt_agent.physical_pipeline.operators.rag_common import retrieval_context_key

# Seed for the sem_join candidate-pair subsampling RNG (see PlanQualityEvaluator.__init__).
# Fixed here rather than plumbed in from the runner: it only has to be stable across runs.
_OP_SAMPLE_SEED = 42

# Max records one operator's per-op quality score is computed from, and so also the max number of
# oracle judge calls it costs: every judge path sends exactly ONE record per call. Records are
# large (a CUAD contract runs to tens of thousands of characters), so this bounds what scoring one
# operator costs, at the price of a noisier score.
#
# It has to be applied per operator type, because what bounded each one before was incidental:
# sem_join keeps every candidate pair (|left| x |right|) and was capped at the datasubset size --
# 40 for CUAD -- while sem_filter/sem_map were bounded only by the pipeline's sample-collection
# cap (_execute_core's max_samples, = NUM_SAMPLES), which is a sampling knob that has no reason
# to double as a judging-cost knob.
MAX_JUDGE_RECORDS = 10

# Concurrency for per-record judge calls. They are independent HTTP requests, so at this default
# an operator's whole MAX_JUDGE_RECORDS-call budget goes out in one wave and costs roughly the
# wall time of a single call.
_JUDGE_MAX_WORKERS = 10

# Attempts per record before it is dropped from the score. Attempts accumulate: each one re-asks
# only the fields still unscored, so a 41-field response that drops a single entry is completed by
# a 1-field follow-up rather than a full re-judge.
_JUDGE_ATTEMPTS = 2


class _JudgeResponseError(Exception):
    """A judge call came back unusable: unparseable, or not aligned with what was asked.

    Raised rather than returned so a malformed response is retried exactly like a failed HTTP
    call, and so the reason reaches the log in one place.
    """


@dataclass
class QualityResult:
    quality: float                          # plan quality vs oracle output (0-1, NaN if failed/N/A)
    # {op_name: 0-1} for each semantic op where quality could be computed
    per_sem_op_quality: dict[str, float] = field(default_factory=dict)
    quality_note: str | None = None         # human-readable reason when quality is not a real score
                                            # (e.g. oracle produced no rows on this subset)


class _MemoizingGenerator:
    """Wraps a Palimpzest ``Generator`` so oracle LLM calls are memoized by
    (operator identity, input content) and reused across plans.

    Two plans that share a semantic operator feed that operator identical inputs (the
    subset is fixed and the shared upstream produces identical records), so its per-record
    oracle calls build identical prompts. We key on the operator's condition / generated
    fields plus the projected input field values, so the second plan reuses the first
    plan's oracle verdicts instead of re-calling the LLM. A cache hit returns a zero-cost
    ``GenerationStats``, so reported oracle cost reflects only real (cache-miss) calls.

    The key is **content-based**: if the upstream differs and the input content changes,
    the key changes and the oracle is re-run — reuse happens only when inputs are truly
    identical. This is the operator-level analogue of PZ's validator ``join_cache`` /
    ``filter_cache``. The operator still builds fresh output DataRecords (correct
    source_indices) from the cached verdict, so downstream scoring is unaffected.
    """

    def __init__(self, inner, cache: dict, model_value, reasoning_effort, stats: dict):
        self._inner = inner
        self._cache = cache
        self._model_value = model_value
        self._reff = reasoning_effort
        self._stats = stats

    def __getattr__(self, name):
        # delegate any other attribute access (e.g. .model) to the wrapped generator
        return getattr(self.__dict__["_inner"], name)

    @staticmethod
    def _content(record, cols) -> tuple | None:
        if record is None:
            return None
        keys = list(cols) if cols else []
        if not keys:
            try:
                schema_cls = record.schema if isinstance(record.schema, type) else type(record.schema)
                keys = list(schema_cls.model_fields)
            except Exception:
                keys = []
        return tuple(sorted((c, repr(getattr(record, c, None))) for c in keys))

    def __call__(self, candidate, fields=None, right_candidate=None, **kwargs):
        from palimpzest.core.models import GenerationStats
        cols = kwargs.get("project_cols")
        cond = kwargs.get("filter_condition") or kwargs.get("join_condition")
        field_sig = tuple(sorted(
            (n, getattr(fi, "description", "") or "") for n, fi in (fields or {}).items()
        ))
        key = (
            self._model_value, self._reff, cond, field_sig,
            self._content(candidate, cols), self._content(right_candidate, cols),
        )
        try:
            hkey = hash(key)
        except TypeError:
            hkey = hash(repr(key))
        if hkey in self._cache:
            self._stats["hits"] += 1
            return copy.deepcopy(self._cache[hkey]), None, GenerationStats(), None
        self._stats["misses"] += 1
        field_answers, reasoning, gen_stats, messages = self._inner(
            candidate, fields, right_candidate=right_candidate, **kwargs
        )
        self._cache[hkey] = copy.deepcopy(field_answers)
        return field_answers, reasoning, gen_stats, messages


class PlanQualityEvaluator:
    """Benchmark-agnostic plan-quality machinery: oracle execution and memoization, per-operator
    oracle judging, ground-truth resolution, and cost accounting.

    Everything here is shared across benchmarks. The one step that is not -- turning a plan's
    output and the ground truth into a quality number -- is left to `score_plan`, implemented by
    each adapter in experiments/*/quality_evaluator.py.
    """

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        subset_path: str | Path,
        normalize_df: Callable[[pd.DataFrame], pd.DataFrame],
        llm_judge_dir: str | Path,
        oracle_reasoning_effort: str | None = None,
        use_oracle_ground_truth: bool = True,
        ground_truth_loader: Callable[[], pd.DataFrame] | None = None,
        ground_truth_path: str | Path | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model          # string, used for OpenRouterClient judge calls
        self._oracle_reasoning_effort = oracle_reasoning_effort
        self._query_id = query_id
        self._subset_path = Path(subset_path)
        self._normalize_df_fn = normalize_df
        self._llm_judge_dir = Path(llm_judge_dir)
        # What scoring this run cost: the oracle run(s) that produced the ground truth, plus every
        # per-operator judge call. Deliberately NOT the oracle runs of later plans -- those are
        # consistency checks, kept separately below so the run's real spend is still recoverable.
        self.total_oracle_cost_usd = 0.0
        self.oracle_consistency_cost_usd = 0.0
        self._canonical_oracle_df: "pd.DataFrame | None" = None
        # Set by evaluate() on each call, for per-plan debug dumps.
        self.last_oracle_df: "pd.DataFrame | None" = None
        self.last_ground_truth_df: "pd.DataFrame | None" = None

        # When False, evaluate() skips running the oracle-SUBSTITUTED PLAN and scores plan-level
        # quality against ground_truth_loader's output instead. Per-operator quality still runs
        # either way -- it only needs the plan's own execution samples plus per-operator oracle
        # judging -- so this flag does not mean "no oracle LLM calls".
        self._use_oracle_ground_truth = use_oracle_ground_truth
        self._ground_truth_loader = ground_truth_loader
        self._direct_ground_truth_df: "pd.DataFrame | None" = None
        self._direct_ground_truth_loaded = False

        # The benchmark's real ground-truth file (benchmark.yaml's ground_truth_path). In direct
        # mode this IS the file scored against, so it is passed straight through to the
        # evaluator -- nothing is copied. Only oracle ground truth, which is generated rather
        # than pre-existing, has to be materialised (see _oracle_ground_truth_path below).
        self._ground_truth_path = Path(ground_truth_path) if ground_truth_path else None

        # The run's opt_results directory (opt_results/Q{id}_{runcount}). The oracle ground
        # truth is materialised here as ONE file shared by every plan in the run, and read back
        # from it -- so it is an inspectable artifact rather than in-memory state, and a re-run
        # plan reuses it instead of regenerating.
        #
        # It lives here rather than beside the benchmark's own ground truth because a
        # benchmark's ground_truth_path may point inside an external checkout (SemBench's does),
        # and those are read-only inputs.
        self._run_dir = Path(run_dir) if run_dir else None
        self._oracle_ground_truth_path = (
            self._run_dir / "oracle_ground_truth.csv" if self._run_dir else None
        )

        # Cross-plan per-operator oracle cache: (operator identity, input content) ->
        # field_answers. Shared by every oracle pipeline this evaluator runs, so a semantic
        # operator that appears in multiple plans is judged by the oracle only once.
        self._oracle_call_cache: dict = {}
        self._oracle_cache_stats: dict = {"hits": 0, "misses": 0}

        # Guards total_oracle_cost_usd: per-record judge calls run on a thread pool
        # (_judge_in_parallel), and `+=` on a float is a read-modify-write that silently drops
        # concurrent updates -- i.e. under-reports what scoring actually cost.
        self._oracle_cost_lock = threading.Lock()

        # Private RNG for subsampling what a judge call scores down to MAX_JUDGE_RECORDS (see
        # _cap_judge_items). Fixed seed so two runs of the same query judge the same records and
        # their per-op quality numbers are comparable; an unseeded global `random` made that
        # number drift run to run for reasons unrelated to the plan. Not a knob — nothing
        # benefits from varying it per run.
        self._op_sample_rng = random.Random(_OP_SAMPLE_SEED)

        # Resolve oracle model string → pz.Model enum once at init so make_oracle_copy
        # receives the enum directly (avoids re-resolving on every execute_plan call and
        # surfaces bad model names early).
        self._oracle_pz_model = None
        try:
            from agent_cost_model.opt_agent.physical_pipeline import _str_to_pz_model
            self._oracle_pz_model = _str_to_pz_model(oracle_model)
        except Exception as e:
            print(f"[QualityEvaluator] could not resolve oracle pz model '{oracle_model}': {e}")


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        plan,                         # PhysicalPipeline
        plan_name: str,
        plan_context,                 # SubsetExecutionContext
        plan_output_df: pd.DataFrame, # already normalized for evaluator
    ) -> QualityResult:
        """Run oracle evaluation and return QualityResult."""
        oracle_df = None
        if self._use_oracle_ground_truth:
            # The FIRST plan's oracle output becomes the canonical ground truth every later
            # plan in this run is scored against. Persist it once, and prefer the persisted
            # copy so a resumed/re-run plan scores against the same ground truth.
            #
            # Resolved BEFORE the oracle runs, because "do we already have a ground truth?" is
            # also what decides whether this run is billed: producing the ground truth is a real
            # cost of scoring the run, while every oracle run after that one is a consistency
            # check kept for debugging (see _get_oracle_context).
            if self._canonical_oracle_df is None or self._canonical_oracle_df.empty:
                self._canonical_oracle_df = self._read_persisted(self._oracle_ground_truth_path)
            have_ground_truth = (
                self._canonical_oracle_df is not None and not self._canonical_oracle_df.empty
            )
            oracle_df = self._get_oracle_context(
                plan, plan_name, charge_cost=not have_ground_truth
            )
            if not have_ground_truth and oracle_df is not None and not oracle_df.empty:
                self._canonical_oracle_df = oracle_df
                self._persist(oracle_df, self._oracle_ground_truth_path)
            gt_df = self._canonical_oracle_df
        else:
            gt_df = self._get_direct_ground_truth()

        # Exposed for per-plan debug dumps (see _dump_opt_debug_artifacts). These are two
        # different things whenever a plan is not the first one evaluated: `last_oracle_df` is
        # THIS plan's own oracle-substituted output, while `last_ground_truth_df` is whatever
        # was actually scored against -- the FIRST plan's oracle output, reused as the shared
        # canonical ground truth, or the benchmark's real ground truth in direct mode.
        self.last_oracle_df = oracle_df
        self.last_ground_truth_df = gt_df

        # Per-op quality: identical regardless of use_oracle_ground_truth. sem_map only ever
        # needed the plan's own input/output samples. sem_filter/sem_join ask the oracle to
        # directly judge each of the plan's own inputs against the operator's condition (one call
        # per record), then score agreement with the plan's own pass/fail decisions —
        # in oracle-ground-truth mode this piggybacks on judging done in the same run rather than
        # comparing against a separately-run oracle-substituted pipeline.
        per_sem_op_quality: dict[str, float] = {}
        for stage_idx, info in plan_context.per_sem_op_info.items():
            op_name = info["op_name"]
            op_type = info["op_type"]
            try:
                if op_type in ("sem_filter", "sem_join", "rag_filter"):
                    condition = info["attributes"].get("condition")
                    if condition:
                        q = self._oracle_judge_filter_join_op(
                            op_type, condition, info["samples"], op_name=op_name,
                            retrieval_contexts=info.get("retrieval_contexts"),
                        )
                        if q is not None:
                            per_sem_op_quality[op_name] = q
                    else:
                        print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no condition in attributes")
                elif op_type in ("sem_map", "rag_map"):
                    q = self._score_map_op(info)
                    if q is not None:
                        per_sem_op_quality[op_name] = q
            except Exception as e:
                print(f"[QualityEvaluator] per-op quality failed for {op_name}: {e}")

        # Plan quality
        quality = float("nan")
        quality_note = None
        gt_path = self.ground_truth_path_in_use()
        if not self.has_ground_truth(gt_df, gt_path):
            # Nothing to score the plan against — quality is undefined. Keep it NaN but attach
            # a note so the agent's observation can explain why (rather than showing a bare NaN).
            quality_note = (
                "subset produces empty result, so plan quality is N/A" if self._use_oracle_ground_truth
                else "ground truth is empty or unavailable, so plan quality is N/A"
            )
        else:
            try:
                # Search-time plans always run on the optimization subset; the
                # full-dataset evaluation passes its own dataset instead.
                quality = self.score_plan(
                    plan_output_df, gt_df, gt_path, self._subset_path
                )
            except Exception as e:
                print(f"[QualityEvaluator] plan quality evaluation failed: {e}")

        return QualityResult(quality=quality, per_sem_op_quality=per_sem_op_quality, quality_note=quality_note)

    def ground_truth_path_in_use(self) -> "Path | None":
        """The FILE quality is scored against for this run.

        Oracle mode: the materialised oracle ground truth in the run directory, written by
        `write_oracle_ground_truth` in whatever shape this benchmark's scorer reads back. Direct mode:
        the benchmark's own ground-truth file, used in place -- nothing is copied.
        """
        if self._use_oracle_ground_truth:
            return self._oracle_ground_truth_path
        return self._ground_truth_path

    # ------------------------------------------------------------------
    # Benchmark-specific hooks (implemented by each adapter)
    # ------------------------------------------------------------------

    def has_ground_truth(self, *args) -> bool:
        """Is there anything to score this run's plans against?

        Called as `has_ground_truth(ground_truth_df, ground_truth_path)`.

        What counts as "a ground truth exists" depends on which form the benchmark's scorer
        consumes, so the implementation decides: a DataFrame scorer asks whether the frame has
        rows, a file scorer asks about the file and would otherwise have to materialise a frame
        it never scores against just to answer this. Returning False makes plan quality NaN with
        an explanatory note rather than an error.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement has_ground_truth(); see "
            "experiments/*/quality_evaluator.py"
        )

    def score_plan(self, *args) -> float:
        """Turn a plan's output plus the ground truth into a single 0-1 quality score.

        Called as `score_plan(plan_output_df, ground_truth_df, ground_truth_path,
        population_path)`.

        The parameters are unnamed here on purpose: the benchmarks' scorers take genuinely
        different inputs -- SemBench's take the ground truth as a DataFrame and return one of its
        own metric dataclasses, docetl's CUAD scorer reads a ground-truth FILE and scores over a
        document population read from the dataset -- so every form is offered on every call and
        the implementation names only the ones it uses. Nothing here is told which ground-truth
        mode is in effect: `ground_truth_path_in_use()` has already resolved to the right file.

        `population_path` is the dataset the plan being scored actually ran over -- the
        optimization subset during search, the full dataset for the final evaluation. A scorer
        that measures over a document population takes it from there rather than from the plan's
        own output, so a plan that drops documents is charged for them.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement score_plan(); see "
            "experiments/*/quality_evaluator.py"
        )

    def write_scoring_input(self, *args) -> None:
        """Write, at a path, exactly what this benchmark's scorer consumes for a plan's output.

        Called as `write_scoring_input(plan_output_df, path)`.

        Purely a debugging artifact -- the run's opt_results/<plan>/ directory keeps it next to
        the plan's raw output so a surprising quality score can be read back to its input. It is
        a hook rather than a plain CSV dump because "what the scorer consumes" is exactly what
        differs per benchmark: SemBench hands its evaluator the normalized frame, while CUAD
        hands docetl a narrow JSON of {filename, clauses} records no frame in the engine holds.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement write_scoring_input(); see "
            "experiments/*/quality_evaluator.py"
        )

    def write_oracle_ground_truth(self, *args) -> None:
        """Materialise the run's oracle ground truth.

        Called as `write_oracle_ground_truth(df, path)`.

        The shape this file takes is a property of the benchmark's scorer, not of the engine: one
        that reads the ground-truth FILE needs it laid out exactly as that reader expects, while
        one that scores against the DataFrame only needs the frame on disk so `_read_persisted`
        can restore it and a human can inspect it. There is deliberately no default -- a plain
        `to_csv` would satisfy the second kind and silently produce an unreadable ground truth
        for the first.

        Whatever is written is read back with `pd.read_csv`, so it must round-trip through CSV --
        which every benchmark's frame does today, since a plan's output columns are the scalar
        fields its operators generate. A list/dict cell would not: `to_csv` renders it with
        repr(), which comes back as an unparseable string.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement write_oracle_ground_truth(); see "
            "experiments/*/quality_evaluator.py"
        )

    # ------------------------------------------------------------------
    # Oracle plan execution and caching
    # ------------------------------------------------------------------

    def _install_oracle_cache(self, oracle_pipeline) -> None:
        """Wrap every semantic operator's Generator in the oracle pipeline (including a
        join's right sub-pipeline) with the shared cross-plan memoizing cache, so shared
        operators reuse oracle verdicts instead of re-calling the LLM."""
        for op in oracle_pipeline:
            pz_op = getattr(op, "_pz_op", None)
            gen = getattr(pz_op, "generator", None)
            if gen is None or isinstance(gen, _MemoizingGenerator):
                continue
            model_value = getattr(getattr(op, "model", None), "value", None)
            reff = getattr(pz_op, "reasoning_effort", None)
            pz_op.generator = _MemoizingGenerator(
                gen, self._oracle_call_cache, model_value, reff, self._oracle_cache_stats
            )

    def _get_oracle_context(
        self, plan, plan_name: str, charge_cost: bool = True
    ) -> "pd.DataFrame | None":
        self._llm_judge_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._llm_judge_dir / f"Q{self._query_id}_{plan_name}_gt.csv"

        if csv_path.exists():
            return pd.read_csv(csv_path)

        try:
            # Prefer the resolved pz.Model enum; fall back to string (triggers _str_to_pz_model)
            oracle_model_arg = self._oracle_pz_model if self._oracle_pz_model is not None else self._oracle_model
            oracle_pipeline = plan.make_oracle_copy(oracle_model_arg, self._oracle_reasoning_effort)
            # Point every semantic operator's Generator at the shared cross-plan cache, so
            # operators this plan shares with an already-evaluated plan reuse the oracle's
            # verdicts instead of re-calling the LLM.
            self._install_oracle_cache(oracle_pipeline)
            subset_cache_path = self._subset_path
            if not subset_cache_path.exists():
                raise FileNotFoundError(
                    f"Subset CSV not found at {subset_cache_path}. "
                    "A plan must be executed before the oracle can run."
                )
            # Oracle wall-clock latency is irrelevant (the oracle exists only for quality
            # scoring and its result is cached), so plan_dict is ignored here.
            hits0, misses0 = self._oracle_cache_stats["hits"], self._oracle_cache_stats["misses"]
            oracle_per_op_list, oracle_context, _ = oracle_pipeline.run_subset(
                subset_cache_path=str(subset_cache_path)
            )
            # The oracle's ACTUAL cost for this run (per-operator cache hits contribute ~0, so
            # the cache's savings show up in the totals). It is billed to the query only while
            # the run is still producing the ground truth -- usually just the first plan, but
            # more when an early plan's oracle output came back empty. Once a ground truth
            # exists, later plans' oracle runs are consistency checks nothing is scored against,
            # and billing them grew the reported oracle cost linearly in the number of plans the
            # agent happened to try.
            run_cost_usd = sum(
                float(row.get("cost_usd", 0.0) or 0.0) for row in oracle_per_op_list
            )
            if charge_cost:
                self.total_oracle_cost_usd += run_cost_usd
            else:
                self.oracle_consistency_cost_usd += run_cost_usd
            hits = self._oracle_cache_stats["hits"] - hits0
            misses = self._oracle_cache_stats["misses"] - misses0
            # print(f"[QualityEvaluator] oracle op-cache for {plan_name}: {hits} hits, {misses} misses "
            #       f"({hits + misses} LLM calls, avoided so far: {self._oracle_cache_stats['hits']})")
        except Exception as e:
            print(f"[QualityEvaluator] oracle pipeline run failed: {e}")
            return None

        oracle_df = pd.DataFrame(oracle_context.output_records)
        oracle_df = self._normalize_df_fn(oracle_df)
        # Cached so a plan re-executed in the same run reuses this oracle output. The frame is
        # all scalar cells (a plan's output columns are the scalar fields its operators
        # generate), so it survives the CSV round trip read back at the top of this method.
        oracle_df.to_csv(csv_path, index=False)

        # Debug artifact only -- nothing reads it back. Kept next to the subset it was produced
        # from so the oracle's output for a query is easy to eyeball against that subset. It is
        # NOT the ground truth (that is the run-level file written by _persist) and it is NOT a
        # cache, so CostModelAgent._setup_run deletes it at the start of every run to keep a
        # previous run's oracle output from being mistaken for this one's.
        oracle_result_path = subset_cache_path.with_name(
            f"{subset_cache_path.stem}_oracle_result.csv"
        )
        if not oracle_result_path.exists():
            oracle_result_path.parent.mkdir(parents=True, exist_ok=True)
            oracle_df.to_csv(oracle_result_path, index=False)

        return oracle_df

    def _read_persisted(self, path: "Path | None") -> "pd.DataFrame | None":
        """Read a previously materialised ground truth, or None if there isn't one."""
        if path is None or not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[QualityEvaluator] could not read ground truth at {path}: {e}")
            return None

    def _persist(self, df: pd.DataFrame, path: "Path | None") -> None:
        """Materialise the ground truth for this run: one file, shared by every plan."""
        if path is None or df is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.write_oracle_ground_truth(df, path)
        except Exception as e:
            print(f"[QualityEvaluator] could not write ground truth to {path}: {e}")

    def _get_direct_ground_truth(self) -> "pd.DataFrame | None":
        """The real ground truth, materialised once per run and reused across plans.

        Used instead of the oracle-substituted plan when use_oracle_ground_truth=False. The
        benchmark's `ground_truth_loader` is what produces it -- for CUAD that reads the real
        annotations, for SemBench it regenerates the gold query over the datasubset -- and the
        result is cached in memory for the rest of the run. It is NOT copied into the run
        directory: the benchmark's ground_truth_path already points at the real file, and that
        path is what gets handed to the evaluator (CUAD's is 3.8 MB -- copying it per run would
        reintroduce exactly the duplicate that prepare_cuad_data.py was changed to stop making).

        Restricting the ground truth to the optimization subset is the LOADER's job, not this
        method's. Every dataset/datasubset this agent runs on names its identifier column in its
        benchmark's `dataset.id_col` (`idx` for CUAD and SemBench ecomm, `reviewId` for SemBench
        movie), but a benchmark's ground truth does not: CUAD's is keyed by `Filename` (bridged by
        prepare_cuad_data.py's idx -> filename mapping), and SemBench's is gold-SQL output whose
        columns its evaluator matches by position rather than by name. There is no id column to
        join on that means the same thing across benchmarks, so each loader handles its own --
        SemBench by regenerating the gold query over the subset rather than filtering after the
        fact, CUAD by letting evaluate_results restrict the annotations to the documents the plan
        actually produced.
        """
        if not self._direct_ground_truth_loaded:
            self._direct_ground_truth_loaded = True
            if self._ground_truth_loader is not None:
                try:
                    self._direct_ground_truth_df = self._ground_truth_loader()
                except Exception as e:
                    print(f"[QualityEvaluator] direct ground-truth load failed: {e}")
        return self._direct_ground_truth_df

    # ------------------------------------------------------------------
    # Per-operator quality scoring
    # ------------------------------------------------------------------

    def _oracle_generate(self, content: str | list) -> str:
        """Call oracle client; handle both str and (str, reasoning) return types.

        `content` may be a plain string or an OpenAI-style multimodal content
        list (mix of {"type": "text", ...} and {"type": "image_url", ...} parts).
        """
        result = self._oracle_client.generate(
            system="You are a rigorous evaluator. Return only valid JSON.",
            messages=[{"role": "user", "content": content}],
        )
        if isinstance(result, tuple):
            if len(result) >= 3 and isinstance(result[2], dict):
                with self._oracle_cost_lock:
                    self.total_oracle_cost_usd += float(result[2].get("cost_usd", 0.0) or 0.0)
            return result[0]
        return result

    @staticmethod
    def _balanced_json_objects(text: str):
        """Yield each top-level ``{...}`` span in `text`, brace-matched and string-aware.

        Needed because the judge's payload is NESTED (`{"scores": {...}}`), which neither a greedy
        nor a lazy regex handles safely: `\\{.*\\}` spans from the first brace in the response to
        the last -- so any prose containing a brace ("field 1 {matches}") swallows the real object
        and fails to parse -- while `\\{.*?\\}` stops at the inner object's closing brace and cuts
        the outer one in half. Quoted strings are skipped so a brace inside an extracted contract
        span cannot unbalance the scan.
        """
        depth, start, in_string, escaped = 0, None, False, False
        for i, ch in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start: i + 1]
                    start = None

    @classmethod
    def _parse_judge_json(cls, response: str, required_key: str) -> dict:
        """The judge's JSON object, as a dict carrying `required_key`.

        Tried in order of how the response is most likely shaped: the bare object (what the prompt
        asks for, and what a compliant model returns), then the contents of a ``` fence, then any
        brace-balanced object found in the text. Raises _JudgeResponseError if none parses.
        """
        candidates: list[str] = [response.strip()]
        candidates += [m.strip() for m in re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)]
        candidates += list(cls._balanced_json_objects(response))

        first_dict: dict | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                if required_key in parsed:
                    return parsed
                if first_dict is None:
                    first_dict = parsed
        if first_dict is not None:
            # Parsed, but not the shape asked for -- report what did come back.
            raise _JudgeResponseError(
                f"response JSON has no {required_key!r} key (got keys {sorted(first_dict)[:10]})"
            )
        raise _JudgeResponseError(
            f"response contained no parseable JSON object (len={len(response)}): {response[:200]!r}"
        )

    @staticmethod
    def _verdict(value) -> float:
        """One judge verdict as 0.0 or 1.0.

        Accepts the JSON spellings a model actually produces for a binary answer -- `true`/`false`,
        `1`/`0`, and those same values quoted -- and raises on anything else rather than letting
        e.g. a `null` or a prose answer silently read as 0.
        """
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return min(1.0, max(0.0, float(value)))
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "false"):
                return 1.0 if text == "true" else 0.0
            return min(1.0, max(0.0, float(text)))  # "1" / "0"; ValueError otherwise
        raise TypeError(f"cannot read {value!r} as a 0/1 verdict")

    def _judge_with_retry(self, op_name: str, judge_once: Callable):
        """Run one record's judge call, retrying up to _JUDGE_ATTEMPTS times.

        Returns judge_once()'s value, or None if every attempt failed -- the caller drops those
        records rather than scoring them partially.
        """
        last_error = None
        for attempt in range(1, _JUDGE_ATTEMPTS + 1):
            try:
                return judge_once()
            except Exception as e:
                last_error = e
                if attempt < _JUDGE_ATTEMPTS:
                    print(f"[QualityEvaluator] {op_name}: retrying one record -- {e}")
        print(
            f"[QualityEvaluator] {op_name}: skipping one record after {_JUDGE_ATTEMPTS} "
            f"attempts -- {last_error}"
        )
        return None

    def _judge_in_parallel(self, items: list, judge_one: Callable) -> list:
        """Run one judge call per item concurrently, preserving input order.

        Every judge path sends a single record per call, so an operator costs
        len(items) <= MAX_JUDGE_RECORDS independent HTTP requests; running them serially would
        make per-op scoring the slowest part of evaluating a plan. Exceptions are the caller's
        to handle -- judge_one is expected to return None for a record it could not score.
        """
        if len(items) <= 1:
            return [judge_one(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(_JUDGE_MAX_WORKERS, len(items))) as pool:
            return list(pool.map(judge_one, items))

    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})

    @classmethod
    def _maybe_image_url(cls, value) -> str | None:
        """If `value` is a path to an image file, return a base64 data URL; else None.

        sem_map inputs of type ImageFilepath hold the on-disk path to the image.
        The oracle judge call is otherwise text-only, so without this the oracle
        never sees the pixels and scores vision maps ~0.
        """
        if not isinstance(value, str):
            return None
        if os.path.splitext(value)[1].lower() not in cls._IMAGE_EXTS:
            return None
        try:
            if not os.path.isfile(value):
                return None
            with open(value, "rb") as f:
                data = f.read()
            mime = mimetypes.guess_type(value)[0] or "image/jpeg"
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    @staticmethod
    def _dr_to_dict(dr) -> dict:
        schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
        return {k: getattr(dr, k, None) for k in schema_cls.model_fields}

    @classmethod
    def _record_content_parts(
        cls, record, index: int | None = None, trailer: str = "", input_view: dict | None = None,
    ) -> list[dict]:
        """One record's OpenAI-style multimodal content parts, shared by both oracle judge calls:
        a text part with the record's INPUT fields (plus an optional `trailer` line), followed by
        an image_url part for every image-valued field.

        `input_view` overrides field values with what the operator's LLM call actually received.
        It is how a RAG operator gets judged on its retrieved chunks rather than the full source
        field: retrieval runs on a COPY of the candidate (palimpzest RAGConvert.convert), so
        `record` -- the input the pipeline sampled -- still holds the whole document. Judging
        against that document turns every chunk the retrieval failed to surface into an apparent
        extraction error, which is the opposite of what a per-operator score is for. Non-RAG ops
        pass None and are unaffected.

        An image field is replaced in the JSON by a placeholder and sent as a real image part, so
        the oracle actually sees the pixels -- without this it scores vision-derived operators ~0.
        """
        image_urls: list[str] = []
        text_inp: dict = {}
        fields = cls._dr_to_dict(record)
        if input_view:
            fields.update(input_view)
        for k, v in fields.items():
            url = cls._maybe_image_url(v)
            if url is not None:
                text_inp[k] = "<image attached below>"
                image_urls.append(url)
            else:
                text_inp[k] = v
        label = "\n  INPUT: " if index is None else f"\nRecord {index}:\n  INPUT: "
        text = label + json.dumps(text_inp, default=str)
        return [
            {"type": "text", "text": text + trailer},
            *({"type": "image_url", "image_url": {"url": url}} for url in image_urls),
        ]

    def _cap_judge_items(self, items: list) -> list:
        """Subsample what one oracle judge call scores down to MAX_JUDGE_RECORDS.

        Random rather than first-N: samples arrive in execution order, which follows the
        datasubset's row order, so a prefix would judge the same handful of records for every
        operator of every plan -- a biased sample, and one that never sees the rest of the
        subset. Drawn from the run-stable _op_sample_rng so a re-run judges the same records.
        """
        if len(items) <= MAX_JUDGE_RECORDS:
            return items
        return self._op_sample_rng.sample(items, MAX_JUDGE_RECORDS)

    def _oracle_judge_filter_join_op(
        self, op_type: str, condition: str, samples: list, op_name: str = "?",
        retrieval_contexts: dict | None = None,
    ) -> float | None:
        """Directly ask the oracle to judge each of the plan's own filter/join inputs against
        `condition` — one call per record, same as _score_map_op — then score agreement between
        the oracle's decisions and the plan's own pass/fail decisions.

        One record per call rather than one call for all of them: a single prompt holding every
        record's full input and answering with a bare positional array gives the judge nowhere to
        reason per record and no way to signal a partial answer, and a short array used to be
        rejected outright, throwing away every verdict in it. Per record, a failure costs only
        that record.

        For rag_filter, `retrieval_contexts` supplies the post-retrieval input the operator's LLM
        call actually saw, so the condition is judged against the retrieved chunks rather than the
        full source field (see _record_content_parts).

        Used for per-operator quality regardless of use_oracle_ground_truth: it only needs the
        plan's own execution samples, never a separately-run oracle-substituted pipeline.
        """
        # Dedupe by source_indices (samples may repeat the same record across batches) and
        # capture the plan's own pass/fail decision (out is not None) per candidate.
        candidates: dict[str, tuple] = {}  # key -> (input_record, plan_passed)
        for item in samples:
            if not (isinstance(item, tuple) and len(item) == 2):
                continue
            inp, out = item
            if not hasattr(inp, "_source_indices"):
                continue
            candidates[str(inp._source_indices)] = (inp, out is not None)
        if not candidates:
            print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no samples with source_indices")
            return None

        keys = self._cap_judge_items(list(candidates))
        contexts = retrieval_contexts or {}
        unit = "pair of records satisfies the join condition" if op_type == "sem_join" else "record satisfies the filter condition"

        def judge_once(key):
            record, plan_passed = candidates[key]
            content: list[dict] = [{
                "type": "text",
                "text": (
                    f"You are evaluating whether the {unit} below.\n"
                    f"Condition: {condition}\n"
                    "Judge the condition ONLY against the INPUT shown -- it is exactly what "
                    "the operator was given."
                ),
            }]
            content.extend(self._record_content_parts(
                record, input_view=contexts.get(retrieval_context_key(record)),
            ))
            content.append({
                "type": "text",
                "text": (
                    "\n\nDecide true if this record satisfies the condition, false otherwise.\n"
                    'Return ONLY valid JSON: {"decision": true or false}'
                ),
            })
            decision = self._parse_judge_json(self._oracle_generate(content), "decision")["decision"]
            try:
                oracle_passed = self._verdict(decision) == 1.0
            except (TypeError, ValueError) as e:
                raise _JudgeResponseError(f"'decision' is {decision!r}, expected true or false: {e}") from e
            return 1.0 if oracle_passed == plan_passed else 0.0

        agreements = [
            a for a in self._judge_in_parallel(
                keys, lambda key: self._judge_with_retry(op_name, lambda: judge_once(key))
            )
            if a is not None
        ]
        if not agreements:
            print(f"[QualityEvaluator] per-op quality failed for {op_name} ({op_type}): no record could be judged")
            return None
        if len(agreements) < len(keys):
            print(
                f"[QualityEvaluator] {op_name}: scored {len(agreements)}/{len(keys)} records "
                "(the rest could not be judged)"
            )
        return sum(agreements) / len(agreements)

    @staticmethod
    def _judge_cols(raw) -> list[tuple[str, str]]:
        """(name, description) for each generated column, from `attributes["cols"]`.

        Also accepts the older names-only form of that attribute, so per-op scoring still runs
        against an execution context produced before descriptions were added there.
        """
        cols: list[tuple[str, str]] = []
        for col in raw or []:
            if isinstance(col, dict):
                name = col.get("name")
                if name:
                    cols.append((str(name), str(col.get("description") or "")))
            elif col:
                cols.append((str(col), ""))
        return cols

    def _judge_map_record(self, op_name, cols, contexts, item) -> tuple[float, int] | None:
        """Score one record's output fields, re-asking only what is still unscored.

        Returns (sum of field verdicts, number of fields) or None if the record could not be fully
        judged — the caller drops those. All-or-nothing per record is what keeps every field backed
        by the same set of records, and so keeps the flat mean equal to the
        per-field-then-across-fields mean.

        Attempts accumulate rather than restart: each call's readable verdicts are kept and only
        the fields still missing are re-asked. A 41-field response that drops one entry then costs
        a 1-field follow-up instead of a full re-judge, and a second response that drops a
        different field still completes the record.

        Every failure is contained here rather than left to propagate: these run on a thread pool
        whose map() re-raises, so one malformed record would otherwise cost the whole operator its
        score instead of just its own verdict.
        """
        inp, out = item
        try:
            out_d = self._dr_to_dict(out)
        except Exception as e:
            print(f"[QualityEvaluator] {op_name}: skipping one record -- unreadable output: {e}")
            return None

        # Split the columns into ones the operator actually answered and ones it MISSED. A miss is
        # a field absent from the output record, or None/NaN -- how a generation that produced
        # nothing for that field surfaces (see base._patch_convert_empty_field_answers). That is
        # not the same as the operator deliberately returning "": an empty string is an answer,
        # and on an extraction task it is usually the right one, so it goes to the judge like any
        # other value. Misses score 0 outright and are never sent.
        judged: list[tuple[str, str, object]] = []
        for name, description in cols:
            value = out_d.get(name)
            if name not in out_d or value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            judged.append((name, description, value))
        if not judged:
            return 0.0, len(cols)

        verdicts: dict[str, float] = {}
        outstanding = judged
        for attempt in range(1, _JUDGE_ATTEMPTS + 1):
            reason = None
            try:
                verdicts.update(self._judge_map_fields(inp, contexts, outstanding))
            except Exception as e:
                reason = e
            outstanding = [f for f in judged if f[0] not in verdicts]
            if not outstanding:
                # Missed fields contribute 0 to the numerator but still count in the denominator.
                return sum(verdicts.values()), len(cols)
            if reason is None:
                missing = [name for name, _, _ in outstanding]
                reason = _JudgeResponseError(
                    f"no usable verdict for {len(missing)} of {len(judged)} field(s): {missing[:5]}"
                )
            if attempt < _JUDGE_ATTEMPTS:
                print(
                    f"[QualityEvaluator] {op_name}: re-asking {len(outstanding)} unscored "
                    f"field(s) of {len(judged)} -- {reason}"
                )
        print(
            f"[QualityEvaluator] {op_name}: skipping one record after {_JUDGE_ATTEMPTS} "
            f"attempts -- {reason}"
        )
        return None

    def _judge_map_fields(self, inp, contexts, fields) -> dict[str, float]:
        """One judge call covering `fields` of one record → {field name: 0.0/1.0}.

        Returns only the verdicts that came back readable AND were actually asked for; the caller
        re-asks whatever is missing, so a dropped or unreadable entry costs that field rather than
        the whole record. Raises _JudgeResponseError when the response yields nothing usable, so
        the reason reaches the log instead of being flattened into "no verdict".
        """
        field_list = "\n".join(
            f'- "{name}": {description}' if description else f'- "{name}"'
            for name, description, _ in fields
        )
        content: list[dict] = [{
            "type": "text",
            "text": (
                "You are evaluating whether a semantic map operator produced correct outputs for "
                "ONE input record.\n"
                "Judge each field ONLY against the INPUT shown below -- it is exactly what the "
                "operator was given. Do not mark a field wrong for information that is not in "
                "this input.\n\n"
                f"Fields to evaluate, with the instruction the operator was given for each:\n{field_list}"
            ),
        }]
        content.extend(self._record_content_parts(
            inp,
            trailer="\n  OPERATOR output: " + json.dumps(
                {name: value for name, _, value in fields}, default=str
            ),
            input_view=contexts.get(retrieval_context_key(inp)),
        ))
        content.append({
            "type": "text",
            "text": (
                "\n\nScore each field above: 1 if the operator's value is correct for this input, "
                "0 if it is not. Where a field's instruction says to return an empty string when "
                "the information is absent, an empty value is CORRECT if this input genuinely "
                "does not contain it.\n"
                'Return ONLY valid JSON of the form {"scores": {"<field name>": 0 or 1, ...}}, '
                "keyed by the EXACT field names listed above, copied verbatim -- including "
                "spaces, slashes and capitalisation.\n"
                f'The "scores" object must have exactly {len(fields)} '
                f"{'entry' if len(fields) == 1 else 'entries'}, one per field listed above: "
                "no field omitted, none added."
            ),
        })

        # Keyed by field name rather than positional: a 41-entry array only has to lose or gain one
        # element for every later verdict to be attributed to the wrong field, and the mismatch is
        # only visible as a length that is off by one, with no way to tell WHICH field went
        # missing. Names make each verdict self-identifying, so a partial response is both safe to
        # use and precise about what still needs asking.
        scores = self._parse_judge_json(self._oracle_generate(content), "scores").get("scores")
        if not isinstance(scores, dict):
            raise _JudgeResponseError(
                f"'scores' is {type(scores).__name__}, expected an object keyed by field name"
            )
        asked = {name for name, _, _ in fields}
        usable: dict[str, float] = {}
        unreadable: list[str] = []
        for name in asked:
            if name not in scores:
                continue
            try:
                usable[name] = self._verdict(scores[name])
            except (TypeError, ValueError):
                unreadable.append(name)
        if not usable:
            raise _JudgeResponseError(
                f"no readable verdict among the {len(asked)} field(s) asked about "
                f"(returned keys {sorted(scores)[:5]}, unreadable {unreadable[:5]})"
            )
        return usable

    def _score_map_op(self, info: dict) -> float | None:
        """Oracle judges whether a sem_map/rag_map's output fields are correct, one call per record.

        One record per call rather than one call for the whole operator: the batched form had to
        hold every record's full input in a single prompt and answer with an
        n_records x n_fields nested array (410 cells for CUAD's 41 clause types over 10 records)
        with no room to reason per field, and the answers degenerated -- one plan scored 0.000 on
        a BM25 rag_map whose output was 89% verbatim source spans, while the same plan's embedding
        rag_map scored 0.951 in the same run.

        Verdicts come back keyed by field NAME, not as a positional array, so a response that
        omits or adds a field is caught by name instead of surviving as a silent off-by-one that
        shifts every later verdict onto the wrong field.

        What each call shows the judge:
          - the operator's OWN input. For rag_map that is the retrieved chunks rather than the
            source document (`retrieval_contexts`), so this scores the LLM step and not the
            retrieval step -- a clause no chunk surfaced is not charged to the map.
          - every field's NAME AND DESCRIPTION. The description is the operator's own instruction
            for that field, and it is where conventions like "return an empty string if the
            category is not present" are stated; with bare names the judge has to guess what ""
            means on a task where most fields are legitimately empty.

        Score is the flat mean over all (record, field) cells of the records that were judged.
        That equals averaging per field across records and then across fields, because every
        judged record contributes a verdict for every field.

        Image-valued input fields are attached to the judge call as vision inputs so the oracle
        can actually verify vision-derived output fields.
        """
        op_name = info.get("op_name", "?")
        cols = self._judge_cols(info["attributes"].get("cols", []))
        if not cols:
            print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no 'cols' in attributes")
            return None

        valid: list[tuple] = [
            (inp, out) for inp, out in info["samples"]
            if out is not None and hasattr(inp, "_source_indices")
        ]
        if not valid:
            print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no samples with a non-None output")
            return None
        valid = self._cap_judge_items(valid)
        contexts = info.get("retrieval_contexts") or {}

        scored = [
            r for r in self._judge_in_parallel(
                valid, lambda item: self._judge_map_record(op_name, cols, contexts, item)
            )
            if r is not None
        ]
        if not scored:
            print(f"[QualityEvaluator] per-op quality failed for {op_name} (sem_map/rag_map): no record could be judged")
            return None
        if len(scored) < len(valid):
            print(
                f"[QualityEvaluator] {op_name}: scored {len(scored)}/{len(valid)} records "
                "(the rest could not be judged)"
            )
        total_cells = sum(n for _, n in scored)
        return sum(s for s, _ in scored) / total_cells if total_cells else None
