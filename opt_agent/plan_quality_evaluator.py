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
    against the operator's condition (one batched call per operator); per-op quality is the
    agreement rate between the oracle's decisions and the plan's own pass/fail decisions.
  sem_map / rag_map: the oracle judges whether each output field is correct (batched LLM call).
  Any other op_type (rag_join doesn't exist; project, filter, groupby, join, map, ...) is
  non-semantic and never scored here — an empty per_sem_op_quality for a plan built only from
  those isn't a bug, just nothing semantic to score.

Plan quality: plan output vs. ground truth, using the original quality metrics (f1,
relative_error, spearman_correlation, accuracy). The ground truth is either:
  - oracle-generated (use_oracle_ground_truth=True, the default): the plan rebuilt with every
    semantic operator replaced by the oracle model, run once and cached per plan_name in
    llm_judge_dir. Oracle LLM calls are also memoized per-operator across plans (see
    _MemoizingGenerator): a semantic operator shared by two plans is judged by the oracle only
    once.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


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
        op_sample_seed: int | None = None,
    ) -> None:
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model          # string, used for OpenRouterClient judge calls
        self._oracle_reasoning_effort = oracle_reasoning_effort
        self._query_id = query_id
        self._subset_path = Path(subset_path)
        self._normalize_df_fn = normalize_df
        self._llm_judge_dir = Path(llm_judge_dir)
        self.total_oracle_cost_usd = 0.0
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

        # Private RNG for subsampling a sem_join's candidate pairs down to max_pairs. Seeded
        # (the runner passes RANDOM_OP_SAMPLER_SEED) so two runs of the same query judge the
        # same pairs and their per-op quality numbers are comparable; an unseeded global
        # `random` made that number drift run to run for reasons unrelated to the plan.
        self._op_sample_rng = random.Random(op_sample_seed)

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
            oracle_df = self._get_oracle_context(plan, plan_name)
            # The FIRST plan's oracle output becomes the canonical ground truth every later
            # plan in this run is scored against. Persist it once, and prefer the persisted
            # copy so a resumed/re-run plan scores against the same ground truth.
            if self._canonical_oracle_df is None or self._canonical_oracle_df.empty:
                self._canonical_oracle_df = self._read_persisted(self._oracle_ground_truth_path)
            if (self._canonical_oracle_df is None or self._canonical_oracle_df.empty) and (
                oracle_df is not None and not oracle_df.empty
            ):
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
        # directly judge each of the plan's own inputs against the operator's condition (a
        # single batched call), then score agreement with the plan's own pass/fail decisions —
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
                        max_pairs = len(plan_context.sampled_records) if op_type == "sem_join" else None
                        q = self._oracle_judge_filter_join_op(
                            op_type, condition, info["samples"], max_pairs=max_pairs, op_name=op_name,
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

    def _get_oracle_context(self, plan, plan_name: str) -> "pd.DataFrame | None":
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
            # Count the oracle's ACTUAL cost on every run (cache hits contribute ~0), so the
            # per-operator cache's savings show up in total_oracle_cost_usd.
            self.total_oracle_cost_usd += sum(
                float(row.get("cost_usd", 0.0) or 0.0) for row in oracle_per_op_list
            )
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
            f"Q{self._query_id}_oracle_result.csv"
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
        method's. Every dataset/datasubset this agent runs on names its identifier column `idx`,
        but a benchmark's ground truth does not: CUAD's is keyed by `Filename` (bridged by
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
                self.total_oracle_cost_usd += float(result[2].get("cost_usd", 0.0) or 0.0)
            return result[0]
        return result

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
    def _record_content_parts(cls, record, index: int, trailer: str = "") -> list[dict]:
        """One record's OpenAI-style multimodal content parts, shared by both oracle judge calls:
        a text part with the record's INPUT fields (plus an optional `trailer` line), followed by
        an image_url part for every image-valued field.

        An image field is replaced in the JSON by a placeholder and sent as a real image part, so
        the oracle actually sees the pixels -- without this it scores vision-derived operators ~0.
        """
        image_urls: list[str] = []
        text_inp: dict = {}
        for k, v in cls._dr_to_dict(record).items():
            url = cls._maybe_image_url(v)
            if url is not None:
                text_inp[k] = "<image attached below>"
                image_urls.append(url)
            else:
                text_inp[k] = v
        text = f"\nRecord {index}:\n  INPUT: {json.dumps(text_inp, default=str)}"
        return [
            {"type": "text", "text": text + trailer},
            *({"type": "image_url", "image_url": {"url": url}} for url in image_urls),
        ]

    def _oracle_judge_filter_join_op(
        self, op_type: str, condition: str, samples: list, max_pairs: int | None = None,
        op_name: str = "?",
    ) -> float | None:
        """Directly ask the oracle to judge each of the plan's own filter/join inputs against
        `condition` (a single batched call — same technique as _score_map_op), then score
        agreement between the oracle's decisions and the plan's own pass/fail decisions.

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

        keys = list(candidates)
        if max_pairs is not None and len(keys) > max_pairs:
            keys = self._op_sample_rng.sample(keys, max_pairs)

        unit = "pair of records satisfies the join condition" if op_type == "sem_join" else "record satisfies the filter condition"
        content: list[dict] = [{
            "type": "text",
            "text": f"You are evaluating whether each {unit} below.\nCondition: {condition}",
        }]
        for i, key in enumerate(keys):
            content.extend(self._record_content_parts(candidates[key][0], i))

        content.append({
            "type": "text",
            "text": (
                f"\n\nFor each record, decide true if it satisfies the condition, false otherwise.\n"
                f"Return ONLY valid JSON: {{\"decisions\": [true_or_false, ...]}} "
                f"with {len(keys)} entries, in the same order as the records above."
            ),
        })

        try:
            response = self._oracle_generate(content)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                print(
                    f"[QualityEvaluator] per-op quality failed for {op_name}: oracle response "
                    f"had no JSON object (len={len(response)}): {response[:300]!r}"
                )
                return None
            decisions = json.loads(m.group()).get("decisions", [])
            if len(decisions) < len(keys):
                print(
                    f"[QualityEvaluator] per-op quality failed for {op_name}: expected "
                    f"{len(keys)} decisions, got {len(decisions)}"
                )
                return None
            return sum(
                1 for key, dec in zip(keys, decisions) if bool(dec) == candidates[key][1]
            ) / len(keys)
        except Exception as e:
            print(f"[QualityEvaluator] per-op quality failed for {op_name} ({op_type}): {e}")
            return None

    def _score_map_op(self, info: dict) -> float | None:
        """Oracle judges whether plan's sem_map output fields are correct (batched call).

        Image-valued input fields are attached to the judge call as vision
        inputs so the oracle can actually verify vision-derived output fields.
        """
        op_name = info.get("op_name", "?")
        samples = info["samples"]
        cols_names: list[str] = info["attributes"].get("cols", [])
        if not cols_names:
            print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no 'cols' in attributes")
            return None

        valid: list[tuple] = [
            (inp, out) for inp, out in samples
            if out is not None and hasattr(inp, "_source_indices")
        ]
        if not valid:
            print(f"[QualityEvaluator] per-op quality skipped for {op_name}: no samples with a non-None output")
            return None

        n_fields = len(cols_names)
        # Build an OpenAI-style multimodal content list: per-record text, with any
        # image-valued input fields attached as image_url parts right after it.
        content: list[dict] = [{
            "type": "text",
            "text": (
                "You are evaluating whether a semantic map operator produced correct outputs.\n"
                f"Output fields to evaluate: {cols_names}"
            ),
        }]
        for i, (inp, out) in enumerate(valid):
            out_d = self._dr_to_dict(out)
            mapped = {}
            for k in cols_names:
                if k not in out_d:
                    continue
                value = out_d[k]
                is_missing = value is None or (isinstance(value, float) and math.isnan(value))
                mapped[k] = "" if is_missing else value
            content.extend(self._record_content_parts(
                inp, i,
                trailer=f"\n  OPERATOR output fields {cols_names}: {json.dumps(mapped, default=str)}",
            ))

        content.append({
            "type": "text",
            "text": (
                f"\n\nFor each record and each output field, score 1 if correct, 0 if incorrect.\n"
                f"Return ONLY valid JSON: "
                f'{{\"scores\": [[field0_rec0, field1_rec0, ...], [field0_rec1, ...], ...]}} '
                f"with {len(valid)} inner lists each of length {n_fields}."
            ),
        })

        try:
            response = self._oracle_generate(content)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                print(
                    f"[QualityEvaluator] per-op quality failed for {op_name}: oracle response "
                    f"had no JSON object (len={len(response)}): {response[:300]!r}"
                )
                return None
            data = json.loads(m.group())
            scores_2d: list[list] = data.get("scores", [])
            if not scores_2d:
                print(f"[QualityEvaluator] per-op quality failed for {op_name}: 'scores' was empty/missing in {data!r}")
                return None
            n_records = len(valid)
            # Average per field across records, then average across fields
            field_avgs = []
            for j in range(n_fields):
                field_vals = [
                    float(scores_2d[r][j])
                    for r in range(min(len(scores_2d), n_records))
                    if j < len(scores_2d[r])
                ]
                if field_vals:
                    field_avgs.append(sum(field_vals) / len(field_vals))
            if not field_avgs:
                print(
                    f"[QualityEvaluator] per-op quality failed for {op_name}: got {len(scores_2d)} score "
                    f"row(s) for {n_records} record(s) x {n_fields} field(s), none usable"
                )
                return None
            return sum(field_avgs) / len(field_avgs)
        except Exception as e:
            print(f"[QualityEvaluator] per-op quality failed for {op_name} (sem_map/rag_map): {e}")
            return None
