"""Plan-quality evaluation for the cost model agent.

`PlanQualityEvaluator` owns what is the same for every benchmark: loading the ground truth once
per run and turning a plan's output into a `QualityResult`. The benchmark-specific steps are hooks,
implemented in experiments/*/quality_evaluator.py:

  `score_plan(plan_output_df, ground_truth_df, ground_truth_path, population_path)` -- turn a
    plan's output plus the ground truth into a 0-1 score. The benchmarks' scorers take genuinely
    different inputs: SemBench's take a DataFrame and return its own metric dataclasses, while
    CUAD's reads a ground-truth FILE. Both forms are handed over on every call and each adapter
    uses only what its scorer needs.
  `has_ground_truth(ground_truth_df, ground_truth_path)` -- is there anything to score against?
    Asked of whichever form that benchmark actually consumes.
  `write_scoring_input(plan_output_df, path)` -- a debugging artifact of what the scorer reads.
  `prompt_fill_values(population_path)` -- placeholders to fill in every operator LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class QualityResult:
    quality: float                          # plan quality vs the ground truth (0-1, NaN if failed/N/A)
    quality_note: str | None = None         # human-readable reason when quality is not a real score
    quality_counts: dict | None = None      # TP/FP/FN/TN behind quality, for metrics that have them
    # What the benchmark's own scorer spent on LLM calls for this score (see
    # PlanQualityEvaluator.quality_judge_cost_usd). Never an optimization cost.
    quality_judge_cost_usd: float = 0.0


class PlanQualityEvaluator:
    """Benchmark-agnostic plan-quality machinery: ground-truth loading and the QualityResult.

    The one step that differs per benchmark -- turning a plan's output and the ground truth into a
    quality number -- is left to `score_plan`, implemented by each adapter in
    experiments/*/quality_evaluator.py.
    """

    def __init__(
        self,
        query_id: int,
        # The optimization subset search-time plans run on; the final evaluation passes its own
        # dataset to score_plan instead.
        subset_path: str | Path | None = None,
        normalize_df: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        ground_truth_loader: Callable[[], pd.DataFrame] | None = None,
        ground_truth_path: str | Path | None = None,
    ) -> None:
        self._query_id = query_id
        self._subset_path = Path(subset_path) if subset_path else None
        # Identity when the adapter supplies none: a benchmark whose scorer already accepts the
        # plan's natural output has nothing to reshape.
        self._normalize_df_fn = normalize_df if normalize_df is not None else (lambda df: df)
        # What the benchmark's OWN scorer spent calling an LLM -- a metric that is itself
        # LLM-judged (SummHay's coverage judge). It is the cost of measuring quality, so it is
        # tracked but never counted as optimization or plan cost. `last_...` is the most recent
        # score_plan call's share; an adapter whose scorer calls no LLM leaves both at 0.
        self.quality_judge_cost_usd = 0.0
        self.last_quality_judge_cost_usd = 0.0
        # SummHay: the last scored summary's insight bullets minus [N_insights]; None elsewhere.
        self.last_bullet_diff = None
        # The judge itself, for adapters that have one: exposes `.model` and `.reasoning_effort`,
        # which the run's metrics record next to its spend.
        self.quality_judge = getattr(self, "quality_judge", None)
        # Set by evaluate() on each call, for per-plan debug dumps.
        self.last_ground_truth_df: "pd.DataFrame | None" = None

        self._ground_truth_loader = ground_truth_loader
        self._ground_truth_df: "pd.DataFrame | None" = None
        self._ground_truth_loaded = False
        # The benchmark's ground-truth file (benchmark.yaml's ground_truth_path), handed straight
        # to the scorer -- nothing is copied.
        self._ground_truth_path = Path(ground_truth_path) if ground_truth_path else None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, plan_output_df: pd.DataFrame) -> QualityResult:
        """Score a plan's output (already normalized) on the optimization subset."""
        gt_df = self._ground_truth()
        self.last_ground_truth_df = gt_df

        quality = float("nan")
        self.last_quality_counts = None     # score_plan sets it when its metric has counts
        self.last_quality_judge_cost_usd = 0.0
        quality_note = None
        if not self.has_ground_truth(gt_df, self._ground_truth_path):
            # Nothing to score the plan against — quality is undefined. Keep it NaN but attach
            # a note so the agent's observation can explain why (rather than showing a bare NaN).
            quality_note = "ground truth is empty or unavailable, so plan quality is N/A"
        else:
            try:
                quality = self.score_plan(
                    plan_output_df, gt_df, self._ground_truth_path, self._subset_path
                )
            except Exception as e:
                print(f"[QualityEvaluator] plan quality evaluation failed: {e}")

        return QualityResult(quality=quality, quality_note=quality_note,
                             quality_counts=self.last_quality_counts,
                             quality_judge_cost_usd=self.last_quality_judge_cost_usd)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reshape a plan's raw output into what this benchmark's scorer reads.

        The adapter supplies the reshaping function as `normalize_df` and binds whatever it needs
        (SemBench's takes the use case and query id); everything that produces a frame to be
        scored -- a plan's output on the subset or on the full dataset -- goes through here, so
        there is one reshaping step rather than one per call site.
        """
        return self._normalize_df_fn(df)

    # ------------------------------------------------------------------
    # Benchmark-specific hooks (implemented by each adapter)
    # ------------------------------------------------------------------

    def prompt_fill_values(self, population_path) -> dict[str, str]:
        """Placeholders to fill in every operator LLM call of a plan running over
        `population_path`, as {placeholder: value} (see llm_client.set_prompt_fill).

        For a benchmark whose prompt states a count of the documents it runs over -- SummHay's
        [N_insights] -- which differs between the optimization subset and the full dataset. The
        runner sets it before the search and again before the full-dataset runs. None by default.
        """
        return {}

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
        the implementation names only the ones it uses.

        `population_path` is the dataset the plan being scored actually ran over -- the
        optimization subset during search, the full dataset for the final evaluation. A scorer
        that measures over a document population takes it from there rather than from the plan's
        own output, so a plan that drops documents is charged for them.

        A scorer whose metric is built from TP/FP/FN/TN counts also sets `self.last_quality_counts`
        to them (split by reason where a Jaccard gate applies; see prompts._QUALITY_METRIC_DOCS),
        and `evaluate` shows them to the agent next to `quality`.
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

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _ground_truth(self) -> "pd.DataFrame | None":
        """The benchmark's ground truth as a frame, loaded once per run and reused across plans.

        The benchmark's `ground_truth_loader` produces it -- SemBench regenerates the gold query
        over the datasubset -- and an adapter whose scorer reads the ground-truth FILE supplies no
        loader, so this is None for it. Restricting the ground truth to the optimization subset is
        the loader's job: there is no id column that means the same thing across benchmarks.
        """
        if not self._ground_truth_loaded:
            self._ground_truth_loaded = True
            if self._ground_truth_loader is not None:
                try:
                    self._ground_truth_df = self._ground_truth_loader()
                except Exception as e:
                    print(f"[QualityEvaluator] ground-truth load failed: {e}")
        return self._ground_truth_df
