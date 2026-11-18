"""Summary of a Haystack adapter for plan-quality evaluation.

A plan's quality is SummHay's joint score (coverage x citation F1, averaged over insights), computed
by scoring.score_summary over the insights the documents the plan ran on hold: the optimization
subset during search, the full haystack in the final runs.

The scorer calls an LLM, SummHay's coverage judge. Its spend is tracked as quality_judge_cost_usd,
never as plan or optimization cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarks.summary_haystack.paths import TOPICS, dataset_csv, ground_truth_csv
from benchmarks.summary_haystack.scoring import (
    QualityJudge,
    agent_counts,
    exposed_insight,
    log_line,
    prompt_fill_values,
    score_summary,
    summary_bullets,
    with_example,
)
from raptor.plan_quality_evaluator import PlanQualityEvaluator

class QualityEvaluator(PlanQualityEvaluator):
    """Scores one subtopic of one haystack: the use case picks the haystack, the query id the
    subtopic."""

    # run_baseline.py scores each baseline run this many times and reports the mean, to measure
    # how much the quality judge's verdicts vary on the same summary.
    BASELINE_SCORINGS = 3

    def __init__(self, query_id: int, use_case: str, ground_truth_path: str | Path | None = None,
                 subset_path: str | Path | None = None, **_ignored) -> None:
        if use_case not in TOPICS:
            raise ValueError(f"use case {use_case!r} is not one of this benchmark's haystacks: {sorted(TOPICS)}")
        self.use_case = use_case
        self.quality_judge = QualityJudge()
        super().__init__(
            query_id=query_id,
            subset_path=subset_path,
            ground_truth_path=Path(ground_truth_path) if ground_truth_path
            else ground_truth_csv(use_case, int(query_id)),
        )

    def prompt_fill_values(self, population_path) -> dict[str, str]:
        return prompt_fill_values(self.use_case, int(self._query_id), population_path)

    def with_exposed_insight(self, plan_code: str) -> str:
        """`plan_code` with the query's prompt showing the exposed insight as an example (the
        benchmark.yaml's `expose_insight`, applied to run_baseline's plans; the insight stays in
        the score)."""
        return with_example(plan_code, exposed_insight(self._ground_truth_path)["insight"])

    def has_ground_truth(self, ground_truth_df, ground_truth_path) -> bool:
        if not ground_truth_path or not Path(ground_truth_path).exists():
            return False
        return not pd.read_csv(ground_truth_path, nrows=1).empty

    def write_scoring_input(self, plan_output_df: pd.DataFrame, path: Path) -> None:
        """The bullets the judge is shown, as read from the plan's output."""
        Path(path).write_text(json.dumps({"bullets": summary_bullets(plan_output_df)}, indent=2))

    def score_plan(self, plan_output_df, ground_truth_df, ground_truth_path, population_path) -> float:
        population = population_path or dataset_csv(self.use_case)
        result = score_summary(plan_output_df, ground_truth_path, population, self.quality_judge)
        self.last_quality_counts = agent_counts(result)
        self.last_scoring_details = result      # every insight's verdict, for run_baseline.py
        self.last_quality_judge_cost_usd = result["quality_judge_cost_usd"]
        self.last_bullet_diff = result["bullet_diff"]
        self.quality_judge_cost_usd += result["quality_judge_cost_usd"]
        print(f"[QualityEvaluator] {self.use_case} Q{self._query_id}: {log_line(result)} "
              f"(total ${self.quality_judge_cost_usd:.4f})")
        return float(result["score"])


__all__ = ["QualityEvaluator"]
