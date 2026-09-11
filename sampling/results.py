"""What a sampling run produces: the subset CSV, the results record, and the round trace.

Three artifacts, deliberately separated:

  subset CSV            the actual product -- the rows every candidate plan is scored on.
  sampling_results.json one record per (query, mode): specs, score summaries, costs. Small
                        enough to read, and nested by mode so the three are comparable.
  rounds/Q{id}_{mode}   full prompts, raw replies, and rendered feedback reports. These run to
                        100k+ characters and would make the results file unreadable.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

__all__ = [
    "RoundRecord",
    "StratumRecord",
    "CostLedger",
    "SamplingResult",
    "mode_subset_path",
    "write_subset",
    "write_scores_csv",
    "merge_results_json",
    "write_rounds_trace",
]


@dataclass
class CostLedger:
    """Every dollar the sampler spends, split by what it was spent on.

    `embedding_rows_sunk` is the crucial distinction: row embeddings are cached per (use case,
    model, column set) and shared across every query of that use case, so a run that reuses
    them pays nothing. The old sampler reported that sunk total as this run's cost, which made
    the sampler look orders of magnitude more expensive than it is. `total_paid_now` is the
    number a mode comparison should use.
    """

    llm: float = 0.0
    llm_by_round: list[float] = field(default_factory=list)
    image_describe: float = 0.0
    embedding_query: float = 0.0
    embedding_rows: float = 0.0
    embedding_rows_sunk: float = 0.0
    n_row_embeddings_computed: int = 0
    n_row_embeddings_reused: int = 0

    llm_latency: float = 0.0
    # `embedding_latency` is the SUM of per-call service times (what a serial run would have
    # spent); `embedding_wall` is elapsed wall clock. With N embedding workers the two diverge
    # by roughly N, so both are kept and `embed_workers` is recorded alongside -- neither
    # number is comparable across runs without the concurrency that produced it.
    embedding_latency: float = 0.0
    embedding_wall: float = 0.0
    # Serial time the cache saved: what the reused vectors took when first computed.
    embedding_sunk_latency: float = 0.0
    scoring_latency: float = 0.0
    # Measured start-to-end, not summed from the components above.
    total_wall: float = 0.0
    embed_workers: int = 1

    def to_json(self, mode: str = "") -> dict:
        """The cost lines this mode can actually incur.

        `baseline` makes no LLM call, and only `agentic` runs more than one round or describes
        images. Emitting those lines everywhere puts zeros in the record that read as "this ran
        and was free" rather than "this never ran".
        """
        paid = self.llm + self.image_describe + self.embedding_query + self.embedding_rows
        out: dict = {}
        if mode != "baseline":
            out["llm_usd"] = round(self.llm, 8)
        if mode == "agentic":
            out["llm_usd_by_round"] = [round(x, 8) for x in self.llm_by_round]
            out["image_describe_usd"] = round(self.image_describe, 8)
        out.update({
            "embedding_query_usd": round(self.embedding_query, 8),
            "embedding_rows_usd": round(self.embedding_rows, 8),
            "embedding_rows_usd_sunk": round(self.embedding_rows_sunk, 8),
            "total_usd_paid_now": round(paid, 8),
            "n_row_embeddings_computed": self.n_row_embeddings_computed,
            "n_row_embeddings_reused": self.n_row_embeddings_reused,
        })
        return out

    def latency_json(self, mode: str = "") -> dict:
        out = {}
        if mode != "baseline":
            out["llm_s"] = round(self.llm_latency, 4)
        out.update({
            "embed_workers": self.embed_workers,
            # Elapsed vs. summed per-call time. At embed_workers=1 the two coincide; at N they
            # differ by roughly N, which is the point of recording the worker count with them.
            "embedding_wallclock_s": round(self.embedding_wall, 4),
            "embedding_iterative_s": round(self.embedding_latency, 4),
            # What a cold, serial run would have spent = iterative + sunk. On a warm cache
            # `iterative` collapses to the query embeddings alone, so without this the record
            # shows no trace of the row-embedding work the cache skipped.
            "embedding_sunk_s": round(self.embedding_sunk_latency, 4),
            "scoring_s": round(self.scoring_latency, 4),
            # Start to end for the whole run, measured. It exceeds the sum of the parts above,
            # and should: the components do not time cache load/save, LLM retries, image
            # description, or frame work.
            "total_s": round(self.total_wall, 4),
        })
        return out


@dataclass
class StratumRecord:
    """What one stratum of a round asked for, and what it retrieved.

    Every round has at least one. Keeping the per-stratum scores separate is the point of
    stratifying: a pooled summary would average away exactly the differences the agent split
    the subset to capture.
    """

    m: int
    spec: dict
    label: str = ""
    filter_info: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    chosen_ids: list[str] = field(default_factory=list)
    chosen_scores: list[dict] = field(default_factory=list)

    def to_json(self, mode: str = "") -> dict:
        out = asdict(self)
        if not self.label:
            out.pop("label", None)
        if mode == "baseline":
            out.pop("filter_info", None)
        return out


@dataclass
class RoundRecord:
    round: int
    spec_raw: dict = field(default_factory=dict)
    strata: list[StratumRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1
    chosen_ids: list[str] = field(default_factory=list)
    # Rows no stratum selected, drawn uniformly to top the subset up to k. They belong to the
    # plan rather than to any one stratum, so they are recorded here instead of being
    # attributed to whichever stratum happened to run last.
    fill_ids: list[str] = field(default_factory=list)
    llm_cost_usd: float = 0.0
    llm_latency_s: float = 0.0
    action: str = ""
    reason: str = ""

    def to_json(self, mode: str = "") -> dict:
        """The round's fields that this mode can populate.

        Baseline plans nothing, so it has no raw LLM reply, no spec-repair warnings, no parse
        retries, no filter and no per-round LLM cost; only agentic decides between rounds, so
        only agentic has an action and a reason for it.
        """
        out = asdict(self)
        out["strata"] = [s.to_json(mode) for s in self.strata]
        if not self.fill_ids:
            out.pop("fill_ids", None)
        if mode == "baseline":
            for key in ("spec_raw", "warnings", "attempts", "llm_cost_usd", "llm_latency_s"):
                out.pop(key, None)
        if mode != "agentic":
            out.pop("action", None)
            out.pop("reason", None)
        return out


@dataclass
class SamplingResult:
    sampled_ids: list[str]
    subset_path: pathlib.Path
    results_path: pathlib.Path
    scores_path: pathlib.Path
    mode: str
    rounds: list[RoundRecord] = field(default_factory=list)
    accepted_round: int = 1
    stop_reason: str = "accepted"
    cost: CostLedger = field(default_factory=CostLedger)
    record: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sampled_ids)


def mode_subset_path(path: pathlib.Path | str, mode: str) -> pathlib.Path:
    """`.../Q6_subset.csv` -> `.../Q6_subset_agentic.csv`: one subset file per sampling mode.

    Modes are compared against each other on the same query, so a single shared filename means
    the last run silently overwrites the subset every earlier mode was scored on.
    """
    p = pathlib.Path(path)
    return p.with_name(f"{p.stem}_{mode}{p.suffix}")


def write_subset(
    df: pd.DataFrame, id_col: str, sampled_ids: list[str], out_path: pathlib.Path
) -> None:
    """Write the sampled rows, in selection order, to the benchmark's subset path.

    Written on every code path including the error fallback: downstream quality scoring reads
    this file, so leaving a previous run's subset in place would silently score new plans
    against stale rows.
    """
    order = {rid: pos for pos, rid in enumerate(sampled_ids)}
    subset = df[df[id_col].isin(set(order))].copy()
    subset = (
        subset.assign(_ord=subset[id_col].map(order))
        .sort_values("_ord")
        .drop_duplicates(subset=id_col, keep="first")
        .drop(columns="_ord")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)


def write_scores_csv(path: pathlib.Path, frames: list[pd.DataFrame]) -> None:
    """One row per scored row per round: every subscore, its normalized form, and the total.

    Kept out of the results JSON deliberately -- this is thousands of rows per query, and it is
    meant to be read with pandas, not by eye. The scalar hyperparameters (mode, alpha, the spec
    strings) are repeated on every row so the file stands on its own in an analysis notebook
    without having to be joined back to sampling_results.json.
    """
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    """Write via a pid-suffixed temp file plus os.replace.

    Queries of one use case are often run in parallel and share this file; a plain open-write
    leaves a truncated file visible to a concurrent reader and loses the loser's record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def merge_results_json(
    path: pathlib.Path, query_id: str, mode: str, record: dict
) -> None:
    """Merge one record into `results[query_id][mode]`, preserving everything else.

    Records written by the previous sampler were stored flat at `results[query_id]`. Those are
    migrated to `results[query_id]["legacy-topk"]` on first touch rather than overwritten --
    the runs they describe still happened, and the analysis scripts can still read them.
    """
    results: dict = {}
    if path.exists():
        try:
            with open(path) as fh:
                results = json.load(fh)
        except Exception:
            results = {}
    if not isinstance(results, dict):
        results = {}

    existing = results.get(str(query_id))
    if isinstance(existing, dict) and "sampled_ids" in existing:
        # A flat, pre-nesting record: the modes it could have been written under are gone, so
        # label it by what the old sampler actually did.
        existing = {existing.get("sample_method") or "legacy-topk": existing}
    elif not isinstance(existing, dict):
        existing = {}

    existing[mode] = record
    results[str(query_id)] = existing
    _atomic_write_json(path, results)


def write_rounds_trace(path: pathlib.Path, payload: dict) -> None:
    _atomic_write_json(path, payload)
