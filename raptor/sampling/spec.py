"""The scoring spec: what to search for, over which columns, and how to weight it.

`SamplingSpec` is the only thing that differs between the three modes -- `baseline` builds it
from the raw query text, `one-shot` gets it from one LLM call, `agentic` revises it. Everything
downstream consumes a validated spec and cannot tell which mode produced it.

`validate_spec` repairs rather than rejects. A spec that names a column that does not exist, or
carries an uncompilable filter, still yields a usable spec plus a list of warnings -- shown to
the agent on its next round and recorded in the results. Sampling failure must never take down
the plan search that depends on it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from agent_cost_model.sampling.constants import DEFAULT_ALPHA, IMAGE_FIELD, MAX_KEYPHRASES
from agent_cost_model.sampling.text import all_text_columns, normalize_phrase, resolve_columns

__all__ = ["ScorerColumns", "SamplingSpec", "Stratum", "validate_spec", "validate_plan"]


@dataclass(frozen=True)
class ScorerColumns:
    bm25: tuple[str, ...] = ()
    keyphrase: tuple[str, ...] = ()
    dense: tuple[str, ...] = ()


@dataclass
class SamplingSpec:
    bm25_query: str = ""
    dense_query: str = ""
    keyphrases: tuple[str, ...] = ()
    cols: ScorerColumns = field(default_factory=ScorerColumns)
    alpha: float = DEFAULT_ALPHA
    filter: str | None = None
    rationale: str = ""

    @classmethod
    def default(
        cls,
        *,
        df: pd.DataFrame,
        id_col: str,
        query_text: str,
        alpha: float = DEFAULT_ALPHA,
        has_images: bool = False,
        text_columns: Sequence[str] | None = None,
    ) -> "SamplingSpec":
        """The baseline-mode spec: the raw query against everything, no filter, no keyphrases.

        This is baseline 1 in full -- there is no separate implementation of it, just this spec
        run through the same scoring core the other two modes use.

        `text_columns` is the caller's idea of what counts as scorable text, which is not always
        every non-numeric column: an image-only query passes `[]` so that no text is scored at
        all. Falls back to the frame's own text columns when not given.
        """
        text_cols = tuple(
            all_text_columns(df, id_col) if text_columns is None else text_columns
        )
        dense = text_cols + ((IMAGE_FIELD,) if has_images else ())
        return cls(
            bm25_query=query_text,
            dense_query=query_text,
            keyphrases=(),
            cols=ScorerColumns(bm25=text_cols, keyphrase=text_cols, dense=dense),
            alpha=alpha,
            filter=None,
            # No LLM wrote this spec, so there is no reasoning to record.
            rationale="",
        )

    def scored_components(self) -> list[str]:
        """Which scorers this spec actually turns on.

        A component needs both a query and columns to target; either one empty means the agent
        asked for it not to be scored, and it contributes nothing rather than contributing
        zeros. This is spec-level only -- `_score` decides again against the real frame, where
        a column set can still turn out to hold no usable text.
        """
        out = []
        if self.bm25_query and self.cols.bm25:
            out.append("bm25")
        if self.keyphrases and self.cols.keyphrase:
            out.append("keyphrase")
        if self.dense_query and self.cols.dense:
            out.append("dense")
        return out

    def to_json(self, configured: Sequence[str] | None = None) -> dict:
        """Serialize for the results record.

        `configured` is the set of components that were actually scored. A query string for a
        component that never ran -- baseline sets `bm25_query` even on an image-only table --
        reads as though it influenced the ranking, so it is left out. Passing None keeps
        everything, which is what the feedback report wants.
        """
        out = asdict(self)
        out["keyphrases"] = list(self.keyphrases)
        cols = {
            "bm25": list(self.cols.bm25),
            "keyphrase": list(self.cols.keyphrase),
            "dense": list(self.cols.dense),
        }
        if configured is None:
            out["cols"] = cols
            return out
        live = set(configured)
        for name, keys in (("bm25", ["bm25_query"]), ("keyphrase", ["keyphrases"]),
                           ("dense", ["dense_query"])):
            if name not in live:
                cols.pop(name, None)
                for key in keys:
                    out.pop(key, None)
        out["cols"] = cols
        if not out.get("rationale"):
            out.pop("rationale", None)
        if out.get("filter") is None:
            out.pop("filter", None)
        return out


@dataclass
class Stratum:
    """One spec plus how many of the subset's k rows it is responsible for.

    A plan is a list of these. One stratum is the ordinary case -- retrieve the k rows most
    relevant to a single idea. Several let the agent cover distinct ideas a query needs
    represented, in stated proportions, instead of collapsing them into one ranking where the
    strongest idea crowds the others out.
    """

    spec: SamplingSpec
    m: int
    label: str = ""

    def to_json(self, configured: Sequence[str] | None = None) -> dict:
        out: dict = {"m": self.m}
        if self.label:
            out["label"] = self.label
        out["spec"] = self.spec.to_json(configured)
        return out


def _repair_allocations(
    wanted: list[int], k: int, warnings: list[str]
) -> list[int]:
    """Force per-stratum allocations to sum to exactly k, preserving their proportions.

    The agent is told the m values must sum to k, but a spec that misses that constraint is
    still a usable plan -- the intent (these ideas, in roughly this ratio) survives rescaling.
    Rejecting it would throw away a good plan over arithmetic.
    """
    if not wanted:
        return []
    total = sum(wanted)
    if total == k:
        return wanted
    warnings.append(
        f"stratum allocations sum to {total}, not the required {k}; rescaled proportionally"
    )
    if total <= 0:
        # No usable ratio to preserve, so spread k as evenly as the count allows.
        base, extra = divmod(k, len(wanted))
        return [base + (1 if i < extra else 0) for i in range(len(wanted))]
    scaled = [max(1, round(w * k / total)) for w in wanted]
    # Rounding and the floor of 1 can overshoot or undershoot; settle the difference against
    # the largest strata so the smallest requested ideas keep at least one row.
    while sum(scaled) > k:
        i = max(range(len(scaled)), key=lambda j: scaled[j])
        if scaled[i] == 1:
            scaled.pop()          # more strata than rows: drop the last rather than zero one
            continue
        scaled[i] -= 1
    while sum(scaled) < k:
        i = min(range(len(scaled)), key=lambda j: scaled[j])
        scaled[i] += 1
    return scaled


def _coerce_query(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_keyphrases(value: Any, warnings: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        warnings.append(f"keyphrases: expected a list, got {type(value).__name__}; ignored")
        return ()
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            warnings.append(f"keyphrases: dropped non-string entry {raw!r}")
            continue
        # Normalizing here means the recorded spec shows exactly what will be matched, rather
        # than the raw phrasing that only looks like it will match.
        if not normalize_phrase(raw):
            continue
        out.append(raw.strip())
    out = list(dict.fromkeys(out))
    if len(out) > MAX_KEYPHRASES:
        warnings.append(
            f"keyphrases: {len(out)} given, kept the first {MAX_KEYPHRASES} — beyond that the "
            "hit count stops being a precision signal"
        )
        out = out[:MAX_KEYPHRASES]
    return tuple(out)


def _coerce_alpha(value: Any, default: float, warnings: list[str]) -> float:
    if value is None:
        return default
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        warnings.append(f"alpha: {value!r} is not a number; used {default}")
        return default
    if not 0.0 <= alpha <= 1.0:
        clamped = min(1.0, max(0.0, alpha))
        warnings.append(f"alpha: {alpha} out of range, clamped to {clamped}")
        return clamped
    return alpha


def validate_spec(
    raw: dict,
    *,
    df: pd.DataFrame,
    id_col: str,
    query_text: str,
    default_alpha: float = DEFAULT_ALPHA,
    has_images: bool = False,
    pinned_dense: tuple[str, ...] | None = None,
    text_columns: Sequence[str] | None = None,
) -> tuple[SamplingSpec, list[str]]:
    """Turn one raw LLM payload into a usable spec. Never raises.

    `pinned_dense` holds the dense column set fixed on rounds >= 2: changing it would mean
    re-embedding the table, so the refinement schema omits the field and any value the model
    sends anyway is ignored.
    """
    warnings: list[str] = []
    # One definition of "all text columns" for every default below, so a column set the caller
    # excluded cannot reappear through a field the model happened to omit.
    text_cols = list(all_text_columns(df, id_col) if text_columns is None else text_columns)
    if not isinstance(raw, dict):
        warnings.append(f"expected a JSON object, got {type(raw).__name__}; used the default spec")
        return SamplingSpec.default(
            df=df, id_col=id_col, query_text=query_text, alpha=default_alpha,
            has_images=has_images, text_columns=text_cols,
        ), warnings

    # An empty query means "do not score this component" and is honoured as written. Filling
    # it in with the raw query text, as this used to, made it impossible for the agent to turn
    # a scorer off -- it would ask for dense-only retrieval and get BM25 over the raw prompt
    # anyway. The all-empty case is caught below.
    bm25_query = _coerce_query(raw.get("bm25_query"))
    dense_query = _coerce_query(raw.get("dense_query"))

    keyphrases = _coerce_keyphrases(raw.get("keyphrases"), warnings)
    alpha = _coerce_alpha(raw.get("alpha"), default_alpha, warnings)

    bm25_cols, w = resolve_columns(
        raw.get("bm25_columns"), df=df, default=text_cols, field="bm25_columns",
        allow_empty=True,
    )
    warnings.extend(w)
    kp_cols, w = resolve_columns(
        raw.get("keyphrase_columns"), df=df, default=text_cols, field="keyphrase_columns",
        allow_empty=True,
    )
    warnings.extend(w)

    if pinned_dense is not None:
        if raw.get("dense_columns"):
            warnings.append(
                "dense_columns: ignored — the dense target columns are fixed after round 1 "
                "because changing them means re-embedding the table"
            )
        dense_cols = list(pinned_dense)
    else:
        default_dense = text_cols + ([IMAGE_FIELD] if has_images else [])
        dense_cols, w = resolve_columns(
            raw.get("dense_columns"),
            df=df,
            default=default_dense,
            field="dense_columns",
            allow_image=True,
            has_images=has_images,
            allow_empty=True,
        )
        warnings.extend(w)

    filter_expr = raw.get("filter")
    if filter_expr is not None and not isinstance(filter_expr, str):
        warnings.append(f"filter: expected a string, got {type(filter_expr).__name__}; ignored")
        filter_expr = None

    rationale = raw.get("rationale") or raw.get("reason") or ""
    spec = SamplingSpec(
        bm25_query=bm25_query,
        dense_query=dense_query,
        keyphrases=keyphrases,
        cols=ScorerColumns(
            bm25=tuple(bm25_cols), keyphrase=tuple(kp_cols), dense=tuple(dense_cols)
        ),
        alpha=alpha,
        filter=filter_expr,
        rationale=str(rationale).strip(),
    )
    # An all-empty spec is a valid answer, not a broken one: it says the query gives no reason
    # to prefer one row over another, so the caller should sample uniformly. It is deliberately
    # NOT repaired into the raw-query default -- doing that would mean the agent can never
    # decline to target, and the pressure to always return *something* is exactly what
    # manufactures spurious keyphrases on a "for each row, extract X" query.
    return spec, warnings


def validate_plan(
    raw: Any,
    *,
    df: pd.DataFrame,
    id_col: str,
    query_text: str,
    sample_size: int,
    default_alpha: float = DEFAULT_ALPHA,
    has_images: bool = False,
    pinned_dense: tuple[str, ...] | None = None,
    text_columns: Sequence[str] | None = None,
) -> tuple[list[Stratum], list[str]]:
    """Turn one raw LLM payload into a stratified sampling plan. Never raises.

    Accepts either shape:

      {"strata": [{...spec..., "m": 6, "label": "..."}, ...]}   stratified
      {...spec...}                                              one stratum, m = sample_size

    The single-spec shape is not just backward compatibility -- it is the right answer whenever
    a query has one idea, and demanding a one-element list would be noise.

    Every stratum is validated by `validate_spec`, so each is independently repairable, and a
    stratum that ends up scoring nothing is dropped rather than silently contributing its
    allocation to an arbitrary slice of the pool.
    """
    warnings: list[str] = []
    entries: list[dict]
    if isinstance(raw, dict) and isinstance(raw.get("strata"), list) and raw["strata"]:
        entries = [e for e in raw["strata"] if isinstance(e, dict)]
        if len(entries) != len(raw["strata"]):
            warnings.append("strata: dropped entries that were not JSON objects")
    else:
        entries = [raw if isinstance(raw, dict) else {}]

    if len(entries) > sample_size:
        warnings.append(
            f"strata: {len(entries)} given but only {sample_size} rows to allocate; "
            f"kept the first {sample_size}"
        )
        entries = entries[:sample_size]

    specs: list[SamplingSpec] = []
    wanted: list[int] = []
    labels: list[str] = []
    for i, entry in enumerate(entries):
        sub_warnings: list[str] = []
        spec, w = validate_spec(
            entry,
            df=df,
            id_col=id_col,
            query_text=query_text,
            default_alpha=default_alpha,
            has_images=has_images,
            pinned_dense=pinned_dense,
            text_columns=text_columns,
        )
        sub_warnings.extend(w)
        label = str(entry.get("label") or "").strip()
        try:
            m = int(entry.get("m", 0))
        except (TypeError, ValueError):
            m = 0
            sub_warnings.append(f"m: {entry.get('m')!r} is not an integer; treated as 0")
        prefix = f"stratum {i + 1}{f' ({label})' if label else ''}: "
        warnings.extend(prefix + w for w in sub_warnings)
        specs.append(spec)
        wanted.append(max(0, m))
        labels.append(label)

    # A stratum with no live scorer would hand its allocation to an arbitrary slice, so drop it
    # and let the rest absorb the rows. If that empties the plan, the caller samples uniformly.
    keep = [i for i, sp in enumerate(specs) if sp.scored_components()]
    if len(keep) != len(specs) and keep:
        dropped = [i + 1 for i in range(len(specs)) if i not in keep]
        warnings.append(
            f"strata {dropped}: no scorer is set, so they cannot rank anything; dropped and "
            "their rows given to the remaining strata"
        )
    if not keep:
        # Either a deliberate "nothing to target here" or every stratum was unusable. Both mean
        # the same thing downstream: there is no ranking, so sample uniformly.
        return [], warnings

    specs = [specs[i] for i in keep]
    wanted = [wanted[i] for i in keep]
    labels = [labels[i] for i in keep]

    if len(specs) == 1 and wanted[0] in (0, sample_size):
        allocations = [sample_size]      # the ordinary single-idea plan needs no arithmetic
    else:
        allocations = _repair_allocations(wanted, sample_size, warnings)
    specs, labels = specs[: len(allocations)], labels[: len(allocations)]

    return [
        Stratum(spec=sp, m=m, label=lb)
        for sp, m, lb in zip(specs, allocations, labels)
    ], warnings
