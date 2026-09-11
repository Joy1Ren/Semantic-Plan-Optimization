"""The scoring core, shared unchanged by all three sampling modes.

    dense  = minmax(mean over dense field-groups of cosine(embed(dense_query), group_vec))
    sparse = minmax( minmax(BM25(bm25_query, bm25 docs)) + n_keyphrase_hits )
    score  = alpha*sparse + (1 - alpha)*dense

The two-stage sparse normalization is deliberate: `minmax(BM25)` alone lands in [0,1] while the
hit count does not, so adding them raw would let a single keyphrase hit outweigh the entire
BM25 range and turn `alpha` into a no-op. Renormalizing the sum keeps both components on the
same scale, which is what makes `alpha` mean what it looks like it means.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from agent_cost_model.sampling.text import bm25_tokenize, normalize_phrase

__all__ = [
    "ScoredPool",
    "minmax",
    "bm25_scores",
    "dead_bm25_terms",
    "keyphrase_hits",
    "cosine_scores",
    "combine",
    "select_topk",
    "percentiles",
]

_PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)


@dataclass
class ScoredPool:
    """Every per-row quantity the report, the results record, and selection need."""

    ids: list[str]
    bm25_raw: np.ndarray
    bm25_norm: np.ndarray
    hits: np.ndarray
    hit_names: list[list[str]]
    dense_raw: np.ndarray
    dense_norm: np.ndarray
    sparse: np.ndarray
    total: np.ndarray
    alpha: float
    live: list[str] = field(default_factory=list)
    # `configured` is what the spec asked to be scored; `live` is the subset of that which came
    # back non-degenerate. The distinction matters for reporting: a BM25 pass that ran and
    # matched nothing is a finding (see `dead_terms`), while one that was never set up -- an
    # image-only query has no text to match -- has nothing to say and is omitted entirely.
    configured: list[str] = field(default_factory=list)
    dead_terms: list[str] = field(default_factory=list)
    zero_hit_keyphrases: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1]; return all zeros when every value is equal.

    An all-equal component carries no ranking information, so it must contribute nothing.
    Returning ones instead would let a degenerate component dominate the blend -- e.g. a BM25
    query matching no document would add a constant `alpha` to every row and swamp nothing,
    but a *dense* component stuck at 1.0 would erase the sparse ranking entirely.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def bm25_scores(query: str, docs: Sequence[str]) -> np.ndarray:
    """BM25 relevance of `query` against each document.

    Scores can come back negative -- rank_bm25's IDF floor allows it -- and that is fine:
    `minmax` maps the minimum to 0 either way, and clipping would flatten real distinctions
    among the weakest rows.
    """
    if not docs:
        return np.zeros(0, dtype=np.float64)
    tokenized = [bm25_tokenize(d) for d in docs]
    query_tokens = bm25_tokenize(query or "")
    # BM25Okapi divides by the average document length, so an entirely empty corpus is a
    # ZeroDivisionError rather than a degenerate-but-valid ranking.
    if not query_tokens or not any(tokenized):
        return np.zeros(len(docs), dtype=np.float64)
    from rank_bm25 import BM25Okapi

    return np.asarray(BM25Okapi(tokenized).get_scores(query_tokens), dtype=np.float64)


def dead_bm25_terms(query: str, docs: Sequence[str]) -> list[str]:
    """Query terms that appear in no document at all.

    A zero-document-frequency term contributes nothing to BM25 and is the most common way a
    hand-written `bm25_query` silently fails, so the agent is shown these by name.
    """
    query_tokens = list(dict.fromkeys(bm25_tokenize(query or "")))
    if not query_tokens:
        return []
    vocabulary: set[str] = set()
    for doc in docs:
        vocabulary.update(bm25_tokenize(doc))
    return [t for t in query_tokens if t not in vocabulary]


def keyphrase_hits(
    phrases: Sequence[str], padded_texts: Sequence[str]
) -> tuple[np.ndarray, list[list[str]]]:
    """Per row: how many distinct keyphrases matched, and which ones.

    Each phrase counts at most once per row regardless of how often it occurs -- this is a
    "does the row contain the thing" signal, not a term-frequency one (BM25 already covers
    frequency).
    """
    n = len(padded_texts)
    if not phrases:
        return np.zeros(n, dtype=np.float64), [[] for _ in range(n)]
    needles = [(p, f" {normalize_phrase(p)} ") for p in phrases if normalize_phrase(p)]
    counts = np.zeros(n, dtype=np.float64)
    names: list[list[str]] = []
    for i, text in enumerate(padded_texts):
        matched = [original for original, needle in needles if needle in text]
        names.append(matched)
        counts[i] = len(matched)
    return counts, names


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against each row of `matrix`."""
    if matrix.size == 0:
        return np.zeros(matrix.shape[0] if matrix.ndim == 2 else 0, dtype=np.float64)
    q = np.asarray(query_vec, dtype=np.float64)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1)
    # A zero-norm row (nothing embedded) scores 0 rather than producing a nan.
    safe = np.where(norms == 0.0, 1.0, norms)
    # errstate, not a correctness guard: some BLAS backends (Accelerate on macOS) raise
    # spurious divide-by-zero/overflow/invalid FPE flags from matmul even for finite,
    # unit-norm inputs. Left unsuppressed they print on every query, and under `-W error`
    # they abort scoring outright. The `nan_to_num` below is the actual safety net.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = (m @ q) / (safe * qn)
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(norms == 0.0, 0.0, sims)


def combine(
    *,
    ids: Sequence[str],
    bm25_raw: np.ndarray,
    hits: np.ndarray,
    hit_names: list[list[str]],
    dense_raw: np.ndarray,
    alpha: float,
    live: Sequence[str],
    configured: Sequence[str] = (),
    dead_terms: Sequence[str] = (),
    zero_hit_keyphrases: Sequence[str] = (),
) -> ScoredPool:
    bm25_norm = minmax(bm25_raw)
    sparse = minmax(bm25_norm + hits)
    dense_norm = minmax(dense_raw)
    # The blend runs only over the halves that carry a signal. A half that does not contributes
    # nothing rather than a constant zero: with alpha=0.5 and no sparse scorer, `alpha*0 +
    # 0.5*dense` caps every recorded score at 0.5 and makes the percentiles unreadable. The
    # ranking is unchanged either way -- this is about the recorded numbers meaning what they
    # say.
    #
    # Read from the arrays, not from `configured`, for two reasons: a caller cannot forget to
    # pass it and silently get an all-zero ranking, and `minmax` already returns zeros for a
    # component that ran but is degenerate -- a BM25 pass matching nothing, or one where every
    # row ties. Those carry no ranking information and should not consume weight either.
    has_sparse = bool(np.any(sparse != 0.0))
    has_dense = bool(np.any(dense_norm != 0.0))
    if has_sparse and has_dense:
        total = alpha * sparse + (1.0 - alpha) * dense_norm
    elif has_sparse:
        total = sparse
    elif has_dense:
        total = dense_norm
    else:
        total = np.zeros(len(ids), dtype=np.float64)
    return ScoredPool(
        ids=list(ids),
        bm25_raw=np.asarray(bm25_raw, dtype=np.float64),
        bm25_norm=bm25_norm,
        hits=np.asarray(hits, dtype=np.float64),
        hit_names=hit_names,
        dense_raw=np.asarray(dense_raw, dtype=np.float64),
        dense_norm=dense_norm,
        sparse=sparse,
        total=total,
        alpha=alpha,
        live=list(live),
        configured=list(configured),
        dead_terms=list(dead_terms),
        zero_hit_keyphrases=list(zero_hit_keyphrases),
    )


def select_topk(pool: ScoredPool, k: int, exclude: Sequence[str] = ()) -> list[str]:
    """The k highest-scoring ids, skipping any in `exclude`.

    Ties need no tie-break: when a tie straddles the cutoff, whichever of the tied rows fills
    k is as good as any other, so `argsort`'s ordering is accepted as-is.

    `exclude` is what makes strata compose: each one draws its own m rows from the same pool,
    and a row already claimed by an earlier stratum is passed over rather than filling two
    allocations with one row.
    """
    if k <= 0 or not pool.ids:
        return []
    order = np.argsort(-pool.total, kind="stable")
    if not exclude:
        return [pool.ids[i] for i in order[:k]]
    skip = set(exclude)
    out: list[str] = []
    for i in order:
        rid = pool.ids[i]
        if rid in skip:
            continue
        out.append(rid)
        if len(out) == k:
            break
    return out


def percentiles(x: np.ndarray, ps: Sequence[int] = _PERCENTILES) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in ps}
    values = np.percentile(arr, ps)
    return {f"p{p}": round(float(v), 6) for p, v in zip(ps, values)}


def summarize(pool: ScoredPool) -> dict:
    """The per-round score summary written to sampling_results.json.

    Only the components the spec actually asked for appear. Emitting all of them unconditionally
    filled the record with all-zero percentiles and empty term lists for passes that never ran
    -- an image-only query would carry nine BM25 percentiles of 0.0 and a keyphrase histogram
    reading `{"0": 500}`, which looks like a result and is merely an absence.
    """
    def _range(x: np.ndarray) -> dict[str, float]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return {"min": 0.0, "max": 0.0}
        return {"min": round(float(arr.min()), 6), "max": round(float(arr.max()), 6)}

    configured = set(pool.configured)
    out: dict = {"n_pool": len(pool), "live_components": pool.live}
    if "bm25" in configured:
        out["bm25_norm_pct"] = percentiles(pool.bm25_norm)
        out["bm25_raw_range"] = _range(pool.bm25_raw)
        out["dead_bm25_terms"] = pool.dead_terms
    if "keyphrase" in configured:
        histogram: dict[str, int] = {}
        for h in pool.hits.astype(int).tolist():
            histogram[str(h)] = histogram.get(str(h), 0) + 1
        out["keyphrase_hit_histogram"] = histogram
        out["zero_hit_keyphrases"] = pool.zero_hit_keyphrases
    if "dense" in configured:
        out["dense_norm_pct"] = percentiles(pool.dense_norm)
        out["dense_raw_range"] = _range(pool.dense_raw)
    out["total_pct"] = percentiles(pool.total)
    return out
