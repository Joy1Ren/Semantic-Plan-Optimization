"""Text materialization, normalization, and column bookkeeping.

Pure functions over a DataFrame and its rows: no network, no LLM, no palimpzest. Everything
here is shared by the BM25, keyphrase, and embedding paths so the three agree on exactly what
"the text of a row over these columns" means.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from agent_cost_model.sampling.constants import IMAGE_FIELD

__all__ = [
    "bm25_tokenize",
    "normalize_phrase",
    "row_text",
    "row_keyphrase_text",
    "all_text_columns",
    "resolve_columns",
    "column_fingerprint",
    "norm_id",
    "norm_id_series",
    "truncate_middle",
]

# Duplicated from `physical_pipeline/operators/rag_common.py::_rag_bm25_tokenize` rather than
# imported: that module does `from palimpzest.constants import Model` at module scope, and this
# package must stay importable -- and its CLI and the analysis scripts runnable -- without
# paying for a full palimpzest import. Keep the two byte-identical: both mirror docetl's
# `operations/sample.py::_sample_top_fts` preprocessing, so BM25 here ranks the way the
# doc_chunking_topk directive does.
_BM25_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric characters, split on whitespace."""
    return _BM25_NON_ALNUM.sub(" ", text.lower()).split()


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_phrase(text: str) -> str:
    """Space-joined lowercase full-word tokens, punctuation stripped.

    Normalizing a phrase and a row's text the same way turns a substring test into full-word
    phrase matching: 'dressing' stays 'dressing' (so a 'dress' keyphrase does not match it),
    and both 'crew-neck' and 'crew neck' become 'crew neck', so the phrase matches only where
    the two words appear contiguously.
    """
    return " ".join(_WORD_RE.findall(text.lower()))


def row_text(row: pd.Series, cols: Sequence[str]) -> str:
    """The row's text over `cols`, in column order, skipping missing cells.

    Cells are skipped rather than stringified, so a NaN never contributes the literal token
    'nan' to a BM25 document or an embedded string.
    """
    parts = []
    for col in cols:
        if col == IMAGE_FIELD or col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        parts.append(str(value))
    return " ".join(parts)


def row_keyphrase_text(row: pd.Series, cols: Sequence[str]) -> str:
    """Normalized, space-padded token string of a row's text over `cols`.

    The padding makes `f" {phrase} " in text` an exact full-word / phrase match at any
    position, including the very start and end.
    """
    return f" {normalize_phrase(row_text(row, cols))} "


def all_text_columns(df: pd.DataFrame, id_col: str) -> list[str]:
    """Every non-numeric column except the id column, in DataFrame order."""
    return [
        c
        for c in df.columns
        if c != id_col and not pd.api.types.is_numeric_dtype(df[c])
    ]


def resolve_columns(
    requested: Any,
    *,
    df: pd.DataFrame,
    default: Sequence[str],
    field: str,
    allow_image: bool = False,
    has_images: bool = False,
    allow_empty: bool = False,
) -> tuple[list[str], list[str]]:
    """Coerce and validate one scorer's column list. Returns (columns, warnings).

    Repairs rather than rejects: a missing/null/"all" value falls back to `default`, a bare
    string is wrapped, and unknown names are dropped.

    With `allow_empty`, an explicitly empty list is honoured as "do not score this component"
    rather than repaired into `default`. Omitting the field and passing `[]` are then different
    requests: the first means "no preference", the second means "off". A list whose names were
    all unknown still falls back to `default` -- that is a mistake to repair, not a choice.
    """
    warnings: list[str] = []
    if requested is None or (isinstance(requested, str) and requested.strip().lower() == "all"):
        return list(default), warnings
    if allow_empty and isinstance(requested, (list, tuple)) and len(requested) == 0:
        return [], warnings
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, (list, tuple)):
        warnings.append(f"{field}: expected a list of column names, got {type(requested).__name__}; used all text columns")
        return list(default), warnings

    known = set(df.columns)
    out: list[str] = []
    for raw in requested:
        name = str(raw)
        if name == IMAGE_FIELD:
            if not allow_image:
                warnings.append(f"{field}: {IMAGE_FIELD} is only valid for dense_columns; dropped")
            elif not has_images:
                warnings.append(f"{field}: {IMAGE_FIELD} dropped, this query has no images")
            else:
                out.append(name)
            continue
        if name not in known:
            warnings.append(f"{field}: unknown column {name!r} dropped")
            continue
        out.append(name)

    # Preserve order but drop repeats, so a fingerprint is stable and a column is not embedded
    # into the same document twice.
    out = list(dict.fromkeys(out))
    if not out:
        warnings.append(f"{field}: no usable columns left; used all text columns")
        return list(default), warnings
    return out, warnings


def column_fingerprint(cols: Iterable[str]) -> str:
    """Stable short hash of an ORDERED column list.

    Order matters: the embedded text is a `" ".join` of the cells in this order, so two
    different orders produce different vectors and must not share a cache group.
    """
    joined = "\x1f".join(str(c) for c in cols)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


_seen_float_id_warning = False


def norm_id(value: Any) -> str:
    """The single id-normalization choke point.

    A float64 id column would otherwise stringify as '10110.0', which never matches the
    int64-derived '10110' keys already written into the embedding cache. Integral floats lose
    the trailing '.0'; everything else is passed through as-is.
    """
    global _seen_float_id_warning
    if isinstance(value, float):
        if value.is_integer():
            if not _seen_float_id_warning:
                _seen_float_id_warning = True
                print(
                    "[sampling] id column holds floats; normalizing e.g. 10110.0 -> '10110' so "
                    "ids match the embedding cache. Read the CSV with dtype={id_col: str} to "
                    "avoid the conversion entirely."
                )
            return str(int(value))
        return str(value)
    return str(value)


def norm_id_series(series: pd.Series) -> pd.Series:
    return series.map(norm_id)


def truncate_middle(text: str, allowance: int) -> str:
    """Keep the head and tail of `text`, labeling how much was dropped.

    Head-and-tail rather than a plain prefix so the reader sees how a document opens *and*
    whether it ends mid-sentence.
    """
    if allowance <= 0 or len(text) <= allowance:
        return text
    marker_len = 40  # rough width of the "[N chars omitted]" marker
    if allowance <= marker_len:
        return text[:allowance]
    keep = allowance - marker_len
    head = (keep * 2) // 3
    tail = keep - head
    omitted = len(text) - head - tail
    tail_part = text[-tail:] if tail else ""
    return f"{text[:head]} …[{omitted:,} chars omitted]… {tail_part}"
