"""Compiling an LLM-written row predicate into a safe callable.

Lifted from `LLM_Sampler._compile_filter` with one change: the compile error is *returned*
rather than swallowed, because the agentic loop shows it to the model so it can fix the
expression on the next round.

Trust model: the expression comes from the LLM and is evaluated with empty `__builtins__`.
That is the same seam the rest of the codebase already accepts -- `CostModelAgent` executes
LLM-written Python in `local_python_executor.LocalPythonExecutor` -- so this is not a new
exposure, just a much narrower one (a single boolean expression over one row).
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

__all__ = ["CompiledFilter", "compile_filter", "filter_mask"]

_SAFE_GLOBALS: dict[str, Any] = {"__builtins__": {}}


class CompiledFilter:
    """A row predicate that never raises: a row it cannot evaluate simply does not match."""

    __slots__ = ("expr", "_fn")

    def __init__(self, expr: str, fn: Callable[[Any], Any]) -> None:
        self.expr = expr
        self._fn = fn

    def __call__(self, row: pd.Series) -> bool:
        try:
            return bool(self._fn(row))
        except Exception:
            return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CompiledFilter({self.expr!r})"


def compile_filter(expr: Any) -> tuple[CompiledFilter | None, str | None]:
    """Compile a filter expression. Returns (filter, error).

    A null/blank expression is "no filter" -- (None, None), not an error. A malformed one is
    (None, "<message>"), which the caller records and shows to the agent.
    """
    if expr is None:
        return None, None
    if not isinstance(expr, str):
        return None, f"expected a string expression, got {type(expr).__name__}"
    text = expr.strip()
    if not text or text.lower() in ("null", "none"):
        return None, None

    try:
        if text.startswith("lambda"):
            raw = eval(text, _SAFE_GLOBALS, {})  # noqa: S307 - empty __builtins__
            if not callable(raw):
                return None, "lambda expression did not evaluate to a callable"
            return CompiledFilter(text, raw), None
        code = compile(text, "<filter>", "eval")
        return CompiledFilter(text, lambda row, _c=code: eval(_c, _SAFE_GLOBALS, {"row": row})), None  # noqa: S307
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def filter_mask(df: pd.DataFrame, flt: CompiledFilter | None) -> pd.Series:
    """Boolean mask of the rows `flt` accepts; all-True when there is no filter."""
    if flt is None:
        return pd.Series(True, index=df.index)
    return df.apply(flt, axis=1).astype(bool)
