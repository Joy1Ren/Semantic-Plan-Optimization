"""Exact (non-LLM) row filter operator."""
from __future__ import annotations

from typing import Callable

from palimpzest.core.elements.filters import Filter
from palimpzest.query.operators.filter import NonLLMFilter

from ..base import Operator, _compute_op_id, _fn_source


class ExactFilter(Operator):
    """Exact (non-LLM) row filter."""
    stage_type = "filter"
    op_type = "filter"

    def __init__(self, fn: Callable, schema):
        super().__init__()
        self._pz_op = NonLLMFilter(
            filter=Filter(filter_fn=fn),
            output_schema=schema,
            input_schema=schema,
        )
        self._fn = fn  # preserved for make_oracle_copy
        self.attributes = {"condition": _fn_source(fn)}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
