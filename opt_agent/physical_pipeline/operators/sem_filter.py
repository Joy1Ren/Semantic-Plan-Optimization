"""LLM-based row filter operator."""
from __future__ import annotations

from palimpzest.constants import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.query.operators.filter import LLMFilter

from ..base import Operator, _compute_op_id, _resolve_reasoning_effort


class SemFilter(Operator):
    """LLM-based row filter."""
    stage_type = "filter"
    op_type = "sem_filter"

    def __init__(self, condition: str, model: Model, schema, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        self._pz_op = LLMFilter(
            model=model,
            filter=Filter(filter_condition=condition),
            output_schema=schema,
            input_schema=schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.attributes = {"condition": condition, "model": model.value, "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, {"model": model.value})
        # self.params_id = _compute_op_id(self.op_type, self.attributes)
