"""LLM-based column derivation operator."""
from __future__ import annotations

from palimpzest.constants import Model, PromptStrategy
from palimpzest.query.operators.convert import LLMConvertBonded

from .base import Operator, _cols_attribute, _compute_op_id, _no_reasoning_prompt, _resolve_reasoning_effort


class SemMap(Operator):
    """LLM-based column derivation."""
    stage_type = "convert"
    op_type = "sem_map"

    def __init__(self, cols: list[dict], model: Model, input_schema, output_schema, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None, skip_if=None):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        prompt_strategy = PromptStrategy.MAP_NO_REASONING if _no_reasoning_prompt(model, eff) else PromptStrategy.MAP
        self._pz_op = LLMConvertBonded(
            model=model,
            prompt_strategy=prompt_strategy,
            output_schema=output_schema,
            input_schema=input_schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.attributes = {"model": model.value, "reasoning_effort": eff, "cols": _cols_attribute(cols), "depends_on": depends_on}
        self.op_id = _compute_op_id(self.op_type, self.attributes)
        self._set_skip_if(skip_if)
