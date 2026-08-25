"""LLM-based column derivation operator."""
from __future__ import annotations

from palimpzest.constants import Model, PromptStrategy
from palimpzest.query.operators.convert import LLMConvertBonded

from ..base import Operator, _compute_op_id, _has_image_field, _resolve_reasoning_effort


class SemMap(Operator):
    """LLM-based column derivation."""
    stage_type = "convert"
    op_type = "sem_map"

    def __init__(self, cols: list[dict], model: Model, input_schema, output_schema, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        # Installed palimpzest (1.5.3) collapsed COT_QA/COT_QA_IMAGE into a single MAP
        # strategy -- there's no separate *_IMAGE variant; the Generator now detects image
        # fields from the schema itself rather than from prompt_strategy.
        is_image = _has_image_field(input_schema, depends_on)  # noqa: F841 -- kept for parity/future use
        prompt_strategy = PromptStrategy.MAP_NO_REASONING if (model.is_reasoning_model() and eff in (None, "minimal", "low", "disable")) else PromptStrategy.MAP
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
        self._cols_full = cols  # preserved for make_oracle_copy
        self.attributes = {"model": model.value, "cols": sorted(col["name"] for col in cols), "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
