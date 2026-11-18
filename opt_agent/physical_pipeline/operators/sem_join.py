"""LLM-based nested-loops join operator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from palimpzest.constants import Model, PromptStrategy
from palimpzest.query.operators.join import NestedLoopsJoin

from .base import Operator, _compute_op_id, _no_reasoning_prompt, _resolve_reasoning_effort

if TYPE_CHECKING:
    from ..pipeline import PhysicalPipeline


class SemJoin(Operator):
    """LLM-based nested-loops join."""
    stage_type = "join"
    op_type = "sem_join"

    def __init__(
        self,
        other: "PhysicalPipeline | None",
        condition: str,
        model: Model,
        join_parallelism: int,
        depends_on: list[str] | None,
        schema,
        reasoning_effort_override: str | None = None,
        self_join: bool = False,
    ):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        self._pz_op = NestedLoopsJoin(
            model=model,
            prompt_strategy=PromptStrategy.JOIN_NO_REASONING if _no_reasoning_prompt(model, eff) else PromptStrategy.JOIN,
            condition=condition,
            output_schema=schema,
            input_schema=schema,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.other = other
        self.self_join = self_join
        self.attributes = {"condition": condition, "model": model.value, "reasoning_effort": eff, "join_parallelism": join_parallelism, "depends_on": depends_on}
        self.op_id = _compute_op_id(self.op_type, self.attributes)
