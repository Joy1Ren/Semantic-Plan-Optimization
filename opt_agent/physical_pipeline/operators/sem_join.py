"""LLM-based nested-loops join operator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from palimpzest.constants import Model
from palimpzest.query.operators.join import NestedLoopsJoin

from ..base import Operator, _compute_op_id, _resolve_reasoning_effort

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
            condition=condition,
            output_schema=schema,
            input_schema=schema,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        # For a self-join `other` is None: the left upstream is run ONCE and its output is
        # joined with itself (see PhysicalPipeline.sem_join / _execute_core). This is a
        # common-subexpression optimization *beyond* PZ — PZ's Cascades groups dedupe only
        # in the optimizer memo, and its extracted physical plan runs the upstream twice.
        self.other = other
        self.self_join = self_join
        self.attributes = {"condition": condition, "model": model.value, "reasoning_effort": eff, "join_parallelism": join_parallelism, "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
