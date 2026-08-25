"""Column projection operator."""
from __future__ import annotations

from palimpzest.query.operators.project import ProjectOp

from ..base import Operator, _compute_op_id


class Project(Operator):
    """Column projection."""
    stage_type = "project"
    op_type = "project"

    def __init__(self, cols: list[str], input_schema, output_schema):
        super().__init__()
        self._pz_op = ProjectOp(
            project_cols=cols,
            output_schema=output_schema,
            input_schema=input_schema,
        )
        self.attributes = {"project_cols": sorted(cols)}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
