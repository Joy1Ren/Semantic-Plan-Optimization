"""Row limit operator."""
from __future__ import annotations

from palimpzest.query.operators.limit import LimitScanOp

from ..base import Operator, _compute_op_id


class Limit(Operator):
    """Row limit."""
    stage_type = "limit"
    op_type = "limit"

    def __init__(self, n: int, schema):
        super().__init__()
        self._pz_op = LimitScanOp(
            limit=n,
            output_schema=schema,
            input_schema=schema,
        )
        self.n = n
        self.attributes = {"limit": n}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
