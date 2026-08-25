"""Group-by aggregation operator (barrier operator)."""
from __future__ import annotations

from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.query.operators.aggregate import ApplyGroupByOp

from ..base import Operator, _compute_op_id


class GroupBy(Operator):
    """Group-by aggregation (barrier operator)."""
    stage_type = "groupby"
    op_type = "groupby"

    def __init__(
        self,
        group_by_fields: list[str],
        agg_funcs: list[str],
        agg_fields: list[str],
        input_schema,
    ):
        super().__init__()
        sig = GroupBySig(
            group_by_fields=group_by_fields,
            agg_funcs=agg_funcs,
            agg_fields=agg_fields,
        )
        self.output_schema = sig.output_schema()
        self._pz_op = ApplyGroupByOp(
            group_by_sig=sig,
            output_schema=self.output_schema,
            input_schema=input_schema,
        )
        self.attributes = {
            "group_by_fields": sorted(group_by_fields),
            "agg_pairs": sorted(zip(agg_funcs, agg_fields)),
        }
        self.params_id = _compute_op_id(self.op_type, self.attributes)
