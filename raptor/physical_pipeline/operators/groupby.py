"""Group-by aggregation operator (barrier operator)."""
from __future__ import annotations

from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.query.operators.aggregate import ApplyGroupByOp

from .base import Operator, _compute_op_id


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
        self.group_by_fields = list(group_by_fields)
        self.attributes = {
            "group_by_fields": sorted(group_by_fields),
            "agg_pairs": sorted(zip(agg_funcs, agg_fields)),
        }
        self.op_id = _compute_op_id(self.op_type, self.attributes)

    def split_groups(self, records: list) -> list[list]:
        """All records go to one call: ApplyGroupByOp groups them itself."""
        return [records]

    def __call__(self, records: list, **kwargs):
        # List values can't be group keys: group on tuples, then restore the lists.
        list_fields = [f for f in self.group_by_fields
                       if any(isinstance(getattr(r, f, None), list) for r in records)]
        for r in records:
            for f in list_fields:
                if isinstance(getattr(r, f, None), list):
                    setattr(r, f, tuple(getattr(r, f)))
        try:
            result = super().__call__(records, **kwargs)
        finally:
            for r in records:
                for f in list_fields:
                    if isinstance(getattr(r, f, None), tuple):
                        setattr(r, f, list(getattr(r, f)))
        for dr in result.data_records:
            for f in list_fields:
                if isinstance(getattr(dr, f, None), tuple):
                    setattr(dr, f, list(getattr(dr, f)))
        return result
