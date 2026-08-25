"""Non-LLM column-suffix rename operator."""
from __future__ import annotations

import time

from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator

from ..base import Operator, _compute_op_id


# ------------------------------------------------------------------
# Non-LLM column-suffix physical operator: append a fixed suffix to every column name
# (a deterministic rename). Useful before a join to disambiguate columns — e.g.
# left.add_col_suffix("_dish"); right.add_col_suffix("_table") — so the join's inputs
# have no colliding names.
# ------------------------------------------------------------------

class NonLLMColSuffix(PhysicalOperator):
    """Rename every field `name -> f"{name}{suffix}"`. 1:1, deterministic, zero cost."""

    def __init__(self, suffix: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix = suffix

    def get_id_params(self):
        return {"suffix": self.suffix, **super().get_id_params()}

    def get_op_params(self):
        return {"suffix": self.suffix, **super().get_op_params()}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=0.0, cost_per_record=0.0, quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()
        field_vals = {}
        for field_name in [f.split(".")[-1] for f in candidate.get_field_names()]:
            field_vals[f"{field_name}{self.suffix}"] = candidate[field_name]
        new_dr = DataRecord(
            self.output_schema(**field_vals),
            source_indices=candidate._source_indices,
            parent_ids=[candidate._id],
        )
        record_op_stats = RecordOpStats(
            record_id=new_dr._id,
            record_parent_ids=new_dr._parent_ids,
            record_source_indices=new_dr._source_indices,
            record_state=new_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            fn_call_duration_secs=time.time() - start_time,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return DataRecordSet([new_dr], [record_op_stats])


class AddColSuffix(Operator):
    """Append a fixed suffix to every column name (non-LLM rename)."""
    stage_type = "convert"   # per-record transform; dispatched like map/convert
    op_type = "add_col_suffix"

    def __init__(self, suffix: str, input_schema, output_schema):
        super().__init__()
        self._pz_op = NonLLMColSuffix(
            suffix=suffix,
            output_schema=output_schema,
            input_schema=input_schema,
        )
        self.suffix = suffix   # preserved for make_oracle_copy
        self.attributes = {"suffix": suffix}
        self.params_id = _compute_op_id(self.op_type, {"suffix": suffix})
