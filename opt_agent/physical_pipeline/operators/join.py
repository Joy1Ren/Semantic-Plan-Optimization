"""Exact (non-LLM) nested-loops join operator."""
from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Callable

from palimpzest.constants import NAIVE_EST_JOIN_SELECTIVITY
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.join import JoinOp

from ..base import Operator, _compute_op_id

if TYPE_CHECKING:
    from ..pipeline import PhysicalPipeline


# ------------------------------------------------------------------
# Non-LLM join physical operator (PZ has no non-LLM join; this mirrors PZ's
# NestedLoopsJoin structure but decides each pair with a Python predicate — the
# join analogue of PZ's NonLLMFilter / NonLLMConvert).
# ------------------------------------------------------------------

class NonLLMJoin(JoinOp):
    """Non-LLM nested-loops join.

    Same execution structure as PZ's ``NestedLoopsJoin`` — nested loops over
    new/stored left×right with input accumulation across calls (so the total pairs
    across calls stay |L|×|R|), and a ``DataRecordSet`` (or ``None`` when empty) as
    output — but each candidate pair is accepted/rejected by a deterministic predicate
    ``join_fn(left_dict, right_dict) -> bool`` instead of an LLM call. The predicate is
    fast and I/O-free, so pairs are evaluated sequentially (no ``ThreadPoolExecutor``).
    """

    def __init__(self, join_fn: Callable[[dict, dict], bool], *args, **kwargs):
        # JoinOp.__init__ consumes condition/desc and forwards output_schema/input_schema/
        # depends_on to PhysicalOperator. No model or Generator is created.
        super().__init__(*args, **kwargs)
        self.join_fn = join_fn
        self.join_idx = 0
        self._left_input_records: list[DataRecord] = []
        self._right_input_records: list[DataRecord] = []
        try:
            self._fn_src = inspect.getsource(join_fn).strip()
        except (OSError, TypeError):
            self._fn_src = repr(join_fn)

    def is_image_join(self) -> bool:
        return False

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"join_fn": self._fn_src, **id_params}

    def naive_cost_estimates(
        self,
        left_source_op_cost_estimates: OperatorCostEstimates,
        right_source_op_cost_estimates: OperatorCostEstimates,
    ) -> OperatorCostEstimates:
        # deterministic predicate: ~1 ms/pair, no LLM cost, perfect quality
        cardinality = NAIVE_EST_JOIN_SELECTIVITY * (
            left_source_op_cost_estimates.cardinality * right_source_op_cost_estimates.cardinality
        )
        return OperatorCostEstimates(
            cardinality=cardinality, time_per_record=0.001, cost_per_record=0.0, quality=1.0,
        )

    def _process_join_candidate_pair(
        self, left_candidate: DataRecord, right_candidate: DataRecord,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        start_time = time.time()
        try:
            passed_operator = bool(self.join_fn(left_candidate.to_dict(), right_candidate.to_dict()))
        except Exception as e:
            print(f"Error invoking user-defined function for join: {e}")
            raise
        join_dr = DataRecord.from_join_parents(self.output_schema, left_candidate, right_candidate)
        join_dr._passed_operator = passed_operator
        elapsed = time.time() - start_time
        record_op_stats = RecordOpStats(
            record_id=join_dr._id,
            record_parent_ids=join_dr._parent_ids,
            record_source_indices=join_dr._source_indices,
            record_state=join_dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=elapsed,
            cost_per_record=0.0,
            model_name=None,
            join_condition=self.condition,
            fn_call_duration_secs=elapsed,
            answer={"passed_operator": passed_operator},
            passed_operator=passed_operator,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )
        return [join_dr], [record_op_stats]

    def __call__(
        self, left_candidates: list[DataRecord], right_candidates: list[DataRecord],
    ) -> tuple[DataRecordSet | None, int]:
        # Mirror NestedLoopsJoin.__call__: join new×new, new×stored, stored×new, then
        # accumulate this call's inputs for future calls.
        output_records, output_record_op_stats, num_inputs_processed = [], [], 0

        def _join_all(lefts, rights):
            nonlocal num_inputs_processed
            for left_candidate in lefts:
                for right_candidate in rights:
                    recs, stats = self._process_join_candidate_pair(left_candidate, right_candidate)
                    output_records.extend(recs)
                    output_record_op_stats.extend(stats)
                    num_inputs_processed += 1

        _join_all(left_candidates, right_candidates)            # new left × new right
        _join_all(left_candidates, self._right_input_records)   # new left × stored right
        _join_all(self._left_input_records, right_candidates)   # stored left × new right

        # store input records to join with new records added later
        self._left_input_records.extend(left_candidates)
        self._right_input_records.extend(right_candidates)

        # return None if no output records were produced (matches NestedLoopsJoin)
        if len(output_records) == 0:
            return None, num_inputs_processed
        return DataRecordSet(output_records, output_record_op_stats), num_inputs_processed


class Join(Operator):
    """Exact (non-LLM) nested-loops join via a Python predicate."""
    stage_type = "join"
    op_type = "join"

    def __init__(self, other: "PhysicalPipeline | None", condition_fn: Callable[[dict, dict], bool], schema, depends_on: list[str] | None = None, self_join: bool = False):
        super().__init__()
        try:
            fn_src = inspect.getsource(condition_fn).strip()
        except (OSError, TypeError):
            fn_src = repr(condition_fn)
        self._pz_op = NonLLMJoin(
            join_fn=condition_fn,
            condition=fn_src,
            output_schema=schema,
            input_schema=schema,
            depends_on=depends_on,
        )
        self._fn = condition_fn  # preserved for make_oracle_copy
        # For a self-join `other` is None (left run once, joined with itself). See SemJoin.
        self.other = other
        self.self_join = self_join
        self.depends_on = depends_on
        self.attributes = {"condition": fn_src, "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
