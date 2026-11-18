"""LLM aggregation over groups of records (barrier operator)."""
from __future__ import annotations

import time

from palimpzest.constants import Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.aggregate import SemanticAggregate

from ..pz_patches import _coerce_answer
from .base import Operator, _cols_attribute, _compute_op_id, _no_reasoning_prompt, _resolve_reasoning_effort


class _SemAggOp(SemanticAggregate):
    """PZ SemanticAggregate that keeps every generated field, carries
    the group's `group_by` values onto its output, and can return several records per group
    (Cardinality.ONE_TO_MANY), splitting the call's stats evenly across them.
    With keep_inputs, each output is instead written onto a copy of every input record."""

    def __init__(self, group_by: list[str], cardinality: Cardinality, *args, keep_inputs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_by = group_by
        self.cardinality = cardinality
        self.keep_inputs = keep_inputs
        self.generator = Generator(self.model, self.prompt_strategy, self.reasoning_effort, cardinality)

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()
        if len(candidates) == 0:
            return DataRecordSet([], [])

        input_fields = self.get_input_fields()
        fields_to_generate = self.get_fields_to_generate(candidates[0])
        fields = {field: field_type for field, field_type in self.output_schema.model_fields.items() if field in fields_to_generate}
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema, "agg_instruction": self.agg_str}
        field_answers, _, generation_stats, _ = self.generator(candidates, fields, **gen_kwargs)
        if any(field_answers.get(field) is None for field in fields):
            raise ValueError(f"Could not parse the aggregation answer for fields {list(fields)}")

        # A list (or object, bool, number) for a str field, or a single value for a list field, would
        # fail output_schema's validation and drop the group; pz_patches.py's coercion covers only ConvertOp.
        for field, answers in field_answers.items():
            if fields.get(field) is not None:
                field_answers[field] = [_coerce_answer(answer, fields[field].annotation) for answer in answers]

        group_values = {col: getattr(candidates[0], col, None) for col in self.group_by}
        n_records = max(len(answers) for answers in field_answers.values())
        input_names = list(self.input_schema.model_fields)
        drs = []
        for idx in range(max(n_records, 1)):
            values = {field: answers[idx] if idx < len(answers) else None for field, answers in field_answers.items()}
            if self.keep_inputs:
                # One record per (input record, output): the input's columns plus this output.
                for candidate in candidates:
                    data_item = self.output_schema(**{name: candidate[name] for name in input_names}, **values)
                    drs.append(DataRecord(data_item, source_indices=candidate._source_indices,
                                          parent_ids=[candidate._id], cardinality_idx=idx))
                continue
            data_item = self.output_schema(**group_values, **values)
            drs.append(DataRecord.from_agg_parents(data_item, parent_records=candidates, cardinality_idx=idx))

        per_record_stats = generation_stats / len(drs)
        time_per_record = (time.time() - start_time) / len(drs)
        record_op_stats = [
            RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=time_per_record,
                cost_per_record=per_record_stats.cost_per_record,
                model_name=self.get_model_name(),
                answer={field: getattr(dr, field, None) for field in fields_to_generate},
                input_fields=input_fields,
                generated_fields=fields_to_generate,
                input_text_tokens=per_record_stats.input_text_tokens,
                input_audio_tokens=per_record_stats.input_audio_tokens,
                input_image_tokens=per_record_stats.input_image_tokens,
                cache_read_tokens=per_record_stats.cache_read_tokens,
                cache_creation_tokens=per_record_stats.cache_creation_tokens,
                output_text_tokens=per_record_stats.output_text_tokens,
                embedding_input_tokens=per_record_stats.embedding_input_tokens,
                llm_call_duration_secs=per_record_stats.llm_call_duration_secs,
                fn_call_duration_secs=per_record_stats.fn_call_duration_secs,
                total_llm_calls=per_record_stats.total_llm_calls,
                total_embedding_llm_calls=per_record_stats.total_embedding_llm_calls,
                image_operation=self.is_image_op(),
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            for dr in drs
        ]
        return DataRecordSet(drs, record_op_stats)


class SemAgg(Operator):
    """LLM aggregation: one LLM call per group of records sharing the `group_by` values
    (None = all records in one group), producing one record per group, or several with one_to_many.
    keep_inputs: every input record of the group, with each output's columns added."""
    stage_type = "groupby"
    op_type = "sem_agg"

    def __init__(
        self,
        cols: list[dict],
        agg: str,
        model: Model,
        input_schema,
        output_schema,
        group_by: list[str] | None = None,
        depends_on: list[str] | None = None,
        one_to_many: bool = False,
        reasoning_effort_override: str | None = None,
        keep_inputs: bool = False,
    ):
        super().__init__()
        self.model = model
        self.group_by = list(group_by or [])
        self.depends_on = depends_on
        self.one_to_many = one_to_many
        self.keep_inputs = keep_inputs
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        prompt_strategy = PromptStrategy.AGG_NO_REASONING if _no_reasoning_prompt(model, eff) else PromptStrategy.AGG
        self._pz_op = _SemAggOp(
            group_by=self.group_by,
            cardinality=Cardinality.ONE_TO_MANY if one_to_many else Cardinality.ONE_TO_ONE,
            agg_str=agg,
            model=model,
            prompt_strategy=prompt_strategy,
            reasoning_effort=eff,
            output_schema=output_schema,
            input_schema=input_schema,
            depends_on=depends_on,
            keep_inputs=keep_inputs,
        )
        self.attributes = {
            "model": model.value,
            "reasoning_effort": eff,
            "cols": _cols_attribute(cols),
            "agg": agg,
            "group_by": self.group_by,
            "depends_on": depends_on,
        }
        if one_to_many:
            self.attributes["one_to_many"] = True
        if keep_inputs:
            self.attributes["keep_inputs"] = True
        self.op_id = _compute_op_id(self.op_type, self.attributes)

    def split_groups(self, records: list[DataRecord]) -> list[list[DataRecord]]:
        """Partition records by exact `group_by` values (list values compared as sorted tuples),
        in first-seen order. No group_by: all records form one group."""
        if not self.group_by:
            return [records]
        groups: dict[tuple, list[DataRecord]] = {}
        for record in records:
            key = []
            for col in self.group_by:
                value = getattr(record, col, None)
                key.append(tuple(sorted(value)) if isinstance(value, list) else value)
            groups.setdefault(tuple(key), []).append(record)
        return list(groups.values())
