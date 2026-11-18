from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import pandas as pd
from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.lib.schemas import create_schema_from_df

from .execution import PipelineExecution
from .operators import (
    AddColSuffix,
    ExactFilter,
    GroupBy,
    Join,
    Limit,
    Map,
    Project,
    RagFilter,
    RagMap,
    SemAgg,
    SemFilter,
    SemJoin,
    SemMap,
)
from .operators.base import _make_schema
from .operators.rag import DEFAULT_RAG_EMBEDDING_MODEL

if TYPE_CHECKING:
    from .operators.base import Operator


class PhysicalPipeline(PipelineExecution):
    """
    Fluent interface for chaining PZ physical operators, each with its own model.

    skip_if (sem_filter, sem_map, rag_filter, rag_map): records it returns True for skip the LLM
    call at no cost; a filter keeps them, a map leaves its new columns None.

    Usage:
        pipeline = PhysicalPipeline("p1", "reviews.csv", reviews_df)
        pipeline.sem_filter("the review is clearly positive", model=pz.Model.GPT_6_LUNA)
        pipeline.project(["review_id"])
        records, per_op_list, plan_dict = pipeline.run()
    """

    def __init__(self, plan_name, source: str, data: pd.DataFrame, max_workers: int = 20):
        self.plan_name = plan_name
        self._source = source
        self._df = data
        self._max_workers = max_workers
        self._initial_schema = create_schema_from_df(data)
        # extract (annotation, FieldInfo) tuples for schema evolution
        self._defs = {
            name: (field.annotation, field)
            for name, field in self._initial_schema.model_fields.items()
        }
        self._schema = self._initial_schema
        self._ops: list["Operator"] = []
        # For _check_no_copied_cols: whether the pipeline holds one row, which columns repeat one
        # value across rows (copied by a join or a keep_inputs sem_agg), and whether those copies
        # come from a single row.
        self._one_row = False
        self._copied_cols: set[str] = set()
        self._copies_one_row = False

    def _flat_ops(self) -> list[tuple[str, "Operator"]]:
        """All operators in topological order with display names.

        The display name is `{plan_name}_op{N}_{op_type}` — e.g. `p1_op3_sem_filter`."""
        result = []
        main_idx = 1
        for op in self._ops:
            if op.stage_type == "join" and not getattr(op, "self_join", False):
                for name, right_op in op.other._flat_ops():
                    result.append((name, right_op))
            result.append((f"{self.plan_name}_op{main_idx}_{op.op_type}", op))
            main_idx += 1
        return result

    def __str__(self) -> str:
        flat = self._flat_ops()
        if not flat:
            return "EmptyPipeline"
        op_names = [name for name, _ in flat]
        pretty_text = f"Plan: {self.plan_name}: ({', '.join(op_names)})\nOperators in topological order:"
        for name, op in flat:
            pretty_text += f"\n ====={name}=====\n   {str(op)}"
        return pretty_text


    def _check_new_cols(self, cols: list[dict], op_name: str) -> None:
        """Reject a new column name the pipeline already has, or one declared twice in the call."""
        seen, clashes, repeats = set(), [], []
        for col in cols:
            name = col["name"]
            if name in self._defs:
                clashes.append(name)
            elif name in seen:
                repeats.append(name)
            seen.add(name)
        if not clashes and not repeats:
            return
        problem = []
        if clashes:
            problem.append(f"column(s) {clashes} already exist in this pipeline")
        if repeats:
            problem.append(f"column(s) {repeats} are declared twice in this same call")
        raise ValueError(
            f"{op_name}: {'; '.join(problem)}. An operator can only ADD columns — it never "
            "overwrites one, because PZ generates only the fields that are not already in its "
            "input schema. A repeated name is therefore either silently ignored (its computed "
            "value is discarded and the old value kept) or, if every declared name already "
            "exists, fatal: the operator drops every record and the plan returns 0 rows. "
            "Rewrite the plan so no column name is ever declared twice. To combine several "
            "sources into one final column, give each source its own name (e.g. 'X_rag', "
            "'X_hdr') and produce 'X' itself — a name nothing has used yet — in one later map "
            "that reads them."
        )

    def _check_cols_exist(self, names: list[str] | None, op_name: str, arg: str,
                          defs: dict | None = None) -> None:
        """Reject a column name the pipeline does not have at this point (in `project`, or in an
        operator's `depends_on`), at plan-construction time."""
        available = self._defs if defs is None else defs
        missing = [name for name in names or [] if name not in available]
        if missing:
            raise ValueError(
                f"{op_name}: {arg} names column(s) {missing} that this pipeline does not have at "
                f"this point. Available columns: {list(available)}. (sem_agg, groupby and project "
                "keep only the columns they output; a join adds '_right' to a right-side column "
                "only when its name collides with a left-side one.)"
            )

    def _check_no_copied_cols(self, depends_on: list[str] | None) -> None:
        """Reject a sem_agg that reads a column repeated across rows (copied by a join or a
        keep_inputs sem_agg), which it would read once per row."""
        read = depends_on if depends_on is not None else list(self._defs)
        repeated = [col for col in read if col in self._copied_cols]
        if repeated:
            raise ValueError(
                f"sem_agg: column(s) {repeated} are repeated in all row copies in the sem_agg: a join "
                "copied them from a one-row result (e.g. an earlier sem_agg's output) onto every row, "
                "or a keep_inputs sem_agg put its output on every input row, "
                f"so this sem_agg would read them once per row. To keep one copy, groupby on "
                f"{repeated} with list first (e.g. p.groupby({repeated}, ['list'], ['<per-row column>'])), "
                "or leave them out of this sem_agg's depends_on."
            )

    def _note_join(self, other: "PhysicalPipeline", is_self: bool, right_names: list[str]) -> None:
        """After a join: columns of a one-row side are now copied onto every row of the other."""
        if is_self:
            return
        if other._one_row and not self._one_row:
            self._add_copied(right_names, one_row=True)
        elif self._one_row and not other._one_row:
            self._copied_cols = set(self._defs) - set(right_names)
            self._copies_one_row = True
        self._one_row = self._one_row and other._one_row

    def _add_copied(self, names, one_row: bool) -> None:
        """Mark `names` as repeated across rows; `one_row`: their values come from a single row."""
        self._copies_one_row = one_row and (self._copies_one_row or not self._copied_cols)
        self._copied_cols |= set(names)

    def sem_filter(self, condition: str, model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None, skip_if: Callable[[dict], bool] | None = None) -> "PhysicalPipeline":
        """LLM-based row filter. Keeps records where condition is true."""
        self._check_cols_exist(depends_on, "sem_filter", "depends_on")
        self._ops.append(SemFilter(condition=condition, model=model, schema=self._schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override, skip_if=skip_if))
        return self

    def sem_map(self, cols: list[dict], model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None, skip_if: Callable[[dict], bool] | None = None) -> "PhysicalPipeline":
        """
        LLM-based column derivation. Adds new fields to each record.

        cols: list of {"name": str, "type": type, "description": str}
            description is passed as FieldInfo and used by the LLM generator.
            Every name must be NEW — see _check_new_cols.
        """
        self._check_cols_exist(depends_on, "sem_map", "depends_on")
        self._check_new_cols(cols, "sem_map")
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col["description"]),
            )
            for col in cols
        }
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(SemMap(cols=cols, model=model, input_schema=self._schema, output_schema=output_schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override, skip_if=skip_if))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    def sem_agg(
        self,
        cols: list[dict],
        agg: str,
        model: Model,
        group_by: list[str] | None = None,
        depends_on: list[str] | None = None,
        one_to_many: bool = False,
        reasoning_effort_override: str | None = None,
        keep_inputs: bool = False,
    ) -> "PhysicalPipeline":
        """
        LLM aggregation across records. Records are grouped by exact value of the `group_by` columns
        (None = all records in one group); one LLM call per group reads all of its records and follows `agg`.

        cols: list of {"name": str, "type": type, "description": str}, as for sem_map.
            Every name must be NEW — see _check_new_cols.
        Output: one record per group (several with one_to_many=True) holding only the
        `group_by` columns plus `cols`.
        keep_inputs=True: instead, every input record of the group is kept with all its columns,
        plus `cols` -- (group's records) x (group's outputs) records per group, so the outputs sit
        next to the exact records (and intermediate columns) they were computed from.
        """
        self._check_cols_exist(depends_on, "sem_agg", "depends_on")
        self._check_no_copied_cols(depends_on)
        self._check_new_cols(cols, "sem_agg")
        group_by = list(group_by or [])
        missing = [col for col in group_by if col not in self._defs]
        if missing:
            raise ValueError(f"sem_agg: group_by column(s) {missing} are not in this pipeline")
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col["description"]),
            )
            for col in cols
        }
        if keep_inputs:
            out_defs = {**self._defs, **new_defs}
        else:
            out_defs = {**{col: self._defs[col] for col in group_by}, **new_defs}
        output_schema = _make_schema(out_defs)
        self._ops.append(SemAgg(cols=cols, agg=agg, model=model, input_schema=self._schema, output_schema=output_schema, group_by=group_by, depends_on=depends_on, one_to_many=one_to_many, reasoning_effort_override=reasoning_effort_override, keep_inputs=keep_inputs))
        self._defs = out_defs
        self._schema = output_schema
        if keep_inputs:
            # Each output is repeated on every record of its group.
            self._add_copied(new_defs, one_row=not group_by and not one_to_many)
            self._one_row = self._one_row and not one_to_many
        else:
            self._one_row = not group_by and not one_to_many
            self._copied_cols = set()
        return self

    def sem_join(
        self,
        other: "PhysicalPipeline",
        condition: str,
        model: Model,
        join_parallelism: int = 20,
        depends_on: list[str] | None = None,
        reasoning_effort_override: str | None = None,
    ) -> "PhysicalPipeline":
        """
        LLM-based join. Keeps pairs of (self record, other record) where condition holds.

        On a name collision, the left (self) field keeps its name and the right (other)
        field is suffixed with "_right" (repeatedly, if needed).

        Self-join: `pipeline.sem_join(pipeline, ...)` is supported. The upstream is executed
        ONCE and joined with its own output.
        """
        is_self = other is self
        right_defs = self._defs if is_self else other._defs
        merged_defs = dict(self._defs)  # left fields keep their names
        right_names = []
        for name, field_def in right_defs.items():
            new_name = name
            while new_name in merged_defs:
                new_name = f"{new_name}_right"
            merged_defs[new_name] = field_def
            right_names.append(new_name)
        self._check_cols_exist(depends_on, "sem_join", "depends_on", merged_defs)
        joined_schema = _make_schema(merged_defs)
        self._ops.append(SemJoin(
            other=None if is_self else other,
            self_join=is_self,
            condition=condition,
            model=model,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            schema=joined_schema,
            reasoning_effort_override=reasoning_effort_override,
        ))
        self._defs = merged_defs
        self._schema = joined_schema
        self._note_join(other, is_self, right_names)
        return self

    def rag_filter(
        self,
        condition: str,
        embedding_query: str,
        model: Model,
        chunk_size: "int | str" = 20000,
        num_chunks_per_field: int | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        similarity_method: str = "embedding",
        depends_on: list[str] | None = None,
        reasoning_effort_override: str | None = None,
        skip_if: Callable[[dict], bool] | None = None,
    ) -> "PhysicalPipeline":
        """
        Chunk long fields, retrieve their most relevant chunks, then LLM-filter over just
        those chunks. Keeps records where condition is true.
        """
        self._check_cols_exist(depends_on, "rag_filter", "depends_on")
        self._ops.append(RagFilter(
            condition=condition, embedding_query=embedding_query, model=model, schema=self._schema,
            chunk_size=chunk_size, num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold, embedding_model=embedding_model,
            similarity_method=similarity_method,
            depends_on=depends_on, reasoning_effort_override=reasoning_effort_override,
            skip_if=skip_if,
        ))
        return self

    def rag_map(
        self,
        cols: list[dict],
        embedding_query: str,
        model: Model,
        chunk_size: "int | str" = 20000,
        num_chunks_per_field: int | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        similarity_method: str = "embedding",
        depends_on: list[str] | None = None,
        reasoning_effort_override: str | None = None,
        skip_if: Callable[[dict], bool] | None = None,
    ) -> "PhysicalPipeline":
        """
        Chunk long fields, retrieve their most relevant chunks, then LLM-derive new columns
        (cols, same shape as sem_map) from just those chunks.
        """
        self._check_cols_exist(depends_on, "rag_map", "depends_on")
        self._check_new_cols(cols, "rag_map")
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col["description"]),
            )
            for col in cols
        }
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(RagMap(
            cols=cols, embedding_query=embedding_query, model=model, input_schema=self._schema,
            output_schema=output_schema, chunk_size=chunk_size, num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold, embedding_model=embedding_model,
            similarity_method=similarity_method,
            depends_on=depends_on, reasoning_effort_override=reasoning_effort_override,
            skip_if=skip_if,
        ))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self


    def filter(self, fn: Callable[[dict], bool]) -> "PhysicalPipeline":
        """Exact (non-LLM) row filter. fn receives a record dict and returns bool."""
        self._ops.append(ExactFilter(fn=fn, schema=self._schema))
        return self

    def map(self, udf: Callable[[dict], dict], cols: list[dict]) -> "PhysicalPipeline":
        """
        Deterministic (non-LLM) column derivation via UDF.

        udf: callable receiving a record dict, returning a dict with the new field values.
        cols: list of {"name": str, "type": type, "description": str}
            The type annotation is preserved in the schema — use pz.ImageFilepath to
            signal image columns so PZ's prompt factory encodes them for LLM calls.
        """
        self._check_new_cols(cols, "map")
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col.get("description", "")),
            )
            for col in cols
        }
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(Map(udf=udf, cols=cols, input_schema=self._schema, output_schema=output_schema))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    def add_col_suffix(self, suffix: str) -> "PhysicalPipeline":
        """Append `suffix` to every column name (a non-LLM, deterministic rename).

        Useful before a join to disambiguate columns so there are no name collisions, e.g.
            left.add_col_suffix("_dish")
            right.add_col_suffix("_table")
            left.join(right, lambda l, r: l["brand_dish"] == r["brand_table"])
        """
        renamed_defs = {f"{name}{suffix}": field_def for name, field_def in self._defs.items()}
        output_schema = _make_schema(renamed_defs)
        self._ops.append(AddColSuffix(suffix=suffix, input_schema=self._schema, output_schema=output_schema))
        self._defs = renamed_defs
        self._schema = output_schema
        self._copied_cols = {f"{name}{suffix}" for name in self._copied_cols}
        return self

    def join(
        self,
        other: "PhysicalPipeline",
        condition_fn: Callable[[dict, dict], bool],
        depends_on: list[str] | None = None,
    ) -> "PhysicalPipeline":
        """
        Exact (non-LLM) join. Keeps pairs (self record, other record) where
        condition_fn(left_dict, right_dict) is True.
        """
        is_self = other is self
        right_defs = self._defs if is_self else other._defs
        merged_defs = dict(self._defs)  # left fields keep their names
        right_names = []
        for name, field_def in right_defs.items():
            new_name = name
            while new_name in merged_defs:
                new_name = f"{new_name}_right"
            merged_defs[new_name] = field_def
            right_names.append(new_name)
        self._check_cols_exist(depends_on, "join", "depends_on", merged_defs)
        joined_schema = _make_schema(merged_defs)
        self._ops.append(Join(other=None if is_self else other, self_join=is_self, condition_fn=condition_fn, schema=joined_schema, depends_on=depends_on))
        self._defs = merged_defs
        self._schema = joined_schema
        self._note_join(other, is_self, right_names)
        return self

    def project(self, cols: list[str]) -> "PhysicalPipeline":
        """Select a subset of columns."""
        self._check_cols_exist(cols, "project", "cols")
        projected_defs = {k: self._defs[k] for k in cols if k in self._defs}
        projected_schema = _make_schema(projected_defs)
        self._ops.append(Project(cols=cols, input_schema=self._schema, output_schema=projected_schema))
        self._defs = projected_defs
        self._schema = projected_schema
        self._copied_cols &= set(projected_defs)
        return self

    def limit(self, n: int) -> "PhysicalPipeline":
        """Keep at most n records."""
        self._ops.append(Limit(n=n, schema=self._schema))
        return self

    def groupby(
        self,
        group_by_fields: list[str],
        agg_funcs: list[str],
        agg_fields: list[str],
    ) -> "PhysicalPipeline":
        """
        Group records and aggregate.

        group_by_fields: fields to group on
        agg_funcs:       per-agg-field function — count, average, sum, min, max, list or set
        agg_fields:      fields to aggregate (parallel to agg_funcs)

        Output fields are the group_by_fields plus "<func>(<field>)" columns.
        """
        op = GroupBy(
            group_by_fields=group_by_fields,
            agg_funcs=agg_funcs,
            agg_fields=agg_fields,
            input_schema=self._schema,
        )
        self._ops.append(op)
        self._defs = {k: (f.annotation, f) for k, f in op.output_schema.model_fields.items()}
        self._schema = op.output_schema
        # Grouping only on copied columns collapses the rows back to one.
        self._one_row = not group_by_fields or (self._copies_one_row and set(group_by_fields) <= self._copied_cols)
        self._copied_cols = set()
        self._copies_one_row = False
        return self


    def __iter__(self):
        """Yield all operators in topological order (right-branch ops before their join)."""
        for _, op in self._flat_ops():
            yield op
