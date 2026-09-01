"""PhysicalPipeline: chain PZ physical operators directly with per-operator model selection.

This module defines the overall pipeline structure (the fluent builder API and the
execution engine). The individual operators it dispatches to live in operators/.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Optional

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.elements.groupbysig import GroupBySig  # noqa: F401 -- re-exported for callers that reach through here
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.lib.schemas import create_schema_from_df
from palimpzest.core.models import ExecutionStats

from .base import (
    DEFAULT_RAG_EMBEDDING_MODEL,
    NUM_SAMPLES,
    SUBSET_SEED,
    SubsetExecutionContext,
    _make_schema,
    _str_to_pz_model,
)
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
    SemFilter,
    SemJoin,
    SemMap,
)

if TYPE_CHECKING:
    from .base import Operator


class PhysicalPipeline:
    """
    Fluent interface for chaining PZ physical operators with per-operator model selection.

    Semantic (require model):     sem_filter, sem_map, sem_join
    Non-semantic (no model):      filter, project, limit, groupby

    Usage:
        pipeline = PhysicalPipeline("plan1", "Reviews.csv", self.load_data("Reviews.csv"))
        pipeline.sem_filter("the review is clearly positive", model=pz.Model.CLAUDE_3_5_HAIKU)
        pipeline.project(["reviewId"])
        pipeline.limit(5)
        return pipeline.run()
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
        self._last_exec_stats = None      # populated by run(); used by to_results_row()

    def _flat_ops(self) -> list[tuple[str, "Operator"]]:
        """All operators in topological order with display names.

        The display name is `{plan_name}_op{N}_{op_type}` — e.g. `p1_shoe_op3_sem_filter`.
        It carries (a) which pipeline the op belongs to (`plan_name`, which the caller should
        set meaningfully, e.g. "p1_shoe"; a two-input join's right branch uses its own
        pipeline's plan_name), (b) the op's position `N` within that pipeline, and (c) the
        op type, so plan printouts and per-operator results/samples are self-describing. The
        name is only a display string / dict key (uniqueness is the only requirement) — it is
        never used to match plan↔oracle ops (that is done by stage index) or for cost-model
        lookup (that uses `params_id`), so the format is free to change.

        Right-branch operators of a two-input join appear before the join itself. A
        self-join has NO right branch: its upstream runs once (shared by both sides), so it
        already appears once in the main chain (a common-subexpression optimization beyond
        PZ, whose physical plan would instead have two instances of the upstream)."""
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


    # ------------------------------------------------------------------
    # Semantic operators
    # ------------------------------------------------------------------

    def sem_filter(self, condition: str, model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None) -> "PhysicalPipeline":
        """LLM-based row filter. Keeps records where condition is true."""
        self._ops.append(SemFilter(condition=condition, model=model, schema=self._schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override))
        return self

    def sem_map(self, cols: list[dict], model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None) -> "PhysicalPipeline":
        """
        LLM-based column derivation. Adds new fields to each record.

        cols: list of {"name": str, "type": type, "description": str}
            description is passed as FieldInfo and used by the LLM generator.
        """
        new_defs = {}
        for col in cols:
            name = col["name"]
            if name in self._defs:
                # Preserve the existing annotation to avoid union_schemas type mismatch
                ann = self._defs[name][0]
            else:
                ann = Optional[col.get("type", Any)]
            new_defs[name] = (ann, FieldInfo(default=None, description=col["description"]))
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(SemMap(cols=cols, model=model, input_schema=self._schema, output_schema=output_schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
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
        field is suffixed with "_right" (repeatedly, if needed). This mirrors Palimpzest's
        own join semantics (see union_schemas(join=True) and DataRecord.from_join_parents),
        so downstream operators can reference right-side columns as e.g. "idx_right".

        Self-join: `pipeline.sem_join(pipeline, ...)` is supported. The upstream is executed
        ONCE and joined with its own output (see _execute_core), so it is not recomputed and
        its cost is counted once. This is a common-subexpression optimization *beyond* PZ:
        PZ's Cascades groups dedupe only in the optimizer memo, and its extracted physical
        plan runs the shared upstream twice (two instances, differentiated by topo index).
        """
        is_self = other is self
        right_defs = self._defs if is_self else other._defs
        merged_defs = dict(self._defs)  # left fields keep their names
        for name, field_def in right_defs.items():
            new_name = name
            while new_name in merged_defs:
                new_name = f"{new_name}_right"
            merged_defs[new_name] = field_def
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
    ) -> "PhysicalPipeline":
        """
        Chunk long fields, retrieve their most relevant chunks, then LLM-filter over just
        those chunks. Keeps records where condition is true.

        embedding_query is the retrieval query text (used for either similarity_method) and
        can (should) be phrased differently from condition: condition instructs the LLM, while
        embedding_query should just be the terms/phrases expected to score highest against the
        relevant passage -- keyword-dense for similarity_method="bm25", or phrased to sit close
        to the relevant passage in embedding space for similarity_method="embedding".

        similarity_method: "embedding" (cosine similarity, default) or "bm25" (keyword/full-text
        search via rank_bm25 -- no embedding calls, so cheaper, and often stronger when the
        target content hinges on specific terminology rather than paraphrase-level meaning).

        chunk_size: a fixed int (characters), or a string expression evaluated per field against
        that field's own length via the name `input_length`, e.g. "max(10000, input_length / 5)"
        (only arithmetic + max/min/abs/round are allowed -- see operators/rag_common.py).

        Specify exactly one of num_chunks_per_field (keep the top-k most similar chunks) or
        similarity_threshold (keep every chunk with similarity >= threshold, always keeping at
        least the single best-scoring chunk). Retrieved chunks are labeled with their rank and
        similarity score when passed to the LLM (see operators/rag_common.py's _format_ranked_chunks).
        """
        self._ops.append(RagFilter(
            condition=condition, embedding_query=embedding_query, model=model, schema=self._schema,
            chunk_size=chunk_size, num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold, embedding_model=embedding_model,
            similarity_method=similarity_method,
            depends_on=depends_on, reasoning_effort_override=reasoning_effort_override,
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
    ) -> "PhysicalPipeline":
        """
        Chunk long fields, retrieve their most relevant chunks, then LLM-derive new columns
        (cols, same shape as sem_map) from just those chunks.

        See rag_filter for embedding_query, similarity_method ("embedding" or "bm25"),
        chunk_size (fixed int or an "input_length"-based expression), and the
        num_chunks_per_field / similarity_threshold chunk-selection parameters (exactly one
        must be given).
        """
        new_defs = {}
        for col in cols:
            name = col["name"]
            if name in self._defs:
                # Preserve the existing annotation to avoid union_schemas type mismatch
                ann = self._defs[name][0]
            else:
                ann = Optional[col.get("type", Any)]
            new_defs[name] = (ann, FieldInfo(default=None, description=col["description"]))
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(RagMap(
            cols=cols, embedding_query=embedding_query, model=model, input_schema=self._schema,
            output_schema=output_schema, chunk_size=chunk_size, num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold, embedding_model=embedding_model,
            similarity_method=similarity_method,
            depends_on=depends_on, reasoning_effort_override=reasoning_effort_override,
        ))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    # ------------------------------------------------------------------
    # Non-semantic operators
    # ------------------------------------------------------------------

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

        condition_fn receives two dicts — the left record and the right record, each with
        their own (pre-join) field names — and returns a bool, e.g.
        `lambda l, r: l["brand"] == r["brand"] and l["category"] == r["category"]`.

        Schema merging mirrors sem_join: on a name collision the left field keeps its name
        and the right field is suffixed with "_right", so downstream operators reference
        right-side columns as e.g. "idx_right".

        Self-join (`pipeline.join(pipeline, ...)`) is supported: the upstream is executed
        ONCE and joined with its own output (not recomputed) — a common-subexpression
        optimization beyond PZ, which runs the shared upstream twice. See sem_join.
        """
        is_self = other is self
        right_defs = self._defs if is_self else other._defs
        merged_defs = dict(self._defs)  # left fields keep their names
        for name, field_def in right_defs.items():
            new_name = name
            while new_name in merged_defs:
                new_name = f"{new_name}_right"
            merged_defs[new_name] = field_def
        joined_schema = _make_schema(merged_defs)
        self._ops.append(Join(other=None if is_self else other, self_join=is_self, condition_fn=condition_fn, schema=joined_schema, depends_on=depends_on))
        self._defs = merged_defs
        self._schema = joined_schema
        return self

    def project(self, cols: list[str]) -> "PhysicalPipeline":
        """Select a subset of columns."""
        projected_defs = {k: self._defs[k] for k in cols if k in self._defs}
        projected_schema = _make_schema(projected_defs)
        self._ops.append(Project(cols=cols, input_schema=self._schema, output_schema=projected_schema))
        self._defs = projected_defs
        self._schema = projected_schema
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
        agg_funcs:       per-agg-field function — "count" or "average"
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
        return self

    # ------------------------------------------------------------------
    # Plan introspection (compatible with cost_model_agent inspect_plan)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yield all operators in topological order (right-branch ops before their join)."""
        for _, op in self._flat_ops():
            yield op

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _build_initial_records(self, df: pd.DataFrame) -> list[DataRecord]:
        """Build DataRecord objects from a DataFrame using this pipeline's initial schema."""
        records: list[DataRecord] = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            dr = DataRecord(data_item=self._initial_schema(**row), source_indices=f"{self._source}-{i}")
            records.append(dr)
        return records

    def _execute(self) -> tuple[list[DataRecord], list, dict, dict]:
        """Execute using all rows in self._df. Delegates to _execute_core."""
        initial_records = self._build_initial_records(self._df)
        return self._execute_core(initial_records, max_samples=5)

    def _execute_core(self, initial_records: list[DataRecord], max_samples: int, skip_limit: bool = False, _executor=None) -> tuple[list[DataRecord], list, dict, dict]:
        from concurrent.futures import wait as fut_wait
        _POLL_INTERVAL = 0.3

        all_record_op_stats = []

        # Assign logical_op_ids and build stage_map for per-operator stat attribution.
        # _flat_ops() includes right-branch operators so all ops appear in per_op_list.
        stage_map: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            op.logical_op_id = op.params_id
            op._pz_op.logical_op_id = op.params_id
            # Reset a join operator's cross-call accumulation before each run. NestedLoopsJoin /
            # NonLLMJoin keep _left/_right_input_records to build the full cross product from
            # incrementally-fed batches WITHIN a single run. But the pipeline is cached and
            # reused (e.g. execute_subplan stores it per plan), so without this reset a SECOND
            # execution would join the new records against the STALE ones from the first —
            # wrong results and a runaway counter (a 2nd run_subset on a 10-row join prints
            # "…400 JOINED" instead of ≤100). join_idx (PZ's per-pair print counter) is reset too.
            if op.stage_type == "join":
                op._pz_op._left_input_records = []
                op._pz_op._right_input_records = []
                op._pz_op._left_joined_record_ids = set()
                op._pz_op._right_joined_record_ids = set()
                op._pz_op.join_idx = 0
                op._pz_op.finished = False
            stage_map[flat_idx] = {
                "logical_op_id": op.params_id,
                "op_name": op_name,
                "op_type": op.op_type,
                "attributes": op.attributes,
                "params_id": op.params_id,
            }

        n_ops = len(self._ops)
        if n_ops == 0:
            return initial_records, all_record_op_stats, stage_map, {}

        _MAX_SAMPLES = max_samples
        # (input_record, Future) for per-record ops; keyed by id(op) so right-branch ops
        # are captured alongside main-pipeline ops.
        op_sample_pairs: dict[int, list] = {}  # id(op) -> [(input_dr, future)]

        # Join ops can't link inputs→outputs the same way (their "input" is a pair of
        # records), so they're collected separately as already-resolved
        # (pair_record, pair_record | None) tuples — None when the pair failed the join.
        # Unlike op_sample_pairs (capped at _MAX_SAMPLES for agent exploration), we keep
        # EVERY join pair: the oracle quality judge scores the sem_join over all of its
        # input→output pairs, keyed by each pair record's source_indices (left + right).
        join_op_samples: dict[int, list[tuple]] = {}  # id(join_op) -> [(pair_dr, pair_dr|None)]

        def _collect_join_samples(op_key: int, result_set) -> None:
            if result_set is None or not result_set.data_records:
                return
            bucket = join_op_samples.setdefault(op_key, [])
            for dr in result_set.data_records:
                passed = bool(dr._passed_operator)
                bucket.append((dr, dr if passed else None))

        # Main pipeline queues
        input_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        future_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        input_queues[0] = initial_records[:]
        output_records: list[DataRecord] = []
        limit_val = None if skip_limit else next((op.n for op in self._ops if op.stage_type == "limit"), None)
        # batch_size for filter/convert/project: limit value when present, else None (submit all)
        batch_size = limit_val

        # Precompute which joins have a downstream limit op (enables incremental join, matching PZ).
        join_has_downstream_limit = {
            i: any(op.stage_type == "limit" for op in self._ops[i + 1:])
            for i, op in enumerate(self._ops) if op.stage_type == "join"
        }
        _join_call_counts: dict[int, int] = {}  # diagnostic: how many times each join fires

        # Initialize right-pipeline state for each two-input join stage. (A self-join has no
        # separate right pipeline — its upstream runs once and is joined with itself — so it
        # gets no right_state and is handled directly in the join branch below.)
        # Each tick the right pipeline advances by batch_size records, matching PZ's scan batching.
        right_state: dict[int, dict] = {}
        for i, op in enumerate(self._ops):
            if op.stage_type != "join" or getattr(op, "self_join", False):
                continue
            other = op.other
            r_initial: list[DataRecord] = other._build_initial_records(other._df)
            # If the right pipeline itself contains a join op, the tick-by-tick right
            # advancement loop (which only handles filter/convert/project/groupby) cannot
            # dispatch it. We mark has_right_join=True and submit the right pipeline as a
            # future once the executor starts, so it runs in parallel with the left pipeline.
            # The outer join uses incremental mode for this case, firing as right records
            # arrive rather than waiting for rs["done"].
            has_right_join = any(r_op.stage_type == "join" for r_op in other._ops)
            if not has_right_join:
                for r_op in other._ops:
                    if r_op.logical_op_id is None:
                        r_op.logical_op_id = r_op.params_id
                    r_op._pz_op.logical_op_id = r_op.logical_op_id
            n_r = 0 if has_right_join else len(other._ops)
            right_state[i] = {
                "other": other,
                "n": n_r,
                "initial": r_initial,
                "n_fed": 0,                # how many right initial records fed into r_iq[0] so far
                "iq": {j: [] for j in range(n_r)},
                "fq": {j: [] for j in range(n_r)},
                "pending": [],             # right records ready for this tick's join call (incremental)
                "all_right": [],           # all right records produced so far (barrier join)
                "done": n_r == 0 and len(r_initial) == 0,
                "has_right_join": has_right_join,
                "right_future": None,      # set below once executor is live
            }

        def any_pending() -> bool:
            if any(input_queues[i] or future_queues[i] for i in range(n_ops)):
                return True
            # A join's right sub-pipeline keeps its OWN queues (right_state), separate from the
            # main input/future queues. Keep the loop alive while any right side is still
            # producing. This matters now that the incremental join pops ALL available left at
            # once (mirroring PZ): the main queues can empty before the right side finishes, and
            # without this check the loop would exit early and drop the remaining right records.
            #
            # NOTE: we intentionally do NOT keep the loop alive on a non-empty rs["pending"].
            # The incremental path consumes pending the same tick it is produced, whereas the
            # barrier path drains rs["all_right"] and leaves rs["pending"] populated — checking
            # pending here would livelock a barrier join after it has already finished.
            for rstate in right_state.values():
                if not rstate["done"]:
                    return True
                if rstate.get("right_future") is not None:
                    return True
                if any(rstate["iq"].get(j) or rstate["fq"].get(j) for j in range(rstate["n"])):
                    return True
            return False

        def upstream_done(stage_idx: int) -> bool:
            return all(not input_queues[i] and not future_queues[i] for i in range(stage_idx))

        def drain(fq_dict: dict, key: int, timeout: float = _POLL_INTERVAL) -> list[DataRecord]:
            if not fq_dict.get(key):
                return []
            done, not_done = fut_wait(fq_dict[key], timeout=timeout)
            fq_dict[key] = list(not_done)
            passing = []
            for future in done:
                try:
                    result = future.result()
                    all_record_op_stats.extend(result.record_op_stats)
                    passing.extend(dr for dr in result.data_records if dr._passed_operator)
                except Exception as _exc:
                    print(f"[pipeline] record-level operator error (skipping record): {type(_exc).__name__}: {_exc}")
            return passing

        def _run(executor):
            # Submit right pipelines that contain joins as background futures so they run
            # in parallel with the left pipeline. The shared executor is passed down so the
            # right pipeline's operators compete for the same worker slots — at most
            # max_workers threads are active at any time. One slot is consumed by the right
            # pipeline's scheduling loop while it runs; the remaining slots service operators
            # from both branches.
            for i, rs in right_state.items():
                if rs["has_right_join"]:
                    r_snap = rs["initial"][:]
                    other_pipe = rs["other"]
                    rs["right_future"] = executor.submit(
                        other_pipe._execute_core, r_snap, max_samples, skip_limit, executor
                    )
                    rs["done"] = False

            while any_pending():
                # Drain completed right-pipeline futures into pending / all_right so the
                # incremental join path can consume them this tick.
                for rs in right_state.values():
                    if rs.get("right_future") is not None and rs["right_future"].done():
                        recs, stats, _, _ = rs["right_future"].result()
                        all_record_op_stats.extend(stats)
                        rs["pending"].extend(recs)
                        rs["all_right"].extend(recs)
                        rs["right_future"] = None
                        rs["done"] = True


                # Advance main pipeline
                for stage_idx, op in enumerate(self._ops):

                    # Step 1: harvest upstream futures into this operator's input queue
                    if stage_idx > 0:
                        input_queues[stage_idx].extend(drain(future_queues, stage_idx - 1))

                    # Step 2: final operator — drain own future queue BEFORE submission
                    # to collect previous-tick results (matches PZ's _process_future_results ordering).
                    if stage_idx == n_ops - 1:
                        output_records.extend(drain(future_queues, stage_idx))

                    # Step 3: submit work — operator type determines the path
                    if op.stage_type == "limit" and input_queues[stage_idx]:
                        if skip_limit:
                            # Subset execution: pass all records through without truncation
                            to_pass = input_queues[stage_idx][:]
                            input_queues[stage_idx] = []
                        else:
                            space = limit_val - len(output_records)
                            to_pass = input_queues[stage_idx][:max(space, 0)]
                            input_queues[stage_idx] = input_queues[stage_idx][len(to_pass):]
                        if stage_idx == n_ops - 1:
                            output_records.extend(to_pass)
                            # Drain own future queue after limit to mirror PZ's second harvest,
                            # ensuring output_records is up to date before the early-stop check.
                            output_records.extend(drain(future_queues, stage_idx))
                        else:
                            input_queues[stage_idx + 1].extend(to_pass)

                    elif op.stage_type == "groupby" and upstream_done(stage_idx) and input_queues[stage_idx]:
                        # Aggregate barrier: wait for all upstream, then submit as single batch future
                        batch = input_queues[stage_idx][:]
                        input_queues[stage_idx].clear()
                        future_queues[stage_idx].append(executor.submit(op, batch))

                    elif op.stage_type == "join" and getattr(op, "self_join", False):
                        # Self-join: the upstream ran ONCE and its output is now in this
                        # stage's input queue. Join that output with ITSELF — feed the same
                        # records as both left and right — so the upstream is not recomputed.
                        # (This is a common-subexpression optimization beyond PZ, which unfolds
                        # the shared side into two instances and runs it twice.) The operator's
                        # accumulation builds the full L×L cross product from the streamed
                        # batches, exactly as it does for a two-input join.
                        join_pz_op = op._pz_op
                        has_dl = join_has_downstream_limit[stage_idx]
                        fire = (input_queues[stage_idx] and
                                (has_dl or upstream_done(stage_idx)))
                        if fire:
                            left_batch = input_queues[stage_idx][:]
                            input_queues[stage_idx].clear()
                            _join_call_counts[stage_idx] = _join_call_counts.get(stage_idx, 0) + 1
                            result_set, _ = join_pz_op(left_batch, left_batch)
                            _collect_join_samples(id(op), result_set)
                            if result_set is not None:
                                future_queues[stage_idx].append(
                                    executor.submit(lambda rset=result_set: rset)
                                )

                    elif op.stage_type == "join":
                        join_pz_op = op._pz_op
                        other = op.other
                        rs = right_state[stage_idx]
                        n_r = rs["n"]
                        r_iq = rs["iq"]
                        r_fq = rs["fq"]
                        has_dl = join_has_downstream_limit[stage_idx]

                        # Advance right pipeline by one tick (mirrors PZ's scan batching).
                        # Feed the next batch_size right records through the right ops each tick,
                        # so the join sees at most batch_size left × batch_size right per call.
                        # Skip for has_right_join pipelines: they run as a background future and
                        # must not be touched inline (r_iq/r_fq are empty, other._ops is non-empty,
                        # and initial records are raw/unprocessed — advancing inline would both
                        # KeyError on r_iq and feed wrong records into pending).
                        if not rs["done"] and not rs["has_right_join"]:
                            n_remaining = len(rs["initial"]) - rs["n_fed"]
                            n_feed = min(batch_size if batch_size is not None else n_remaining, n_remaining)
                            new_right = rs["initial"][rs["n_fed"]:rs["n_fed"] + n_feed]
                            rs["n_fed"] += n_feed
                            if n_r == 0:
                                # No right ops: records are immediately ready
                                rs["pending"].extend(new_right)
                                rs["all_right"].extend(new_right)
                            elif new_right:
                                r_iq[0].extend(new_right)

                            # Advance right ops: harvest upstream → drain final → submit
                            for r_stage, r_op in enumerate(other._ops):
                                if r_stage > 0:
                                    r_iq[r_stage].extend(drain(r_fq, r_stage - 1, timeout=0))
                                if r_stage == n_r - 1:
                                    new_ready = drain(r_fq, r_stage)
                                    rs["pending"].extend(new_ready)
                                    rs["all_right"].extend(new_ready)
                                if r_op.stage_type in ("filter", "convert", "project") and r_iq.get(r_stage):
                                    r_bs = batch_size if batch_size is not None else len(r_iq[r_stage])
                                    r_batch = r_iq[r_stage][:r_bs]
                                    r_iq[r_stage] = r_iq[r_stage][r_bs:]
                                    for rec in r_batch:
                                        f = executor.submit(r_op, rec)
                                        r_fq[r_stage].append(f)
                                        r_op_key = id(r_op)
                                        if len(op_sample_pairs.get(r_op_key, [])) < _MAX_SAMPLES:
                                            op_sample_pairs.setdefault(r_op_key, []).append((rec, f))
                                elif r_op.stage_type == "groupby":
                                    r_upstream_done = all(not r_iq.get(j) and not r_fq.get(j) for j in range(r_stage))
                                    if r_upstream_done and r_iq.get(r_stage):
                                        r_gb = r_iq[r_stage][:]
                                        r_iq[r_stage].clear()
                                        r_fq[r_stage].append(executor.submit(r_op, r_gb))

                            rs["done"] = (
                                rs["n_fed"] >= len(rs["initial"])
                                and not any(r_iq.get(j) or r_fq.get(j) for j in range(n_r))
                            )

                        # Fire join
                        if has_dl:
                            # Incremental path — mirrors PZ's join_has_downstream_limit_op branch.
                            # Pop ALL available left records and ALL pending right records, then call
                            # the join whenever EITHER side is non-empty. NestedLoopsJoin accumulates
                            # _left_input_records / _right_input_records, so a call with one side
                            # empty simply stores the other side for future pairs — nothing is
                            # dropped — and later calls only compute the new cross-terms (total
                            # across calls stays L*R).
                            #
                            # The join runs SYNCHRONOUSLY, exactly like PZ: we do not submit it to
                            # the executor because it mutates shared accumulation state and
                            # concurrent join calls would race (see PZ's note at
                            # parallel_execution_strategy.py:149-151). Popping all available left
                            # (instead of a batch_size slice) gives the join's own thread pool
                            # larger, better-saturated batches; the amount actually available is
                            # already throttled by the upstream operators' batch_size feeding.
                            #
                            # For has_right_join pipelines the right records arrive all at once
                            # when the background future completes (drained into rs["pending"]
                            # above). The incremental path handles this naturally: it fires with
                            # whatever left is available vs the newly-arrived right batch, then
                            # keeps firing on subsequent ticks as more left records come in.
                            left_batch = input_queues[stage_idx][:]
                            right_batch = rs["pending"][:]
                            if left_batch or right_batch:
                                input_queues[stage_idx].clear()
                                rs["pending"] = []
                                _join_call_counts[stage_idx] = _join_call_counts.get(stage_idx, 0) + 1
                                result_set, _ = join_pz_op(left_batch, right_batch)
                                _collect_join_samples(id(op), result_set)
                                if result_set is not None:
                                    future_queues[stage_idx].append(
                                        executor.submit(lambda rset=result_set: rset)
                                    )
                        elif upstream_done(stage_idx) and rs["done"] and input_queues[stage_idx]:
                            # Barrier: wait for all left upstream + right pipeline done,
                            # then join all left × all right in one call.
                            left_batch = input_queues[stage_idx][:]
                            input_queues[stage_idx].clear()
                            _join_call_counts[stage_idx] = _join_call_counts.get(stage_idx, 0) + 1
                            prev_l = len(join_pz_op._left_input_records)
                            prev_r = len(join_pz_op._right_input_records)
                            pairs = (len(left_batch) * len(rs["all_right"])
                                     + len(left_batch) * prev_r
                                     + prev_l * len(rs["all_right"]))
                            # print(f"[join call #{_join_call_counts[stage_idx]} stage={stage_idx} BARRIER] "
                            #       f"left={len(left_batch)} right={len(rs['all_right'])} "
                            #       f"prev_l={prev_l} prev_r={prev_r} => {pairs} pairs")
                            result_set, _ = join_pz_op(left_batch, rs["all_right"])
                            _collect_join_samples(id(op), result_set)
                            if result_set is not None:
                                future_queues[stage_idx].append(
                                    executor.submit(lambda rset=result_set: rset)
                                )

                    elif input_queues[stage_idx] and op.stage_type != "groupby":
                        # filter / convert / project: submit up to batch_size records per tick
                        # (batch_size=None means submit all ready records)
                        # groupby is a barrier and must never be dispatched per-record
                        batch = input_queues[stage_idx][:batch_size]
                        input_queues[stage_idx] = [] if batch_size is None else input_queues[stage_idx][batch_size:]
                        op_key = id(op)
                        for r in batch:
                            f = executor.submit(op, r)
                            future_queues[stage_idx].append(f)
                            if len(op_sample_pairs.get(op_key, [])) < _MAX_SAMPLES:
                                op_sample_pairs.setdefault(op_key, []).append((r, f))

                # Early stop once limit is satisfied
                if limit_val is not None and len(output_records) >= limit_val:
                    break

        # _run is defined above and closes here. Dispatch: if a shared executor was
        # passed in (we're already inside a parent's ThreadPoolExecutor), use it directly
        # so both pipelines share the same worker pool. Otherwise create our own.
        if _executor is not None:
            _run(_executor)
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                _run(executor)

        # if _join_call_counts:
        #     print(f"[join summary] calls per stage: {_join_call_counts}")

        # Resolve per-record sample futures → (input_dr, output_dr | None) pairs.
        # Groupby stages have no samples (can't link inputs to outputs). Join stages
        # are collected separately (join_op_samples) as already-resolved pair tuples.
        # Keyed by id(op) so both main-pipeline and right-branch ops are included.
        op_samples: dict[int, list[tuple]] = {}
        for op_key, pairs in op_sample_pairs.items():
            op_samples[op_key] = []
            for inp, f in pairs:
                try:
                    result = f.result()
                    out_recs = [dr for dr in result.data_records if dr._passed_operator]
                    op_samples[op_key].append((inp, out_recs[0] if out_recs else None))
                except Exception:
                    op_samples[op_key].append((inp, None))

        # Merge in join pair samples (already resolved to (pair_dr, pair_dr | None)).
        op_samples.update(join_op_samples)

        return output_records[:limit_val] if limit_val is not None else output_records, all_record_op_stats, stage_map, op_samples

    def run(self) -> tuple[DataRecordCollection, list[dict], dict]:
        """Execute the pipeline and return (records, per_op_list, plan_dict).

        per_op_list: one dict per operator in pipeline order, each with keys:
            op_id, cost_usd, latency_s, input_tokens, output_tokens,
            num_records, num_passed, op_type, attributes, params_id

        num_records: number of input records the operator processed.
        num_passed:  number that passed (equal to num_records for non-filter ops).
        op_id:       alias for params_id; used by SampleBasedCostModel for lookup.

        plan_dict keys (each value is a single scalar):
            cost_usd, latency_s, input_tokens, output_tokens
        """
        def _fmt_record(dr: DataRecord | None) -> str:
            if dr is None:
                return "∅"
            schema_cls: type[BaseModel] = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            return repr({k: getattr(dr, k, None) for k in schema_cls.model_fields})

        start = time.time()
        records, all_record_op_stats, stage_map, op_samples = self._execute()
        elapsed = time.time() - start

        # Aggregate per-operator stats keyed by logical_op_id (= params_id).
        # num_records counts input invocations; num_passed counts records that
        # passed the operator (for selectivity estimation in filter ops).
        op_agg: dict[str, dict] = {}
        for r in all_record_op_stats:
            lid = r.logical_op_id
            if lid not in op_agg:
                op_agg[lid] = {
                    "cost_usd": 0.0, "latency_s": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                    "num_records": 0, "num_passed": 0,
                }
            op_agg[lid]["cost_usd"] += r.cost_per_record
            op_agg[lid]["latency_s"] += r.time_per_record
            op_agg[lid]["input_tokens"] += int(r.input_text_tokens + r.input_image_tokens + r.input_audio_tokens)
            op_agg[lid]["output_tokens"] += int(r.output_text_tokens)
            op_agg[lid]["num_records"] += 1
            op_agg[lid]["num_passed"] += int(bool(getattr(r, "passed_operator", True)))

        _zero = {
            "cost_usd": 0.0, "latency_s": 0.0,
            "input_tokens": 0, "output_tokens": 0,
            "num_records": 0, "num_passed": 0,
        }
        per_op_list: list[dict] = []
        for stage_idx in sorted(stage_map):
            meta = stage_map[stage_idx]
            s = op_agg.get(meta["logical_op_id"], _zero)
            per_op_list.append({
                "op_name": meta["op_name"],
                "op_type": meta["op_type"],
                "op_id": meta["params_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        # for entry in per_op_list:
        #     print(
        #         f"[pipeline] {entry['op_name']} ({entry['op_type']}): "
        #         f"{entry['num_records']} in, {entry['num_passed']} passed",
        #         flush=True,
        #     )

        total_cost = sum(r.cost_per_record for r in all_record_op_stats)
        total_input_tokens = int(sum(r.input_text_tokens + r.input_image_tokens + r.input_audio_tokens for r in all_record_op_stats))
        total_output_tokens = int(sum(r.output_text_tokens for r in all_record_op_stats))

        op_id_to_name = {id(op): name for name, op in self._flat_ops()}
        op_samples_dict: dict[str, list[dict]] = {}
        for op_key, pairs in op_samples.items():
            if not pairs:
                continue
            op_name = op_id_to_name.get(op_key)
            if op_name is None:
                continue
            op_samples_dict[op_name] = [
                {"input": _fmt_record(inp), "output": _fmt_record(out)}
                for inp, out in pairs
            ]

        plan_dict: dict = {
            "plan_name": self.plan_name,
            "latency_s": elapsed,
            "cost_usd": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "plan_str": str(self),
            "op_samples": op_samples_dict,
        }

        exec_stats = ExecutionStats(
            plan_execution_time=elapsed,
            total_execution_time=elapsed,
            plan_execution_cost=total_cost,
            total_execution_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )
        self._last_exec_stats = exec_stats
        return DataRecordCollection(records, execution_stats=exec_stats), per_op_list, plan_dict

    # ------------------------------------------------------------------
    # Subset execution
    # ------------------------------------------------------------------

    def run_subset(
        self,
        num_samples: int = NUM_SAMPLES,
        seed: int = SUBSET_SEED,
        subset_cache_path: str | None = None,
    ) -> tuple[list[dict], "SubsetExecutionContext", dict]:
        """Execute on a random sample of num_samples records.

        If subset_cache_path is given and the file already exists, the cached
        rows are loaded instead of resampling.  If the file does not exist yet
        the sample is drawn as usual and written to that path.

        Returns (per_op_list, SubsetExecutionContext, plan_dict).
        per_op_list: same schema as run() but stats are from the subset; each op's
          latency_s is a SUM of per-record times (serial-equivalent; used by the cost model).
        SubsetExecutionContext: sampled records, output records, and per-op samples
          (all operators, semantic and non-semantic) for use by QualityEvaluator and get_op_samples.
        plan_dict: plan-level totals; plan_dict["latency_s"] is the WALL-CLOCK time of the
          subset execution (real elapsed, reflecting parallelism).
        """
        import os as _os
        if subset_cache_path is not None and _os.path.exists(subset_cache_path):
            sampled_df = pd.read_csv(subset_cache_path).reset_index(drop=True)
            for col in sampled_df.columns:
                try:
                    sampled_df[col] = sampled_df[col].astype(self._df[col].dtype)
                except (ValueError, TypeError):
                    pass
        else:
            n = min(num_samples, len(self._df))
            sampled_df = self._df.sample(n=n, random_state=seed).reset_index(drop=True)
            if subset_cache_path is not None:
                _os.makedirs(_os.path.dirname(_os.path.abspath(subset_cache_path)), exist_ok=True)
                sampled_df.to_csv(subset_cache_path, index=False)
        sampled_records_list = sampled_df.to_dict(orient="records")

        has_join = any(op.stage_type == "join" for op in self._ops)
        right_df_backup: dict[int, pd.DataFrame] = {}
        right_sampled_records: list[dict] | None = None

        if has_join:
            # Load the cached subset once so a self-join draws both sides from the
            # SAME rows. Independently resampling the right side (below) would give a
            # different row set whenever the cache is stale w.r.t. the source, silently
            # dropping pairs that only exist on one side.
            cached_subset_df: "pd.DataFrame | None" = None
            if subset_cache_path is not None and _os.path.exists(subset_cache_path):
                cached_subset_df = pd.read_csv(subset_cache_path).reset_index(drop=True)

            right_sampled_records = []
            for op in self._ops:
                if op.stage_type == "join" and getattr(op, "self_join", False):
                    # Self-join: right side IS the left side (run once); no separate right _df.
                    right_sampled_records.extend(sampled_records_list)
                    continue
                if op.stage_type == "join":
                    other = op.other
                    oid = id(other)
                    if oid not in right_df_backup:
                        right_df_backup[oid] = other._df
                        # Reuse the cached subset when it covers the right source's
                        # columns (self-join); otherwise fall back to sampling.
                        if cached_subset_df is not None and set(other._df.columns).issubset(cached_subset_df.columns):
                            right_df = cached_subset_df[list(other._df.columns)].copy().reset_index(drop=True)
                            for col in right_df.columns:
                                try:
                                    right_df[col] = right_df[col].astype(other._df[col].dtype)
                                except (ValueError, TypeError):
                                    pass
                            other._df = right_df
                        else:
                            n_r = min(num_samples, len(other._df))
                            other._df = other._df.sample(n=n_r, random_state=seed).reset_index(drop=True)
                    right_sampled_records.extend(other._df.to_dict(orient="records"))

        subset_wall_s = 0.0
        try:
            initial_records = self._build_initial_records(sampled_df)
            _subset_start = time.time()
            records, all_record_op_stats, stage_map, op_samples = self._execute_core(
                initial_records, max_samples=num_samples, skip_limit=True
            )
            subset_wall_s = time.time() - _subset_start
        finally:
            for op in self._ops:
                if op.stage_type == "join" and op.other is not None:
                    oid = id(op.other)
                    if oid in right_df_backup:
                        op.other._df = right_df_backup[oid]

        # Build per_op_list (same aggregation logic as run())
        op_agg: dict[str, dict] = {}
        _zero: dict = {"cost_usd": 0.0, "latency_s": 0.0, "input_tokens": 0,
                       "output_tokens": 0, "num_records": 0, "num_passed": 0}
        for r in all_record_op_stats:
            lid = r.logical_op_id
            if lid not in op_agg:
                op_agg[lid] = dict(_zero)
            op_agg[lid]["cost_usd"] += r.cost_per_record
            op_agg[lid]["latency_s"] += r.time_per_record
            op_agg[lid]["input_tokens"] += int(r.input_text_tokens + r.input_image_tokens + r.input_audio_tokens)
            op_agg[lid]["output_tokens"] += int(r.output_text_tokens)
            op_agg[lid]["num_records"] += 1
            op_agg[lid]["num_passed"] += int(bool(getattr(r, "passed_operator", True)))

        per_op_list: list[dict] = []
        for stage_idx in sorted(stage_map):
            meta = stage_map[stage_idx]
            s = op_agg.get(meta["logical_op_id"], _zero)
            per_op_list.append({
                "op_name": meta["op_name"],
                "op_type": meta["op_type"],
                "op_id": meta["params_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        # Convert output DataRecords to plain dicts; replace None/NaN in str-typed
        # fields with "" so downstream evaluators (e.g. adjusted-rand-index) don't crash.
        def _is_str_ann(ann) -> bool:
            import typing as _t
            if ann is str:
                return True
            if _t.get_origin(ann) is _t.Union:
                return str in _t.get_args(ann)
            return False

        def _dr_to_dict(dr: DataRecord) -> dict:
            import math as _math
            schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            result = {}
            for k, finfo in schema_cls.model_fields.items():
                v = getattr(dr, k, None)
                if _is_str_ann(finfo.annotation) and (
                    v is None or (isinstance(v, float) and _math.isnan(v))
                ):
                    v = ""
                result[k] = v
            return result

        output_records = [_dr_to_dict(dr) for dr in records]

        # Collect per-op info for all operators (semantic and non-semantic): keyed by
        # flat_idx from _flat_ops() so that op names are consistent with per_op_list.
        # op_samples is keyed by id(op), so right-branch ops are included automatically.
        per_sem_op_info: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            per_sem_op_info[flat_idx] = {
                "op_name": op_name,
                "op_type": op.op_type,
                "attributes": op.attributes,
                # {source_indices -> {input_field: text}} for rag_map/rag_filter: the
                # post-retrieval input view the op's LLM call actually received. Recorded by
                # rag_common._record_retrieval_context on the chunked COPY of each candidate,
                # which is why it can't be recovered from `samples` (those hold the untouched
                # original). None for every non-RAG op. See _score_map_op.
                "retrieval_contexts": getattr(
                    getattr(op, "_pz_op", None), "_retrieval_contexts", None
                ),
                "samples": op_samples.get(id(op), []),
            }

        context = SubsetExecutionContext(
            sampled_records=sampled_records_list,
            output_records=output_records,
            per_sem_op_info=per_sem_op_info,
            has_join=has_join,
            right_sampled_records=right_sampled_records,
        )
        # plan_dict mirrors run(): latency_s is the WALL-CLOCK time of the subset
        # _execute_core (real elapsed time, reflecting parallelism), as opposed to the
        # per-op latency_s in per_op_list, which is a SUM of per-record times (serial-
        # equivalent, used by the cost model). Callers that want to report the real
        # subset execution time should use plan_dict["latency_s"].
        plan_dict = {
            "plan_name": self.plan_name,
            "latency_s": subset_wall_s,
            "cost_usd": sum(op["cost_usd"] for op in per_op_list),
            "input_tokens": sum(op["input_tokens"] for op in per_op_list),
            "output_tokens": sum(op["output_tokens"] for op in per_op_list),
        }
        return per_op_list, context, plan_dict

    # ------------------------------------------------------------------
    # Oracle copy (for QualityEvaluator)
    # ------------------------------------------------------------------

    def make_oracle_copy(self, oracle_model: "Model | str", oracle_reasoning_effort: str | None = None) -> "PhysicalPipeline":
        """Return a copy of this pipeline with all LLM operators using oracle_model.

        oracle_model may be a pz.Model enum (preferred) or a model string
        that will be resolved via _str_to_pz_model.
        """
        if isinstance(oracle_model, str):
            oracle_model = _str_to_pz_model(oracle_model)
        oracle = PhysicalPipeline(
            plan_name=self.plan_name + "_oracle",
            source=self._source,
            data=self._df,
            max_workers=self._max_workers,
        )
        for op in self._ops:
            if isinstance(op, SemFilter):
                oracle.sem_filter(
                    condition=op.attributes["condition"],
                    model=oracle_model,
                    depends_on=getattr(op, "depends_on", None),
                    reasoning_effort_override=oracle_reasoning_effort,
                )
            elif isinstance(op, SemMap):
                cols = getattr(op, "_cols_full", None)
                if cols:
                    oracle.sem_map(
                        cols=cols,
                        model=oracle_model,
                        depends_on=getattr(op, "depends_on", None),
                        reasoning_effort_override=oracle_reasoning_effort,
                    )
            elif isinstance(op, RagFilter):
                oracle.rag_filter(
                    condition=op.attributes["condition"],
                    embedding_query=op.attributes["embedding_query"],
                    model=oracle_model,
                    chunk_size=op.attributes.get("chunk_size", 20000),
                    num_chunks_per_field=op.attributes.get("num_chunks_per_field"),
                    similarity_threshold=op.attributes.get("similarity_threshold"),
                    embedding_model=op.attributes.get("embedding_model", DEFAULT_RAG_EMBEDDING_MODEL),
                    similarity_method=op.attributes.get("similarity_method", "embedding"),
                    depends_on=getattr(op, "depends_on", None),
                    reasoning_effort_override=oracle_reasoning_effort,
                )
            elif isinstance(op, RagMap):
                cols = getattr(op, "_cols_full", None)
                if cols:
                    oracle.rag_map(
                        cols=cols,
                        embedding_query=op.attributes["embedding_query"],
                        model=oracle_model,
                        chunk_size=op.attributes.get("chunk_size", 20000),
                        num_chunks_per_field=op.attributes.get("num_chunks_per_field"),
                        similarity_threshold=op.attributes.get("similarity_threshold"),
                        embedding_model=op.attributes.get("embedding_model", DEFAULT_RAG_EMBEDDING_MODEL),
                        similarity_method=op.attributes.get("similarity_method", "embedding"),
                        depends_on=getattr(op, "depends_on", None),
                        reasoning_effort_override=oracle_reasoning_effort,
                    )
            elif isinstance(op, SemJoin):
                # For a self-join, join the oracle with itself; else oracle-copy the right side.
                oracle_other = oracle if op.self_join else op.other.make_oracle_copy(oracle_model, oracle_reasoning_effort)
                oracle.sem_join(
                    other=oracle_other,
                    condition=op.attributes["condition"],
                    model=oracle_model,
                    join_parallelism=op.attributes.get("join_parallelism", 20),
                    depends_on=getattr(op, "depends_on", None),
                    reasoning_effort_override=oracle_reasoning_effort,
                )
            elif isinstance(op, Join):
                oracle_other = oracle if op.self_join else op.other.make_oracle_copy(oracle_model, oracle_reasoning_effort)
                fn = getattr(op, "_fn", None)
                if fn is not None:
                    oracle.join(
                        other=oracle_other,
                        condition_fn=fn,
                        depends_on=getattr(op, "depends_on", None),
                    )
            elif isinstance(op, ExactFilter):
                fn = getattr(op, "_fn", None)
                if fn is not None:
                    oracle.filter(fn)
            elif isinstance(op, Map):
                udf = getattr(op, "_udf", None)
                cols = getattr(op, "_cols_full", None)
                if udf is not None and cols:
                    oracle.map(udf, cols)
            elif isinstance(op, AddColSuffix):
                oracle.add_col_suffix(op.suffix)
            elif isinstance(op, Project):
                oracle.project(op.attributes["project_cols"])
            elif isinstance(op, Limit):
                oracle.limit(op.attributes["limit"])
            elif isinstance(op, GroupBy):
                group_by_fields = op.attributes["group_by_fields"]
                agg_pairs = list(op.attributes["agg_pairs"])
                agg_funcs = [p[0] for p in agg_pairs]
                agg_fields = [p[1] for p in agg_pairs]
                oracle.groupby(group_by_fields, agg_funcs, agg_fields)
        return oracle

    # def to_results_row(self, plan_id: str) -> dict:
    #     """Return a ResultsStore-compatible dict from the most recent run().

    #     Captures plan-level totals only (not per-operator breakdowns).
    #     Canonical columns match the ResultsStore schema in cost_model_agent:
    #         plan_id, cost_usd, latency_s, input_tokens, output_tokens
    #     """
    #     if self._last_exec_stats is None:
    #         raise RuntimeError("Call run() before to_results_row()")
    #     s = self._last_exec_stats
    #     return {
    #         "plan_id": plan_id,
    #         "cost_usd": s.total_execution_cost,
    #         "latency_s": s.plan_execution_time,
    #         "input_tokens": int(s.total_input_tokens),
    #         "output_tokens": int(s.total_output_tokens),
    #     }
