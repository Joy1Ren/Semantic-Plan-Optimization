"""The execution engine behind PhysicalPipeline: runs the plan's operators over a thread pool and
collects per-operator stats and samples."""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import pandas as pd

from palimpzest.core.elements.records import DataRecord, DataRecordCollection

from raptor.subsets import load_subset

from .operators import RagFilter, RagMap

MAX_ERRORS_PER_OP = 3
MAX_ERROR_CHARS = 300

# A full-run prompt estimated at this share of its model's long-prompt threshold gets a warning.
LONG_PROMPT_WARN_SHARE = 0.8

# Input/output pairs kept per operator: the most get_op_samples can show.
MAX_OP_SAMPLES = 5


@dataclass
class SubsetExecutionContext:
    """What run_subset() returns besides the per-op and plan stats."""
    sampled_records: list[dict]               # sampled left-side input rows as plain dicts
    output_records: list[dict]                # final pipeline output rows as plain dicts
    per_sem_op_info: dict[int, dict]          # op index → {op_name, op_type, attributes, samples, retrieval_contexts}
    has_join: bool
    right_sampled_records: list[dict] | None  # right-side sampled rows if join, else None


class PipelineExecution:
    """Execution methods of PhysicalPipeline."""

    def _build_initial_records(self, df: pd.DataFrame) -> list[DataRecord]:
        """Build DataRecord objects from a DataFrame using this pipeline's initial schema."""
        records: list[DataRecord] = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            dr = DataRecord(data_item=self._initial_schema(**row), source_indices=f"{self._source}-{i}")
            records.append(dr)
        return records

    def _execute_core(self, initial_records: list[DataRecord], max_samples: int, skip_limit: bool = False) -> tuple[list[DataRecord], list, dict, dict]:
        from concurrent.futures import wait as fut_wait
        _POLL_INTERVAL = 0.3

        all_record_op_stats = []
        stage_map: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            op.errors = []
            if isinstance(op, (RagFilter, RagMap)):
                # Reset the retrieval contexts captured by the last run
                op._pz_op._retrieval_contexts = {}
            op._pz_op.logical_op_id = str(id(op))
            if op._skip_op is not None:
                op._skip_op.logical_op_id = str(id(op))
            # Reset a join operator's cross-call accumulation before each run
            if op.stage_type == "join":
                op._pz_op._left_input_records = []
                op._pz_op._right_input_records = []
                op._pz_op._left_joined_record_ids = set()
                op._pz_op._right_joined_record_ids = set()
                op._pz_op.join_idx = 0
                op._pz_op.finished = False
            stage_map[flat_idx] = {
                "op_name": op_name,
                "op_type": op.op_type,
                "attributes": op.attributes,
                "op_id": op.op_id,
                "stat_key": str(id(op)),
            }

        if not self._ops:
            return initial_records, all_record_op_stats, stage_map, {}
        ops = [op for _, op in self._flat_ops()]
        node_of = {id(op): i for i, op in enumerate(ops)}
        slots: list[dict[str, list]] = [{} for _ in ops]
        producers: list[list[int]] = [[] for _ in ops]       # nodes feeding node i
        consumer: dict[int, tuple[int, str] | None] = {}     # node -> (node, slot) it feeds; None = plan output
        upstream: list[set[int]] = [set() for _ in ops]      # every node upstream of node i

        def wire(pipe, records: list[DataRecord]) -> int | list[DataRecord]:
            """Chain `pipe`'s operators onto its source `records`; return the node producing its
            output, or the records themselves if it has no operators."""
            prev = records
            for op in pipe._ops:
                i = node_of[id(op)]
                if slots[i]:
                    raise ValueError(
                        f"Pipeline {pipe.plan_name!r} feeds more than one join; build a separate pipeline for each use."
                    )
                inputs = {"in": prev}
                if op.stage_type == "join" and not getattr(op, "self_join", False):
                    inputs = {"left": prev, "right": wire(op.other, op.other._build_initial_records(op.other._df))}
                for slot, src in inputs.items():
                    if isinstance(src, int):
                        slots[i][slot] = []
                        producers[i].append(src)
                        consumer[src] = (i, slot)
                        upstream[i] |= upstream[src] | {src}
                    else:
                        slots[i][slot] = list(src)
                prev = i
            return prev

        consumer[wire(self, initial_records)] = None

        def downstream_limit(i: int) -> bool:
            dest = consumer[i]
            while dest is not None:
                if ops[dest[0]].stage_type == "limit":
                    return True
                dest = consumer[dest[0]]
            return False

        # A join with a limit downstream fires incrementally; otherwise it waits for all upstream records
        incremental = {i: downstream_limit(i) for i, op in enumerate(ops) if op.stage_type == "join"}

        op_sample_pairs: dict[int, list] = {}  # id(op) -> [(input_dr, future)]
        join_op_samples: dict[int, list[tuple]] = {}  # id(join_op) -> [(pair_dr, pair_dr if passed else None)]

        def _collect_join_samples(op_key: int, result_set) -> None:
            if result_set is None or not result_set.data_records:
                return
            bucket = join_op_samples.setdefault(op_key, [])
            for dr in result_set.data_records[:max_samples - len(bucket)]:
                passed = bool(dr._passed_operator)
                bucket.append((dr, dr if passed else None))

        futures: list[list] = [[] for _ in ops]
        limit_passed = [0] * len(ops)
        output_records: list[DataRecord] = []
        limit_val = None if skip_limit else next((op.n for op in self._ops if op.stage_type == "limit"), None)
        batch_size = limit_val

        def any_pending() -> bool:
            return any(futures[i] or any(slots[i].values()) for i in range(len(ops)))

        def upstream_done(i: int) -> bool:
            return all(not futures[u] and not any(slots[u].values()) for u in upstream[i])

        def take(i: int, slot: str, n: int | None = None) -> list[DataRecord]:
            records = slots[i][slot]
            slots[i][slot] = [] if n is None else records[n:]
            return records if n is None else records[:n]

        def deliver(i: int, records: list[DataRecord]) -> None:
            dest = consumer[i]
            if dest is None:
                output_records.extend(records)
            else:
                slots[dest[0]][dest[1]].extend(records)

        def drain(i: int) -> list[DataRecord]:
            if not futures[i]:
                return []
            done, not_done = fut_wait(futures[i], timeout=_POLL_INTERVAL)
            futures[i] = list(not_done)
            passing = []
            for future in done:
                try:
                    result = future.result()
                    all_record_op_stats.extend(result.record_op_stats)
                    passing.extend(dr for dr in result.data_records if dr._passed_operator)
                except Exception as _exc:
                    print(f"[pipeline] record-level operator error (skipping record): {type(_exc).__name__}: {_exc}")
            return passing

        def _schedule(executor):
            def submit_records(op, i: int, records: list[DataRecord]) -> None:
                for r in records:
                    f = executor.submit(op, r)
                    futures[i].append(f)
                    if len(op_sample_pairs.get(id(op), [])) < max_samples:
                        op_sample_pairs.setdefault(id(op), []).append((r, f))

            def submit_barrier(op, i: int, records: list[DataRecord]) -> None:
                # for aggregation operators, one future per group once all upstream records are done
                for group in op.split_groups(records):
                    f = executor.submit(op, group)
                    futures[i].append(f)
                    if len(op_sample_pairs.get(id(op), [])) < max_samples:
                        op_sample_pairs.setdefault(id(op), []).append((group, f))

            def fire_join(op, i: int, left: list[DataRecord], right: list[DataRecord]) -> None:
                result_set, _ = op._pz_op(left, right)
                _collect_join_samples(id(op), result_set)
                if result_set is not None:
                    futures[i].append(executor.submit(lambda rset=result_set: rset))

            while any_pending():
                for i, op in enumerate(ops):
                    for p in producers[i]:
                        deliver(p, drain(p))
                    if consumer[i] is None:
                        deliver(i, drain(i))

                    if op.stage_type == "limit":
                        records = take(i, "in")
                        if records and not skip_limit:
                            records = records[:max(op.n - limit_passed[i], 0)]
                            limit_passed[i] += len(records)
                        deliver(i, records)

                    elif op.stage_type == "groupby":
                        if slots[i]["in"] and upstream_done(i):
                            submit_barrier(op, i, take(i, "in"))

                    elif op.stage_type == "join" and getattr(op, "self_join", False):
                        # the upstream ran once; join its output with itself
                        if slots[i]["in"] and (incremental[i] or upstream_done(i)):
                            records = take(i, "in")
                            fire_join(op, i, records, records)

                    elif op.stage_type == "join":
                        if (slots[i]["left"] or slots[i]["right"]) and (incremental[i] or upstream_done(i)):
                            fire_join(op, i, take(i, "left"), take(i, "right"))

                    elif slots[i]["in"]:
                        # filter / convert / project: submit up to batch_size records per tick
                        submit_records(op, i, take(i, "in", batch_size))

                # Early stop once limit is satisfied
                if limit_val is not None and len(output_records) >= limit_val:
                    break

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            _schedule(executor)

        op_samples: dict[int, list[tuple]] = {}
        for op_key, pairs in op_sample_pairs.items():
            op_samples[op_key] = []
            for inp, f in pairs:
                try:
                    result = f.result()
                    out_recs = [dr for dr in result.data_records if dr._passed_operator]
                    if isinstance(inp, list):
                        op_samples[op_key].append((inp, out_recs or None))
                    else:
                        op_samples[op_key].append((inp, out_recs[0] if out_recs else None))
                except Exception:
                    op_samples[op_key].append((inp, None))

        # Merge in join pair samples
        op_samples.update(join_op_samples)

        return output_records[:limit_val] if limit_val is not None else output_records, all_record_op_stats, stage_map, op_samples

    def _run(self, df: pd.DataFrame, skip_limit: bool) -> tuple[list[DataRecord], list[dict], dict, dict, list]:
        """Execute on `df`; return (records, per_op_list, plan_dict, op_samples, all_record_op_stats).

        per_op_list: one dict per operator in _flat_ops() order, with keys op_name, op_type, op_id,
          cost_usd, latency_s, input_tokens, output_tokens, num_records, num_passed.
          num_records: number of input records the operator processed.
          num_passed:  number that passed (equal to num_records for non-filter ops).
          op_id:       hash of op_type + attributes (Operator.op_id).
          latency_s:   a SUM of per-record times (serial-equivalent).
        plan_dict: plan_name, cost_usd, input_tokens, output_tokens, op_errors ({op_name:
          ["<count>x <message>"]}, see _op_errors), and latency_s, the WALL-CLOCK time of the run.
        """
        start = time.time()
        initial_records = self._build_initial_records(df)
        records, all_record_op_stats, stage_map, op_samples = self._execute_core(
            initial_records, max_samples=MAX_OP_SAMPLES, skip_limit=skip_limit
        )
        elapsed = time.time() - start

        # Aggregate per-operator stats by stat_key
        _zero = {"cost_usd": 0.0, "latency_s": 0.0, "input_tokens": 0,
                 "output_tokens": 0, "num_records": 0, "num_passed": 0}
        op_agg: dict[str, dict] = {}
        for r in all_record_op_stats:
            agg = op_agg.setdefault(r.logical_op_id, dict(_zero))
            agg["cost_usd"] += r.cost_per_record
            agg["latency_s"] += r.time_per_record
            agg["input_tokens"] += int(r.input_text_tokens + r.input_image_tokens + r.input_audio_tokens)
            agg["output_tokens"] += int(r.output_text_tokens)
            agg["num_records"] += 1
            agg["num_passed"] += int(bool(getattr(r, "passed_operator", True)))

        per_op_list: list[dict] = []
        for stage_idx in sorted(stage_map):
            meta = stage_map[stage_idx]
            s = op_agg.get(meta["stat_key"], _zero)
            per_op_list.append({
                "op_name": meta["op_name"],
                "op_type": meta["op_type"],
                "op_id": meta["op_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        plan_dict = {
            "plan_name": self.plan_name,
            "latency_s": elapsed,
            "cost_usd": sum(op["cost_usd"] for op in per_op_list),
            "input_tokens": sum(op["input_tokens"] for op in per_op_list),
            "output_tokens": sum(op["output_tokens"] for op in per_op_list),
            "op_errors": self._op_errors(),
        }
        return records, per_op_list, plan_dict, op_samples, all_record_op_stats

    def run(self) -> tuple[DataRecordCollection, list[dict], dict]:
        """Execute on the full dataset, applying limits; return (records, per_op_list, plan_dict).
        See _run for per_op_list and plan_dict."""
        records, per_op_list, plan_dict, _, _ = self._run(self._df, skip_limit=False)
        return DataRecordCollection(records), per_op_list, plan_dict

    def run_subset(self, subset_path: str) -> tuple[list[dict], "SubsetExecutionContext", dict]:
        """Execute on the optimization subset saved at `subset_path`.

        Limits are ignored. Returns (per_op_list, SubsetExecutionContext, plan_dict).
        per_op_list, plan_dict: as in _run, from the subset; plan_dict also holds
          long_prompt_warnings.
        SubsetExecutionContext: sampled records, output records, and per-op samples.
        """
        import os as _os
        if not _os.path.exists(subset_path):
            raise FileNotFoundError(f"No optimization subset at {subset_path}: build it with the sampler first.")
        sampled_df = load_subset(subset_path, self._df).reset_index(drop=True)
        for col in sampled_df.columns:
            try:
                sampled_df[col] = sampled_df[col].astype(self._df[col].dtype)
            except (ValueError, TypeError):
                pass
        sampled_records_list = sampled_df.to_dict(orient="records")

        has_join = any(op.stage_type == "join" for op in self._ops)
        right_df_backup: dict[int, pd.DataFrame] = {}
        right_sampled_records: list[dict] | None = None

        if has_join:
            # Every non-self join's right table must have its columns in the subset
            cached_subset_df = load_subset(subset_path, self._df).reset_index(drop=True)
            for op in self._ops:
                if op.stage_type == "join" and not getattr(op, "self_join", False):
                    missing = set(op.other._df.columns) - set(cached_subset_df.columns)
                    if missing:
                        raise ValueError(
                            f"The right side of join {op.op_type!r} reads a table with no subset at "
                            f"{subset_path} (columns not in it: {sorted(missing)})."
                        )

            right_sampled_records = []
            for op in self._ops:
                if op.stage_type == "join" and getattr(op, "self_join", False):
                    right_sampled_records.extend(sampled_records_list)
                    continue
                if op.stage_type == "join":
                    other = op.other
                    oid = id(other)
                    if oid not in right_df_backup:
                        right_df_backup[oid] = other._df
                        right_df = cached_subset_df[list(other._df.columns)].copy().reset_index(drop=True)
                        for col in right_df.columns:
                            try:
                                right_df[col] = right_df[col].astype(other._df[col].dtype)
                            except (ValueError, TypeError):
                                pass
                        other._df = right_df
                    right_sampled_records.extend(other._df.to_dict(orient="records"))

        try:
            records, per_op_list, plan_dict, op_samples, all_record_op_stats = self._run(sampled_df, skip_limit=True)
        finally:
            for op in self._ops:
                if op.stage_type == "join" and op.other is not None:
                    oid = id(op.other)
                    if oid in right_df_backup:
                        op.other._df = right_df_backup[oid]

        # Convert output DataRecords to plain dicts
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

        # Collect per-op info for all operators, keyed by flat_idx from _flat_ops()
        per_sem_op_info: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            per_sem_op_info[flat_idx] = {
                "op_name": op_name,
                "op_type": op.op_type,
                "attributes": op.attributes,
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
        plan_dict["long_prompt_warnings"] = self._long_prompt_warnings(all_record_op_stats, op_samples)
        return per_op_list, context, plan_dict

    def _long_prompt_warnings(self, all_record_op_stats: list, op_samples: dict) -> list[str]:
        """Warnings for LLM operators whose prompts could pass their model's long-prompt threshold
        when the plan runs on the full dataset (self._df), where such a call is billed at higher
        rates. Token counts measured on the subset are scaled up, assuming every full-dataset
        record reaches the operator: a sem_agg's call by the number of source rows behind it, a
        sem_map's or sem_filter's per-record calls by each full-dataset record's length."""
        from raptor.pricing import long_prompt_tier

        stats_by_op: dict[str, list] = {}
        for r in all_record_op_stats:
            stats_by_op.setdefault(r.logical_op_id, []).append(r)

        def prompt_tokens(r) -> int:
            return int(r.input_text_tokens + r.input_image_tokens + r.input_audio_tokens
                       + r.cache_read_tokens + r.cache_creation_tokens)

        n_full = len(self._df)
        warnings = []
        for op_name, op in self._flat_ops():
            stats = stats_by_op.get(str(id(op)))
            if op.op_type not in ("sem_agg", "sem_map", "sem_filter") or not stats:
                continue
            model = op.attributes["model"]
            try:
                tier = long_prompt_tier(model)
            except KeyError:
                continue
            if tier is None:
                continue
            threshold, (base_in, base_out), (long_in, long_out) = tier
            rates = (f"Above it, input is billed at ${long_in * 1e6:.2f}/1M instead of ${base_in * 1e6:.2f}/1M "
                     f"and output at ${long_out * 1e6:.2f}/1M instead of ${base_out * 1e6:.2f}/1M, for the "
                     f"whole call. `cost_usd` is the subset price and does not include this.")
            if op.op_type == "sem_agg":
                n_rows = len({i for r in stats for i in r.record_source_indices})
                n_read = len({pid for r in stats for pid in (r.record_parent_ids or [])})
                if not n_read:
                    continue
                per_record = sum(prompt_tokens(r) for r in stats) / n_read
                largest_group = max((len(group) for group, _ in op_samples.get(id(op), [])), default=n_read)
                subset_call = per_record * largest_group
                full_call = subset_call * max(n_full / n_rows, 1)
                if full_call < LONG_PROMPT_WARN_SHARE * threshold:
                    continue
                where = "above" if full_call > threshold else "close to"
                warnings.append(
                    f"`{op_name}` ({model}) read {subset_call / 1000:.0f}k prompt tokens in one call on "
                    f"the subset. On the full dataset this call could read up to ~{full_call / 1000:.0f}k "
                    f"tokens (if every record reaches it), {where} this model's "
                    f"{threshold // 1000}k-token long-prompt threshold. {rates}")
                continue

            cols = [c for c in (op.depends_on or self._df.columns) if c in self._df.columns]
            subset_chars = sum(len(str(r.record_state.get(c, ""))) for r in stats for c in cols)
            if not subset_chars:
                continue
            per_char = sum(prompt_tokens(r) for r in stats) / subset_chars
            full_tokens = self._df[cols].astype(str).apply(lambda col: col.str.len()).sum(axis=1) * per_char
            n_near = int((full_tokens >= LONG_PROMPT_WARN_SHARE * threshold).sum())
            if not n_near:
                continue
            n_over = int((full_tokens > threshold).sum())
            n_subset_over = sum(prompt_tokens(r) > threshold for r in stats)
            warnings.append(
                f"`{op_name}` ({model}) makes one call per record. On the subset, {n_subset_over} of its "
                f"{len(stats)} calls were above this model's {threshold // 1000}k-token long-prompt "
                f"threshold. On the full dataset, an estimated {n_over} of {n_full} records would be "
                f"above it, and {n_near} at {LONG_PROMPT_WARN_SHARE:.0%} of it or more (if every "
                f"record reaches this operator). {rates}")
        return warnings

    def _op_errors(self) -> dict[str, list[str]]:
        """The errors each operator hit in the last execution, most frequent first, as
        "<count>x <message>". An operator that raised on a record dropped that record from the
        run; a failed or unparseable LLM call let the record through with a failed filter
        verdict or None fields. Empty when nothing went wrong."""
        report = {}
        for op_name, op in self._flat_ops():
            if op.errors:
                counts = Counter(msg[:MAX_ERROR_CHARS] for msg in op.errors)
                report[op_name] = [f"{n}x {msg}" for msg, n in counts.most_common(MAX_ERRORS_PER_OP)]
        return report
