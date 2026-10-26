"""Shared constants, helpers, and the `Operator` base class used across all
physical-pipeline operators (agent_cost_model/physical_pipeline/operators/*.py) and
by the `PhysicalPipeline` fluent interface (pipeline.py)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.lib.schemas import ImageFilepath, _create_pickleable_model

# Re-exported from the palimpzest-free leaf module so `agent_cost_model.sampling` and the
# analysis scripts can import these without pulling in palimpzest via this module.
from .constants import (  # noqa: F401
    DEFAULT_RAG_EMBEDDING_MODEL,
    NUM_SAMPLES,
    SUBSET_SEED,
    _RAG_EMBEDDING_BASE_URL,
)
from .openrouter_models import register_openrouter_models


def _patch_convert_empty_field_answers() -> None:
    """Guard installed palimpzest 1.5.3's `ConvertOp._create_data_records_from_field_answers`
    (query/operators/convert.py) against crashing with `ValueError: max() iterable argument is
    empty` when a Convert/RAGConvert call returns a completely empty `field_answers` dict for a
    record -- i.e. `max([len(lst) for lst in field_answers.values()])` over zero keys.

    Observed on CUAD sem_map/rag_map calls (e.g. gpt-5-nano) whose completion omits some of the
    requested JSON fields entirely: get_json_from_answer/_prepare_field_answers then raises a
    KeyError for the missing field, which generators.py's Generator.__call__ catches and dumps to
    parse-answer-errors/*.txt -- but every fallback path that's supposed to backfill the missing
    fields with None still keys the dict by every requested field name, so field_answers should
    never actually come back with zero keys. Whatever specific path produces the empty dict, this
    guard treats it exactly like the ALREADY-CORRECT "generation failed for a known field" case
    the un-patched code already handles a few lines up (LLMConvert.__call__ replaces a None answer
    with []): it rebuilds field_answers as {field: [] for field in self.generated_fields} and
    delegates to the original method, which -- given a non-empty dict of empty lists -- computes
    n_records = max([0, 0, ...]) = 0 (no crash, since the list itself isn't empty) but still loops
    `range(max(n_records, 1))` once, producing exactly ONE output record with every one of this
    op's own target fields set to None and every OTHER (parent) field carried through unchanged.
    That single-record-with-None-fields shape is what downstream CUAD scoring
    (experiments/cuad/quality_evaluator.py's _predicted_records) already
    interprets as "this clause type is not present in the document" -- it filters None/NaN/blank
    values out of the clauses list entirely, which is exactly the semantics we want.

    Deliberately NOT `return [], False` (zero records): in physical_pipeline's chained pipelines
    (e.g. CUAD's Q1_p1, nine sequential rag_map calls each adding a few columns), zero records out
    of one op means the record never reaches any downstream op -- the WHOLE document silently
    disappears from the plan's final output, not just this op's own columns. That underscores an
    entire document as a blanket false negative on all 41 clause types (it also drops the document
    from the scored population, which is derived from the plan's own predicted rows -- see
    quality_evaluator.py's _evaluate), which is a much bigger and less honest failure
    than "this op's own few clauses are absent". Idempotent.
    """
    from palimpzest.core.elements.records import DataRecord
    from palimpzest.query.operators.convert import ConvertOp

    if getattr(ConvertOp, "_agent_cost_model_empty_field_answers_patched", False):
        return
    _orig = ConvertOp._create_data_records_from_field_answers

    def _guarded(self, field_answers, candidate):
        if not field_answers and not self.generated_fields:
            # A DIFFERENT cause with the same crash: the operator generates no fields at all,
            # because every column it declares was already in its input schema (converts only ADD
            # columns). Rebuilding field_answers below would leave it empty and still hit max([]).
            # PhysicalPipeline._check_new_cols rejects this when the plan is built, so reaching
            # here means the operator was constructed some other way. Pass the record through
            # unchanged -- which is all the op could have done anyway.
            print(
                f"[physical_pipeline] {type(self).__name__} generates NO fields: every column it "
                "declares already exists in its input schema, and a convert can only add columns, "
                "never overwrite one. Passing the record through unchanged; the op is a no-op."
            )
            return [DataRecord.from_parent(self.output_schema, {}, parent_record=candidate, cardinality_idx=0)], True
        if not field_answers:
            print(
                f"[physical_pipeline] {type(self).__name__} got an EMPTY field_answers dict for "
                f"a record (expected fields: {self.generated_fields}) -- the LLM call likely "
                "failed to produce any of the requested fields. Treating every one of this op's "
                "fields as absent for this record (not dropping the record); check "
                "parse-answer-errors/ for the raw completion that triggered this."
            )
            field_answers = {field: [] for field in self.generated_fields}
        return _orig(self, field_answers, candidate)

    ConvertOp._create_data_records_from_field_answers = _guarded
    ConvertOp._agent_cost_model_empty_field_answers_patched = True


_patch_convert_empty_field_answers()


def _flatten_dict_answer(answer: dict) -> str:
    """Render a JSON object answer as 'key: value; key: value'.

    Models answering a multi-part CUAD field (e.g. 'Termination For Convenience') often key the
    spans by the party or sub-clause they came from -- {'Company': "60 days' prior written
    notice", 'Supplier': ...} -- rather than emitting the requested single string. The keys carry
    real information, so keep them as labels instead of dropping to just the values. Pairs are
    joined with '; ' because the values themselves routinely contain commas, and values are run
    back through `_coerce_str_answer` so nested lists/objects/numbers flatten too. Empty and None
    values are skipped, so an empty object flattens to '' (a valid `str`, unlike None).
    """
    parts = []
    for key, value in answer.items():
        rendered = _coerce_str_answer(value)
        rendered = "" if rendered is None else str(rendered).strip()
        if rendered:
            parts.append(f"{key}: {rendered}")
    return "; ".join(parts)


def _coerce_str_answer(answer):
    """Render one LLM answer as the `str` its output_schema field is annotated with.

    Lists become the comma-separated string CUAD's task prompt asks for; dicts are flattened by
    `_flatten_dict_answer`; JSON booleans become lowercase 'true'/'false' (the literal spelling
    the model emitted, and the spelling column descriptions asking for a true/false flag use,
    rather than Python's 'True'/'False'); ints and floats become their str(). Anything else (str,
    None, ...) is returned untouched -- None is valid for the `str | None` annotations
    _make_schema produces.
    """
    if isinstance(answer, list):
        return ", ".join(str(_coerce_str_answer(v)) for v in answer)
    if isinstance(answer, dict):
        return _flatten_dict_answer(answer)
    # bool before int: bool IS an int subclass, and str(True) would give 'True'.
    if isinstance(answer, bool):
        return "true" if answer else "false"
    if isinstance(answer, (int, float)):
        return str(answer)
    return answer


def _patch_convert_coerce_str_field_answers() -> None:
    """Guard installed palimpzest 1.5.3's `ConvertOp._create_data_records_from_field_answers`
    against a pydantic ValidationError when an LLM returns a JSON array, object, boolean, or
    number for a field whose output_schema annotation is `str` (or `str | None`), instead of the
    single string the schema -- and CUAD's task prompt, which asks for "a comma-separated list of
    text spans" -- expects.

    Observed on rag_map calls (e.g. gpt-5-mini) for fields like 'Revenue/Profit Sharing' and
    'Affiliate IP License-Licensor': the model emits `["span one", "span two"]` for a field
    typed `str | None` instead of the requested comma-joined string, and pydantic rejects the
    list outright (`Input should be a valid string [type=string_type]`) inside
    `DataRecord.from_parent`, which currently propagates all the way up to a record-level
    "skipping record" error and drops the whole record's other 40 CUAD fields along with it.
    Coerce to the string form the model was already asked to produce, rather than losing the
    record over one mis-typed field.

    The same ValidationError arrives via booleans whenever a plan adds a `str`-typed presence
    flag whose description asks for 'true'/'false' (a two-stage "classify presence, then extract
    spans" plan, which the agent writes on its own): the model answers with a JSON `true`, and
    pydantic v2 does NOT coerce bool -> str even in lax mode, so that one flag would again take
    the whole record's other 40 fields down with it. Numbers are folded in for the same reason --
    a 'Warranty Duration' or 'Minimum Commitment' answered as `12` rather than `"12"` is a
    correct answer in the wrong JSON type, not a reason to drop the document.

    JSON objects arrive the same way on multi-part fields -- 'Termination For Convenience' as
    {'Company': "60 days' prior written notice", ...} or 'License Grant' as {'Initial Term':
    '10 years', ...} -- where the model breaks its answer out by party or sub-clause. Those are
    flattened to 'key: value; key: value' rather than dropped, so the labels survive alongside
    the spans.

    Note this only rescues fields the plan declared as `str`; declaring a flag column as `bool`
    remains the better plan-side choice, since pydantic coerces both `true` and `'true'` into it.
    """
    from palimpzest.query.operators.convert import ConvertOp

    if getattr(ConvertOp, "_agent_cost_model_coerce_str_field_answers_patched", False):
        return
    _orig = ConvertOp._create_data_records_from_field_answers

    def _coerced(self, field_answers, candidate):
        model_fields = getattr(self.output_schema, "model_fields", {})
        for field, answers in field_answers.items():
            if not isinstance(answers, list):
                continue
            field_info = model_fields.get(field)
            if field_info is None or field_info.annotation not in (str, str | None):
                continue
            field_answers[field] = [_coerce_str_answer(answer) for answer in answers]
        return _orig(self, field_answers, candidate)

    ConvertOp._create_data_records_from_field_answers = _coerced
    ConvertOp._agent_cost_model_coerce_str_field_answers_patched = True


_patch_convert_coerce_str_field_answers()


def _patch_get_json_from_answer() -> None:
    """Make installed palimpzest 1.5.3's `get_json_from_answer` recover two answers json.loads
    rejects, which otherwise give the record None for every field the op generates. Palimpzest
    handles both only for Llama models.

    "Extra data": gpt-4o-mini often answers a map with the JSON, then the `---` palimpzest's
    prompt asks it to finish with, then the same JSON again. The parser keeps everything from the
    first `{` to the last `}`, so keep just the first value.

    "Expecting value": the model writes Python's True/False/None instead of JSON's
    true/false/null. Read it as a Python literal, which leaves those words inside strings alone.

    A recovered value counts only if it is the object (one-to-one) or array (one-to-many) the
    parser was looking for; a bare answer like `11` keeps the original error."""
    from palimpzest.constants import Cardinality
    from palimpzest.query.generators import generators

    if getattr(generators, "_agent_cost_model_get_json_patched", False):
        return
    _orig = generators.get_json_from_answer

    def _parse(answer, model, cardinality):
        try:
            return _orig(answer, model, cardinality)
        except json.JSONDecodeError as e:
            expected = dict if cardinality == Cardinality.ONE_TO_ONE else list
            try:
                if e.msg == "Extra data":
                    value = json.loads(e.doc[: e.pos])
                elif e.msg == "Expecting value":
                    value = ast.literal_eval(e.doc)
                else:
                    raise e
            except (ValueError, SyntaxError):
                raise e from None
            if not isinstance(value, expected):
                raise e
            return value

    generators.get_json_from_answer = _parse
    generators._agent_cost_model_get_json_patched = True


_patch_get_json_from_answer()
register_openrouter_models()


def _str_to_pz_model(model_str: str) -> Model:
    """Map an OpenRouter/PZ model string to pz.Model enum value.

    Accepts exact matches or unambiguous prefix matches so callers can pass
    short names like "openai/o4-mini" for "openai/o4-mini-2025-04-16".
    """
    all_models = Model.get_all_models()
    for m in all_models:
        if m.value == model_str:
            return m
    # Prefix fallback: "openai/o4-mini" → "openai/o4-mini-2025-04-16"
    matches = [m for m in all_models if m.value.startswith(model_str)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous oracle model {model_str!r}: matches "
            f"{[m.value for m in matches]}. Use a more specific string."
        )
    raise ValueError(
        f"Unknown oracle model: {model_str!r}. "
        f"Available: {[m.value for m in all_models]}"
    )


@dataclass
class SubsetExecutionContext:
    """Execution context returned by run_subset(), used by QualityEvaluator."""
    sampled_records: list[dict]               # sampled left-side input rows as plain dicts
    output_records: list[dict]                # final pipeline output rows as plain dicts
    per_sem_op_info: dict[int, dict]          # stage_idx → {op_name, op_type, attributes, samples} (all ops, not just semantic)
    has_join: bool
    right_sampled_records: list[dict] | None  # right-side sampled rows if join, else None


def _resolve_reasoning_effort(model: Model) -> str | None:
    """Every reasoning model's operators run at low effort, whatever the provider (MOAR's
    operators too, see docetl's sembench_run.py)."""
    if model is None or not model.is_reasoning_model():
        return None
    return "low"


def _no_reasoning_prompt(model: Model, effort: str | None) -> bool:
    """Whether an operator gets pz's prompt without a written REASONING section: a reasoning
    model at low effort or below already reasons internally (pz's own rule, see
    palimpzest.utils.model_helpers.use_reasoning_prompt)."""
    return model.is_reasoning_model() and effort in (None, "minimal", "low", "disable")


def _compute_op_id(op_type: str, params: dict) -> str:
    """Stable 10-char hex id derived from op_type and id params."""
    payload = json.dumps({"op_type": op_type, **{k: str(v) for k, v in params.items()}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _fn_source(fn) -> str:
    """Source of a UDF, for `attributes`. Functions built by the plan sandbox are interpreter
    wrappers that all share one body, so their real source is read from `__source__`."""
    src = getattr(fn, "__source__", None)
    if src:
        return src
    try:
        return inspect.getsource(fn).strip()
    except (OSError, TypeError):
        return repr(fn)


def _cols_attribute(cols: list[dict]) -> list[dict]:
    """The `cols` entry of a sem_map/rag_map's `attributes`: name, type, and DESCRIPTION.

    Carrying the description (not just the name) is what makes `attributes` a complete
    identity for the operator, which two things depend on:

      - `params_id` (= `_compute_op_id(op_type, attributes)`) is the `op_id` on every
        per-operator row in `op_results`, which is how one operator's measurements are
        matched across plans. A column's description IS the operator's prompt for that
        column, so two maps with the same column names but different descriptions run
        different prompts at different token counts -- giving them one id would make one
        op's cost look like the other's.
      - the oracle judge reads these descriptions to learn what each field was asked for
        (see plan_quality_evaluator._score_map_op); with names alone it has to guess the
        output convention, e.g. whether "" means "absent" or "the operator failed".

    `type` is included for the same reason: it shapes the generated output schema. Rendered
    as its name ("str") so `attributes` stays JSON-safe, and sorted by column name so two
    ops that declare the same columns in a different order still hash alike.
    """
    return [
        {
            "name": col["name"],
            "type": getattr(col.get("type"), "__name__", None) or str(col.get("type")),
            "description": col.get("description") or "",
        }
        for col in sorted(cols, key=lambda c: c["name"])
    ]


def _make_schema(field_defs: dict):
    # Use palimpzest's pickleable/cached schema so all schemas live in the same
    # registry and work correctly with union_schemas, from_parent, etc.
    safe_defs = {
        k: (ann, fi if isinstance(fi, FieldInfo) else FieldInfo(default=None))
        for k, (ann, fi) in field_defs.items()
    }
    return _create_pickleable_model(safe_defs)


def _has_image_field(schema, field_names: list[str] | None) -> bool:
    """Return True if any of the given fields (or all fields if None) in schema have ImageFilepath type."""
    fields_to_check = field_names if field_names is not None else list(schema.model_fields)
    for name in fields_to_check:
        fi = schema.model_fields.get(name)
        if fi is not None and fi.annotation in (ImageFilepath, ImageFilepath | None):
            return True
    return False


# The operator a worker thread is running a record through, so an error palimpzest only logs can
# be charged to it (see _OpErrorLog).
_current_op = threading.local()


class _OpErrorLog(logging.Handler):
    """Palimpzest handles a failed LLM call or an unparseable answer itself -- the record just
    fails the filter or gets None fields -- and only logs it. Charge each such log line to the
    operator whose call produced it, so the error reaches the plan's `op_errors`.

    A filter's unparseable answer arrives with the whole completion, which palimpzest already
    saves to parse-answer-errors/. Keep only what follows its last "ANSWER:" -- the verdict the
    parser rejected -- so the console stays readable and repeats of the same verdict group into
    one `op_errors` count."""

    def emit(self, record: logging.LogRecord) -> None:
        msg, sep, completion = record.getMessage().partition(" from completion text: ")
        if sep:
            answer = re.split(r"(?i)answer:", completion)[-1].replace("---", "").strip()
            msg += f": the answer {answer[-100:]!r} has neither TRUE nor FALSE, so the record failed the filter"
        # A handler here stops logging's fallback printing to stderr; keep that output.
        sys.stderr.write(msg + "\n")
        op = getattr(_current_op, "op", None)
        if op is not None:
            op.errors.append(msg)


logging.getLogger("palimpzest").addHandler(_OpErrorLog(level=logging.ERROR))


class Operator:
    """Base class for PhysicalPipeline operators.

    Each subclass sets class-level `stage_type` (execution dispatch key) and
    `op_type` (human-readable name used in stats), and populates instance
    attributes `attributes`, `params_id`, and `_pz_op` in its __init__.

    Two invariants tie `attributes` and `params_id` together, and every subclass owes both:

      1. `attributes` fully determines the operator's output. Everything that changes what it
         produces belongs there -- a filter's condition, a map's per-column DESCRIPTIONS (that
         is the prompt), the model, the resolved reasoning_effort, a UDF's source, RAG's
         chunking and retrieval settings -- so that two operators agreeing on `attributes` are
         interchangeable up to LLM sampling noise.
      2. `params_id` is ALWAYS `_compute_op_id(self.op_type, self.attributes)` -- the whole
         dict, never a hand-picked subset. Deriving it from a subset makes two operators that
         behave differently share an id, and `params_id` is the `op_id` measured
         cost/latency/selectivity rows are keyed by: a coarse id silently attributes one
         operator's measurements to another.
    """

    stage_type: str  # filter | convert | project | limit | groupby | join
    op_type: str     # sem_filter | sem_map | sem_join | filter | project | limit | groupby
    attributes: dict  # complete behavioural identity; see invariant 1 above
    params_id: str    # 10-char hex of op_type + ALL of attributes; see invariant 2 above

    def __init__(self):
        self.logical_op_id: str | None = None
        self.skip_if = None
        self._skip_op = None
        # Errors from the current execution, one entry per occurrence; the pipeline resets
        # them before each run and reports them as the plan's `op_errors`.
        self.errors: list[str] = []

    def __call__(self, *args, **kwargs):
        _current_op.op = self
        try:
            if self._skip_op is not None and self.skip_if(args[0].to_dict()):
                return self._skip_op(args[0])
            return self._pz_op(*args, **kwargs)
        except Exception as e:
            # The pipeline drops a record whose operator raised; this is the only trace of why.
            self.errors.append(f"{type(e).__name__}: {e}")
            raise
        finally:
            _current_op.op = None

    def _set_skip_if(self, skip_if) -> None:
        """Send the records `skip_if` matches to a zero-cost pass-through instead of the LLM op:
        a filter keeps them, a convert leaves its new columns None. Enters `attributes` (and so
        `params_id`) only when set, so an operator without one keeps the id it always had."""
        if skip_if is None:
            return
        from collections import defaultdict

        from palimpzest.core.elements.filters import Filter
        from palimpzest.query.operators.convert import NonLLMConvert
        from palimpzest.query.operators.filter import NonLLMFilter

        schemas = {"input_schema": self._pz_op.input_schema, "output_schema": self._pz_op.output_schema}
        if self.stage_type == "filter":
            self._skip_op = NonLLMFilter(filter=Filter(filter_fn=lambda r: True), **schemas)
        else:
            self._skip_op = NonLLMConvert(udf=lambda r: defaultdict(lambda: None), **schemas)
        self.skip_if = skip_if
        self.attributes["skip_if"] = _fn_source(skip_if)
        self.params_id = _compute_op_id(self.op_type, self.attributes)

    def __str__(self) -> str:
        if not self.attributes:
            return self.op_type
        attr_str = "\n".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.op_type}({attr_str})"


def row_id_str(value) -> str:
    """A row id as it names files on disk (<image_subdir>/<id>.jpg).

    A UDF can get an integer id back as a float: palimpzest's DataRecord.to_dict() goes through
    pd.Series, which upcasts a record whose fields are all numeric (e.g. ecomm Q9's idx + price)
    to float64, so 50194 arrives as 50194.0.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
