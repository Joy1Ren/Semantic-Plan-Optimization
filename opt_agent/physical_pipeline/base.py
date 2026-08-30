"""Shared constants, helpers, and the `Operator` base class used across all
physical-pipeline operators (agent_cost_model/physical_pipeline/operators/*.py) and
by the `PhysicalPipeline` fluent interface (pipeline.py)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.lib.schemas import ImageFilepath, _create_pickleable_model

NUM_SAMPLES = 10
SUBSET_SEED = 42

# Embedding model + endpoint used by rag_filter/rag_map for chunk retrieval, via OpenRouter's
# /embeddings endpoint (not a palimpzest Model enum value -- see operators/rag_common.py).
DEFAULT_RAG_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
_RAG_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"


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
    from palimpzest.query.operators.convert import ConvertOp

    if getattr(ConvertOp, "_agent_cost_model_empty_field_answers_patched", False):
        return
    _orig = ConvertOp._create_data_records_from_field_answers

    def _guarded(self, field_answers, candidate):
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


def _patch_convert_coerce_str_field_answers() -> None:
    """Guard installed palimpzest 1.5.3's `ConvertOp._create_data_records_from_field_answers`
    against a pydantic ValidationError when an LLM returns a JSON array for a field whose
    output_schema annotation is `str` (or `str | None`), instead of the single string the
    schema -- and CUAD's task prompt, which asks for "a comma-separated list of text spans" --
    expects.

    Observed on rag_map calls (e.g. gpt-5-mini) for fields like 'Revenue/Profit Sharing' and
    'Affiliate IP License-Licensor': the model emits `["span one", "span two"]` for a field
    typed `str | None` instead of the requested comma-joined string, and pydantic rejects the
    list outright (`Input should be a valid string [type=string_type]`) inside
    `DataRecord.from_parent`, which currently propagates all the way up to a record-level
    "skipping record" error and drops the whole record's other 40 CUAD fields along with it.
    Join list values into the same comma-separated-string form the model was already asked to
    produce for multi-span answers, rather than losing the record over one mis-typed field.
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
            field_answers[field] = [
                ", ".join(str(v) for v in answer) if isinstance(answer, list) else answer
                for answer in answers
            ]
        return _orig(self, field_answers, candidate)

    ConvertOp._create_data_records_from_field_answers = _coerced
    ConvertOp._agent_cost_model_coerce_str_field_answers_patched = True


_patch_convert_coerce_str_field_answers()


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
    """Mirror PZ optimizer logic: disable thinking tokens for reasoning models by default."""
    if model is None or not model.is_reasoning_model():
        return None
    if model.is_provider_vertex_ai() or model.is_provider_google_ai_studio():
        if model in (getattr(Model, 'GEMINI_2_5_PRO', None), getattr(Model, 'GOOGLE_GEMINI_2_5_PRO', None)):
            return "low"
        return "disable"
    if model.is_provider_openai() or model.is_provider_azure():
        return "low"
    return None


def _compute_op_id(op_type: str, params: dict) -> str:
    """Stable 10-char hex id derived from op_type and id params."""
    payload = json.dumps({"op_type": op_type, **{k: str(v) for k, v in params.items()}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


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


class Operator:
    """Base class for PhysicalPipeline operators.

    Each subclass sets class-level `stage_type` (execution dispatch key) and
    `op_type` (human-readable name used in stats), and populates instance
    attributes `attributes`, `params_id`, and `_pz_op` in its __init__.
    """

    stage_type: str  # filter | convert | project | limit | groupby | join
    op_type: str     # sem_filter | sem_map | sem_join | filter | project | limit | groupby
    attributes: dict
    params_id: str #10-char hex from op_type and attributes

    def __init__(self):
        self.logical_op_id: str | None = None

    def __call__(self, *args, **kwargs):
        return self._pz_op(*args, **kwargs)

    def __str__(self) -> str:
        if not self.attributes:
            return self.op_type
        attr_str = "\n".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.op_type}({attr_str})"
