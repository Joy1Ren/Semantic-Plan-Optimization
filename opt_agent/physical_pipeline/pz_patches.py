"""Fixes to how installed palimpzest (1.5.3) reads model answers and builds records and prompts.

Each patch wraps one palimpzest method and is idempotent. `apply_pz_patches` applies them all,
in this order, and the package __init__ calls it once.
"""
from __future__ import annotations

import ast
import json
import re
import types
import typing


def apply_pz_patches() -> None:
    _patch_convert_empty_field_answers()
    _patch_convert_coerce_str_field_answers()
    _patch_output_field_types()
    _patch_get_json_from_answer()
    _patch_prepare_field_answers_missing_keys()
    _patch_bare_text_answer()


def _patch_convert_empty_field_answers() -> None:
    """Pass a record through a convert that generates no fields, and treat an empty answer as
    every field missing instead of dropping the record."""
    from palimpzest.core.elements.records import DataRecord
    from palimpzest.query.operators.convert import ConvertOp

    if getattr(ConvertOp, "_agent_cost_model_empty_field_answers_patched", False):
        return
    _orig = ConvertOp._create_data_records_from_field_answers

    def _guarded(self, field_answers, candidate):
        if not field_answers and not self.generated_fields:
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


def _flatten_dict_answer(answer: dict) -> str:
    """Render a JSON object answer as 'key: value; key: value'."""
    parts = []
    for key, value in answer.items():
        rendered = _coerce_str_answer(value)
        rendered = "" if rendered is None else str(rendered).strip()
        if rendered:
            parts.append(f"{key}: {rendered}")
    return "; ".join(parts)


def _coerce_str_answer(answer):
    """Render one LLM answer as the `str` its output_schema field is annotated with."""
    if isinstance(answer, list):
        return "\n".join(str(_coerce_str_answer(v)) for v in answer)
    if isinstance(answer, dict):
        return _flatten_dict_answer(answer)
    # bool before int: bool IS an int subclass, and str(True) would give 'True'.
    if isinstance(answer, bool):
        return "true" if answer else "false"
    if isinstance(answer, (int, float)):
        return str(answer)
    return answer


def _list_item_type(annotation):
    """The item type of a list[...] annotation, optional or not (list[int] -> int); None when the
    annotation is not a list."""
    for ann in (annotation, *typing.get_args(annotation)):
        if typing.get_origin(ann) is list:
            args = typing.get_args(ann)
            return args[0] if args else object
    return None


def _coerce_list_answer(answer, item_type):
    """Render one LLM answer as the list its field is annotated with, when the model answered with a
    single value instead: a JSON list written as a string ('[1, 4]'), item numbers written as text
    for list[int] ('1, 4'; 'none' -> []), or one scalar (4 -> [4]). Other text is left as is, for
    validation to reject."""
    if isinstance(answer, str):
        text = answer.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        if item_type is int and re.fullmatch(r"[\d\s,;]*|none|n/?a", text, re.I):
            return [int(n) for n in re.findall(r"\d+", text)]
        return answer
    if answer is not None and not isinstance(answer, (list, dict)):
        return [answer]
    return answer


def _coerce_answer(answer, annotation):
    """One LLM answer coerced toward its field's declared type."""
    if annotation in (str, str | None):
        return _coerce_str_answer(answer)
    item_type = _list_item_type(annotation)
    if item_type is not None:
        return _coerce_list_answer(answer, item_type)
    return answer


def _patch_convert_coerce_str_field_answers() -> None:
    """Coerce each convert answer toward its field's type, so a list, object, bool or number for a
    `str` field, or a single value for a list field, does not fail validation."""
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
            if field_info is None:
                continue
            field_answers[field] = [_coerce_answer(answer, field_info.annotation) for answer in answers]
        return _orig(self, field_answers, candidate)

    ConvertOp._create_data_records_from_field_answers = _coerced
    ConvertOp._agent_cost_model_coerce_str_field_answers_patched = True


def _type_label(annotation) -> str | None:
    """How an output field's type is named in a prompt ('list[int]', 'bool'); None for str and
    untyped fields, which need no label."""
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and len(args) == 1:
        annotation = args[0]
    if annotation in (str, typing.Any):
        return None
    if isinstance(annotation, type) and not typing.get_args(annotation):
        return annotation.__name__
    return str(annotation)


def _patch_output_field_types() -> None:
    """Show each non-str output field's type next to its description in pz's map and aggregate
    prompts ('- item_ids (type: list[int]): ...')."""
    from palimpzest.prompts.prompt_factory import PromptFactory

    if getattr(PromptFactory, "_agent_cost_model_field_types_patched", False):
        return
    _orig = PromptFactory._get_output_fields_desc

    def _desc(self, output_fields, **kwargs):
        output_schema = kwargs.get("output_schema")
        if output_schema is None or not (self.prompt_strategy.is_map_prompt() or self.prompt_strategy.is_agg_prompt()):
            return _orig(self, output_fields, **kwargs)
        lines = []
        for name in sorted(output_fields):
            field = output_schema.model_fields[name]
            label = _type_label(field.annotation)
            desc = "no description available" if field.description is None else field.description
            lines.append(f"- {name}{f' (type: {label})' if label else ''}: {desc}")
        return "\n".join(lines)

    PromptFactory._get_output_fields_desc = _desc
    PromptFactory._agent_cost_model_field_types_patched = True


_JSON_DECODER = json.JSONDecoder()
_WHITESPACE = re.compile(r"\s*")


def _lenient_json_object(doc: str) -> dict:
    """Parse a flat JSON object of `"key": value` pairs, where a value may be followed by extra bare
    strings (joined onto a string value with ', ')."""
    def skip(i):
        return _WHITESPACE.match(doc, i).end()

    i = skip(doc.index("{") + 1)
    out = {}
    while i < len(doc) and doc[i] != "}":
        key, i = _JSON_DECODER.raw_decode(doc, i)
        i = skip(i)
        if not isinstance(key, str) or doc[i:i + 1] != ":":
            raise ValueError("expected a string key and ':'")
        value, i = _JSON_DECODER.raw_decode(doc, skip(i + 1))
        extras = []
        while True:
            j = skip(i)
            if doc[j:j + 1] == ",":
                j = skip(j + 1)
            if doc[j:j + 1] != '"':
                break
            extra, k = _JSON_DECODER.raw_decode(doc, j)
            if doc[skip(k):skip(k) + 1] == ":":
                break
            extras.append(extra)
            i = k
        if extras:
            if isinstance(value, str):
                value = ", ".join(map(str, [value, *extras]))
            else:
                value = [*value, *extras] if isinstance(value, list) else [value, *extras]
        out[key] = value
        i = j
    if i >= len(doc):
        raise ValueError("unterminated object")
    return out


def _patch_get_json_from_answer() -> None:
    """Recover answers `get_json_from_answer` rejects, which would leave every generated field None:
    - "Extra data": the JSON repeated after the closing `---`; keep the first value.
    - "Expecting value": Python's True/False/None; read the answer as a Python literal.
    - "Expecting ':'/',' delimiter": a value followed by extra bare strings (_lenient_json_object).
    A recovered value counts only if it is the expected object (one-to-one) or array (one-to-many).

    Each `\\n` escape is first rewritten as `\\u000a`, so pz does not turn it into a space."""
    from palimpzest.constants import Cardinality
    from palimpzest.query.generators import generators

    if getattr(generators, "_agent_cost_model_get_json_patched", False):
        return
    _orig = generators.get_json_from_answer

    def _parse(answer, model, cardinality):
        answer = re.sub(r"(?<!\\)\\n", r"\\u000a", answer)
        try:
            return _orig(answer, model, cardinality)
        except json.JSONDecodeError as e:
            expected = dict if cardinality == Cardinality.ONE_TO_ONE else list
            try:
                if e.msg == "Extra data":
                    value = json.loads(e.doc[: e.pos])
                elif e.msg == "Expecting value":
                    value = ast.literal_eval(e.doc)
                elif e.msg in ("Expecting ':' delimiter", "Expecting ',' delimiter") and expected is dict:
                    value = _lenient_json_object(e.doc)
                else:
                    raise e
            except (ValueError, SyntaxError):
                raise e from None
            if not isinstance(value, expected):
                raise e
            return value

    generators.get_json_from_answer = _parse
    generators._agent_cost_model_get_json_patched = True


def _patch_prepare_field_answers_missing_keys() -> None:
    """Give a one-to-one answer's missing fields None, as pz's one-to-many branch already does."""
    from palimpzest.constants import Cardinality
    from palimpzest.query.generators.generators import Generator

    if getattr(Generator, "_agent_cost_model_missing_keys_patched", False):
        return
    _orig = Generator._prepare_field_answers

    def _prepare(self, field_answers, fields):
        if self.cardinality == Cardinality.ONE_TO_ONE and isinstance(field_answers, dict):
            return {field_name: [field_answers.get(field_name)] for field_name in fields}
        return _orig(self, field_answers, fields)

    Generator._prepare_field_answers = _prepare
    Generator._agent_cost_model_missing_keys_patched = True


_ANSWER_SECTION = re.compile(r"answer:(.*?)(?:---|$)", re.IGNORECASE | re.DOTALL)
_TRAILING_RULE = re.compile(r"\s*-{3,}\s*$")


def _patch_bare_text_answer() -> None:
    """Accept a bare-text answer (no JSON) when the op generates exactly one `str` field: the text
    after any "ANSWER:", without the closing `---`, is its value. Text starting with `{` or `[` is
    broken JSON and keeps its error."""
    from palimpzest.constants import Cardinality
    from palimpzest.query.generators.generators import Generator

    if getattr(Generator, "_agent_cost_model_bare_text_patched", False):
        return
    _orig = Generator._parse_convert_answer

    def _parse(self, completion_text, fields, json_output):
        try:
            return _orig(self, completion_text, fields, json_output)
        except Exception:
            if not (json_output and self.cardinality == Cardinality.ONE_TO_ONE and len(fields) == 1
                    and next(iter(fields.values())).annotation in (str, str | None)):
                raise
            match = _ANSWER_SECTION.search(completion_text)
            text = _TRAILING_RULE.sub("", match.group(1) if match else completion_text).strip()
            if not text or text[0] in "{[":
                raise
            return {next(iter(fields)): [text]}

    Generator._parse_convert_answer = _parse
    Generator._agent_cost_model_bare_text_patched = True
