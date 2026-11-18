"""The `Operator` base class every physical-pipeline operator subclasses, and the helpers the
operators share: operator ids and attributes, schemas, reasoning effort, and error capture."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import sys
import threading

from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.lib.schemas import _create_pickleable_model


def _resolve_reasoning_effort(model: Model) -> str | None:
    """Every reasoning model's operators run at low effort, whatever the provider."""
    if model is None or not model.is_reasoning_model():
        return None
    return "low"


def _no_reasoning_prompt(model: Model, effort: str | None) -> bool:
    """Whether an operator gets pz's prompt without a written REASONING section."""
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
    """The `cols` entry of a sem_map/rag_map/sem_agg's `attributes`: name, type, and description."""
    return [
        {
            "name": col["name"],
            "type": getattr(col.get("type"), "__name__", None) or str(col.get("type")),
            "description": col.get("description") or "",
        }
        for col in sorted(cols, key=lambda c: c["name"])
    ]


def _make_schema(field_defs: dict):
    safe_defs = {
        k: (ann, fi if isinstance(fi, FieldInfo) else FieldInfo(default=None))
        for k, (ann, fi) in field_defs.items()
    }
    return _create_pickleable_model(safe_defs)


# The operator each worker thread is running, so an error palimpzest only logs reaches its `op_errors`.
_current_op = threading.local()


class _OpErrorLog(logging.Handler):
    """Charge each error palimpzest only logs (a failed LLM call or unparseable answer, which fails
    the filter or leaves fields None) to the operator that made the call. A filter's unparseable
    answer is cut to what follows its last "ANSWER:", so repeats group into one count."""

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
    attributes `attributes`, `op_id`, and `_pz_op` in its __init__.
    """

    stage_type: str
    op_type: str
    attributes: dict
    op_id: str

    def __init__(self):
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
        `op_id`) only when set, so an operator without one keeps its id."""
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
        self.op_id = _compute_op_id(self.op_type, self.attributes)

    def __str__(self) -> str:
        if not self.attributes:
            return self.op_type
        attr_str = "\n".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.op_type}({attr_str})"
