"""Deterministic (non-LLM) column derivation operator."""
from __future__ import annotations

import inspect
from typing import Callable

from palimpzest.query.operators.convert import NonLLMConvert

from ..base import Operator, _compute_op_id


class Map(Operator):
    """Deterministic (non-LLM) column derivation via UDF."""
    stage_type = "convert"
    op_type = "map"

    def __init__(self, udf: Callable, cols: list[dict], input_schema, output_schema):
        super().__init__()
        self._pz_op = NonLLMConvert(
            udf=udf,
            output_schema=output_schema,
            input_schema=input_schema,
        )
        self._udf = udf          # preserved for make_oracle_copy
        self._cols_full = cols   # preserved for make_oracle_copy
        try:
            fn_src = inspect.getsource(udf).strip()
        except (OSError, TypeError):
            fn_src = repr(udf)
        col_names = sorted(col["name"] for col in cols)
        self.attributes = {"cols": col_names, "udf": fn_src}
        self.params_id = _compute_op_id(self.op_type, self.attributes)
