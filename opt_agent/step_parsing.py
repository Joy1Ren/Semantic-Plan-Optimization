"""Parsing of one LLM reply into either a tool-call step or a final-answer step.

Shared by `CostModelAgent` and `CostHelperAgent` — both agents run the same
"emit exactly one fenced ```python``` or ```json``` block per turn" protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agent_cost_model.opt_agent.errors import ParseError

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)


@dataclass
class _Step:
    code: str | None = None
    result: Any = None
    raw: str = ""


def _parse_step(text: str) -> _Step:
    """First fenced block → python tool call or json final answer."""
    m = _FENCE_RE.search(text)
    if m is None:
        raise ParseError(
            raw=text,
            detail="no fenced block — emit ONE ```python``` block (a tool call) or "
            "ONE ```json``` block (your final answer).",
        )
    lang, body = m.group(1).lower(), m.group(2).strip()
    if lang != "json":
        return _Step(code=body, raw=text)
    try:
        return _Step(result=json.loads(body), raw=text)
    except json.JSONDecodeError as e:
        raise ParseError(raw=text, detail=f"final-answer JSON was malformed — {e}") from e
