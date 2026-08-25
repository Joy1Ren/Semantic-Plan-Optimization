"""Errors raised by the cost-model agent's step loop."""

from __future__ import annotations


class ParseError(Exception):
    """The model's reply was not a single valid fenced block."""

    def __init__(self, raw: str, detail: str):
        super().__init__(detail)
        self.raw = raw
        self.detail = detail


class StepFailed(Exception):
    """The agent ran out of steps without an accepted final answer."""

    def __init__(self, reason: str, diagnostic: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic
