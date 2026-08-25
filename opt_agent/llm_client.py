"""LLM client seam.

The agent only needs a single synchronous call: given a system prompt and a
list of {role, content} messages, return the assistant's text. Implement this
Protocol however you like; `OpenRouterClient` below is the default.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict]) -> Any: ...


def patch_litellm_for_openrouter() -> None:
    """Route every litellm.completion call through OpenRouter and quiet litellm's noisy
    exception-mapping banner ("Give Feedback / Get Help..." / "LiteLLM.Info: ...").

    Required before any PhysicalPipeline execution: palimpzest's own generators call
    litellm.completion directly with bare provider/model strings (e.g. "openai/gpt-4o-mini"),
    which without this patch hit that provider's API directly instead of routing through
    OpenRouter -- failing (and printing that banner) for anyone who only has an
    OPENROUTER_API_KEY set, not a key for every individual provider.

    Idempotent: safe to call more than once (e.g. once per CostModelAgent.run(), or once from
    a standalone script that never calls run() at all -- see experiments/cuad/run_single_plan.py).
    """
    import litellm as _litellm

    _litellm.suppress_debug_info = True
    _litellm.drop_params = True  # OpenRouter rejects reasoning_effort for Google models
    if getattr(_litellm, "_agent_cost_model_openrouter_patched", False):
        return
    _orig_completion = _litellm.completion

    def _openrouter_completion(model, **kwargs):
        if not model.startswith("openrouter/"):
            model = "openrouter/" + model
        return _orig_completion(model=model, **kwargs)

    _litellm.completion = _openrouter_completion
    _litellm._agent_cost_model_openrouter_patched = True


class OpenRouterClient:
    """OpenRouter-backed `LLMClient`, via the OpenAI-compatible endpoint.

    Uses the widely-installed `openai` SDK pointed at OpenRouter (the same
    provider the SearchAgent supports). `pip install openai`, then set
    `OPENROUTER_API_KEY`. `model` is a full OpenRouter id, e.g.
    "openai/gpt-5", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash".

    (If you prefer the native `openrouter` SDK used in skunk's llm_client.py,
    swap `generate` for a `client.chat.send(...)` call — same message shape.)
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,  # "minimal" | "low" | "medium" | "high"
    ) -> None:
        import os

        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.total_cost_usd = 0.0
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    @staticmethod
    def _extract_cost_usd(resp: Any) -> float:
        """Best-effort extraction of provider-reported dollar cost from a chat response."""
        candidates = []
        usage = getattr(resp, "usage", None)
        if usage is not None:
            candidates.extend([
                getattr(usage, "cost", None),
                getattr(usage, "total_cost", None),
                getattr(usage, "estimated_cost", None),
            ])
            usage_extra = getattr(usage, "model_extra", None) or {}
            if isinstance(usage_extra, dict):
                candidates.extend([
                    usage_extra.get("cost"),
                    usage_extra.get("total_cost"),
                    usage_extra.get("estimated_cost"),
                ])
        resp_extra = getattr(resp, "model_extra", None) or {}
        if isinstance(resp_extra, dict):
            candidates.extend([
                resp_extra.get("cost"),
                resp_extra.get("total_cost"),
                resp_extra.get("estimated_cost"),
            ])
            usage_extra = resp_extra.get("usage")
            if isinstance(usage_extra, dict):
                candidates.extend([
                    usage_extra.get("cost"),
                    usage_extra.get("total_cost"),
                    usage_extra.get("estimated_cost"),
                ])

        for value in candidates:
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def generate(self, system: str, messages: list[dict]) -> tuple[str, str | None, dict[str, Any]]:
        msgs = [{"role": "system", "content": system}, *messages]
        extra_body: dict = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            extra_body=extra_body or None,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning")
        cost_usd = self._extract_cost_usd(resp)
        self.total_cost_usd += cost_usd
        return content, reasoning, {"cost_usd": cost_usd}
