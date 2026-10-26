"""LLM client seam.

The agent only needs a single synchronous call: given a system prompt and a
list of {role, content} messages, return the assistant's text. Implement this
Protocol however you like; `OpenRouterClient` below is the default.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, Protocol

from agent_cost_model.opt_agent.pricing import token_cost

RATE_LIMIT_RETRIES = 6


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
        # Palimpzest's Generator turns ANY exception into an empty answer (a filter drops the
        # record, a map returns None) without retrying, so a transient error would otherwise be
        # scored as a wrong answer. litellm's retries are immediate (no wait), which is useless
        # against a per-minute rate limit, so 429s get exponential backoff here (same schedule
        # as docetl's api.py, so pz and MOAR handle rate limits alike).
        kwargs.setdefault("num_retries", 3)
        # litellm silently drops `reasoning_effort` for OpenRouter's Google and Anthropic
        # models, so send OpenRouter's own `reasoning` field, which it maps for every provider.
        effort = kwargs.pop("reasoning_effort", None)
        if effort:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "reasoning": {"effort": effort}}
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = _orig_completion(model=model, **kwargs)
                break
            except _litellm.RateLimitError:
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(4 * 2**attempt, 120))
        # A reply with no text (seen from gemini-3.5-flash-lite) makes pz drop the call's usage,
        # so its cost would go uncounted; as "" it is costed, then fails to parse like any bad answer.
        if response.choices and response.choices[0].message.content is None:
            response.choices[0].message.content = ""
            logging.getLogger("palimpzest").error(
                f"{model} replied with no text (finish_reason={response.choices[0].finish_reason!r})")
        return response

    _litellm.completion = _openrouter_completion
    _litellm._agent_cost_model_openrouter_patched = True


# Wall-clock seconds one OpenRouter request may take before it is abandoned and retried. A deadline
# on the whole call rather than the SDK's read timeout: OpenRouter sends keep-alive bytes while a
# model works, so a request whose model never answers would otherwise wait forever.
REQUEST_TIMEOUT_S = 300
REQUEST_RETRIES = 2


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
        reasoning_effort: str | None = None,  # "minimal" | "low" | "medium" | "high"
    ) -> None:
        import os

        from openai import OpenAI

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.total_cost_usd = 0.0
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
            timeout=REQUEST_TIMEOUT_S,
        )

    def _cost_usd(self, resp: Any) -> float:
        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        return token_cost(
            self.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        )

    def generate(self, system: str, messages: list[dict]) -> tuple[str, str | None, dict[str, Any]]:
        msgs = [{"role": "system", "content": system}, *messages]
        extra_body: dict = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        # Left to route freely, OpenRouter sometimes sends gpt-6.1-sol requests to a provider that
        # accepts them and never answers; pinned to OpenAI they answer every time.
        if self.model.startswith("openai/"):
            extra_body["provider"] = {"order": ["openai"], "allow_fallbacks": False}
        for attempt in range(REQUEST_RETRIES + 1):
            try:
                content, reasoning, usage = self._stream_once(msgs, extra_body)
                break
            except TimeoutError:
                if attempt == REQUEST_RETRIES:
                    raise
                print(f"[OpenRouterClient] {self.model}: no answer within {REQUEST_TIMEOUT_S}s, "
                      f"retrying ({attempt + 1}/{REQUEST_RETRIES})", flush=True)
        cost_usd = self._cost_usd(SimpleNamespace(usage=usage))
        self.total_cost_usd += cost_usd
        return content, reasoning, {"cost_usd": cost_usd}

    def _stream_once(self, msgs: list[dict], extra_body: dict) -> tuple[str, str | None, Any]:
        """One streamed request, closed from a timer if it outlives REQUEST_TIMEOUT_S. The SDK
        skips OpenRouter's keep-alive bytes without yielding, so only closing the stream from
        outside can interrupt a request whose model never answers. No temperature is sent: it
        made no measurable difference to how consistent these models' answers were."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            extra_body=extra_body or None,
            stream=True,
            stream_options={"include_usage": True},
        )
        timed_out = threading.Event()

        def abandon():
            timed_out.set()
            stream.close()

        timer = threading.Timer(REQUEST_TIMEOUT_S, abandon)
        timer.start()
        content, reasoning, usage = [], [], None
        try:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content.append(delta.content or "")
                    reasoning.append((getattr(delta, "model_extra", None) or {}).get("reasoning") or "")
        except Exception:
            if timed_out.is_set():
                raise TimeoutError(f"{self.model}: no answer within {REQUEST_TIMEOUT_S}s") from None
            raise
        finally:
            timer.cancel()
            stream.close()
        if timed_out.is_set():
            raise TimeoutError(f"{self.model}: no answer within {REQUEST_TIMEOUT_S}s")
        if usage is None:
            raise RuntimeError(f"{self.model}: stream ended without a usage chunk")
        return "".join(content), "".join(reasoning) or None, usage
