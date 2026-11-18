"""Which API, and which key, serves each model -- for RAPTOR and MOAR alike.

Models are named by OpenRouter id ("openai/gpt-6-luna", "google/gemini-3.8-flash"; a leading
"openrouter/" is ignored). A provider in DIRECT is called on its own API with its own key; every
other model goes through OpenRouter. Keys are read from the environment by name.

Adding a provider to DIRECT also needs its model names and rejected parameters handled in
litellm_call (OpenAI's are below).
"""
from __future__ import annotations

import os
import sys

import litellm

OPENROUTER = ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
# provider -> (environment variable holding its key, OpenAI-compatible base URL)
DIRECT = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
}

# OpenAI rejects these on a reasoning model; OpenRouter silently dropped them.
_OPENAI_REASONING_UNSUPPORTED = (
    "temperature", "top_p", "logprobs", "top_logprobs", "presence_penalty", "frequency_penalty", "logit_bias",
)


def _split(model: str) -> tuple[str, str, str]:
    model = model.removeprefix("openrouter/")
    provider, _, name = model.partition("/")
    return model, provider, name


def is_direct(model: str) -> bool:
    return _split(model)[1] in DIRECT


def check_keys() -> None:
    """Fail now if a key in the table is unset: otherwise every call on that route fails, and pz
    and docetl turn each failure into an empty answer."""
    missing = [env for env, _ in (OPENROUTER, *DIRECT.values()) if not os.environ.get(env)]
    if missing:
        sys.exit(f"Not set in the environment: {', '.join(missing)}")


def endpoint(model: str) -> tuple[str, str, str]:
    """(model name as its API knows it, base URL, API key) for an OpenAI-compatible client."""
    model, provider, name = _split(model)
    env, base_url = DIRECT.get(provider, OPENROUTER)
    return (name if provider in DIRECT else model), base_url, os.environ[env]


def _is_reasoning(name: str) -> bool:
    return litellm.model_cost.get(name, {}).get("supports_reasoning", True)


def litellm_call(model: str, kwargs: dict, effort: str | None = None) -> tuple[str, dict]:
    """The model string and kwargs for litellm.completion that send `model` down its route at
    reasoning `effort` (a `reasoning_effort` in kwargs takes precedence)."""
    name, base_url, api_key = endpoint(model)
    kwargs = {**kwargs, "api_base": base_url, "api_key": api_key}
    effort = kwargs.pop("reasoning_effort", None) or effort
    extra_body = dict(kwargs.get("extra_body") or {})
    if not is_direct(model):
        # litellm silently drops `reasoning_effort` for OpenRouter's Google and Anthropic models;
        # OpenRouter's own `reasoning` field reaches every provider.
        if effort:
            extra_body["reasoning"] = {"effort": effort}
        model = "openrouter/" + name
    else:
        if _is_reasoning(name):
            for param in _OPENAI_REASONING_UNSUPPORTED:
                kwargs.pop(param, None)
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            # In the body, not as `reasoning_effort`: litellm drops that for gpt-6 models, whose
            # parameter list it has without it.
            if effort:
                extra_body["reasoning_effort"] = effort
        # OpenAI's /v1/chat/completions rejects function tools with a reasoning effort on newer
        # models (docetl's send_output tool); litellm's "responses/" prefix sends them to
        # /v1/responses and converts the answer back.
        if effort and _is_reasoning(name) and kwargs.get("tools"):
            model = "openai/responses/" + name
        else:
            model = "openai/" + name
    if extra_body:
        kwargs["extra_body"] = extra_body
    return model, kwargs


def exit_if_out_of_quota(error: object) -> None:
    """OpenAI reports an exhausted balance as a 429, like a rate limit, so it would be retried with
    backoff; pz and docetl then turn the failed call into a wrong answer. Stop the whole process
    instead -- os._exit, since the call may be on a worker thread whose exceptions are swallowed.
    litellm's exception keeps only the message, so the chain is searched for OpenAI's own."""
    cause = error
    while cause is not None:
        if "insufficient_quota" in str(cause):
            print(f"\nOut of OpenAI credits (insufficient_quota), exiting:\n{cause}", file=sys.stderr, flush=True)
            os._exit(1)
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
