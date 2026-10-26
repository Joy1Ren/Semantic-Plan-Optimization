"""Dollar cost of a call, computed as tokens x litellm's price table.

Cost never comes from the provider's own report (e.g. OpenRouter's `usage.cost`), so it is the
same whichever provider or API key serves the call.
"""
from __future__ import annotations

import re

import litellm

# litellm's table has no entry for these under their OpenRouter names ($ per token).
EMBEDDING_PRICES = {
    "openrouter/openai/text-embedding-3-small": 0.02e-6,
    "openrouter/google/gemini-embedding-2": 0.20e-6,
    "openrouter/qwen/qwen3-embedding-8b": 0.01e-6,
}
litellm.register_model({
    model: {
        "input_cost_per_token": price,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openrouter",
        "mode": "embedding",
    }
    for model, price in EMBEDDING_PRICES.items()
})


def _price_key(model: str) -> str:
    # An OpenRouter id maps to the row of its provider's own API: "openai/gpt-6-luna" ->
    # "gpt-6-luna", "google/gemini-3.8-flash" -> "gemini/gemini-3.8-flash",
    # "anthropic/claude-haiku-5.5" -> "claude-haiku-5-5".
    model = model.removeprefix("openrouter/")
    provider, _, name = model.partition("/")
    provider_row = {
        "google": f"gemini/{name}",
        "anthropic": re.sub(r"-(\d+)\.(\d+)$", r"-\1-\2", name),
    }.get(provider, name)
    for key in (model, f"openrouter/{model}", provider_row):
        if key in litellm.model_cost:
            return key
    raise KeyError(f"No price for model {model!r} in litellm's table; add it to pricing.py")


def token_cost(model: str, prompt_tokens: int, completion_tokens: int = 0, cached_tokens: int = 0) -> float:
    """`prompt_tokens` includes `cached_tokens`, as in OpenAI-style usage. litellm applies
    long-prompt rates (e.g. gpt-5.4 above 272k prompt tokens), as it does for docetl."""
    input_cost, output_cost = litellm.cost_per_token(
        model=_price_key(model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_input_tokens=cached_tokens,
    )
    return input_cost + output_cost
