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

OPENROUTER_CHAT_PRICES = {
    "openrouter/qwen/qwen3.8-flash": (0.15e-6, 0.47e-6, 0.016e-6, 1_000_000),
    "openrouter/qwen/qwen3.8-max-0902": (2.00e-6, 6.00e-6, 0.25e-6, 1_000_000),
}
litellm.register_model({
    model: {
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
        "cache_read_input_token_cost": cache_read_cost,
        "max_input_tokens": context,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_reasoning": True,
        "supports_vision": True,
    }
    for model, (input_cost, output_cost, cache_read_cost, context) in OPENROUTER_CHAT_PRICES.items()
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


# A long-prompt tier's input rate, e.g. "input_cost_per_token_above_272k_tokens"; the variants with
# a suffix (_batches, _priority, _flex) are other service levels and never apply here.
_TIER_KEY = re.compile(r"input_cost_per_token_above_(\d+)k_tokens")


def _rates(row: dict, prompt_tokens: int) -> tuple[float, float, float]:
    """(input, cached input, output) $/token for a call with `prompt_tokens`, from the highest
    long-prompt tier the prompt exceeds, else the base rates."""
    tiers = [int(m.group(1)) for key in row if (m := _TIER_KEY.fullmatch(key))]
    above = max((t for t in tiers if prompt_tokens > t * 1000), default=None)
    sfx = f"_above_{above}k_tokens" if above is not None else ""
    base_in, base_out = row["input_cost_per_token"], row.get("output_cost_per_token", 0.0)
    base_cache = row.get("cache_read_input_token_cost", 0.0)   # as litellm, for rows without one
    return (row.get(f"input_cost_per_token{sfx}", base_in),
            row.get(f"cache_read_input_token_cost{sfx}", base_cache),
            row.get(f"output_cost_per_token{sfx}", base_out))


def long_prompt_tier(model: str) -> tuple[int, tuple[float, float], tuple[float, float]] | None:
    """`model`'s lowest long-prompt tier: (threshold in prompt tokens, (input, output) $/token at
    or below it, (input, output) $/token above it), or None when its rates are flat."""
    row = litellm.model_cost[_price_key(model)]
    tiers = [int(m.group(1)) for key in row if (m := _TIER_KEY.fullmatch(key))]
    if not tiers or "input_cost_per_token" not in row:
        return None
    threshold = min(tiers) * 1000
    base_in, _, base_out = _rates(row, threshold)
    long_in, _, long_out = _rates(row, threshold + 1)
    return threshold, (base_in, base_out), (long_in, long_out)


def token_cost(model: str, prompt_tokens: int, completion_tokens: int = 0, cached_tokens: int = 0) -> float:
    """`prompt_tokens` includes `cached_tokens`, as in OpenAI-style usage.

    Long-prompt rates come from litellm's row (gpt-6 above 272k prompt tokens, claude-haiku-5-5
    above 100k) but are applied here: litellm.cost_per_token applies the first and ignores the
    second. Every LLM cost goes through this function -- the agent's, pz plan operators' (via
    physical_pipeline/models.py) and MOAR's (via docetl/experiments/moar_llm.py) -- so all three are
    priced alike."""
    row = litellm.model_cost[_price_key(model)]
    if "input_cost_per_token" not in row:
        input_cost, output_cost = litellm.cost_per_token(
            model=_price_key(model), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cache_read_input_tokens=cached_tokens,
        )
        return input_cost + output_cost
    in_rate, cache_rate, out_rate = _rates(row, prompt_tokens)
    return (prompt_tokens - cached_tokens) * in_rate + cached_tokens * cache_rate + completion_tokens * out_rate
