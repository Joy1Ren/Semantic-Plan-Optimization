"""Models newer than installed palimpzest's spec table, registered at import as pz.Model.<NAME>.

`pz.Model(id)` only accepts ids in the spec table palimpzest downloads at import, and pz prices
each call from the spec. So every model here gets a spec whose prices are read from the same
litellm row `pricing.token_cost` (and so docetl's MOAR, via sembench_run.py) prices from, with
cached and cache-written input tokens charged as ordinary input tokens.
"""
from __future__ import annotations

import litellm
from palimpzest.constants import Model
from palimpzest.utils.model_info_helpers import ModelMetricsManager

from agent_cost_model.opt_agent.pricing import _price_key

# pz.Model attribute -> OpenRouter id. All take text and images, and all reason.
OPENROUTER_MODELS = {
    "GPT_6_LUNA": "openai/gpt-6-luna",
    "GPT_6_SOL": "openai/gpt-6-sol",
    "GPT_5_6_LUNA": "openai/gpt-5.6-luna",
    "GPT_5_6_TERRA": "openai/gpt-5.6-terra",
    "GPT_5_6_SOL": "openai/gpt-5.6-sol",
    "CLAUDE_HAIKU_5_5": "anthropic/claude-haiku-5.5",
    "CLAUDE_SONNET_5_5": "anthropic/claude-sonnet-5.5",
    "CLAUDE_OPUS_5_5": "anthropic/claude-opus-5.5",
    "GEMINI_3_5_FLASH_LITE": "google/gemini-3.5-flash-lite",
    "GEMINI_3_8_FLASH": "google/gemini-3.8-flash",
    "GEMINI_4_ARGON": "google/gemini-4-argon",
}
_PZ_PROVIDER = {"openai": "openai", "anthropic": "anthropic", "google": "gemini"}


def _spec(model_id: str) -> dict:
    row = litellm.model_cost[_price_key(model_id)]
    provider = _PZ_PROVIDER[model_id.split("/", 1)[0]]
    return {
        "provider": provider,
        "is_text_model": True,
        "is_vision_model": True,
        "is_reasoning_model": True,
        # pz sends no temperature to these: OpenAI's reasoning models take only the default.
        "is_gpt_5_model": provider == "openai",
        # Without it pz leaves its <<cache-boundary>> marker in the prompt text.
        "supports_prompt_caching": True,
        "usd_per_input_token": row["input_cost_per_token"],
        "usd_per_cache_read_token": row["input_cost_per_token"],
        "usd_per_cache_creation_token": row["input_cost_per_token"],
        "usd_per_output_token": row["output_cost_per_token"],
    }


def register_openrouter_models() -> None:
    manager = ModelMetricsManager()
    manager._load_data()
    manager._metrics_cache.update({model_id: _spec(model_id) for model_id in OPENROUTER_MODELS.values()})
    for name, model_id in OPENROUTER_MODELS.items():
        setattr(Model, name, Model(model_id))
    # pz sends any id containing "gemini" to Google's SDK directly, skipping litellm and so
    # OpenRouter; every model goes through litellm instead.
    Model.is_model_gemini = lambda self: False
