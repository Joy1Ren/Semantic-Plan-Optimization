"""Register NEW_MODELS with palimpzest, priced from litellm, and patch how pz routes, prompts
and prices them."""
from __future__ import annotations

import litellm
from palimpzest.constants import Model
from palimpzest.utils.model_info_helpers import ModelMetricsManager

from raptor.pricing import _price_key

# pz.Model attribute -> provider/model id. All take text and images, and all reason.
NEW_MODELS = {
    "GPT_6_LUNA": "openai/gpt-6-luna",
    "GPT_6_SOL": "openai/gpt-6-sol",
    "GPT_5_6_LUNA": "openai/gpt-5.6-luna",
    "GPT_5_6_TERRA": "openai/gpt-5.6-terra",
    "CLAUDE_HAIKU_5_5": "anthropic/claude-haiku-5.5",
    "CLAUDE_SONNET_5_5": "anthropic/claude-sonnet-5.5",
    "GEMINI_3_5_FLASH_LITE": "google/gemini-3.5-flash-lite",
    "GEMINI_3_8_FLASH": "google/gemini-3.8-flash",
    "GEMINI_4_ARGON": "google/gemini-4-argon",
    "QWEN_3_8_FLASH": "qwen/qwen3.8-flash",
    "QWEN_3_8_MAX": "qwen/qwen3.8-max-0902",
}
_PZ_PROVIDER = {"openai": "openai", "anthropic": "anthropic", "google": "gemini", "qwen": "qwen"}


def _spec(model_id: str) -> dict:
    row = litellm.model_cost[_price_key(model_id)]
    provider = _PZ_PROVIDER[model_id.split("/", 1)[0]]
    return {
        "provider": provider,
        "is_text_model": True,
        "is_vision_model": True,
        "is_reasoning_model": True,
        # Without it pz leaves its <<cache-boundary>> marker in the prompt text.
        "supports_prompt_caching": True,
        "usd_per_input_token": row["input_cost_per_token"],
        "usd_per_cache_read_token": row["input_cost_per_token"],
        "usd_per_cache_creation_token": row["input_cost_per_token"],
        "usd_per_output_token": row["output_cost_per_token"],
    }


def patch_palimpzest_for_new_models() -> None:
    manager = ModelMetricsManager()
    manager._load_data()
    manager._metrics_cache.update({model_id: _spec(model_id) for model_id in NEW_MODELS.values()})
    for name, model_id in NEW_MODELS.items():
        setattr(Model, name, Model(model_id))
    # pz sends any id containing "gemini" to Google's SDK; send it through litellm instead.
    Model.is_model_gemini = lambda self: False
    _strip_cache_markers_for_other_providers()
    _patch_generator_cost()


def _strip_cache_markers_for_other_providers() -> None:
    """Strip pz's <<cache-boundary>> marker from every non-Anthropic prompt. pz strips it only for
    the providers it knows, so a Qwen model would get the marker as prompt text."""
    from palimpzest.prompts.prompt_manager import PromptManager

    if getattr(PromptManager, "_raptor_markers_patched", False):
        return
    _orig = PromptManager.update_messages_for_caching

    def update_messages_for_caching(self, messages):
        if self.model.is_provider_anthropic():
            return _orig(self, messages)
        return self._remove_cache_boundary_markers(messages)

    PromptManager.update_messages_for_caching = update_messages_for_caching
    PromptManager._raptor_markers_patched = True


def _patch_generator_cost() -> None:
    """Price every call to a NEW_MODELS model with pricing.token_cost, which applies long-prompt
    tiers; pz itself charges flat per-token rates."""
    from palimpzest.query.generators.generators import Generator

    from raptor.pricing import token_cost

    if getattr(Generator, "_raptor_cost_patched", False):
        return
    _orig = Generator.__call__
    priced = set(NEW_MODELS.values())

    def _call(self, *args, **kwargs):
        output = _orig(self, *args, **kwargs)
        stats = output[2]
        if stats is not None and self.model.value in priced and stats.total_llm_calls:
            prompt_tokens = (stats.input_text_tokens + stats.input_image_tokens + stats.input_audio_tokens
                             + stats.cache_read_tokens + stats.cache_creation_tokens)
            stats.cost_per_record = token_cost(self.model.value, int(prompt_tokens), int(stats.output_text_tokens))
        return output

    Generator.__call__ = _call
    Generator._raptor_cost_patched = True
