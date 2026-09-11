"""Physical-pipeline constants, with no palimpzest dependency.

Split out of `base.py` so consumers that need only these values -- notably
`agent_cost_model.sampling`, its CLI, and the analysis scripts -- can import them without
paying for `base.py`'s module-scope `from palimpzest.constants import Model` and its two
import-time monkeypatches. `base.py` re-exports everything here, so existing
`from ...physical_pipeline.base import NUM_SAMPLES` imports are unaffected.
"""
from __future__ import annotations

NUM_SAMPLES = 10
SUBSET_SEED = 42

# Embedding model + endpoint used by rag_filter/rag_map for chunk retrieval, and by
# agent_cost_model.sampling, via OpenRouter's /embeddings endpoint (not a palimpzest Model
# enum value -- see operators/rag_common.py).
DEFAULT_RAG_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
_RAG_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
