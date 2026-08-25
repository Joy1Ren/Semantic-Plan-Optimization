"""RAG (chunk + retrieve) column-derivation operator. See rag_common.py for the shared
rationale behind the OpenRouter-embedding / decoupled-query / chunk-selection design."""
from __future__ import annotations

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.rag import RAGConvert as _PZRAGConvert

from ..base import DEFAULT_RAG_EMBEDDING_MODEL, Operator, _compute_op_id, _has_image_field, _resolve_reasoning_effort
from .rag_common import _RAG_PLACEHOLDER_MODEL, _rag_embed, rag_get_chunked_candidate


class RAGConvert(_PZRAGConvert):
    """PZ RAGConvert with OpenRouter/qwen3-embedding-8b embeddings, a retrieval query
    (embedding_query) decoupled from the output-field descriptions, a choice of
    similarity_method (embedding cosine similarity or bm25 keyword search), and top-k OR
    similarity-threshold chunk selection. See rag_common.py for the full rationale."""

    def __init__(
        self,
        embedding_query: str,
        chunk_size: "int | str" = 20000,
        num_chunks_per_field: int | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        similarity_method: str = "embedding",
        *args,
        **kwargs,
    ):
        if (num_chunks_per_field is None) == (similarity_threshold is None):
            raise ValueError(
                "Specify exactly one of num_chunks_per_field (top-k) or similarity_threshold "
                "(min cosine similarity), not both/neither."
            )
        if similarity_method not in ("embedding", "bm25"):
            raise ValueError(f"similarity_method must be 'embedding' or 'bm25', got {similarity_method!r}")
        super().__init__(
            embedding_model=_RAG_PLACEHOLDER_MODEL,
            num_chunks_per_field=num_chunks_per_field or 1,
            chunk_size=chunk_size,
            *args,
            **kwargs,
        )
        self.embedding_query = embedding_query
        self.num_chunks_per_field = num_chunks_per_field
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.similarity_method = similarity_method

    def __str__(self):
        # Skip _PZRAGConvert.__str__: it assumes self.embedding_model is a PZ Model.
        op = LLMConvert.__str__(self)
        selection = f"top-{self.num_chunks_per_field}" if self.num_chunks_per_field is not None else f"similarity >= {self.similarity_threshold}"
        op += f"    Embedding Query: {self.embedding_query!r}\n"
        op += f"    Similarity Method: {self.similarity_method}\n"
        op += f"    Chunk Selection: {selection}\n"
        op += f"    Embedding Model: {self.embedding_model}\n"
        op += f"    Chunk Size: {self.chunk_size}\n"
        return op

    def get_id_params(self):
        id_params = LLMConvert.get_id_params(self)
        return {
            "embedding_query": self.embedding_query,
            "num_chunks_per_field": self.num_chunks_per_field,
            "similarity_threshold": self.similarity_threshold,
            "chunk_size": self.chunk_size,
            "embedding_model": self.embedding_model,
            "similarity_method": self.similarity_method,
            **id_params,
        }

    def get_op_params(self):
        op_params = LLMConvert.get_op_params(self)
        return {
            "embedding_query": self.embedding_query,
            "num_chunks_per_field": self.num_chunks_per_field,
            "similarity_threshold": self.similarity_threshold,
            "chunk_size": self.chunk_size,
            "embedding_model": self.embedding_model,
            "similarity_method": self.similarity_method,
            **op_params,
        }

    def compute_embedding(self, text: str) -> tuple[list[float], GenerationStats]:
        return _rag_embed(self.embedding_model, text)

    def get_chunked_candidate(self, candidate: DataRecord, input_fields: list[str], output_fields: list[str]) -> tuple[DataRecord, GenerationStats]:
        # output_fields is unused: PZ's RAGConvert derives the query from the output fields'
        # descriptions, but we use the caller-supplied embedding_query instead.
        return rag_get_chunked_candidate(self, candidate, input_fields)


class RagMap(Operator):
    """Chunk long fields, retrieve the chunks most similar to `embedding_query` (top-k or
    similarity-threshold), then LLM-derive new columns from the retrieved content instead of
    the full field. See the RAGConvert class above."""
    stage_type = "convert"
    op_type = "rag_map"

    def __init__(
        self,
        cols: list[dict],
        embedding_query: str,
        model: Model,
        input_schema,
        output_schema,
        chunk_size: "int | str" = 20000,
        num_chunks_per_field: int | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        similarity_method: str = "embedding",
        depends_on: list[str] | None = None,
        reasoning_effort_override: str | None = None,
    ):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        # Installed palimpzest (1.5.3) collapsed COT_QA/COT_QA_IMAGE into a single MAP
        # strategy -- there's no separate *_IMAGE variant; the Generator now detects image
        # fields from the schema itself rather than from prompt_strategy.
        is_image = _has_image_field(input_schema, depends_on)  # noqa: F841 -- kept for parity/future use
        prompt_strategy = PromptStrategy.MAP_NO_REASONING if (model.is_reasoning_model() and eff in (None, "minimal", "low", "disable")) else PromptStrategy.MAP
        self._pz_op = RAGConvert(
            embedding_query=embedding_query,
            chunk_size=chunk_size,
            num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            similarity_method=similarity_method,
            model=model,
            prompt_strategy=prompt_strategy,
            output_schema=output_schema,
            input_schema=input_schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self._cols_full = cols  # preserved for make_oracle_copy
        self.attributes = {
            "model": model.value,
            "cols": sorted(col["name"] for col in cols),
            "embedding_query": embedding_query,
            "chunk_size": chunk_size,
            "num_chunks_per_field": num_chunks_per_field,
            "similarity_threshold": similarity_threshold,
            "embedding_model": embedding_model,
            "similarity_method": similarity_method,
            "depends_on": depends_on,
        }
        self.params_id = _compute_op_id(self.op_type, self.attributes)
