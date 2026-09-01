"""RAG (chunk + retrieve) filter operator. See rag_common.py for the shared rationale
behind the OpenRouter-embedding / decoupled-query / chunk-selection design."""
from __future__ import annotations

from palimpzest.constants import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.rag import RAGFilter as _PZRAGFilter

from ..base import DEFAULT_RAG_EMBEDDING_MODEL, Operator, _compute_op_id, _resolve_reasoning_effort
from .rag_common import _RAG_PLACEHOLDER_MODEL, _rag_embed, rag_get_chunked_candidate


class RAGFilter(_PZRAGFilter):
    """PZ RAGFilter with OpenRouter/qwen3-embedding-8b embeddings, a retrieval query
    (embedding_query) decoupled from the filter condition, a choice of similarity_method
    (embedding cosine similarity or bm25 keyword search), and top-k OR similarity-threshold
    chunk selection. See rag_common.py for the full rationale."""

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
        # _PZRAGFilter.__init__ requires a valid `Model` for embedding_model and an int
        # num_chunks_per_field; both are placeholders here (see class docstring) and are
        # overwritten with our real values right after.
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
        # Skip _PZRAGFilter.__str__: it assumes self.embedding_model is a PZ Model.
        op = LLMFilter.__str__(self)
        selection = f"top-{self.num_chunks_per_field}" if self.num_chunks_per_field is not None else f"similarity >= {self.similarity_threshold}"
        op += f"    Embedding Query: {self.embedding_query!r}\n"
        op += f"    Similarity Method: {self.similarity_method}\n"
        op += f"    Chunk Selection: {selection}\n"
        op += f"    Embedding Model: {self.embedding_model}\n"
        op += f"    Chunk Size: {self.chunk_size}\n"
        return op

    def get_id_params(self):
        # Skip _PZRAGFilter.get_id_params: it assumes self.embedding_model is a PZ Model.
        id_params = LLMFilter.get_id_params(self)
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
        op_params = LLMFilter.get_op_params(self)
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

    def get_chunked_candidate(self, candidate: DataRecord, input_fields: list[str]) -> tuple[DataRecord, GenerationStats]:
        return rag_get_chunked_candidate(self, candidate, input_fields)


class RagFilter(Operator):
    """Chunk long fields, retrieve the chunks most similar to `embedding_query` (top-k or
    similarity-threshold), then LLM-filter using `condition` over the retrieved content
    instead of the full field. See the RAGFilter class above."""
    stage_type = "filter"
    op_type = "rag_filter"

    def __init__(
        self,
        condition: str,
        embedding_query: str,
        model: Model,
        schema,
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
        self._pz_op = RAGFilter(
            embedding_query=embedding_query,
            chunk_size=chunk_size,
            num_chunks_per_field=num_chunks_per_field,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            similarity_method=similarity_method,
            model=model,
            filter=Filter(filter_condition=condition),
            output_schema=schema,
            input_schema=schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.attributes = {
            "condition": condition,
            "embedding_query": embedding_query,
            "model": model.value,
            "reasoning_effort": eff,
            "chunk_size": chunk_size,
            "num_chunks_per_field": num_chunks_per_field,
            "similarity_threshold": similarity_threshold,
            "embedding_model": embedding_model,
            "similarity_method": similarity_method,
            "depends_on": depends_on,
        }
        self.params_id = _compute_op_id(self.op_type, self.attributes)
