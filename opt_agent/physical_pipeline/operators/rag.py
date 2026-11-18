"""rag_filter and rag_map: chunk long text fields, keep the chunks most relevant to
`embedding_query` (embedding or BM25 similarity), and run the LLM filter/convert on them."""
from __future__ import annotations

import ast
import json
import operator as _operator
import re
import time
from typing import Any

import httpx
from rank_bm25 import BM25Okapi

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.rag import RAGConvert as _PZRAGConvert
from palimpzest.query.operators.rag import RAGFilter as _PZRAGFilter

from agent_cost_model.opt_agent import routing
from agent_cost_model.opt_agent.pricing import token_cost

from .base import Operator, _cols_attribute, _compute_op_id, _no_reasoning_prompt, _resolve_reasoning_effort

# Embedding model rag_filter/rag_map retrieve chunks with, unless the plan names another.
DEFAULT_RAG_EMBEDDING_MODEL = "google/gemini-embedding-2"

# Placeholder for the `embedding_model: Model` PZ's RAG operators require. Never used:
# compute_embedding is overridden to embed with each operator's own embedding_model.
_RAG_PLACEHOLDER_MODEL = Model.TEXT_EMBEDDING_3_SMALL


def _rag_embed(model: str, text: str) -> tuple[list[float], GenerationStats]:
    """Embed `text` at the /embeddings endpoint routing.py assigns `model`; return (vector, stats)."""
    api_model, base_url, api_key = routing.endpoint(model)
    body = {
        "model": api_model,
        "input": text or " ",
        "encoding_format": "float",
    }
    start_time = time.time()
    resp = httpx.post(
        f"{base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120.0,
    )
    latency = time.time() - start_time
    if resp.status_code != 200:
        routing.exit_if_out_of_quota(resp.text)
        raise RuntimeError(f"Embedding request failed ({resp.status_code}) for model {model!r}: {resp.text[:800]}")
    payload = resp.json()
    data = payload.get("data")
    if not data:
        raise RuntimeError(f"No embedding data from {model!r}. Response: {json.dumps(payload)[:800]}")
    vec = [float(x) for x in data[0]["embedding"]]
    usage = payload.get("usage") or {}
    input_tokens = float(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    cost = token_cost(model, int(input_tokens))
    stats = GenerationStats(
        model_name=model,
        embedding_input_tokens=input_tokens,
        cost_per_record=cost,
        llm_call_duration_secs=latency,
        total_llm_calls=1,
        total_embedding_llm_calls=1,
    )
    return vec, stats


def _rag_select_chunk_indices(sims: list[float], num_chunks: int | None, threshold: float | None) -> list[int]:
    """Indices (in original document order) of the chunks to keep, given per-chunk similarities."""
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    if num_chunks is not None:
        keep = order[:num_chunks]
    else:
        keep = [i for i in order if sims[i] >= threshold]
        if not keep and order:
            keep = [order[0]]  # always keep the single best-scoring chunk
    return sorted(keep)


_CHUNK_SIZE_BINOPS = {
    ast.Add: _operator.add, ast.Sub: _operator.sub, ast.Mult: _operator.mul,
    ast.Div: _operator.truediv, ast.FloorDiv: _operator.floordiv, ast.Mod: _operator.mod,
    ast.Pow: _operator.pow,
}
_CHUNK_SIZE_UNARYOPS = {ast.USub: _operator.neg, ast.UAdd: _operator.pos}
_CHUNK_SIZE_FUNCS = {"max": max, "min": min, "abs": abs, "round": round}


def _eval_chunk_size_node(node: ast.AST, input_length: int) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "input_length":
            return input_length
        raise ValueError(f"Unknown name in chunk_size expression: {node.id!r} (only 'input_length' is available)")
    if isinstance(node, ast.BinOp) and type(node.op) in _CHUNK_SIZE_BINOPS:
        return _CHUNK_SIZE_BINOPS[type(node.op)](
            _eval_chunk_size_node(node.left, input_length), _eval_chunk_size_node(node.right, input_length)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CHUNK_SIZE_UNARYOPS:
        return _CHUNK_SIZE_UNARYOPS[type(node.op)](_eval_chunk_size_node(node.operand, input_length))
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in _CHUNK_SIZE_FUNCS and not node.keywords
    ):
        return _CHUNK_SIZE_FUNCS[node.func.id](*[_eval_chunk_size_node(a, input_length) for a in node.args])
    raise ValueError(f"Unsupported syntax in chunk_size expression: {ast.dump(node)}")


def _resolve_chunk_size(chunk_size: "int | str", input_length: int) -> int:
    """Resolve a fixed int chunk_size (unchanged) or evaluate a string expression like
    "max(10000, input_length / 5)" against this field's own text length. Only arithmetic
    operators, max/min/abs/round, and the name 'input_length' are permitted."""
    if isinstance(chunk_size, (int, float)):
        return int(chunk_size)
    node = ast.parse(chunk_size, mode="eval").body
    return max(1, int(_eval_chunk_size_node(node, input_length)))


def _rag_bm25_tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric characters, split on whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


def _rag_bm25_scores(query: str, chunks: list[str]) -> list[float]:
    """BM25 relevance score per chunk, scoring each chunk as its own "document" against `query`."""
    tokenized_chunks = [_rag_bm25_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return [float(s) for s in bm25.get_scores(_rag_bm25_tokenize(query))]


def _format_ranked_chunks(chunks: list[str], keep_idx: list[int], sims: list[float], similarity_method: str) -> str:
    """Format retrieved chunks labeled with rank (1 = most similar) and similarity score,
    ordered by rank."""
    ranked = sorted(keep_idx, key=lambda i: sims[i], reverse=True)
    score_label = "bm25 score" if similarity_method == "bm25" else "cosine similarity"
    return "\n\n".join(
        f"[Chunk rank {rank}/{len(ranked)}, {score_label} {sims[i]:.4f}]\n{chunks[i]}"
        for rank, i in enumerate(ranked, start=1)
    )


def retrieval_context_key(record: DataRecord) -> str | None:
    """Key for a record's retrieval context: its source indices, since PZ chunks a copy of the
    record."""
    indices = getattr(record, "_source_indices", None)
    return None if indices is None else str(indices)


def _record_retrieval_context(op: Any, candidate: DataRecord, input_view: dict[str, str]) -> None:
    """Remember the post-retrieval input this op actually sent to the LLM, so get_op_samples can
    show the agent the retrieved chunks next to each sample.

    Written from pipeline worker threads, one distinct key per record, and read only after
    execution finishes.
    """
    key = retrieval_context_key(candidate)
    if key is None or not input_view:
        return
    store = getattr(op, "_retrieval_contexts", None)
    if store is None:
        store = op._retrieval_contexts = {}
    store[key] = input_view


def rag_get_chunked_candidate(op: Any, candidate: DataRecord, input_fields: list[str]) -> tuple[DataRecord, GenerationStats]:
    """Shared chunk/score/select body for RAGFilter/RAGConvert.get_chunked_candidate."""
    embed_stats = GenerationStats()
    similarity_method = getattr(op, "similarity_method", "embedding")
    query_embedding = None
    if similarity_method == "embedding":
        query_embedding, query_embed_stats = op.compute_embedding(op.embedding_query)
        embed_stats += query_embed_stats

    for field_name in input_fields:
        field = candidate.get_field_type(field_name)
        is_string_field = field.annotation in [str, str | None, str | Any]
        is_list_string_field = field.annotation in [list[str], list[str] | None, list[str] | Any]
        if not (is_string_field or is_list_string_field) or candidate[field_name] is None:
            continue

        if is_list_string_field:
            candidate[field_name] = "[" + ", ".join(candidate[field_name]) + "]"

        field_text = candidate[field_name]
        chunk_size = _resolve_chunk_size(op.chunk_size, len(field_text))
        if len(field_text) < chunk_size:
            continue

        chunks = op.chunk_text(field_text, chunk_size)

        if similarity_method == "bm25":
            sims = _rag_bm25_scores(op.embedding_query, chunks)
        else:
            chunk_embeddings, chunk_embed_stats_lst = zip(*[op.compute_embedding(chunk) for chunk in chunks])
            for chunk_embed_stats in chunk_embed_stats_lst:
                embed_stats += chunk_embed_stats
            sims = [op.compute_similarity(query_embedding, emb) for emb in chunk_embeddings]

        keep_idx = _rag_select_chunk_indices(sims, op.num_chunks_per_field, op.similarity_threshold)
        candidate[field_name] = _format_ranked_chunks(chunks, keep_idx, sims, similarity_method)

    # Snapshot every input field as the LLM sees it.
    _record_retrieval_context(
        op,
        candidate,
        {f: candidate[f] for f in input_fields if candidate[f] is not None},
    )

    return candidate, embed_stats


# ── rag_filter ──────────────────────────────────────────────────────────────────

class RAGFilter(_PZRAGFilter):
    """PZ RAGFilter with a retrieval query separate from the filter condition, embedding or BM25
    similarity, and top-k or similarity-threshold chunk selection."""

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
    """LLM filter on `condition` over the chunks of each long field most similar to
    `embedding_query`, instead of the full field."""
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
        skip_if=None,
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
            prompt_strategy=PromptStrategy.FILTER_NO_REASONING if _no_reasoning_prompt(model, eff) else PromptStrategy.FILTER,
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
        self.op_id = _compute_op_id(self.op_type, self.attributes)
        self._set_skip_if(skip_if)


# ── rag_map ─────────────────────────────────────────────────────────────────────

class RAGConvert(_PZRAGConvert):
    """PZ RAGConvert with a retrieval query separate from the output-field descriptions, embedding
    or BM25 similarity, and top-k or similarity-threshold chunk selection."""

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
    """LLM-derived columns from the chunks of each long field most similar to `embedding_query`,
    instead of the full field."""
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
        skip_if=None,
    ):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        prompt_strategy = PromptStrategy.MAP_NO_REASONING if _no_reasoning_prompt(model, eff) else PromptStrategy.MAP
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
        self.attributes = {
            "model": model.value,
            "reasoning_effort": eff,
            "cols": _cols_attribute(cols),
            "embedding_query": embedding_query,
            "chunk_size": chunk_size,
            "num_chunks_per_field": num_chunks_per_field,
            "similarity_threshold": similarity_threshold,
            "embedding_model": embedding_model,
            "similarity_method": similarity_method,
            "depends_on": depends_on,
        }
        self.op_id = _compute_op_id(self.op_type, self.attributes)
        self._set_skip_if(skip_if)
