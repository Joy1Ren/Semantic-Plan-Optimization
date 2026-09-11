"""Shared RAG (chunk + retrieve) infrastructure used by rag_filter.py and rag_map.py.

Both operators subclass PZ's own RAGFilter/RAGConvert (palimpzest.query.operators.rag)
with two changes:

  1. Embeddings are computed with an OpenRouter-hosted embedding model (default
     "qwen/qwen3-embedding-8b") via a direct HTTP call to OpenRouter's /embeddings
     endpoint -- the same approach agent_cost_model.sampling.embeddings.EmbeddingClient uses when
     scoring rows for the optimization subset. PZ's installed RAGFilter/RAGConvert now embed
     via litellm using a `palimpzest.constants.Model`, but "qwen/qwen3-embedding-8b" isn't
     in PZ's curated model registry, so constructing a Model for it raises; we bypass PZ's
     embedding path (compute_embedding) entirely and call OpenRouter directly instead. PZ
     still requires a valid `embedding_model: Model` positionally, so callers pass a
     placeholder (never actually used to embed anything, see _RAG_PLACEHOLDER_MODEL below)
     and store the real OpenRouter model id separately.
  2. The text embedded as the *retrieval query* is a separate `embedding_query` string,
     decoupled from the filter condition / output-field descriptions used for the actual
     LLM call. PZ's RAGFilter/RAGConvert embed the filter condition or field descriptions
     directly, but that text is phrased to instruct an LLM, not to sit close to relevant
     passages in embedding space, so retrieval quality suffers. embedding_query lets the
     caller phrase the retrieval query differently (e.g. keyword-dense, no instruction
     framing) from the condition/description passed to the LLM.

Chunk selection supports two mutually exclusive modes (exactly one must be given):
  - num_chunks_per_field: keep the top-k chunks by cosine similarity to embedding_query
    (PZ's original behavior).
  - similarity_threshold: keep every chunk with cosine similarity >= threshold, but always
    keep at least the single best-scoring chunk so a field is never dropped entirely.
"""
from __future__ import annotations

import ast
import json
import operator as _operator
import os
import re
import time
from typing import Any

import httpx
from rank_bm25 import BM25Okapi

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats

from ..base import _RAG_EMBEDDING_BASE_URL

# Placeholder passed to PZ RAGFilter/RAGConvert's required `embedding_model: Model` param.
# It is never used to compute an embedding -- compute_embedding is overridden to call
# OpenRouter directly with each operator's own (string) embedding_model instead.
_RAG_PLACEHOLDER_MODEL = Model.TEXT_EMBEDDING_3_SMALL


def _rag_embed(model: str, text: str) -> tuple[list[float], GenerationStats]:
    """POST one input to OpenRouter's /embeddings endpoint; return (vector, GenerationStats).

    Mirrors agent_cost_model.sampling.embeddings.EmbeddingClient._call: raw HTTP with
    encoding_format="float" so provider errors surface clearly instead of an SDK's opaque
    "No embedding data received".
    """
    body = {
        "model": model,
        "input": text or " ",
        "encoding_format": "float",
        "usage": {"include": True},  # ask OpenRouter to report dollar cost
    }
    start_time = time.time()
    resp = httpx.post(
        f"{_RAG_EMBEDDING_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120.0,
    )
    latency = time.time() - start_time
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding request failed ({resp.status_code}) for model {model!r}: {resp.text[:800]}")
    payload = resp.json()
    data = payload.get("data")
    if not data:
        raise RuntimeError(f"No embedding data from {model!r}. Response: {json.dumps(payload)[:800]}")
    vec = [float(x) for x in data[0]["embedding"]]
    usage = payload.get("usage") or {}
    cost = 0.0
    for key in ("cost", "total_cost", "estimated_cost"):
        value = usage.get(key)
        if value is not None:
            try:
                cost = float(value)
                break
            except (TypeError, ValueError):
                continue
    input_tokens = 0.0
    for key in ("total_tokens", "prompt_tokens"):
        value = usage.get(key)
        if value is not None:
            try:
                input_tokens = float(value)
                break
            except (TypeError, ValueError):
                continue
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


# ------------------------------------------------------------------
# Custom chunk_size expressions: chunk_size may be an int (fixed, as before) or a string
# expression evaluated per field against that field's own text length, e.g.
# "max(10000, input_length / 5)". Evaluated with a small whitelisted-AST interpreter rather
# than eval() -- this runs directly during plan EXECUTION, outside the sandboxed
# LocalPythonExecutor plan-construction code normally runs in, so a permissive eval() here on
# an LLM-authored string would be a real arbitrary-code-execution hole.
# ------------------------------------------------------------------

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
    """Lowercase, strip non-alphanumeric characters, split on whitespace -- matches docetl's
    own BM25 preprocessing (docetl/operations/sample.py's _sample_top_fts) so bm25 mode here
    behaves the same way as the doc_chunking_topk directive it mirrors."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return text.split()


def _rag_bm25_scores(query: str, chunks: list[str]) -> list[float]:
    """BM25 relevance score per chunk, scoring each chunk as its own "document" against
    `query` -- rank_bm25.BM25Okapi, the same algorithm docetl's topk(method="fts") uses."""
    tokenized_chunks = [_rag_bm25_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return [float(s) for s in bm25.get_scores(_rag_bm25_tokenize(query))]


def _format_ranked_chunks(chunks: list[str], keep_idx: list[int], sims: list[float], similarity_method: str) -> str:
    """Format retrieved chunks labeled with rank (1 = most similar) and similarity score,
    ordered by rank -- mirrors docetl's topk-operator reduce-step chunk annotations
    (`{{ input._..._rank }}`, `{{ input._..._score }}`), which the LLM call sees directly."""
    ranked = sorted(keep_idx, key=lambda i: sims[i], reverse=True)
    score_label = "bm25 score" if similarity_method == "bm25" else "cosine similarity"
    return "\n\n".join(
        f"[Chunk rank {rank}/{len(ranked)}, {score_label} {sims[i]:.4f}]\n{chunks[i]}"
        for rank, i in enumerate(ranked, start=1)
    )


def retrieval_context_key(record: DataRecord) -> str | None:
    """Identity under which a record's post-retrieval input view is stored and looked up.

    `_source_indices` survives `DataRecord.copy()` (see palimpzest's DataRecord.copy), which is
    what makes this work at all: PZ's RAGConvert/RAGFilter chunk a COPY of the candidate, so the
    record the pipeline captured as the operator's input sample is NOT the object retrieval
    mutated. Keying on source indices is how the two are matched back up -- the same convention
    _oracle_judge_filter_join_op already uses to dedupe filter/join samples.
    """
    indices = getattr(record, "_source_indices", None)
    return None if indices is None else str(indices)


def _record_retrieval_context(op: Any, candidate: DataRecord, input_view: dict[str, str]) -> None:
    """Remember the post-retrieval input this op actually sent to the LLM, for the oracle judge.

    Without this the judge scores a rag_map/rag_filter against the FULL source field while the
    operator only ever saw the retrieved chunks, which silently turns every retrieval miss into
    an apparent extraction error and makes the per-op score unable to tell the two apart.

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

    # Snapshot every input field as it now stands -- retrieved chunks for the fields that were
    # long enough to chunk, and the untouched original for the ones that fell under chunk_size
    # (the `continue` above) or aren't text. That whole view, not just the chunked fields, is
    # what the LLM call below receives, so it is what the oracle judge has to score against.
    _record_retrieval_context(
        op,
        candidate,
        {f: candidate[f] for f in input_fields if candidate[f] is not None},
    )

    return candidate, embed_stats
