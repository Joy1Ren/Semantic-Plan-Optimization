"""Physical pipeline: chain PZ physical operators directly with per-operator model selection.

This package is organized as:
  base.py          - shared constants, helpers, and the Operator base class
  operators/       - one file per operator (SemFilter, SemMap, SemJoin, RagFilter, RagMap,
                      ExactFilter, Map, Join, AddColSuffix, Project, Limit, GroupBy)
  pipeline.py      - the PhysicalPipeline class: the fluent builder API and execution engine

Everything importable from the old single-file `agent_cost_model.opt_agent.physical_pipeline` module
is re-exported here, so `from agent_cost_model.opt_agent.physical_pipeline import PhysicalPipeline`
(etc.) continues to work unchanged.
"""
from .base import (
    DEFAULT_RAG_EMBEDDING_MODEL,
    NUM_SAMPLES,
    SUBSET_SEED,
    Operator,
    SubsetExecutionContext,
    _compute_op_id,
    _has_image_field,
    _make_schema,
    _resolve_reasoning_effort,
    _str_to_pz_model,
)
from .operators import (
    AddColSuffix,
    ExactFilter,
    GroupBy,
    Join,
    Limit,
    Map,
    NonLLMColSuffix,
    NonLLMJoin,
    Project,
    RAGConvert,
    RAGFilter,
    RagFilter,
    RagMap,
    SemFilter,
    SemJoin,
    SemMap,
)
from .pipeline import PhysicalPipeline

__all__ = [
    "PhysicalPipeline",
    "SubsetExecutionContext",
    "Operator",
    "NUM_SAMPLES",
    "SUBSET_SEED",
    "DEFAULT_RAG_EMBEDDING_MODEL",
    "SemFilter",
    "SemMap",
    "SemJoin",
    "RagFilter",
    "RagMap",
    "RAGFilter",
    "RAGConvert",
    "ExactFilter",
    "Map",
    "Join",
    "NonLLMJoin",
    "AddColSuffix",
    "NonLLMColSuffix",
    "Project",
    "Limit",
    "GroupBy",
    "_str_to_pz_model",
    "_resolve_reasoning_effort",
    "_compute_op_id",
    "_make_schema",
    "_has_image_field",
]
