"""
Semantic query plan construction with PZ physical operators.
This module uses PZ's physical operators, replicates its execution engine,
and exposes more operator parameters through its API.
"""
from .execution import SubsetExecutionContext
from .operators import (
    AddColSuffix,
    ExactFilter,
    GroupBy,
    Join,
    Limit,
    Map,
    Project,
    RagFilter,
    RagMap,
    SemAgg,
    SemFilter,
    SemJoin,
    SemMap,
)
from .operators.base import Operator
from .operators.rag import DEFAULT_RAG_EMBEDDING_MODEL
from .models import patch_palimpzest_for_new_models
from .pipeline import PhysicalPipeline
from .pz_patches import apply_pz_patches
from .udf_helpers import keys_by_items, number_items, row_id_str

apply_pz_patches()
patch_palimpzest_for_new_models()

__all__ = [
    "PhysicalPipeline",
    "SubsetExecutionContext",
    "Operator",
    "DEFAULT_RAG_EMBEDDING_MODEL",
    "AddColSuffix",
    "ExactFilter",
    "GroupBy",
    "Join",
    "Limit",
    "Map",
    "Project",
    "RagFilter",
    "RagMap",
    "SemAgg",
    "SemFilter",
    "SemJoin",
    "SemMap",
    "keys_by_items",
    "number_items",
    "row_id_str",
]
