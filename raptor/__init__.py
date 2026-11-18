"""RAPTOR: an LLM agent that optimizes semantic query plans by executing candidate plans on a data
subset and measuring their cost, latency and quality.

    import raptor
    raptor.RAPTOR               the optimization agent
    raptor.PhysicalPipeline     plans built from palimpzest's physical operators
    raptor.DataSampler          builds the optimization subset plans are scored on

Submodules and the names below load on first use, so a light module such as `raptor.routing` or
`raptor.subsets` imports without palimpzest.
"""
from __future__ import annotations

import importlib

_SUBMODULES = {"physical_pipeline", "sampling", "tools"}
_EXPORTS = {
    "RAPTOR": "raptor.core_agent",
    "PhysicalPipeline": "raptor.physical_pipeline",
    "DataSampler": "raptor.sampling",
    "ResultsStore": "raptor.results_store",
    "PlanQualityEvaluator": "raptor.plan_quality_evaluator",
    "QualityResult": "raptor.plan_quality_evaluator",
    "RoutedClient": "raptor.llm_client",
}

__all__ = sorted(_SUBMODULES | _EXPORTS.keys())


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
