"""LLM-guided selection of the optimization subset.

`CostModelAgent` scores every candidate query plan against a small fixed subset of rows rather
than the full table, so that subset decides what the optimizer can see: a subset that misses
the rows a query targets makes every plan look alike. `DataSampler` builds it in one of four
modes:

  baseline  BM25 + embedding cosine over the raw query text. No LLM call.
  one-shot  One LLM call writes a scoring spec -- separate lexical and semantic queries,
            keyphrases, an optional row filter, and which columns each scorer reads.
  agentic   The same, then up to two revisions: the agent inspects its own top-scoring rows
            and the score distribution, and rewrites the spec if retrieval went wrong.
  random    A seeded uniform draw of `sample_size` rows. No LLM call, no scoring -- the
            no-signal floor the other three modes are measured against.

The first three share a single scoring core; random bypasses it entirely.

Every run writes the subset CSV (the only artifact the optimizer reads), a per-row scores CSV,
a cost + per-round record in sampling_results.json, and — for the two LLM modes — a prompt
trace. Run it inline with `run_opt.py --opt-subsample`, or ahead of time with
`python -m agent_cost_model.sampling.cli --config <benchmark.yaml>` so the same subset can be
optimized against repeatedly.

See `data_sampler.py` for the orchestration and `scoring.py` for the formula.
"""
from agent_cost_model.sampling.constants import IMAGE_FIELD, MODES
from agent_cost_model.sampling.data_sampler import DataSampler
from agent_cost_model.sampling.results import SamplingResult, mode_subset_path
from agent_cost_model.sampling.spec import SamplingSpec, ScorerColumns

__all__ = [
    "DataSampler",
    "SamplingResult",
    "mode_subset_path",
    "SamplingSpec",
    "ScorerColumns",
    "IMAGE_FIELD",
    "MODES",
]
