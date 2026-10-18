"""LLM-guided selection of the optimization subset.

`CostModelAgent` scores every candidate query plan against a small fixed subset of rows rather
than the full table, so that subset decides what the optimizer can see: a subset that misses
the rows a query targets makes every plan look alike. `DataSampler` builds it in one of five
modes:

  baseline           BM25 + embedding cosine over the raw query text. No LLM call.
  one-shot           One LLM call writes a scoring spec -- separate lexical and semantic
                     queries, keyphrases, an optional row filter, and which columns each
                     scorer reads.
  one-shot-gradient  The same spec, but each stratum's allocation is spread across its own
                     ranking -- half from the top, a quarter from the median, a quarter from
                     the 90th percentile -- so the subset spans what the query means instead of
                     holding only rows that match it.
  agentic            The same as one-shot, then up to two revisions: the agent inspects its own
                     top-scoring rows and the score distribution, and rewrites the spec if
                     retrieval went wrong.
  random             A seeded uniform draw of `sample_size` rows. No LLM call, no scoring --
                     the no-signal floor the other four are measured against.

All but random share a single scoring core; they differ in who writes the spec and how the
ranking is drawn from.

Every run writes the subset CSV (the only artifact the optimizer reads), a per-row scores CSV,
a cost + per-round record in sampling_results.json, and — for the three LLM modes — a prompt
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
