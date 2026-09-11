"""Tunables and defaults for the data sampler.

Following the codebase convention, policy constants live as module-level names rather than in
a shared config file (see `experiments/run_opt.py`'s MODEL/MAX_STEPS block).
"""
from __future__ import annotations

# Imported unguarded, deliberately. `opt_agent/llm_sampler.py` wrapped the equivalent import in
# a `try/except ImportError` naming a package that has never existed in this layout (a leftover
# from when `agent/` became `agent_cost_model/opt_agent/`). The fallback therefore fired on
# every run, silently pinning constants that are supposed to track the pipeline's. A hard
# import turns the next such rename into an ImportError at import time instead of values that
# are quietly wrong. `physical_pipeline.constants` is a leaf with no palimpzest dependency, so
# this costs nothing at import.
from agent_cost_model.opt_agent.physical_pipeline.constants import (
    DEFAULT_RAG_EMBEDDING_MODEL,
    SUBSET_SEED,
    _RAG_EMBEDDING_BASE_URL,
)

# NUM_SAMPLES is deliberately NOT imported here. It caps per-operator sample collection during
# optimization (`_execute_core`'s max_samples), which is a judging-cost knob, not a statement
# about how many rows the evaluation subset should hold. Borrowing it as this module's default
# tied two unrelated numbers together and hid the subset size behind a constant nobody sets on
# purpose. The subset size has no default: every caller states it.

__all__ = [
    "DEFAULT_RAG_EMBEDDING_MODEL",
    "SUBSET_SEED",
    "_RAG_EMBEDDING_BASE_URL",
    "IMAGE_FIELD",
    "MODES",
    "DEFAULT_ALPHA",
    "DEFAULT_EMBED_WORKERS",
    "EXPLORE_AUTHORIZED_IMPORTS",
    "EXPLORE_OUTPUT_CHARS",
    "EXPLORE_MAX_IMAGES",
    "DEFAULT_MAX_ROUNDS",
    "FEEDBACK_ROW_CHARS",
    "FEEDBACK_CHAR_CEILING",
    "MAX_CELL_CHARS",
    "MAX_KEYPHRASES",
    "PREVIEW_ROWS",
    "PREVIEW_CELL_CHARS",
    "IMAGE_DESCRIBER_MODEL",
    "RESULTS_SCHEMA_VERSION",
]

# Pseudo-column naming the image modality. The agent may name it in `dense_columns` alongside
# real columns; it is never a real DataFrame column.
IMAGE_FIELD = "__image__"

# Embedding is network-bound, so concurrency here is close to a linear speedup. Recorded
# with every run: a latency figure means nothing without the worker count behind it.
DEFAULT_EMBED_WORKERS = 10

# Agentic mode's optional pre-sampling exploration: the agent submits one snippet that is run
# against the dataframe so it can look before it writes a spec. Read-only analysis is the whole
# point, so the import list stays narrow -- enough to count, group and regex, nothing that
# reaches the network or filesystem.
EXPLORE_AUTHORIZED_IMPORTS = ["re", "math", "statistics", "collections", "itertools", "json"]
# The result is pasted into the next prompt, so it is capped rather than allowed to blow the
# context on a stray `print(df)`.
EXPLORE_OUTPUT_CHARS = 6000
# On an image dataset the snippet can only see filenames, so exploration also offers a look at
# the pictures themselves, described by the cheap vision model. Capped: this is orientation
# before the first spec, and every image is a paid vision input.
EXPLORE_MAX_IMAGES = 8

MODES = ("baseline", "one-shot", "agentic", "random")

DEFAULT_ALPHA = 0.5

# Agentic mode: one initial spec plus at most two revisions.
DEFAULT_MAX_ROUNDS = 3

# Cell-content budget for the feedback report, PER SELECTED ROW. Per-row rather than a fixed
# total because the report shows every row of the subset: a fixed total would silently halve
# each row's space when the subset size doubles, and the agent judges relevance from these
# cells. 2,400 shows ~98% of a SemBench ecomm row in full at any k.
FEEDBACK_ROW_CHARS = 2_400

# Absolute ceiling on that content, so a large subset of a wide table cannot run away. The
# formula, distribution table, and per-row score lines are small and always shown in full, so
# this caps the only part that can blow up (a 47-column table at 10k chars/cell is 2.3M chars).
FEEDBACK_CHAR_CEILING = 80_000

# Per-cell ceiling, applied after the budget split. A cell never exceeds this even if the
# budget would allow it.
MAX_CELL_CHARS = 10_000

# Beyond a handful, keyphrases stop being a precision signal and turn `sparse` into a raw
# keyword count.
MAX_KEYPHRASES = 12

# The schema preview shown when asking for the initial spec.
PREVIEW_ROWS = 5
PREVIEW_CELL_CHARS = 100

# Cheap vision model the sampler describes images with, in the feedback report and in the
# pre-spec exploration. CostModelAgent picks its own describer independently.
IMAGE_DESCRIBER_MODEL = "google/gemini-2.5-flash-lite"

# Bumped when the sampling_results.json record shape changes incompatibly.
RESULTS_SCHEMA_VERSION = 2
