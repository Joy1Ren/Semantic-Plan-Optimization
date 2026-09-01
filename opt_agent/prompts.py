"""Static briefing text shared by `CostModelAgent` and `CostHelperAgent`: the physical-operator
catalog, the available-models catalog, and the plan-quality-metric documentation."""

from __future__ import annotations

import pathlib

_PHYSICAL_SEMANTIC_OPERATORS = {
    "sem_filter": "pipeline.sem_filter(condition: str, model: pz.Mode) — LLM filter; keeps rows where condition is true.",
    "sem_map": "pipeline.sem_map(cols: list[dict], model: pz.Model) — Add LLM-derived columns. col is list of dict {'name': str, 'type': type, 'description': str}",
    "sem_join": "pipeline.sem_join(other: PhysicalPipeline, condition: str, model: pz.Model) — LLM join; keeps pairs where condition holds."
                "If specifying columns with `depends_on`, make sure to include both left and right columns (e.g., `depends_on=['col', 'col_right']`)."
                "Performs a self join when `other` is the same instance as `pipeline`",
    "rag_filter": "pipeline.rag_filter(condition: str, embedding_query: str, model: pz.Model, chunk_size: int | str = 20000, similarity_method: str = 'embedding', num_chunks_per_field: int = None, similarity_threshold: float = None, embedding_model: str = 'qwen/qwen3-embedding-8b') — "
                  "For a long text field where only a small part is actually relevant, chunk the field and retrieve just the relevant chunks BEFORE the LLM call, instead of sending the whole field to the LLM — cheaper than sem_filter. It then LLM-filters using `condition` over just the retrieved chunks. "
                  "There are two chunk-selection methods, chosen via similarity_method: 'embedding' (default) scores chunks by cosine similarity of their embedding to embedding_query — best for paraphrase/meaning matches; 'bm25' scores chunks by keyword/full-text overlap with embedding_query, with no embedding cost — best when the target hinges on specific terminology (e.g. legal/technical text). "
                  "embedding_model only matters (and is only used) when similarity_method='embedding'; it is ignored for 'bm25'. "
                  "chunk_size (characters) is required for EITHER method — either a fixed int, or a string expression over `input_length` (that field's own char count), e.g. \"max(10000, input_length / 5)\", to scale chunk size with document length instead of a fixed size. "
                  "For selecting which chunks to keep, specify EXACTLY ONE of: num_chunks_per_field (keep the top-k most similar/relevant chunks — works for either similarity_method), or similarity_threshold (keep every chunk with cosine similarity >= threshold, always keeping at least the single best chunk so a field is never dropped entirely — only available when similarity_method='embedding'). "
                  "IMPORTANT: embedding_query is a SEPARATE string from `condition`, used only to score/retrieve chunks — write it with the terms/phrases you'd expect to score highest against the relevant "
                  "passage itself (keyword-dense, no framing like 'the text that...'), while `condition` stays a natural-language instruction for the LLM. "
                  "E.g. condition=\"the review complains about a broken zipper\" -> embedding_query=\"broken zipper, zipper stuck, zipper fell off\".",
    "rag_map": "pipeline.rag_map(cols: list[dict], embedding_query: str, model: pz.Model, chunk_size: int | str = 20000, similarity_method: str = 'embedding', num_chunks_per_field: int = None, similarity_threshold: float = None, embedding_model: str = 'qwen/qwen3-embedding-8b') — "
               "Same chunk-and-select step as rag_filter (see rag_filter for similarity_method/embedding_model/chunk_size/selection details, and note embedding_query is a SEPARATE string from the column descriptions in `cols`, scored the same way), but the final step is an LLM map instead of an LLM filter: "
               "derives new columns (cols, same shape as sem_map) from just the retrieved chunks — cheaper than sem_map when a field is long but only a small part of it is relevant.",
}

_PHYSICAL_NONSEMANTIC_OPERATORS = {
    "filter": "pipeline.filter(fn: Callable[[dict], bool]) — Exact row filter using a Python callable.",
    "map": "pipeline.map(fn: Callable[[dict], dict], cols: list[dict]) - Exact row map using a Python callable to add new columns. col is list of dict {'name': str, 'type': type, 'description': str}",
    "add_col_suffix": "pipeline.add_col_suffix(suffix: str) — Append `suffix` to EVERY column name."
                      " Use before a join to disambiguate columns so the two sides have no colliding names, e.g."
                      " left.add_col_suffix('_dish'); right.add_col_suffix('_table'); left.join(right, lambda l, r: l['brand_dish'] == r['brand_table']).",
    "join": "pipeline.join(other: PhysicalPipeline, condition_fn: Callable[[dict, dict], bool]) — Exact (non-LLM) join;"
            "keeps pairs where condition_fn(left, right) is True (e.g. lambda l, r: l['brand'] == r['brand'])."
            "Right-side columns are suffixed with '_right' on a name collision. Prefer this over sem_join when the match is an exact/computable predicate."
            "Performs a self join when `other` is the same instance as `pipeline`",
    "project": "pipeline.project(cols: list[str]) — Select a subset of columns.",
    "limit": "pipeline.limit(n: int) — Keep at most n rows.",
    "groupby": "pipeline.groupby(group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]) — Group and aggregate. Produces schema name 'agg_func(agg_field)', e.g. 'count(reviewId)' or 'average(score)'.",
}
_AVAILABLE_MODELS_TEXT = (pathlib.Path(__file__).parent / "available_models.txt").read_text()

# Human-readable explanation of the overall plan-quality metric, keyed by a query's `accuracy_metric`
# (see files/<use_case>/queries/q<id>.toml and QualityEvaluator.evaluate). Rendered into the briefing
# so the agent knows exactly how the 0–1 `quality` score it optimizes is computed.
_QUALITY_METRIC_DOCS = {
    "f1-score": (
        "F1 of the returned id set vs. the oracle/ground-truth id set — the harmonic mean of "
        "precision (fraction of returned ids that are correct) and recall (fraction of correct ids "
        "returned). 1.0 = exactly the right set; both missing and extra ids lower it."
    ),
    "adjusted-rand-index": (
        "Adjusted Rand Index between your per-record class assignment and the oracle's — a "
        "clustering-agreement score corrected for chance. 1.0 = identical grouping, ~0 = chance-level, "
        "and it can go negative. Used for classification/grouping queries."
    ),
    "spearman-rank": (
        "Spearman rank correlation between your scoring and the oracle's -- a monotonic"
        "agreement score. 1.0 = perfect positive correlation, ~0 = no correlation, -1.0 = perfect negative correlation"
    ),
    "relative-error": (
        "A transformed metric from relative error between your value and the oracle's: 1/(1 + relative error)."
        "1.0 = exact match. ~0 = very large difference"
    ),
    "f1_jaccard": (
        "F1 over per-record binary presence/absence decisions vs. the oracle, but a record only "
        "counts as a true positive if BOTH (a) you correctly decided the clause/item is present, AND "
        "(b) the Jaccard similarity (word-level) between your extracted text span and the oracle's "
        "span is > 0.15. Getting the binary decision right with a low-overlap or missing span still "
        "counts as wrong -- not a true positive. 1.0 = every present/absent decision correct and every "
        "extracted span sufficiently overlaps the oracle's; extract the actual relevant clause text, "
        "not just a presence flag."
    ),
}


def _quality_metric_reminder(eval_metric: str | None) -> str:
    """One-line reminder of what `quality` measures, shown on every execute_plan result so the agent
    keeps the metric in mind — and understands why overall quality can diverge from per_sem_op_quality."""
    metric_desc = f"`{eval_metric}`" if eval_metric else "a plan-output-vs-oracle score"
    return (
        f"quality = {metric_desc} (0-1, higher is better). It scores the final plan output using the "
        "evaluation metric; per_sem_op_quality scores the accuracy per semantic operator."
    )


def _quality_metric_section(eval_metric: str | None) -> str:
    """Render the '## Plan Quality Metric' briefing section for this query's metric."""
    if not eval_metric:
        return (
            "## Plan Quality Metric\n"
            "Each executed plan's `quality` (0–1, higher is better) is scored against a strong-LLM "
            "ORACLE running the same logical query. Treat the oracle score as ground truth and "
            "optimize cost/latency at the best achievable quality."
        )
    doc = _QUALITY_METRIC_DOCS.get(
        eval_metric, f"the `{eval_metric}` metric (0–1, higher is better)"
    )
    return (
        "## Plan Quality Metric\n"
        f"The `quality` score (0–1) reported for each executed plan is: {doc}"
    )

_SYSTEM_TEMPLATE = """\
{briefing}

{quality_metric}

## HARD RULES

### No data snooping or hardcoded indexes
`explore_sample` and `explore_schema` help you understand **schema and format**; `explore_data`
lets you study the dataset in **aggregate** (row counts, keyword/keyphrase frequencies, value
distributions) to inform how you design plans. What you learn there may only shape HOW you build a
general plan — it must NEVER be baked into a plan as specific records, and you must not brute-force
the data to solve the query and reverse-engineer a plan from the answer.
You MUST NOT use the data to identify specific records and then hardcode their IDs. For example, writing
plans like `filter(lambda row: row["id"] in [3, 17, 42])` is **cheating** and will produce meaningless results.
However, you can use your observations to construct Python-based and regex functions for deterministic operators. For
example, `filter(lambda row: row["description"].isin(["settee", "sofa", "couch"]))`
and `map(lambda row: {{ "red_freq_count": row["description"].str.count("red")}},
    cols=[{{"name": "red_freq_count", "type": str, "description": "Number of times the word 'red' appears in the description"}}])`
are valid.

{estimate_rule}

## Tools (already imported into your python sandbox)

{tools_doc}

## Plan Writing (using PhysicalPipeline API)
You have access to the following operators to build physical plans.
Make sure to output the desired fields in the correct order, as specified in the task.
PhysicalPipeline already has a execution speedup involving `limit` operators:
once the limit number of records is passed, all previous operator execution calls are cancelled.
Thus, it is unnecessary to include intermediate `limit` operators to reduce cost/latency. The `limit`
operator should only be used at the end of a plan.

Semantic operators — require a model= argument:
All semantic operators have an optional "depends_on: list[str] | None" argument,
which is a list of field names to pass in for the LLM call (instead of the full input schema).
Using `depends_on` will reduce the LLM input and help reduce cost.
{physical_sem_ops}

Non-semantic operators — no model argument:
{physical_nonsem_ops}

## Available Models:
All models listed support both text and image inputs. Use `add_image_data` with a `depends_on`
that includes the image column to pass images to any model.
Within each tier, cheaper options are listed first. Prefer the cheaper models unless improving
quality requires a more expensive model.
{available_models}

## Also available in your sandbox (no import needed)
These run in the python blocks you execute each step (plan code, passed to write_plan, runs in a
separate minimal sandbox — see the write_plan tool description).
- `explore_data(filename)` : read a CSV from the data directory and return the FULL DataFrame for
    DATA EXPLORATION. Use it to compute AGGREGATE statistics — row counts, how many rows contain a
    keyword/keyphrase, `value_counts()` of a column — to gauge how common an item or attribute is.
    This reveals the selectivity of filters, the size of tables feeding a join, and the class
    balance of a classification/search task (rare target vs. common groups). Keep it lightweight
    (a few probes, NOT exhaustive regex/brute-force scans) and use what you learn to GUIDE plan
    design — never to solve the query or reverse-engineer a plan.
    df = explore_data("items.csv")
- `plans`           : dict[str, dict] — {{name: {{"plan": PhysicalPipeline, "description": str, "optimizations": ...}}}}; populated by write_plan (updated by execute_plan)
- `plan_codes`      : dict[str, str] — code strings stored by write_plan (keys = plan names)
- `plan_results`    : the observed-execution store (`plan_results.rows`, `plan_results.df`)
- `op_results`      : the observed-execution store (`op_results.rows`, `op_results.df`, `op_results.summary()`)
- stdlib: `math`, `statistics`, `random`, `collections`, `itertools`, `json`

You have <= {max_steps} steps. On each step output EXACTLY ONE fenced block:
  - a ```python``` block — runs in the sandbox; its stdout/return value comes back as your next observation; or
  - a ```json``` block — your final answer (parsed as data, not executed). Emit this once, when done.
DO NOT emit multiple fenced blocks in each step output. Keep each python block small and focused (one logical action).
Requirements for the final answer:
{final_answer_doc}"""
