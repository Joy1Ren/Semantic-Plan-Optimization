"""System prompts and user-message builders for the spec and revision calls."""
from __future__ import annotations

import pandas as pd

from agent_cost_model.sampling.constants import (
    EXPLORE_AUTHORIZED_IMPORTS,
    EXPLORE_MAX_IMAGES,
    EXPLORE_OUTPUT_CHARS,
    IMAGE_FIELD,
    MAX_KEYPHRASES,
    PREVIEW_CELL_CHARS,
    PREVIEW_ROWS,
)

__all__ = [
    "SCORING_EXPLAINER",
    "SPEC_SYSTEM",
    "REVISE_SYSTEM",
    "spec_system",
    "revise_system",
    "PRESELECT_BLOCK",
    "EXPLORE_SYSTEM",
    "explore_system",
    "build_explore_user",
    "render_explore_result",
    "render_explore_images",
    "REPROMPT",
    "IMAGE_DESCRIBE_SYSTEM",
    "image_describe_system",
    "build_spec_user",
    "build_revise_user",
    "schema_block",
    "preview_block",
]


# Shown to the model in both calls, and again verbatim in the feedback report, so the numbers
# it reads are never separated from the formula that produced them.
SCORING_EXPLAINER = """\
Every row that survives the filter is scored:

  bm25_norm = minmax(BM25(bm25_query, text of bm25_columns))
  hits      = how many DISTINCT keyphrases appear in keyphrase_columns (each counts once)
  sparse    = minmax(bm25_norm + hits)
  dense     = minmax(cosine(embed(dense_query), embedding of dense_columns))
  score     = alpha * sparse + (1 - alpha) * dense

The k highest-scoring rows become the subset. Both `sparse` and `dense` are rescaled to [0,1],
so alpha genuinely trades one against the other.

Any scorer can be turned OFF by leaving its query empty ("") or its columns empty ([]). A
scorer that is off is dropped from the formula rather than contributing zero, and the remaining
half is used on its own -- so dense-only retrieval is `score = dense`, not `(1-alpha) * dense`.
Turn one off when it would only add noise.

Turning ALL of them off is a legitimate answer. It means "this query gives no reason to prefer
one row over another", and the subset is then drawn uniformly at random."""


_SPEC_FIELDS = """\
  "filter":            a Python boolean expression over a variable named `row`, e.g.
                       "row[\\"price\\"] <= 500", or null. Use ONLY deterministic column
                       predicates -- comparisons, membership. Never anything requiring an
                       image or free text to be understood; that is what the other fields
                       are for. The filter runs FIRST and narrows everything after it.
  "bm25_query":        a phrase of the distinct, discriminative KEYWORDS to match on. Lexical,
                       not grammatical -- no query framing, no instructions. "" turns BM25 off.
  "keyphrases":        up to a few exact phrases. Each one that appears in a row adds a full
                       point, so these are strong "this row contains the thing" signals. Use
                       0-4; beyond that the score degenerates into a keyword count. [] uses none.
  "dense_query":       a fluent descriptive phrase for the embedding model. This should NOT be
                       the same string as bm25_query -- it can carry compound meaning and
                       relationships that a bag of keywords cannot. "" turns the dense half off.
  "bm25_columns":      which columns BM25 reads. [] turns BM25 off; omit the field for all
                       text columns.
  "keyphrase_columns": which columns the keyphrases are matched against.
  "dense_columns":     which columns are embedded and compared to dense_query.
  "alpha":             0.0-1.0, the weight on the sparse (lexical) half. 0.5 is a fair start;
                       raise it when the query hinges on specific words, lower it when it
                       hinges on meaning.
  "rationale":         one or two sentences on why this spec should find the right rows."""


_SHARED_GUIDANCE = """\
Choosing columns matters. The three scorers do not have to read the same columns, and they do
not all have to run -- an image-only question has nothing for BM25 to match, and a question
about meaning rather than wording is often better served by the dense half alone.

DECIDE FIRST WHETHER TARGETING HELPS AT ALL. Many tasks apply to every row equally or can perform well
on any random subset. When that is the case, return every field empty:

  {"filter": null, "bm25_query": "", "dense_query": "", "keyphrases": [],
   "bm25_columns": [], "keyphrase_columns": [], "dense_columns": [], "alpha": 0.5,
   "rationale": "the reasoning"}

You are NOT expected to always find something to target, and an empty spec is not a failure or
a non-answer.

Write dense_query as PURELY POSITIVE -- only what you want. Embedding similarity has no notion
of negation: "a bag that is not green" sits close to green bags, because the embedder still
keys on "green bag". Drop the excluded thing entirely, or express it as a `filter` instead.
  good: "a black leather handbag"      bad: "a handbag that is not green"
  good: "a coffee table"               bad: "a table, excluding desks"

BM25 scores are relative to the rows being ranked, so a term that appears in every row carries
almost no weight, and a term in no row carries none at all. Prefer distinctive words."""


_STRATA_GUIDANCE = """\
ONE PLAN, ONE OR SEVERAL STRATA. The default is a single spec that takes all {k} rows. But a
query sometimes needs several *different kinds* of row represented, and a single ranking cannot
deliver that: whichever idea scores highest takes every slot, and the others are squeezed out
entirely. When that is the case, split the subset into strata and say how many rows each gets.

Reach for strata when the query:
  - names two or more distinct conditions, categories or cases that a plan must handle;
  - compares or relates groups, so the subset needs members of each;
  - has an obvious majority case plus rarer ones worth testing against;

Use one stratum when the query is about a single idea. More strata is not better -- each one
gets fewer rows, and a stratum with one or two rows is a weak sample of anything.

  {"strata": [
     {"label": "<short name>", "m": <rows>, ...spec fields...},
     {"label": "<short name>", "m": <rows>, ...spec fields...}
   ]}

The "m" values MUST sum to exactly {k}, the total subset size. Each stratum has its own
filter, queries, columns and alpha -- they are independent retrievals over the same table. A
row already taken by an earlier stratum is skipped by later ones, so strata never duplicate
rows; overlapping strata just means the later ones reach further down their own ranking.

Give every stratum a short "label" naming the idea it covers, so the rows it selects can be
read back against your intent."""


# Appended to both the spec and revision prompts only under `--idx-specification`. Off by
# default: naming rows directly bypasses the retrieval spec entirely, which is the thing being
# measured, so it is an experiment the runner opts into rather than a standing capability.
PRESELECT_BLOCK = """\


YOU MAY ALSO NAME SPECIFIC ROWS TO INCLUDE, by id, in a "preselected_idx" list:

  {"preselected_idx": ["10448", "10689"], "strata": [...]}

Those rows go into the subset outright, without being scored or ranked. Use it for rows you
have concrete reason to want -- a case the retrieval spec demonstrably cannot reach, an edge
case you saw during exploration -- and not as a substitute for writing a spec: a subset you
hand-pick in full teaches the optimizer only what you already believed.

The ids must come from rows you have actually seen; invented ids are dropped. THE "m" VALUES
THEN SUM TO k MINUS THE NUMBER OF PRESELECTED IDS, since those rows already hold their slots.
With 3 preselected ids and k=20, the strata allocate 17. Omit the key to preselect nothing."""


SPEC_SYSTEM = f"""\
You choose how to retrieve a small, informative evaluation subset for a data query.

A downstream optimizer compares candidate query plans by running them on a subset of a large
table. Your job is to write a retrieval spec that surfaces at least a few rows
that the query is actually about. Not all retrievals need to be successful -- some queries
have few rows that are relevant.

{SCORING_EXPLAINER}

{_STRATA_GUIDANCE}

Return EXACTLY ONE fenced ```json``` block. Either a single spec:

{{
{_SPEC_FIELDS}
}}

or a stratified plan, where each entry carries those same fields plus "m" and "label":

{{"strata": [{{"label": "...", "m": <rows>, ...}}, ...]}}

{_SHARED_GUIDANCE}

Keyphrases match whole words contiguously: "crew neck" matches "a crew neck tee" but not "a
crew of sailors", and "dress" does not match "dressing". At most {MAX_KEYPHRASES} are kept.

Emit the JSON block and nothing else."""


REVISE_SYSTEM = f"""\
You are refining the retrieval spec you just wrote, having seen what it actually retrieved.

{SCORING_EXPLAINER}

You will be shown EVERY row the plan selected -- grouped by stratum when there is more than
one -- with its component scores, the score distribution across the whole pool, and diagnostics
about terms that matched nothing. These are exactly the rows the optimizer will receive, so
judge the subset as a whole, not just its strongest members.

Signals that something went wrong:
  - The top rows are plainly not what the query asks for.
  - Raw cosine has almost no range (e.g. every row between 0.29 and 0.41): the dense query is
    not discriminating, so the normalized spread is manufactured from noise.
  - bm25 terms or keyphrases matched zero rows: they are contributing nothing.
  - Nearly every row has the same score: that component is dead weight.
  - A column you can now see in the row contents would have been a better target.
  - The rows are all variations of one thing when the query needs several kinds represented:
    split into strata, or rebalance the allocations you already have.
  - One stratum is doing all the useful work while another retrieved nothing relevant: drop it
    and give its rows to the stratum that is working.

Return EXACTLY ONE fenced ```json``` block, either:

  {{"accept": true, "reason": "..."}}

when the retrieved rows are right, or

  {{"accept": false, "reason": "...",
{_SPEC_FIELDS}
  }}

or, to change how the subset is split,

  {{"accept": false, "reason": "...", "strata": [{{"label": "...", "m": <rows>, ...}}, ...]}}

You may move between a single spec and a stratified plan in either direction. The "m" values
must still sum to exactly {{k}}.

In every case with "dense_columns" OMITTED -- the dense target columns are fixed after the first round,
because changing them means re-embedding the whole table. You may still rewrite dense_query
freely, and retarget bm25_columns and keyphrase_columns at no cost.

{_SHARED_GUIDANCE}

Do not re-send a spec you have already tried. Emit the JSON block and nothing else."""


EXPLORE_SYSTEM = f"""\
You are about to choose how to retrieve a small evaluation subset from a large table. Before
you do, you may run ONE read-only Python snippet against the table to see what is actually in
it. This is optional and happens only once.

The schema and a few sample rows are shown below, but a handful of rows cannot tell you how
values are distributed: which categories exist and how common they are, how a field is
actually spelled and formatted, whether a term you were going to search for appears at all, or
how many rows would survive a filter you were considering. Measuring beats guessing, and a
retrieval spec built on a guess about vocabulary usually retrieves nothing.

Your snippet runs with a pandas DataFrame named `df` already in scope (and `id_col`, the name
of the identifier column). You may import: {", ".join(EXPLORE_AUTHORIZED_IMPORTS)}. Use
`print(...)` for anything you want to see. There is no network and no filesystem access.

Keep it small and targeted: a few aggregate counts answer far more than a dump of rows. The
output is truncated to about {EXPLORE_OUTPUT_CHARS:,} characters, so do not print the frame.

Return EXACTLY ONE fenced ```json``` block, either:

  {{"explore": false, "reason": "the schema and samples already tell me what I need"}}

or:

  {{"explore": true, "code": "print(df['some_column'].value_counts().head(20))"}}

Skip it when the sample rows already answer your question -- the call is not free, and an
uninformative snippet costs a round trip for nothing. Emit the JSON block and nothing else."""


# Appended to EXPLORE_SYSTEM only when the dataset has images. The snippet sees the table, which
# on an image dataset is little more than a list of ids -- the content it must retrieve on is in
# the pixels, and nothing a pandas snippet prints can reach it.
_EXPLORE_IMAGES_BLOCK = f"""


THIS DATASET ALSO HAS AN IMAGE PER ROW, and the table alone cannot tell you what is in them.
Alongside the snippet -- or instead of it -- you may have {EXPLORE_MAX_IMAGES} of the images,
drawn at random, described to you in words by a vision model. Use it to check the assumptions
your `dense_query` rests on: whether the attribute the query turns on is visible at all, how the
subjects are actually photographed, how much the images differ from one another.

Add to the same JSON object:

  "see_images":     true to have {EXPLORE_MAX_IMAGES} random images described. You do not choose
                    WHICH rows -- picking rows is what the retrieval spec is for, and this look
                    comes before you have written one. Omit the key, or false, to see none.
  "image_question": optional, a natural-language question the describer must answer for every
                    image on top of its description, e.g. "is the garment's sleeve length
                    visible?". Ask what you actually need to know to write the spec.

`"explore": true` with only `see_images` and no `code` is a good answer on an image-only table,
where there is no vocabulary for a snippet to measure. Emit the JSON block and nothing else."""


def explore_system(has_images: bool) -> str:
    """EXPLORE_SYSTEM, plus the image-exploration option when the dataset has images.

    Withheld on a text-only table rather than described and then refused: an option the agent
    cannot use is only a way for it to waste a round trip asking for it.
    """
    return EXPLORE_SYSTEM + (_EXPLORE_IMAGES_BLOCK if has_images else "")


def build_explore_user(
    *,
    query_text: str,
    df: pd.DataFrame,
    id_col: str,
    has_images: bool,
    sample_size: int,
    n_rows: int = PREVIEW_ROWS,
    cell_chars: int = PREVIEW_CELL_CHARS,
) -> str:
    return (
        f"Query:\n{query_text}\n\n"
        f"{schema_block(df, id_col, has_images=has_images)}\n\n"
        f"{preview_block(df, n_rows=n_rows, cell_chars=cell_chars)}\n\n"
        f"The table has {len(df):,} rows; {sample_size} of them will be selected.\n"
        "Run one analysis snippet, or skip it."
    )


def render_explore_result(code: str, output: str, error: str) -> str:
    """The exploration block pasted into the spec prompt.

    A failed snippet is reported rather than hidden: knowing an approach did not work is worth
    something, and silently dropping it would leave the agent believing it had seen the data.
    """
    if error:
        return (
            "YOU RAN THIS ANALYSIS, AND IT FAILED:\n"
            f"```python\n{code}\n```\n"
            f"Error: {error}\n"
            "Write the retrieval plan from the schema and sample rows instead."
        )
    return (
        "YOU RAN THIS ANALYSIS ON THE FULL TABLE:\n"
        f"```python\n{code}\n```\n\n"
        f"Result:\n{output}"
    )


def render_explore_images(
    descriptions: dict[str, str], question: str, missing: list[str]
) -> str:
    """The image-exploration block pasted into the spec prompt.

    The question is repeated above the lines so the agent reads each answer against what was
    asked, rather than as an unprompted aside.
    """
    if not descriptions and not missing:
        return ""
    lines = [f"YOU LOOKED AT {len(descriptions)} OF THE DATASET'S IMAGES:"]
    if question:
        lines.append(f"You asked of each: {question}")
    lines.append("")
    lines.extend(f"  {desc}" if desc else f"  {rid}: [no description]"
                 for rid, desc in descriptions.items())
    if missing:
        lines.append(f"\n  no image file found for: {', '.join(missing)}")
    return "\n".join(lines)


def spec_system(sample_size: int, idx_specification: bool = False) -> str:
    """SPEC_SYSTEM with the subset size filled in, plus preselection when it is enabled.

    The allocation constraint has to name a real number -- "the m values must sum to 10" is a
    rule a model can check itself against, while "must sum to k" is one it can only guess at.
    Preselection is the one case where it cannot be filled in ahead of time: the target is k
    minus a count the agent has not chosen yet, so that rule stays stated in terms of k.
    """
    base = SPEC_SYSTEM + (PRESELECT_BLOCK if idx_specification else "")
    return base.replace("{k}", str(sample_size))


def revise_system(sample_size: int, idx_specification: bool = False) -> str:
    base = REVISE_SYSTEM + (PRESELECT_BLOCK if idx_specification else "")
    return base.replace("{k}", str(sample_size))


REPROMPT = (
    "Could not parse your reply: {detail}. Re-send EXACTLY ONE ```json``` block holding a "
    "single object, with no text outside the block."
)


# Copied verbatim from CostModelAgent._IMAGE_DESCRIBE_SYSTEM
IMAGE_DESCRIBE_SYSTEM = (
    "You are a vision assistant helping a query-planning agent explore a dataset's images. "
    "For each attached image, write ONE concise, factual line describing what it shows — main "
    "object/subject, dominant colors, and any attributes useful for filtering (item type, "
    "composition, visible text). Prefix each line with the image's id. No preamble, no summary."
)


def image_describe_system(question: str | None) -> str:
    """IMAGE_DESCRIBE_SYSTEM, plus a question the describer must answer for every image.

    Appended rather than substituted: the default description is what makes the lines
    comparable to each other and to the feedback report's, and the question is an addition to
    it, not a replacement for it.
    """
    if not question:
        return IMAGE_DESCRIBE_SYSTEM
    return (
        f"{IMAGE_DESCRIBE_SYSTEM} In addition, make sure to address the following in every "
        f"line: {question}"
    )


def schema_block(df: pd.DataFrame, id_col: str, *, has_images: bool) -> str:
    lines = [f"  {c} ({df[c].dtype})" for c in df.columns]
    if has_images:
        lines.append(
            f"  {IMAGE_FIELD} (image) — the row's picture. Valid only in dense_columns; it is "
            "embedded and compared to dense_query like any other field."
        )
    return "Columns:\n" + "\n".join(lines)


def preview_block(
    df: pd.DataFrame, *, n_rows: int = PREVIEW_ROWS, cell_chars: int = PREVIEW_CELL_CHARS
) -> str:
    """The first `n_rows` rows, one `column: value` line each, every cell capped.

    One line per column rather than a single truncated `str(row.to_dict())`: on a wide table
    the dict form spends its whole budget on the first two or three columns and silently drops
    the rest, which is exactly the information needed to choose columns.
    """
    chunks = []
    for pos, (_, row) in enumerate(df.head(n_rows).iterrows(), start=1):
        lines = [f"row {pos}:"]
        for col in df.columns:
            value = row[col]
            rendered = "<NA>" if pd.isna(value) else str(value)
            if len(rendered) > cell_chars:
                rendered = f"{rendered[:cell_chars]}… (+{len(rendered) - cell_chars:,} chars)"
            lines.append(f"  {col}: {rendered}")
        chunks.append("\n".join(lines))
    return (
        f"First {len(chunks)} rows (each value truncated to {cell_chars} chars):\n"
        + "\n".join(chunks)
    )


def build_spec_user(
    *,
    query_text: str,
    df: pd.DataFrame,
    id_col: str,
    has_images: bool,
    sample_size: int,
    n_rows: int = PREVIEW_ROWS,
    cell_chars: int = PREVIEW_CELL_CHARS,
    exploration: str = "",
) -> str:
    blocks = [
        f"Query:\n{query_text}",
        schema_block(df, id_col, has_images=has_images),
        preview_block(df, n_rows=n_rows, cell_chars=cell_chars),
    ]
    if exploration:
        blocks.append(exploration)
    blocks.append(
        f"The table has {len(df):,} rows; {sample_size} of them will be selected.\n"
        "Return the JSON plan."
    )
    return "\n\n".join(blocks)


def build_revise_user(report: str, *, attempt: int, max_attempts: int) -> str:
    remaining = max_attempts - attempt
    tail = (
        "This is your last chance to change the spec; after this the current rows are used."
        if remaining <= 0
        else f"You may revise {remaining} more time(s) after this one."
    )
    return f"{report}\n\nAccept these rows, or return a revised spec. {tail}"