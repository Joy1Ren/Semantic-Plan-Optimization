"""The between-rounds report: what the current spec actually retrieved.

The agent needs three things to judge its own spec: the top rows in enough detail to tell
whether they are the right rows, the score distribution to tell whether the components are
discriminating at all, and diagnostics naming the terms that matched nothing.

Only the raw cell content is budgeted. The formula, the distribution table, and the per-row
score lines are small and always shown in full; the cell values are the part that would
otherwise run to millions of characters on a wide table.
"""
from __future__ import annotations

import pathlib
from typing import Any, Sequence

import pandas as pd

from agent_cost_model.sampling.constants import (
    FEEDBACK_CHAR_CEILING,
    IMAGE_FIELD,
    MAX_CELL_CHARS,
)
from agent_cost_model.sampling.prompts import SCORING_EXPLAINER
from agent_cost_model.sampling.scoring import ScoredPool
from agent_cost_model.sampling.spec import SamplingSpec
from agent_cost_model.sampling.text import truncate_middle

__all__ = ["allocate_cell_budgets", "describe_images", "build_stratum_section", "build_feedback"]

_PCTS = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def allocate_cell_budgets(
    lengths: Sequence[int], total: int, cap: int = MAX_CELL_CHARS
) -> list[int]:
    """Split `total` characters across cells of the given lengths.

    One pass, no iteration: every cell that fits inside an equal share is shown in full, and
    the surplus they leave behind is divided evenly among the cells that do not fit. It is
    fine for a long cell not to use its whole allowance -- the point is a predictable ceiling,
    not an exactly-filled budget.
    """
    n = len(lengths)
    if n == 0:
        return []
    if total <= 0:
        return [0] * n
    share = total // n
    short = [i for i, length in enumerate(lengths) if length <= share]
    long_ = [i for i, length in enumerate(lengths) if length > share]
    out = [0] * n
    # `cap` is an absolute per-cell ceiling, so it binds here too: a single enormous cell can
    # fit inside its share on a narrow table and would otherwise swallow the whole budget.
    for i in short:
        out[i] = min(lengths[i], cap)
    if long_:
        surplus = total - sum(out[i] for i in short)
        per_long = max(0, surplus // len(long_))
        for i in long_:
            out[i] = min(lengths[i], per_long, cap)
    return out


def describe_images(
    ids: Sequence[str],
    image_dir: str | pathlib.Path,
    *,
    describer: Any,
    system: str,
) -> tuple[dict[str, str], float]:
    """One batched vision call describing each row's image. Returns (by_id, cost_usd).

    The sampler's own describer, used by the feedback report and by the pre-spec exploration.
    `CostModelAgent._describe_images` is a separate implementation for the optimizer's
    explore_images tool -- same idea, no shared code, deliberately free to diverge. Never
    raises: a missing description is not a reason to abandon a sampling round.
    """
    import base64

    directory = pathlib.Path(image_dir)
    present = [(rid, directory / f"{rid}.jpg") for rid in ids]
    present = [(rid, p) for rid, p in present if p.exists()]
    if not present or describer is None:
        return {}, 0.0

    content: list[dict] = [{
        "type": "text",
        "text": (
            f"Describe each of these {len(present)} image(s), one short line each, "
            "prefixed with its id:"
        ),
    }]
    for rid, path in present:
        try:
            b64 = base64.b64encode(path.read_bytes()).decode()
        except Exception:
            continue
        content.append({"type": "text", "text": f"id {rid}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        result = describer.generate(system, [{"role": "user", "content": content}])
    except Exception as e:
        return {rid: f"[image description unavailable: {type(e).__name__}: {e}]" for rid, _ in present}, 0.0

    text, meta = result, {}
    if isinstance(result, tuple):
        text = result[0] if result else ""
        if len(result) >= 3 and isinstance(result[2], dict):
            meta = result[2]
    cost = float((meta or {}).get("cost_usd", 0.0) or 0.0)

    # The describer returns free text, one line per image, each naming its id. Attribute each
    # line to the id it mentions rather than trusting the order.
    by_id: dict[str, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for rid, _ in present:
            if rid in by_id:
                continue
            if stripped.startswith(rid) or f"id {rid}" in stripped[:40]:
                by_id[rid] = stripped
                break
    for rid, _ in present:
        by_id.setdefault(rid, "")
    return by_id, cost


def _fmt_pcts(values: dict[str, float]) -> str:
    return "  ".join(f"{values[f'p{p}']:.3f}" for p in _PCTS)


def build_stratum_section(
    *,
    spec: SamplingSpec,
    pool: ScoredPool,
    df: pd.DataFrame,
    id_col: str,
    filter_info: dict,
    show_ids: Sequence[str],
    budget: int,
    max_cell: int = MAX_CELL_CHARS,
    image_descriptions: dict[str, str] | None = None,
    warnings: Sequence[str] = (),
    header: str = "",
    seen: dict[str, int] | None = None,
    own_top: Sequence[str] = (),
    index: int = 1,
) -> str:
    """One stratum's diagnostics plus the rows it actually contributed to the subset.

    `show_ids` is what this stratum added -- these are real rows the optimizer will receive, so
    they are rendered in full. Because a stratum never takes a row an earlier one already has,
    no row is ever printed twice across sections.

    `own_top` is what this stratum's spec ranked highest before that exclusion. Where the two
    differ, the difference is reported as a note naming the pre-empted rows and which stratum
    took them: that is how the agent learns its targeting worked even though the rows went
    elsewhere, and that two strata are chasing the same thing.
    """
    seen = {} if seen is None else seen
    rank_of = {rid: i for i, rid in enumerate(pool.ids)}
    order = [rank_of[rid] for rid in show_ids if rid in rank_of]
    top_ids = [pool.ids[i] for i in order]

    parts: list[str] = []
    if header:
        parts.append(header)
    parts.append(f"alpha = {spec.alpha:.2f}   |   live components: {', '.join(pool.live) or 'none'}")

    applied = filter_info.get("expr")
    if filter_info.get("compile_error"):
        parts.append(f"FILTER {applied!r} FAILED TO COMPILE: {filter_info['compile_error']} — ignored.")
    elif applied:
        parts.append(
            f"FILTER {applied!r} matched {filter_info.get('n_pool', 0):,} of "
            f"{filter_info.get('n_total', 0):,} rows."
        )
        if filter_info.get("underfilled"):
            parts.append(
                "  That is fewer than the subset size, so the remaining slots were filled with "
                "random rows from outside the filter."
            )
    else:
        parts.append(f"FILTER none — all {filter_info.get('n_total', 0):,} rows scored.")

    parts.append("")
    parts.append(f"SCORE DISTRIBUTION over the {len(pool):,} scored rows")
    parts.append("               " + "  ".join(f" p{p}" for p in _PCTS))
    from agent_cost_model.sampling.scoring import percentiles

    parts.append("  bm25_norm    " + _fmt_pcts(percentiles(pool.bm25_norm)))
    parts.append("  dense_norm   " + _fmt_pcts(percentiles(pool.dense_norm)))
    parts.append("  score        " + _fmt_pcts(percentiles(pool.total)))
    if pool.bm25_raw.size:
        parts.append(
            f"  raw BM25   range [{pool.bm25_raw.min():.4f}, {pool.bm25_raw.max():.4f}]"
        )
    if pool.dense_raw.size:
        parts.append(
            f"  raw cosine range [{pool.dense_raw.min():.4f}, {pool.dense_raw.max():.4f}]"
            "   <- a narrow range means the dense query is not discriminating; the normalized"
            " spread above is then manufactured from noise"
        )
    histogram: dict[int, int] = {}
    for h in pool.hits.astype(int).tolist():
        histogram[h] = histogram.get(h, 0) + 1
    parts.append(
        "  keyphrase hits: "
        + " | ".join(f"{k} -> {v:,} rows" for k, v in sorted(histogram.items()))
    )

    if pool.dead_terms or pool.zero_hit_keyphrases or warnings:
        parts.append("")
        parts.append("DIAGNOSTICS")
        if pool.dead_terms:
            parts.append(f"  bm25_query terms matching zero rows: {pool.dead_terms}")
        if pool.zero_hit_keyphrases:
            parts.append(f"  keyphrases matching zero rows: {pool.zero_hit_keyphrases}")
        for warning in warnings:
            parts.append(f"  spec repaired: {warning}")

    # Everything above is fixed-size; the budget governs only the cell content below.
    parts.append("")
    parts.append(f"THE {len(top_ids)} ROWS THIS STRATUM CONTRIBUTED TO THE SUBSET")
    mine = set(top_ids)
    preempted = [rid for rid in own_top if rid not in mine]
    if preempted:
        by_rank = {rid: i for i, rid in enumerate(pool.ids)}
        detail = ", ".join(
            f"{id_col}={rid} (score {pool.total[by_rank[rid]]:.4f}, taken by stratum "
            f"{seen.get(rid, '?')})"
            for rid in preempted if rid in by_rank
        )
        parts.append(
            f"  Note: {len(preempted)} of this stratum's own top {len(own_top)} were already "
            f"selected by an earlier stratum — {detail}. Your targeting found them; they are "
            "in the subset under that stratum, and this one drew further down its ranking "
            "instead. Heavy overlap means two strata are chasing the same rows."
        )

    columns = [c for c in df.columns if c != id_col]
    row_budget = max(0, budget) // max(1, len(top_ids))

    for rank, (i, rid) in enumerate(zip(order, top_ids), start=1):
        hit = pool.hit_names[i]
        missed = [p for p in spec.keyphrases if p not in hit]
        parts.append("")
        parts.append(f"[{rank}/{len(top_ids)}] {id_col}={rid}   score={pool.total[i]:.4f}")
        parts.append(
            f"  sparse={pool.sparse[i]:.4f}  (bm25_norm {pool.bm25_norm[i]:.4f} raw "
            f"{pool.bm25_raw[i]:.4f} + {int(pool.hits[i])} keyphrase hit(s))"
        )
        if spec.keyphrases:
            parts.append(f"    hit: {hit or 'none'}    missed: {missed or 'none'}")
        parts.append(
            f"  dense={pool.dense_norm[i]:.4f}  (raw cosine {pool.dense_raw[i]:.4f})"
        )

        row = df.loc[df[id_col] == rid]
        if row.empty:
            continue
        series = row.iloc[0]
        rendered = []
        for col in columns:
            value = series[col]
            rendered.append((col, "<NA>" if pd.isna(value) else str(value)))
        allowances = allocate_cell_budgets([len(v) for _, v in rendered], row_budget, max_cell)
        for (col, value), allowance in zip(rendered, allowances):
            parts.append(f"  {col}: {truncate_middle(value, allowance)}")
        if image_descriptions and rid in image_descriptions:
            parts.append(f"  {IMAGE_FIELD}: {image_descriptions[rid] or '(no description)'}")

    return "\n".join(parts)


def build_feedback(
    *,
    sections: Sequence[tuple[str, SamplingSpec, ScoredPool, dict, Sequence[str]]],
    df: pd.DataFrame,
    id_col: str,
    chosen: Sequence[str],
    own_tops: Sequence[Sequence[str]] = (),
    budget: int = FEEDBACK_CHAR_CEILING,
    max_cell: int = MAX_CELL_CHARS,
    image_descriptions: dict[str, str] | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """The whole round's report: the formula, then one section per stratum.

    The cell budget is split across strata by how many rows each contributed, so a stratum
    holding 8 of 10 rows gets most of the space rather than every stratum getting an equal
    share regardless of size.
    """
    parts: list[str] = [SCORING_EXPLAINER, ""]
    if len(sections) > 1:
        parts.append(
            f"This plan is STRATIFIED into {len(sections)} strata, together contributing the "
            f"{len(chosen)} rows of the subset. Each section shows the rows THAT stratum "
            "contributed, so the sections together are exactly the subset. Where a stratum's "
            "own top rows were already taken by an earlier one, that is noted rather than "
            "repeated."
        )
        parts.append("")
    total_rows = max(1, sum(len(ids) for _, _, _, _, ids in sections))
    # Shared across sections so a row shown under stratum 1 is a one-line reference under
    # stratum 2 instead of a second full copy of the same cells.
    # Which stratum selected each row, so a pre-empted row can be named with its new home.
    # Filled ahead of the loop: stratum 1's note may reference a row stratum 2 ends up with.
    seen: dict[str, int] = {
        rid: i for i, (_, _, _, _, ids) in enumerate(sections, 1) for rid in ids
    }
    own_tops = list(own_tops) or [ids for _, _, _, _, ids in sections]
    for i, (label, spec, pool, filter_info, show_ids) in enumerate(sections, 1):
        header = ""
        if len(sections) > 1:
            name = f" — {label}" if label else ""
            header = (
                f"{'=' * 70}\nSTRATUM {i}/{len(sections)}{name}: allocated "
                f"{len(show_ids)} of {len(chosen)} rows\n{'=' * 70}"
            )
        share = int(budget * len(show_ids) / total_rows)
        parts.append(
            build_stratum_section(
                spec=spec,
                pool=pool,
                df=df,
                id_col=id_col,
                filter_info=filter_info,
                show_ids=show_ids,
                budget=share,
                max_cell=max_cell,
                image_descriptions=image_descriptions,
                # Repair warnings belong to the plan as a whole, so they are printed once
                # under the first stratum rather than repeated under every one.
                warnings=warnings if i == 1 else (),
                header=header,
                seen=seen,
                own_top=own_tops[i - 1],
                index=i,
            )
        )
        parts.append("")
    return "\n".join(parts).rstrip()
