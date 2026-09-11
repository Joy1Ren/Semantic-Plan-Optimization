#!/usr/bin/env python3
"""
Compare our cost-model agent ("pz") against DocETL MOAR on SemBench ecomm at sf_500, one row
per (query, optimization-subset) variation.  Run:  PYTHONPATH=. python ecomm_moar_pz_result.py

Columns, left to right:
  1. During optimization -- every plan pz executed on the optimization subset, against MOAR's
     search frontier.  Both sides are subset-evaluated, so these are per-plan subset costs with
     no search overhead folded in.
  2. Full dataset -- only the plan each system finally selected, re-run on the full dataset.
  3. Full dataset, with optimization cost -- the same points shifted right by what finding them
     cost (pz: total_opt_cost, MOAR: total_search_cost).

Data sources, all under results/SemBench/{use_case}/sf_{scale_factor}/:
  MOAR/outputs/{moar_run}/pareto_frontier.json      flat [{cost, accuracy, ...}] -- already the frontier
  MOAR/outputs/{moar_run}/experiment_summary.json   total_search_cost
  MOAR/outputs/{moar_run}/full_dataset/full_dataset_results.json
                                                    {plan: {full_cost, full_score, status}} -- note
                                                    SemBench names these full_cost/full_score, where
                                                    CUAD uses full15_cost/full15_avg_f1
  {agent_type}/metrics/Q{qid}_{opt_run}_results.csv every plan pz executed during search
  {agent_type}/metrics/final_eval.json              keyed "{qid}_{opt_run}", run1/run2 averaged
"""

import csv
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from agent_cost_model.experiments.config import results_prefix
from agent_cost_model.paths import RESULTS_DIR

csv.field_size_limit(10 * 1024 * 1024)  # plan_str/op_samples blobs exceed the 128 KiB default

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
USE_CASE = "ecomm"
SCALE_FACTOR = 500

# One row per (query, subset-sampling) variation.  agent_type is the pz runner directory, which
# is what encodes random vs. agentic; moar_run is the matching MOAR search under MOAR/outputs/.
ROWS = [
    {"label": "Q5 · random subset",  "query_id": 5, "agent_type": "execute_optimizer_random",  "moar_run": "ecomm_q5_random"},
    {"label": "Q7 · random subset",  "query_id": 7, "agent_type": "execute_optimizer_random",  "moar_run": "ecomm_q7_random"},
    {"label": "Q7 · agentic subset", "query_id": 7, "agent_type": "execute_optimizer_agentic", "moar_run": "ecomm_q7_agentic"},
]

OUTPUT_DIR = RESULTS_DIR / "_analysis" / "ecomm_sf500_moar_pz"
# ─────────────────────────────────────────────────────────────────────────────

MOAR_COLOR = "#2ca02c"
AGENT_RUN_COLORS = ["#9467bd", "#1f77b4", "#8c564b", "#e377c2", "#7f7f7f"]
MOAR_MARKER = "^"
AGENT_FINAL_MARKER = "*"
AGENT_EXPLORED_MARKER = "o"
LEGEND_GRAY = "#555555"

COLUMN_TITLES = [
    "During optimization\n(optimization subset)",
    "Full dataset\n(execution cost only)",
    "Full dataset\n(+ optimization cost)",
]


def _agent_run_color(opt_run):
    return AGENT_RUN_COLORS[(opt_run - 1) % len(AGENT_RUN_COLORS)]


def _moar_dir(moar_run):
    prefix = results_prefix("SemBench", "MOAR", use_case=USE_CASE, scale_factor=SCALE_FACTOR)
    return prefix / "outputs" / moar_run


def _pz_metrics_dir(agent_type):
    prefix = results_prefix("SemBench", agent_type, use_case=USE_CASE, scale_factor=SCALE_FACTOR)
    return prefix / "metrics"


# ── Data loading: DocETL MOAR ─────────────────────────────────────────────────

def load_moar_search(moar_run):
    """Frontier as evaluated on the optimization subset.  Every entry in this file is already
    non-dominated, so the scattered points and the connecting line are the same list."""
    path = _moar_dir(moar_run) / "pareto_frontier.json"
    if not path.exists():
        print(f"[moar] search frontier not found: {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    points = [{"cost": e["cost"], "quality": e["accuracy"]} for e in raw
              if e.get("cost") is not None and e.get("accuracy") is not None]
    return sorted(points, key=lambda p: p["cost"])


def load_moar_full(moar_run):
    """The same frontier plans re-scored on the full dataset."""
    path = _moar_dir(moar_run) / "full_dataset" / "full_dataset_results.json"
    if not path.exists():
        print(f"[moar] full-dataset results not found: {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    points = [{"cost": e["full_cost"], "quality": e["full_score"]} for e in raw.values()
              if e.get("status") == "ok" and e.get("full_cost") is not None and e.get("full_score") is not None]
    return sorted(points, key=lambda p: p["cost"])


def load_moar_search_cost(moar_run):
    path = _moar_dir(moar_run) / "experiment_summary.json"
    if not path.exists():
        print(f"[moar] experiment summary not found: {path}")
        return 0.0
    with open(path) as f:
        return json.load(f).get("total_search_cost") or 0.0


def pareto_frontier(points):
    """Non-dominated subset, maximizing quality and minimizing cost."""
    frontier, best = [], float("-inf")
    for p in sorted(points, key=lambda p: p["cost"]):
        if p["quality"] > best:
            frontier.append(p)
            best = p["quality"]
    return frontier


# ── Data loading: cost-model agent ("pz") ─────────────────────────────────────

def _avg_runs(entry):
    """final_eval.json carries run1/run2 repeats of the same selected plan; average them."""
    runs = [entry[k] for k in ("run1", "run2") if k in entry]
    if not runs:
        return entry.get("cost"), entry.get("quality")
    cost = sum(r.get("cost") or 0.0 for r in runs) / len(runs)
    qualities = [r.get("quality") for r in runs if r.get("quality") is not None]
    return cost, (sum(qualities) / len(qualities) if qualities else None)


def _pz_entries(agent_type, query_id):
    """{opt_run: entry} from final_eval.json, for this query only."""
    path = _pz_metrics_dir(agent_type) / "final_eval.json"
    if not path.exists():
        print(f"[pz] metrics not found: {path}")
        return {}
    with open(path) as f:
        raw = json.load(f)
    entries = {}
    for key, entry in raw.items():
        parts = key.split("_")
        if len(parts) == 2 and parts[0] == str(query_id):
            entries[int(parts[1])] = entry
    return entries


def load_pz_final(agent_type, query_id):
    """Full-dataset numbers for the plan pz actually selected, one point per opt_run."""
    points = []
    for opt_run, entry in sorted(_pz_entries(agent_type, query_id).items()):
        cost, quality = _avg_runs(entry)
        if quality is None:
            continue
        points.append({
            "opt_run": opt_run,
            "cost": cost or 0.0,
            "quality": quality,
            "opt_cost": entry.get("total_opt_cost") or 0.0,
        })
    return points


def load_pz_explored(agent_type, query_id):
    """Every plan pz executed on the optimization subset, from the per-opt-run results CSVs."""
    metrics_dir = _pz_metrics_dir(agent_type)
    pattern = re.compile(rf"^Q{query_id}_(\d+)_results\.csv$")
    points = []
    for path in sorted(metrics_dir.glob(f"Q{query_id}_*_results.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        opt_run = int(m.group(1))
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cost, quality = float(row["cost_usd"]), float(row["quality"])
                except (KeyError, TypeError, ValueError):
                    continue
                points.append({
                    "opt_run": opt_run,
                    "cost": cost,
                    "quality": quality,
                    "final_selected": str(row.get("final_selected")).strip().lower() == "true",
                })
    return points


def load_pz_models(agent_type, query_id):
    for entry in _pz_entries(agent_type, query_id).values():
        return {
            "agent": (entry.get("core_agent") or {}).get("model"),
            "oracle": (entry.get("oracle") or {}).get("oracle_model"),
        }
    return {}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _scatter_moar(ax, points, x_offset=0.0, connect=True, linestyle="-"):
    if not points:
        return
    ax.scatter([p["cost"] + x_offset for p in points], [p["quality"] for p in points],
               marker=MOAR_MARKER, s=95, color=MOAR_COLOR, zorder=3)
    line = pareto_frontier(points) if connect else points
    if len(line) > 1:
        ax.plot([p["cost"] + x_offset for p in line], [p["quality"] for p in line],
                color=MOAR_COLOR, linewidth=1.5, linestyle=linestyle, zorder=2)


def _scatter_pz_final(ax, points, with_opt_cost):
    for p in points:
        x = p["cost"] + (p["opt_cost"] if with_opt_cost else 0.0)
        ax.scatter(x, p["quality"], marker=AGENT_FINAL_MARKER, s=230,
                   color=_agent_run_color(p["opt_run"]), zorder=5)


def _scatter_pz_explored(ax, points):
    for p in points:
        final = p["final_selected"]
        ax.scatter(p["cost"], p["quality"],
                   marker=AGENT_FINAL_MARKER if final else AGENT_EXPLORED_MARKER,
                   s=230 if final else 55,
                   color=_agent_run_color(p["opt_run"]),
                   alpha=1.0 if final else 0.55,
                   zorder=5 if final else 3.5)


def _limits(values, pad_frac=0.12, clamp=None):
    """Padded (lo, hi) over values, widened when every value coincides so the points do not
    land on the axis spine."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = span * pad_frac if span else max(abs(hi) * 0.1, 0.01)
    lo, hi = lo - pad, hi + pad
    if clamp:
        lo, hi = max(clamp[0], lo), min(clamp[1], hi)
    return (lo, hi)


def _col_values(d, column):
    """The (xs, ys) a given column plots for one row's data."""
    if column == 0:
        pts = d["explored"] + d["moar_search"]
        return [p["cost"] for p in pts], [p["quality"] for p in pts]
    if column == 1:
        pts = d["final"] + d["moar_full"]
        return [p["cost"] for p in pts], [p["quality"] for p in pts]
    xs = ([p["cost"] + p["opt_cost"] for p in d["final"]]
          + [p["cost"] + d["search_cost"] for p in d["moar_full"]])
    ys = [p["quality"] for p in d["final"] + d["moar_full"]]
    return xs, ys


def build_figure():
    data = []
    for row in ROWS:
        data.append({
            "row": row,
            "explored": load_pz_explored(row["agent_type"], row["query_id"]),
            "final": load_pz_final(row["agent_type"], row["query_id"]),
            "moar_search": load_moar_search(row["moar_run"]),
            "moar_full": load_moar_full(row["moar_run"]),
            "search_cost": load_moar_search_cost(row["moar_run"]),
        })

    # One quality range per column, shared down every row, so the same column can be read across
    # query/subset variations.  Columns 2 and 3 hold the same plans on two cost bases, so they
    # land on the same range anyway.
    col_ylim = [_limits([v for d in data for v in _col_values(d, c)[1]], clamp=(0.0, 1.02))
                for c in range(3)]

    # The two Q7 rows differ only in how the optimization subset was sampled, so pin their x
    # ranges together per column -- random vs. agentic is then a like-for-like read across rows.
    q7 = [d for d in data if d["row"]["query_id"] == 7]
    q7_xlim = [_limits([v for d in q7 for v in _col_values(d, c)[0]]) for c in range(3)]

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(17, 4.6 * len(ROWS)))
    fig.suptitle("SemBench ecomm (sf_500): cost-model agent (pz) vs. DocETL MOAR",
                 fontsize=15, fontweight="bold", y=0.985)

    models = load_pz_models(ROWS[0]["agent_type"], ROWS[0]["query_id"])
    subtitle = ("Each row is one query × optimization-subset variation; each column re-plots that "
                "row's plans on a different cost basis")
    if models.get("agent"):
        subtitle += f"    ·    pz agent={models['agent']}  oracle={models['oracle']}"
    fig.text(0.5, 0.958, subtitle, ha="center", fontsize=9.5, color="#555555")

    opt_runs_seen = set()

    for r, d in enumerate(data):
        row = d["row"]
        opt_runs_seen.update(p["opt_run"] for p in d["explored"] + d["final"])

        # Column 1 -- optimization subset, no search overhead on either side.
        ax = axes[r][0]
        _scatter_moar(ax, d["moar_search"], connect=True, linestyle="--")
        _scatter_pz_explored(ax, d["explored"])
        ax.set_title(f"pz: {len(d['explored'])} plans explored   ·   "
                     f"MOAR: {len(d['moar_search'])} frontier plans",
                     fontsize=9, color="#555555", pad=6)

        # Column 2 -- full dataset, execution cost only.
        ax = axes[r][1]
        _scatter_moar(ax, d["moar_full"], connect=True, linestyle="-")
        _scatter_pz_final(ax, d["final"], with_opt_cost=False)
        ax.set_title("selected plans, re-run on the full dataset", fontsize=9, color="#555555", pad=6)

        # Column 3 -- same points, plus what the search cost to find them.
        ax = axes[r][2]
        _scatter_moar(ax, d["moar_full"], x_offset=d["search_cost"], connect=True, linestyle="-")
        _scatter_pz_final(ax, d["final"], with_opt_cost=True)
        pz_opt = ", ".join(f"\\${p['opt_cost']:.2f}" for p in d["final"]) or "n/a"
        ax.set_title(f"opt cost added — pz: {pz_opt}   ·   MOAR: \\${d['search_cost']:.2f}",
                     fontsize=9, color="#555555", pad=6)

        for c in range(3):
            ax = axes[r][c]
            if col_ylim[c]:
                ax.set_ylim(*col_ylim[c])
            if row["query_id"] == 7 and q7_xlim[c]:
                ax.set_xlim(*q7_xlim[c])
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Cost ($)", fontsize=9)
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel("Quality", fontsize=9)
                ax.text(-0.22, 0.5, row["label"], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold")

    for c, title in enumerate(COLUMN_TITLES):
        axes[0][c].annotate(title, xy=(0.5, 1.14), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=12, fontweight="bold")

    role_handles = [
        mlines.Line2D([], [], color=MOAR_COLOR, marker=MOAR_MARKER, linestyle="--", markersize=9,
                      label="MOAR — search frontier (subset)"),
        mlines.Line2D([], [], color=MOAR_COLOR, marker=MOAR_MARKER, linestyle="-", markersize=9,
                      label="MOAR — full-dataset frontier"),
        mlines.Line2D([], [], color=LEGEND_GRAY, marker=AGENT_FINAL_MARKER, linestyle="None",
                      markersize=16, label="pz — final selected plan"),
        mlines.Line2D([], [], color=LEGEND_GRAY, marker=AGENT_EXPLORED_MARKER, linestyle="None",
                      markersize=7, alpha=0.55, label="pz — explored plan, during search"),
    ]
    fig.legend(handles=role_handles, loc="lower center", ncol=len(role_handles),
               bbox_to_anchor=(0.5, -0.012), fontsize=10, framealpha=0.95)

    color_handles = [mpatches.Patch(color=MOAR_COLOR, label="MOAR")]
    color_handles += [mpatches.Patch(color=_agent_run_color(n), label=f"pz (opt run {n})")
                      for n in sorted(opt_runs_seen)]
    fig.legend(handles=color_handles, loc="upper right", bbox_to_anchor=(0.997, 0.945),
               fontsize=9, framealpha=0.95, title="Color key", title_fontsize=9)

    fig.tight_layout(rect=[0.02, 0.035, 1, 0.945])
    return fig

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = OUTPUT_DIR / "ecomm_sf500_moar_vs_pz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved:\n  {out}")


if __name__ == "__main__":
    main()
