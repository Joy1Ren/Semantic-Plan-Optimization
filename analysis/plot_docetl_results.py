#!/usr/bin/env python3
"""
Plot our cost-model agent's CUAD plans ("pz") against DocETL MOAR's pareto frontier.
Edit the CONFIGURATION block below, then run:  python plot_docetl_results.py

Produces two comparisons, each as a with/without-opt-cost pair:
  - Full dataset:      the search's frontier plans, re-run on the full 15-doc dataset
                        (results/CUAD/MOAR/outputs/{full_run}/) vs. our agent's final selected
                        plan, also re-run on the full dataset ({runner}/metrics/final_eval.json).
  - During optimization: the search's own frontier on the optimization subset
                        (results/CUAD/MOAR/outputs/{search_run}/pareto_frontier.json) vs. EVERY plan our
                        agent executed during search, not just the one it finally picked
                        ({runner}/metrics/Q{qid}_{opt_run}_results.csv).

Data sources:
  - DocETL MOAR search frontier: results/CUAD/MOAR/outputs/{search_run}/pareto_frontier.json (flat list of
    {cost, accuracy, ...}) + experiment_summary.json's total_search_cost.
  - DocETL MOAR full-dataset results: results/CUAD/MOAR/outputs/{full_run}/full15_results.json -- the
    search frontier's plans re-evaluated on the full dataset (falls back to the older
    pareto_frontier*.json schema if present instead). We compute the non-dominated subset
    ourselves for the connecting line, since full15_results.json has no precomputed frontier.
    search cost is still attributed to {search_run} (full_run is just a verification pass,
    not a fresh search).
  - Cost-model agent ("pz"), final plan on full dataset: agent_cost_model/results/metrics/
    CUAD/{runner}/metrics/final_eval.json -- written by
    CostModelAgent._run_final_evaluation.
  - Cost-model agent ("pz"), every plan tried during search: agent_cost_model/results/metrics/
    CUAD/{runner}/metrics/Q{query_id}_{opt_run}_results.csv -- one row per
    plan the agent executed on the optimization subset, written by CostModelAgent._save_results_df.
    "final_selected" marks which row was the one actually chosen.
  CUAD has no real scale-factor axis -- sf_15 is just the 15-document full-dataset size, passed
  as --scale-factor to run_opt.py.
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

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# DocETL MOAR run pairs to compare: search_run is the optimization run (small subset,
# produces pareto_frontier.json + experiment_summary.json under docetl/outputs/{search_run}/);
# full_run is that search's frontier plans re-evaluated on the full dataset, under
# docetl/outputs/{full_run}/. Add more pairs here as additional runs are produced.
RUN_PAIRS = [
    {"search_run": "cuad_small_smoke_api", "full_run": "cuad_small_smoke_api/full15"},
]

# Our cost-model agent's variant, and the CUAD query id (benchmark.yaml only
# defines query "1" for CUAD).
USE_CASE = "cuad"
AGENT_TYPE = "execute_oracle_sampler"  # matches run_opt.py's AGENT_TYPE (no "_agent" suffix -- that's agent_dir, a different path segment used by metrics/raw_results, not final_answer_path)
AGENT_DIR = AGENT_TYPE                 # runner name; the `_agent` suffix is gone
QUERY_ID = 1
SCALE_FACTOR = 15  # CUAD has no real scale-factor axis; 15 is the full-dataset doc count, passed as --scale-factor to run_opt.py

# MOAR's own outputs now live inside this repo, under its runner prefix -- run_moar.py is
# pointed here via --output_dir, so nothing is written into the docetl checkout.
DOCETL_OUTPUTS_DIR = results_prefix("CUAD", "MOAR") / "outputs"
PZ_PREFIX = results_prefix("CUAD", AGENT_TYPE)
PZ_METRICS_DIR = PZ_PREFIX / "metrics"
# Same file plot_results.py's load_agent() reads for SemBench, written by
# CostModelAgent._run_final_evaluation -- carries cost/quality/agent_model/oracle_model.
PZ_METRICS_PATH = PZ_METRICS_DIR / "final_eval.json"
OUTPUT_DIR = RESULTS_DIR / "_analysis" / "cuad_docetl"
# ─────────────────────────────────────────────────────────────────────────────

# One color per system everywhere, one marker per role everywhere -- consistent across all
# three subplots regardless of dataset scale. pz is further split by opt_run (a separate,
# independent invocation of the whole agent search) since different opt_runs are not
# comparable trials of the same search.
MOAR_COLOR = "#2ca02c"      # green -- every DocETL MOAR series
# opt_run 1 -> purple, opt_run 2 -> blue, further opt_runs (if any) cycle through the rest.
AGENT_RUN_COLORS = ["#9467bd", "#1f77b4", "#8c564b", "#e377c2", "#7f7f7f"]
MOAR_MARKER = "^"        # triangle -- every MOAR plan
AGENT_FINAL_MARKER = "*"    # star -- pz's selected plan (full dataset or during search)
AGENT_EXPLORED_MARKER = "o"  # small circle -- pz plans explored but not selected (during search)
AGENT_LEGEND_COLOR = "#555555"  # neutral gray for the shape-only legend entries (actual color varies by opt_run)

DOCETL_FULL_DISPLAY_NAME = "DocETL MOAR -- full-dataset frontier"
DOCETL_SEARCH_DISPLAY_NAME = "DocETL MOAR -- search frontier"
PZ_FINAL_DISPLAY_NAME = "pz -- final selected plan"
PZ_EXPLORED_DISPLAY_NAME = "pz -- explored plan, during search"


def _agent_run_color(opt_run):
    return AGENT_RUN_COLORS[(opt_run - 1) % len(AGENT_RUN_COLORS)]


# ── Data loading: DocETL ──────────────────────────────────────────────────────

def _entries_to_points(entries):
    points = []
    for entry in entries:
        cost, quality = entry.get("cost"), entry.get("accuracy")
        if cost is None or quality is None:
            continue
        points.append({"cost": cost, "quality": quality})
    points.sort(key=lambda p: p["cost"])
    return points


def _pareto_frontier(points):
    """Non-dominated subset (maximize quality, minimize cost): sorted by cost ascending,
    keeping only points whose quality beats every cheaper point's quality so far."""
    frontier = []
    best_quality = float("-inf")
    for p in sorted(points, key=lambda p: p["cost"]):
        if p["quality"] > best_quality:
            frontier.append(p)
            best_quality = p["quality"]
    return frontier


def load_docetl_search_frontier(search_run):
    """
    Parse docetl/outputs/{search_run}/pareto_frontier.json -- the search's own frontier,
    evaluated on the optimization subset. Flat list of {cost, accuracy, ...}; every entry in
    this file is already on the frontier, so "points" (scattered) and "frontier" (connected
    with a line) are the same list.
    """
    path = DOCETL_OUTPUTS_DIR / search_run / "pareto_frontier.json"
    if not path.exists():
        print(f"[docetl] search frontier not found: {path}")
        return {"points": [], "frontier": []}
    with open(path) as f:
        raw = json.load(f)
    points = _entries_to_points(raw)
    return {"points": points, "frontier": points}


def load_docetl_full_frontier(full_run):
    """
    The SAME plans that made up the search frontier, now re-evaluated on the full dataset.
    "points" (scattered, all of them) should match load_docetl_search_frontier's count --
    these are the identical plans, just re-scored on more data. "frontier" (connected with a
    line) is the non-dominated subset: a plan that was non-dominated on the small optimization
    subset can be dominated once evaluated on the full dataset, so it can be smaller than
    "points".

    Tries two schemas, in order:
      1. full15_results.json: {plan_name: {full15_cost, full15_avg_f1, status, ...}, ...} --
         current format. We compute the frontier ourselves (no precomputed one in this file).
      2. pareto_frontier*.json: {"all_points": [...], "frontier_points": [...]} -- older format,
         frontier precomputed by docetl itself.
    """
    run_dir = DOCETL_OUTPUTS_DIR / full_run

    results_path = run_dir / "full15_results.json"
    if results_path.exists():
        with open(results_path) as f:
            raw = json.load(f)
        points = []
        for entry in raw.values():
            if entry.get("status") != "ok":
                continue
            cost, quality = entry.get("full15_cost"), entry.get("full15_avg_f1")
            if cost is None or quality is None:
                continue
            points.append({"cost": cost, "quality": quality})
        points.sort(key=lambda p: p["cost"])
        return {"points": points, "frontier": _pareto_frontier(points)}

    path = run_dir / "pareto_frontier.json"
    if not path.exists():
        candidates = sorted(run_dir.glob("pareto_frontier*.json"))
        path = candidates[0] if candidates else None
    if path is None or not path.exists():
        print(f"[docetl] full-dataset results not found under {run_dir}")
        return {"points": [], "frontier": []}
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        points = _entries_to_points(raw.get("all_points", raw.get("frontier_points", [])))
        frontier = _entries_to_points(raw.get("frontier_points", raw.get("all_points", [])))
    else:
        points = _entries_to_points(raw)
        frontier = points
    return {"points": points, "frontier": frontier}


def load_docetl_search_cost(search_run):
    """Return total_search_cost from experiment_summary.json (0.0 if missing/absent)."""
    path = DOCETL_OUTPUTS_DIR / search_run / "experiment_summary.json"
    if not path.exists():
        print(f"[docetl] experiment summary not found: {path}")
        return 0.0
    with open(path) as f:
        raw = json.load(f)
    return raw.get("total_search_cost") or 0.0


# ── Data loading: cost-model agent ("pz") ─────────────────────────────────────

def _extract_metrics(entry):
    """
    Average run1/run2 sub-dicts if present; fall back to top-level cost/quality/latency.
    Mirrors plot_results.py's _avg_runs.
    """
    if "run1" in entry and "run2" in entry:
        r1, r2 = entry["run1"], entry["run2"]
        cost = ((r1.get("cost") or 0) + (r2.get("cost") or 0)) / 2
        q1, q2 = r1.get("quality"), r2.get("quality")
        qual = (q1 + q2) / 2 if (q1 is not None and q2 is not None) else (q1 if q1 is not None else q2)
        lat = ((r1.get("latency") or 0) + (r2.get("latency") or 0)) / 2
        return cost, qual, lat
    return entry.get("cost"), entry.get("quality"), entry.get("latency")


def _load_pz_metrics_entries():
    """Load PZ_METRICS_PATH once, returning {opt_run: entry} for QUERY_ID (empty if missing)."""
    if not PZ_METRICS_PATH.exists():
        print(f"[pz] metrics not found yet: {PZ_METRICS_PATH}")
        return {}
    with open(PZ_METRICS_PATH) as f:
        raw = json.load(f)

    entries = {}
    for key, entry in raw.items():
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            qid, opt_run = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if qid == QUERY_ID:
            entries[opt_run] = entry
    return entries


def _model_metadata(entry):
    return {
        "agent_model":  entry.get("agent_model"),
        "oracle_model": entry.get("oracle_model"),
        "helper_model": entry.get("helper_model"),
    }


def load_pz_final_plans():
    """
    Full-dataset performance of the plan our agent actually picked, one point per opt_run.
    Returns (points, model_metadata).
    """
    entries = _load_pz_metrics_entries()
    points = []
    model_metadata = {}
    for opt_run, entry in entries.items():
        cost, quality, latency = _extract_metrics(entry)
        points.append({
            "opt_run":               opt_run,
            "exec_cost":             cost if cost is not None else 0.0,
            "quality":               quality,
            "latency":               latency,
            "agent_cost":            entry.get("agent_cost") or 0.0,
            "subset_execution_cost": entry.get("subset_execution_cost") or 0.0,
            "oracle_cost":           entry.get("oracle_cost") or 0.0,
            "final_selected":        True,
        })
        model_metadata = _model_metadata(entry)
    return points, model_metadata


def load_pz_all_plans():
    """
    Every plan our agent executed on the optimization subset during search (not just the one
    it finally picked), read from the per-opt-run results CSVs. Each point also carries that
    opt_run's shared agent_cost/subset_execution_cost/oracle_cost (for the with-opt-cost view)
    and a final_selected flag pulled from the CSV.
    """
    entries = _load_pz_metrics_entries()
    csv_pattern = re.compile(rf"^Q{QUERY_ID}_(\d+)_{re.escape(AGENT_DIR)}_results\.csv$")

    points = []
    model_metadata = {}
    if not PZ_METRICS_DIR.exists():
        print(f"[pz] metrics dir not found yet: {PZ_METRICS_DIR}")
        return points, model_metadata

    for csv_path in sorted(PZ_METRICS_DIR.glob(f"Q{QUERY_ID}_*_results.csv")):
        m = csv_pattern.match(csv_path.name)
        if not m:
            continue
        opt_run = int(m.group(1))
        entry = entries.get(opt_run, {})
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cost = float(row["cost_usd"])
                    quality = float(row["quality"])
                except (KeyError, TypeError, ValueError):
                    continue
                points.append({
                    "opt_run":               opt_run,
                    "plan_name":             row.get("plan_name"),
                    "exec_cost":             cost,
                    "quality":               quality,
                    "final_selected":        str(row.get("final_selected")).strip().lower() == "true",
                    "agent_cost":            entry.get("agent_cost") or 0.0,
                    "subset_execution_cost": entry.get("subset_execution_cost") or 0.0,
                    "oracle_cost":           entry.get("oracle_cost") or 0.0,
                })
        if entry:
            model_metadata = _model_metadata(entry)
    return points, model_metadata


# ── Scatter figure ────────────────────────────────────────────────────────────

def _style_axis(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Cost ($)", fontsize=9)
    ax.set_ylabel("Quality", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3)


def _plot_moar(ax, data_by_run, run_names, with_opt_cost, linestyle):
    for run_name in run_names:
        d, search_cost = data_by_run[run_name]
        add = search_cost if with_opt_cost else 0.0
        # scatter every plan (parity with pz's point count -- see load_docetl_full_frontier)
        xs = [p["cost"] + add for p in d["points"]]
        ys = [p["quality"] for p in d["points"]]
        if xs:
            ax.scatter(xs, ys, marker=MOAR_MARKER, s=90, color=MOAR_COLOR, zorder=3)
        # connect only the true (non-dominated) frontier subset
        fxs = [p["cost"] + add for p in d["frontier"]]
        fys = [p["quality"] for p in d["frontier"]]
        if fxs:
            ax.plot(fxs, fys, color=MOAR_COLOR, linewidth=1.5, linestyle=linestyle, zorder=2)


def _plot_full(ax, run_pairs, docetl_full_data, pz_final_points, with_opt_cost):
    """docetl's full-dataset frontier vs. pz's final selected plan, also on the full dataset."""
    _plot_moar(ax, docetl_full_data, [p["full_run"] for p in run_pairs], with_opt_cost, linestyle="-")

    for pt in pz_final_points:
        x = pt["exec_cost"]
        if with_opt_cost:
            x += pt["agent_cost"] + pt["subset_execution_cost"] + pt["oracle_cost"]
        y = pt["quality"]
        if y is None:
            continue
        ax.scatter(x, y, marker=AGENT_FINAL_MARKER, s=200, color=_agent_run_color(pt["opt_run"]), zorder=5)


def _plot_during_opt(ax, run_pairs, docetl_search_data, pz_all_points, with_opt_cost):
    """docetl's search-time frontier vs. every plan pz executed during search."""
    _plot_moar(ax, docetl_search_data, [p["search_run"] for p in run_pairs], with_opt_cost, linestyle="--")

    for pt in pz_all_points:
        x = pt["exec_cost"]
        if with_opt_cost:
            x += pt["agent_cost"] + pt["subset_execution_cost"] + pt["oracle_cost"]
        y = pt["quality"]
        if y is None:
            continue
        final = pt.get("final_selected", False)
        ax.scatter(
            x, y,
            marker=AGENT_FINAL_MARKER if final else AGENT_EXPLORED_MARKER,
            s=200 if final else 50,
            color=_agent_run_color(pt["opt_run"]),
            alpha=1.0 if final else 0.5,
            zorder=5 if final else 3.5,
        )


def make_combined_figure(run_pairs, docetl_full_data, docetl_search_data, pz_final_points,
                          pz_all_points, pz_models):
    """One figure, three subplots:
      1. During optimization (no opt cost): docetl's search frontier vs. ALL of pz's plan points.
      2. Full dataset, without opt cost: the same docetl frontier plans (now full-dataset
         numbers) vs. only pz's final selected plan (also full-dataset numbers).
      3. Same as (2), with opt cost added.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle("CUAD: cost model agent (pz) vs. DocETL MOAR", fontsize=13, fontweight="bold", y=0.98)
    if pz_models:
        subtitle = f"pz agent={pz_models.get('agent_model')}  oracle={pz_models.get('oracle_model')}"
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=8.5, color="#555555")

    _style_axis(axes[0], "During optimization -- without opt cost")
    _plot_during_opt(axes[0], run_pairs, docetl_search_data, pz_all_points, with_opt_cost=False)

    _style_axis(axes[1], "Full dataset -- without opt cost")
    _plot_full(axes[1], run_pairs, docetl_full_data, pz_final_points, with_opt_cost=False)

    _style_axis(axes[2], "Full dataset -- with opt cost")
    _plot_full(axes[2], run_pairs, docetl_full_data, pz_final_points, with_opt_cost=True)

    # Shape/role legend, in neutral gray -- actual color depends on opt_run (see color key).
    handles = [
        mlines.Line2D([], [], color=MOAR_COLOR, marker=MOAR_MARKER, linestyle="-",
                      markersize=9, label=DOCETL_FULL_DISPLAY_NAME),
        mlines.Line2D([], [], color=MOAR_COLOR, marker=MOAR_MARKER, linestyle="--",
                      markersize=9, label=DOCETL_SEARCH_DISPLAY_NAME),
        mlines.Line2D([], [], color=AGENT_LEGEND_COLOR, marker=AGENT_FINAL_MARKER, linestyle="None",
                      markersize=15, label=PZ_FINAL_DISPLAY_NAME),
        mlines.Line2D([], [], color=AGENT_LEGEND_COLOR, marker=AGENT_EXPLORED_MARKER, linestyle="None",
                      markersize=7, alpha=0.5, label=PZ_EXPLORED_DISPLAY_NAME),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05), fontsize=9.5, framealpha=0.95)

    # Color-only legend: which color is which system / which pz opt_run.
    opt_runs = sorted({pt["opt_run"] for pt in pz_final_points + pz_all_points})
    color_handles = [mpatches.Patch(color=MOAR_COLOR, label="MOAR")]
    for opt_run in opt_runs:
        color_handles.append(mpatches.Patch(color=_agent_run_color(opt_run), label=f"PZ-agent (run {opt_run})"))
    fig.legend(handles=color_handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               fontsize=9, framealpha=0.95, title="Color key", title_fontsize=9)

    fig.tight_layout(rect=[0, 0.1, 1, 0.87])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    docetl_full_data = {
        p["full_run"]: (load_docetl_full_frontier(p["full_run"]), load_docetl_search_cost(p["search_run"]))
        for p in RUN_PAIRS
    }
    docetl_search_data = {
        p["search_run"]: (load_docetl_search_frontier(p["search_run"]), load_docetl_search_cost(p["search_run"]))
        for p in RUN_PAIRS
    }

    pz_final_points, pz_final_models = load_pz_final_plans()
    pz_all_points, pz_all_models = load_pz_all_plans()
    pz_models = pz_final_models or pz_all_models

    fig = make_combined_figure(RUN_PAIRS, docetl_full_data, docetl_search_data,
                                pz_final_points, pz_all_points, pz_models)
    out = OUTPUT_DIR / "costQuality_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"Saved:\n  {out}")
    plt.show()


if __name__ == "__main__":
    main()
