from __future__ import annotations

import ast
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from agent_cost_model.experiments.summary_haystack.paths import (
    SUMMHAY_PROMPTS_DIR,
    ground_truth_csv,
    summhay_utils_eval,
)

_SUMMHAY = summhay_utils_eval()
EVAL_PROMPT = (SUMMHAY_PROMPTS_DIR / "eval_summhay.txt").read_text()

QUALITY_JUDGE_MODEL = "openai/gpt-6-sol"
QUALITY_JUDGE_REASONING_EFFORT = "high"
JUDGE_ATTEMPTS = 3
JUDGE_WORKERS = 10

SUMMARY_COL = "summary"
# Columns that identify a row rather than hold summary text: the dataset id, and `_all`, the key
# a whole-table reduce adds to its output.
_ID_COLS = {"idx", "_all"}

FULL, PARTIAL, NONE = "FULL_COVERAGE", "PARTIAL_COVERAGE", "NO_COVERAGE"
_LABELS = {
    "FULL": FULL, "FULL_COVERAGE": FULL, "FULLY_COVERED": FULL,
    "PARTIAL": PARTIAL, "PARTIAL_COVERAGE": PARTIAL, "PARTIALLY_COVERED": PARTIAL,
    "NO": NONE, "NONE": NONE, "NO_COVERAGE": NONE, "NOT_COVERED": NONE,
}


class JudgeError(RuntimeError):
    """The quality judge never returned a usable verdict for an insight."""


# ---------------------------------------------------------------------------------------------
# Ground truth over a population
# ---------------------------------------------------------------------------------------------

def load_ground_truth(ground_truth_path: str | Path) -> list[dict]:
    """Q{id}_gt.csv's rows, one per insight, with doc_idxs as a set of ints."""
    rows = pd.read_csv(ground_truth_path, dtype=str).fillna("").to_dict("records")
    for row in rows:
        row["doc_idxs"] = {int(i) for i in json.loads(row["doc_idxs"])}
    return rows


def population_idx(population_path: str | Path) -> set[int]:
    """The idx of every document the plan ran over."""
    return {int(i) for i in pd.read_csv(population_path, usecols=["idx"])["idx"]}


def insights_in_population(ground_truth_path, population_path) -> list[dict]:
    """The insights at least one population document holds, each with `ref`: those documents."""
    population = population_idx(population_path)
    present = []
    for row in load_ground_truth(ground_truth_path):
        ref = row["doc_idxs"] & population
        if ref:
            present.append({**row, "ref": ref})
    return present


def prompt_fill_values(use_case: str, query_id: int, population_path: str | Path) -> dict[str, str]:
    """[N_insights] for a plan running over `population_path`: how many of the query's insights its
    documents hold. On the full haystack this is what populate_summaries.summarize fills in."""
    return {"[N_insights]": str(len(insights_in_population(ground_truth_csv(use_case, query_id), population_path)))}


# ---------------------------------------------------------------------------------------------
# Exposing one insight as an example (benchmark.yaml's task prompts and `expose_insight`)
# ---------------------------------------------------------------------------------------------

EXAMPLE_ANCHOR = "\nCareful:\n"


def exposed_insight(ground_truth_path) -> dict:
    """The insight shown to plans as an example of what an insight is: the query's first. Within a
    query the insights share one form (a specific fact, or a specific exchange with its detail), so
    the first stands for the rest. It stays in the score, so runs with and without it compare; the
    other insights are what tells plans apart."""
    return load_ground_truth(ground_truth_path)[0]


# A plan's own extra LLM step that should see the example too declares `EXAMPLE_HINT = ""` and
# appends it to its instruction.
EXAMPLE_HINT_LINE = 'EXAMPLE_HINT = ""'


def example_sentence(insight: str) -> str:
    """The sentence quoting the exposed insight in a query prompt."""
    end = "" if insight.rstrip().endswith((".", "!", "?")) else "."
    return (f'One of the insights to be summarized is "{insight.rstrip()}{end}" Return all insights on the '
            f'topic, including this one; the others are about different aspects of the topic than this example.')


def with_example(text: str, insight: str) -> str:
    """`text` (a query prompt, or plan code holding one) with a sentence quoting `insight` put just
    before the prompt's "Careful:" list. The prompt's own words are unchanged. In plan code that
    declares `EXAMPLE_HINT = ""`, the hint is set to the same insight, worded as an example."""
    if text.count(EXAMPLE_ANCHOR) != 1:
        raise ValueError(f"expected one line {EXAMPLE_ANCHOR.strip()!r} to put the example before, "
                         f"found {text.count(EXAMPLE_ANCHOR)}")
    text = text.replace(EXAMPLE_ANCHOR,
                        "\n" + example_sentence(insight) + EXAMPLE_ANCHOR)
    hint = f' For example, one insight on this topic is: "{insight}"'
    return text.replace(EXAMPLE_HINT_LINE, f"EXAMPLE_HINT = {hint!r}", 1)


# ---------------------------------------------------------------------------------------------
# Bullets
# ---------------------------------------------------------------------------------------------

def bullet_processor(summary: str) -> list[str]:
    """populate_summaries.bullet_processor, copied: that module imports SummHay's internal `anyllm`
    and so cannot be imported."""
    lines = summary.split("\n")
    lines = [l.strip() for l in lines if l.strip() != ""]
    return lines


# A bullet marker ("- ", "• ", "* " or "– ", or a number "1. " / "1) " before a capitalized
# sentence) after a sentence end or a citation, in a summary with no newlines: pz's JSON parser
# (generators.parse_json_output) replaces every newline in the answer with a space. The capital
# keeps "(105 vs. 91) supports" from splitting.
_INLINE_BULLET = re.compile(r"(?<=[.\]])\s+(?=[-•*–]\s|\d{1,2}[.)]\s+[A-Z\"“])")


def _cell_items(value) -> list[str]:
    """One output cell as summary text items: a list (or a string holding a JSON or Python list)
    gives its items, a string with escaped but no real newlines is unescaped, and a string with
    neither has a newline put back before each inline bullet marker."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value)
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(stripped)
            except Exception:
                continue
            if isinstance(parsed, list) and all(isinstance(v, str) for v in parsed):
                return parsed
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n")
    if "\n" not in text:
        text = _INLINE_BULLET.sub("\n", text)
    return [text]


def summary_bullets(plan_output_df: pd.DataFrame) -> list[str]:
    """The plan's summary as SummHay's bullets. Read from the `summary` column, or every text column
    when the plan named it otherwise; rows are taken in order, so a plan that emits one bullet per
    row is read the same as one that emits a single summary."""
    if plan_output_df is None or plan_output_df.empty:
        return []
    if SUMMARY_COL in plan_output_df.columns:
        cols = [SUMMARY_COL]
    else:
        cols = [c for c in plan_output_df.columns if c not in _ID_COLS
                and plan_output_df[c].map(lambda v: isinstance(v, (str, list))).any()]
    bullets = []
    for _, row in plan_output_df[cols].iterrows():
        for col in cols:
            for item in _cell_items(row[col]):
                bullets += bullet_processor(item)
    return bullets


# ---------------------------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------------------------

_BRACKET = re.compile(r"\[([^\[\]]{1,200})\]")
_CITE_LABEL = re.compile(r"\b(?:articles?|conversations?|documents?|docs?|idx|ids?)\b\s*[:=#]?|#", re.I)
_CITE_BODY = re.compile(r"\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;\s]\s*\d+(?:\s*[-–]\s*\d+)?)*")
# What SummHay's extract_citations reads correctly as is.
_SUMMHAY_CITATION = re.compile(r"[\d, ]+")
MAX_RANGE = 100


def _citation_numbers(body: str) -> list[int]:
    """The documents a citation body names, ranges expanded: "3-5; 8" -> [3, 4, 5, 8]."""
    numbers = []
    for start, end in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", body):
        if end and 0 < int(end) - int(start) <= MAX_RANGE:
            numbers += range(int(start), int(end) + 1)
        else:
            numbers += [int(start)] + ([int(end)] if end else [])
    return numbers


def normalize_citations(bullet: str) -> tuple[str, int]:
    """`bullet` with each unambiguous citation variant rewritten to SummHay's [a, b] form, and how
    many were rewritten. A bracket is a citation when, with labels like "Article" or "idx" taken
    out, it holds only numbers, ranges and , ; or space separators; anything else ([doctor]) is
    left alone."""
    rewritten = 0

    def fix(match: re.Match) -> str:
        nonlocal rewritten
        content = match.group(1)
        if _SUMMHAY_CITATION.fullmatch(content) and not re.search(r"\d\s+\d", content):
            return match.group(0)
        body = _CITE_LABEL.sub(" ", content).strip()
        if not _CITE_BODY.fullmatch(body):
            return match.group(0)
        rewritten += 1
        return "[" + ", ".join(str(n) for n in _citation_numbers(body)) + "]"

    return _BRACKET.sub(fix, bullet), rewritten


# ---------------------------------------------------------------------------------------------
# The quality judge
# ---------------------------------------------------------------------------------------------

class QualityJudge:
    """The LLM call inside SummHay's metric: `judge(prompt) -> (reply, cost_usd)`.

    Built lazily so a scorer that never judges (an empty summary, a stubbed test) needs no API key.
    """

    def __init__(self, model: str = QUALITY_JUDGE_MODEL,
                 reasoning_effort: str = QUALITY_JUDGE_REASONING_EFFORT) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = None

    def __call__(self, prompt: str) -> tuple[str, float]:
        if self._client is None:
            from agent_cost_model.opt_agent.llm_client import RoutedClient
            self._client = RoutedClient(self.model, reasoning_effort=self.reasoning_effort)
        # User-only, as populate_eval.evaluate_insights sends it.
        reply, _, info = self._client.generate(system="", messages=[{"role": "user", "content": prompt}])
        return reply, float(info.get("cost_usd", 0.0) or 0.0)


def _balanced_json_objects(text: str):
    """Each top-level, brace-matched {...} span in `text`, skipping braces inside strings."""
    depth, start, in_string, escaped = 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start: i + 1]
                start = None


def _reply_json(reply: str) -> dict | None:
    """The judge's JSON object: the bare reply, else a ``` fence, else any brace-balanced object."""
    candidates = [reply.strip()]
    candidates += [m.strip() for m in re.findall(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)]
    candidates += list(_balanced_json_objects(reply))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and "coverage" in parsed:
            return parsed
    return None


def _bullet_id(value, n_bullets: int) -> int | None:
    """A 1-based bullet number in range, from an int, a numeric string, or a list of them."""
    for v in value if isinstance(value, list) else [value]:
        try:
            i = int(str(v).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n_bullets:
            return i
    return None


def parse_verdict(reply: str, n_bullets: int) -> tuple[str | None, int | None]:
    """(coverage label, bullet id) from one judge reply. The label is None when unreadable; the
    bullet id is None when no valid one was given."""
    parsed = _reply_json(reply)
    if parsed is None:
        return None, None
    label = _LABELS.get(str(parsed.get("coverage", "")).strip().upper().replace(" ", "_").replace("-", "_"))
    return label, _bullet_id(parsed.get("bullet_id"), n_bullets)


def judge_insight(insight: str, bullets: list[str], judge: Callable) -> dict:
    """One insight's verdict, asked with populate_eval.evaluate_insights' exact prompt.

    Retried up to JUDGE_ATTEMPTS times while the reply is unusable: unparseable, an unknown
    coverage label, or covered with no valid bullet id. Covered but never with a valid bullet id
    keeps the coverage and scores citation 0 (`bullet_id` None); never readable at all raises
    JudgeError, so a failed judge is not scored as an uncovered insight."""
    bullets_str = json.dumps({"bullets": [{"bullet_id": i + 1, "text": b} for i, b in enumerate(bullets)]}, indent=1)
    prompt = EVAL_PROMPT.replace("[[BULLETS]]", bullets_str).replace("[[INSIGHT]]", insight)
    cost, label, bullet_id = 0.0, None, None
    for _ in range(JUDGE_ATTEMPTS):
        reply, call_cost = judge(prompt)
        cost += call_cost
        attempt_label, attempt_bullet = parse_verdict(reply, len(bullets))
        if attempt_label is None:
            continue
        label, bullet_id = attempt_label, attempt_bullet
        if label == NONE or bullet_id is not None:
            break
    if label is None:
        raise JudgeError(f"no readable verdict after {JUDGE_ATTEMPTS} attempts for insight {insight[:80]!r}")
    return {"coverage": label, "bullet_id": bullet_id, "cost_usd": cost,
            "invalid": label != NONE and bullet_id is None}


# ---------------------------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------------------------

def insight_bullets(bullets: list[str]) -> int:
    """How many bullets state something: lines with at least 5 words besides their citations.
    Format lines a plan leaves in ("summary:", "---", "<summary>") are not insights."""
    return sum(len(re.findall(r"[A-Za-z0-9]+", re.sub(r"\[[\d,\s]+\]", "", b))) >= 5 for b in bullets)


def bullet_diff(plan_output_df: pd.DataFrame, ground_truth_path, population_path) -> int:
    """score_summary's bullet_diff without judging: the summary's insight bullets minus how many
    insights the population holds ([N_insights]); negative when it has fewer."""
    bullets = [normalize_citations(b)[0] for b in summary_bullets(plan_output_df)]
    return insight_bullets(bullets) - len(insights_in_population(ground_truth_path, population_path))


def score_summary(plan_output_df: pd.DataFrame, ground_truth_path, population_path,
                  judge: Callable) -> dict:
    """SummHay's scores for one plan output, over the insights its population holds.

    Returns `score` (mean joint, NaN when the population holds none of the subtopic's insights),
    the coverage and citation breakdown, formatting counts, and `quality_judge_cost_usd`.
    """
    insights = insights_in_population(ground_truth_path, population_path)
    n_ground_truth = len(load_ground_truth(ground_truth_path))
    bullets, rewritten = [], 0
    for bullet in summary_bullets(plan_output_df):
        fixed, n = normalize_citations(bullet)
        bullets.append(fixed)
        rewritten += n

    if bullets:
        with ThreadPoolExecutor(max_workers=max(1, min(JUDGE_WORKERS, len(insights)))) as pool:
            verdicts = list(pool.map(lambda ins: judge_insight(ins["insight"], bullets, judge), insights))
    else:
        verdicts = [{"coverage": NONE, "bullet_id": None, "cost_usd": 0.0, "invalid": False}
                    for _ in insights]

    # A covered verdict with no usable bullet id points at an empty sentinel bullet: SummHay's
    # function then scores its citation 0 without being changed.
    scored_bullets = bullets + [""] if any(v["invalid"] for v in verdicts) else bullets
    evals = [{"insight_id": ins["insight_id"], "coverage": v["coverage"],
              "bullet_id": v["bullet_id"] if v["bullet_id"] is not None else len(scored_bullets)}
             for ins, v in zip(insights, verdicts)]
    ref_map = {ins["insight_id"]: ins["ref"] for ins in insights}
    scores = _SUMMHAY.compute_single_sample_scores(
        scored_bullets, evals, topic=None, insightid2ref_citations=ref_map, partial_score=0.5,
        cite_offset=0,
    ) if insights else {"coverage_score": [], "citation_score": [], "joint_score": [],
                        "citation_precision": [], "citation_recall": []}

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    return {
        "score": mean(scores["joint_score"]),
        "coverage": mean(scores["coverage_score"]),
        # over covered insights, as SummHay averages them
        "citation_precision": mean(scores["citation_precision"]),
        "citation_recall": mean(scores["citation_recall"]),
        "citation_f1": mean(scores["citation_score"]),
        "n_insights": len(insights),
        "fully_covered": sum(v["coverage"] == FULL for v in verdicts),
        "partially_covered": sum(v["coverage"] == PARTIAL for v in verdicts),
        "insights_absent": n_ground_truth - len(insights),
        "n_bullets": len(bullets),
        # insight bullets minus [N_insights]: how far the summary is from the count it was asked for
        "bullet_diff": insight_bullets(bullets) - len(insights),
        "n_words": sum(len(b.split()) for b in bullets),
        "citations_rewritten": rewritten,
        "judge_invalid": sum(v["invalid"] for v in verdicts),
        "quality_judge_cost_usd": sum(v["cost_usd"] for v in verdicts),
        "evals": [{**e, "insight": ins["insight"], "ref": sorted(ins["ref"])}
                  for e, ins in zip(evals, insights)],
        "bullets": bullets,
    }


def agent_counts(result: dict) -> dict:
    """What the optimizing agent is shown next to `quality`."""
    n, covered = result["n_insights"], result["fully_covered"] + result["partially_covered"]
    counts = {"fully_covered": f"{result['fully_covered']}/{n}",
              "partially_covered": f"{result['partially_covered']}/{n}"}
    if not covered:
        return {**counts, "citation_means": "n/a (no insight covered)"}
    return {
        **counts,
        "mean_citation_precision": f"{result['citation_precision']:.2f}",
        "mean_citation_recall": f"{result['citation_recall']:.2f}",
        "mean_citation_f1": f"{result['citation_f1']:.2f}",
        "citation_means_over": f"{covered} covered insight{'' if covered == 1 else 's'}",
    }


def log_line(result: dict) -> str:
    """One line for the run log: the score, its breakdown, formatting counts and judge spend."""
    n = result["n_insights"]
    return (f"joint {result['score']:.4f} | full {result['fully_covered']}/{n}, partial "
            f"{result['partially_covered']}/{n}, citation P/R/F1 {result['citation_precision']:.2f}/"
            f"{result['citation_recall']:.2f}/{result['citation_f1']:.2f} "
            f"({result['fully_covered'] + result['partially_covered']} covered) | absent "
            f"{result['insights_absent']} | bullets {result['n_bullets']}, words {result['n_words']}, "
            f"citations rewritten {result['citations_rewritten']}, judge invalid "
            f"{result['judge_invalid']} | quality judge ${result['quality_judge_cost_usd']:.4f}")
