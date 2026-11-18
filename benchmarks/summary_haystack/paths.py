"""Where the Summary of a Haystack (SummHay) benchmark reads from.

One use case per haystack, one query per subtopic. The haystacks, prompts and scoring code are
SummHay's own, read from the summary-of-a-haystack checkout beside this repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BENCHMARK_DIR.parents[2]
SUMMHAY_REPO = WORKSPACE_ROOT / "summary-of-a-haystack"
SUMMHAY_DATA_DIR = SUMMHAY_REPO / "data"
SUMMHAY_PROMPTS_DIR = SUMMHAY_REPO / "prompts"

# use case -> haystack file. Query k of a use case is the haystack's k-th subtopic (1-based).
TOPICS = {
    "news1": "topic_news1.json",
    "conv1": "topic_conv1.json",
}

DATASET_DIR = BENCHMARK_DIR / "dataset"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"
DATASUBSET_DIR = BENCHMARK_DIR / "datasubset_opt"
DATASET_CSV_NAME = "haystack.csv"


def domain(use_case: str) -> str:
    """"news" or "conversation", as populate_summaries.py's --domain names them."""
    return "conversation" if TOPICS[use_case].startswith("topic_conv") else "news"


def topic_json(use_case: str) -> Path:
    return SUMMHAY_DATA_DIR / TOPICS[use_case]


def dataset_csv(use_case: str) -> Path:
    return DATASET_DIR / use_case / DATASET_CSV_NAME


def ground_truth_csv(use_case: str, query_id: int) -> Path:
    return GROUND_TRUTH_DIR / use_case / f"Q{query_id}_gt.csv"


def subset_csv(use_case: str, query_id: int, subset: str) -> Path:
    """An optimization subset as run_opt writes it, e.g. subset "random-1"."""
    return DATASUBSET_DIR / use_case / f"Q{query_id}_subset_{subset}.csv"


def summhay_utils_eval():
    """SummHay's utils_eval.py, loaded from the checkout. It is imported by file path rather than
    by putting the checkout on sys.path, where its generic module names (utils_eval, utils_misc)
    could shadow others."""
    spec = importlib.util.spec_from_file_location("summhay_utils_eval", SUMMHAY_REPO / "utils_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
