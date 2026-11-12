"""Where the sustainability benchmark reads from.

The 140 company documents are docetl's sustainability splits (40 train + 100 test). Their labels
come from the Hugging Face dataset DataNeed/company-reports, version 2024_01_22, where GPT-3.5-turbo
extracted every field from the first 2,500 characters of each report. See ground_truth/llm_grading/.
"""
from __future__ import annotations

from pathlib import Path

SUSTAINABILITY_DIR = Path(__file__).resolve().parent
DOCETL_DATA_DIR = SUSTAINABILITY_DIR.parents[2] / "docetl" / "experiments" / "reasoning" / "data"

DATASET_DIR = SUSTAINABILITY_DIR / "dataset"
DATASET_CSV = DATASET_DIR / "sustainability_140.csv"
DATASUBSET_DIR = SUSTAINABILITY_DIR / "datasubset_opt"
OPTIMIZE_CSV = DATASUBSET_DIR / "sustainability_optimize.csv"

# ground_truth/ holds only what the scorer reads: sustainability_gt.csv and the Q{id}_gt.csv files.
GROUND_TRUTH_DIR = SUSTAINABILITY_DIR / "ground_truth"
SUSTAINABILITY_GT_CSV = GROUND_TRUTH_DIR / "sustainability_gt.csv"
# Everything that produced them: the frontier-model labeling runs, their analysis and the build.
LLM_GRADING_DIR = GROUND_TRUTH_DIR / "llm_grading"
# The source dataset's own labels, kept for comparison: document_category, economic_activity and
# company_name per idx. Never shown to a plan.
ORIGINAL_LABELS_CSV = LLM_GRADING_DIR / "original_labels.csv"
MODEL_LABELS_DIR = LLM_GRADING_DIR / "model_labels"


def per_query_ground_truth(query_id: int) -> Path:
    return GROUND_TRUTH_DIR / f"Q{query_id}_gt.csv"
