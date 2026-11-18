"""Where the sustainability benchmark reads from.

The 140 company documents are docetl's sustainability splits (40 train + 100 test), joined in
dataset/source/sustainability.json. Their source labels come from the Hugging Face dataset
DataNeed/company-reports, version 2024_01_22, where GPT-3.5-turbo extracted every field from the
first 2,500 characters of each report.
"""
from __future__ import annotations

from pathlib import Path

SUSTAINABILITY_DIR = Path(__file__).resolve().parent

DATASET_DIR = SUSTAINABILITY_DIR / "dataset"
# docetl's test/ and train/ sustainability.json, joined. Read only by prepare_sustainability_data.py.
SOURCE_JSON = DATASET_DIR / "source" / "sustainability.json"
DATASET_CSV = DATASET_DIR / "sustainability_140.csv"
DATASUBSET_DIR = SUSTAINABILITY_DIR / "datasubset_opt"

# ground_truth/: the Q{id}_gt.csv files the scorer reads, and sustainability_gt.csv, the labels they
# are built from.
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
