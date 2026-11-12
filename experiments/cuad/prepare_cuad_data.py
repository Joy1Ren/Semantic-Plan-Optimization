"""Build the idx -> contract name map every CUAD experiment shares (IDX_TO_NAME_JSON).

A plan never sees a contract's real filename (e.g. "...ContentLicenseAgreement.txt"): the CUAD
task asks a plan to extract "Document Name", and a filename in an input column would let it copy
that instead of reading the title from the text. Each contract gets a small integer "idx" instead,
and this map re-attaches the name at eval time (see quality_evaluator.py's normalize_eval_df). It
lives under ground_truth/ since that directory is eval-only and never fed to a plan.

The datasets themselves are built elsewhere: cuad/dataset/cuad_clean_140.csv is a copy of
cuad_categories/dataset/cuad_clean_140.csv, written by
cuad_categories/prepare_cuad_categories_data.py from this map.
"""
from __future__ import annotations

import json

from agent_cost_model.experiments.cuad.paths import (
    BENCHMARK_DIR,
    GROUND_TRUTH_DIR,
    IDX_TO_NAME_JSON,
    cuad_data_dir,
    cuad_ground_truth_csv,
)

# docetl's own train/test splits, relative to the docetl checkout's experiments/reasoning/data/.
# Their contracts take the first idx values, so the numbering stays what it has always been.
SOURCES = ["test/cuad.json", "train/cuad.json"]  # 100 + 40 documents, disjoint
# Every CUAD contract, for the rest of the map.
CUAD_JSON = BENCHMARK_DIR / "CUADv1.json"


def _build_name_to_idx() -> dict[str, int]:
    """Assign every one of CUAD's 510 contracts a small sequential idx, keyed by its real "name"
    (source filename).

    The documents in docetl's splits come first, pooled and sorted. The rest of CUADv1 follows,
    sorted, so adding them never renumbered a document already in use (or in the embedding cache
    keyed by idx).
    """
    names = set()
    for source in SOURCES:
        path = cuad_data_dir() / source
        if not path.exists():
            raise FileNotFoundError(f"CUAD source data not found: {path}")
        names |= {r["name"] for r in json.loads(path.read_text())}
    names = sorted(names)
    titles = [f"{doc['title']}.txt" for doc in json.loads(CUAD_JSON.read_text())["data"]]
    names += sorted(set(titles) - set(names))
    return {name: i for i, name in enumerate(names)}


def main() -> None:
    name_to_idx = _build_name_to_idx()
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    idx_to_name = {str(idx): name for name, idx in name_to_idx.items()}
    IDX_TO_NAME_JSON.write_text(json.dumps(idx_to_name, indent=2))
    print(f"idx -> name mapping: wrote {len(idx_to_name)} entries -> {IDX_TO_NAME_JSON}")

    if not cuad_ground_truth_csv().exists():
        raise FileNotFoundError(f"CUAD ground truth not found: {cuad_ground_truth_csv()}")
    print(f"ground truth: read in place from {cuad_ground_truth_csv()} (not copied)")


if __name__ == "__main__":
    main()
