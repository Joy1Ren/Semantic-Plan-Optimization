"""Build CUAD's dataset and ground truth from the raw release. Run from the repo root, in order:

    python -m benchmarks.cuad.ground_truth.cleaning.clean_cuad_gt  # cuad_gt.csv, cuad_gt_issues.csv
    python -m benchmarks.cuad.prepare_cuad_data                    # idx_to_name.json, dataset/cuad_clean_140.csv
    python -m benchmarks.cuad.ground_truth.cleaning.per_query_gt   # ground_truth/Q{1..8}_gt.csv

Inputs: CUADv1.json and category_descriptions.csv here, plus dataset/source/: CUAD-master_clauses.csv
and cuad.json (docetl's 140 contracts).
"""
