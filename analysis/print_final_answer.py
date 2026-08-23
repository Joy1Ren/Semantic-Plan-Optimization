import json
import sys
from pathlib import Path
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_cost_model.paths import RESULTS_DIR
qid = 5
run = "1"                      # which run's final answer to print (key inside Q{qid}.json)
use_case = "ecomm"
scale_factor = 500
agent_dir = "customCost_oracle_helper_agent"
final_answer_path = RESULTS_DIR / "final_answer" / use_case / agent_dir / f"sf_{scale_factor}" / f"Q{qid}.json"
results_path = RESULTS_DIR / "metrics" / use_case / f"sf_{scale_factor}" / f"Q{qid}_{run}_{agent_dir}_results.csv"

def main() -> None:
    results_df = pd.read_csv(results_path)
    quality_metrics = ["f1_score", "precision", "recall"]
    print(results_df[["plan_name", "latency_s", "cost_usd", "input_tokens", "output_tokens"] + quality_metrics])

    with final_answer_path.open() as f:
        answers_by_run = json.load(f)

    # Final answers are nested by run number; fall back to the first run present.
    answer = answers_by_run.get(run) or next(iter(answers_by_run.values()))
    for plan_name, plan_code in answer["plan_codes"].items():
        print(f"=============={plan_name}==============")
        print(plan_code)


if __name__ == "__main__":
    main()
