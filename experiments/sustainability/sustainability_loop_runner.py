"""Run run_opt.py sequentially over sustainability runcounts and queries: every query of one
runcount before the next runcount, so results can be checked for drift as each runcount finishes.
A copy of cuad_categories_loop_runner.py with this benchmark's settings.

The optimization subsets must already exist (datasubset_opt/Q{id}_subset_{method}-{k}.csv, from
sampling_loop_runner.py); run_opt reuses them rather than sampling.

The runcounts to run are given on the command line, one terminal per group:
    PYTHONPATH=. python -m agent_cost_model.experiments.sustainability.sustainability_loop_runner 50 51
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

CORE_MODEL = "openai/gpt-5.6-terra"
ORACLE_MODEL = "openai/gpt-5.6-terra"
# The catalog of plan-operator models shown to the agent and the exploration checker.
MODEL_CATALOG = "available_models.txt"
USE_CASE = "sustainability"
QUERY_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
SAMPLE_METHODS = ["one-shot"]
# Which of each method's subsets to optimize against: 1 -> Q{id}_subset_one-shot-1.csv, with
# results under execute_optimizer_one-shot-1/.
SUBSET_ID = 1
COST_AWARE = False
# runcount -> (lower_variance, exploration_checker)
RUN_CONFIGS = {
    # 50: (True, True),
    51: (True, True),
    52: (True, True),
}
# The ground truth is the benchmark's own, so the oracle is never used (see quality_evaluator.py).
ORACLE_OPERATOR_QUALITY = False
PER_OPERATOR_INFO = True

SKIP_RUNS = []  # (runcount, query_id, method)

CONFIG = Path(__file__).resolve().parent / "benchmark.yaml"
REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    if ORACLE_OPERATOR_QUALITY and not PER_OPERATOR_INFO:
        sys.exit("ORACLE_OPERATOR_QUALITY cannot be on with PER_OPERATOR_INFO off.")
    runcounts = [int(r) for r in sys.argv[1:]]
    unknown = [r for r in runcounts if r not in RUN_CONFIGS]
    if not runcounts or unknown:
        sys.exit(f"Pass runcounts from {sorted(RUN_CONFIGS)}; unknown: {unknown}")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "AGENT_MODEL_CATALOG": MODEL_CATALOG}
    failures = []
    runs = [run for run in itertools.product(runcounts, QUERY_IDS, SAMPLE_METHODS) if run not in SKIP_RUNS]
    for i, (runcount, query_id, method) in enumerate(runs, 1):
        lower_variance, exploration_checker = RUN_CONFIGS[runcount]
        cmd = [
            sys.executable, "-m", "agent_cost_model.experiments.run_opt",
            "--config", str(CONFIG),
            "--core-model", CORE_MODEL,
            "--oracle-model", ORACLE_MODEL,
            "--use-exploration-checker" if exploration_checker else "--no-use-exploration-checker",
            "--use-case", USE_CASE,
            "--query-id", str(query_id),
            "--agent_type", f"execute_optimizer_{method}-{SUBSET_ID}",
            "--sample-method", method,
            "--subset-id", str(SUBSET_ID),
            "--runcount", str(runcount),
            "--no-use-oracle-ground-truth",
            "--cost-aware-opt" if COST_AWARE else "--no-cost-aware-opt",
        ]
        if exploration_checker:
            cmd += ["--exploration-checker-model", CORE_MODEL]
        if not ORACLE_OPERATOR_QUALITY:
            cmd.append("--no-oracle-operator-quality")
        if not PER_OPERATOR_INFO:
            cmd.append("--no-per-operator-info")
        if lower_variance:
            cmd.append("--lower-variance")
        print(f"\n=== [{i}/{len(runs)}] run {runcount} {USE_CASE} Q{query_id} {method}-{SUBSET_ID} ===\n{' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode != 0:
            failures.append((runcount, query_id, method, exploration_checker))
    print(f"\nDone: {len(runs) - len(failures)}/{len(runs)} succeeded.")
    for runcount, query_id, method, exploration_checker in failures:
        print(f"  FAILED: run {runcount} {USE_CASE} Q{query_id} {method} cost_aware={COST_AWARE} exploration-checker={exploration_checker}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
