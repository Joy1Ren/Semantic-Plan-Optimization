# benchmarks/

One directory per benchmark. A benchmark says **what to optimize** (queries, data, ground truth,
how quality is scored); the optimizer in `raptor/` says **how**. Adding a benchmark changes
nothing in `raptor/`.

These exist today:

| directory | what it is | ground truth |
|---|---|---|
| `cuad/` | 8 queries over 140 CUAD contracts: Q1 extracts all 41 categories, Q2-8 each ask one question | real annotations, cleaned and split per query by `ground_truth/cleaning/` |
| `SemBench/` | 10 `ecomm` queries at scale factor 2000 | SemBench's gold SQL, regenerated over the subset |
| `summary_haystack/` | SummHay: one use case per haystack (`news1`, `conv1`), one query per subtopic, scored with SummHay's own LLM judge; each prompt quotes one ground-truth insight as an example | SummHay's insight-to-document labels, restricted to the documents the plan ran on |
| `sustainability/` | 8 questions over 140 company reports (sector, sustainability commitments, fatalities) | labels verified by frontier models, built by `ground_truth/llm_grading/` |

Run one:

```bash
python -m benchmarks.run_opt \
    --config benchmarks/cuad/benchmark.yaml \
    --use-case cuad --query-id 2 \
    --agent_type execute_optimizer_random \
    --sample-method random --opt-subsample --sample-size 40 \
    --runcount 1
```

---

## The four files a benchmark needs

| file | responsibility |
|---|---|
| `benchmark.yaml` | every path, prompt and metric. The only file `run_opt.py` reads directly. |
| `quality_evaluator.py` | a `PlanQualityEvaluator` subclass: how a plan's output becomes a 0–1 score. |
| `paths.py` | file locations only this benchmark's own scripts read. The runner never imports it. |
| a data-prep script | writes the per-query source CSV that `benchmark.yaml` points at. |

Nothing else is required. `__init__.py` should exist so the adapter is importable.

Every benchmark here is laid out the same way:

| | |
|---|---|
| `benchmark.yaml`, `quality_evaluator.py`, `paths.py` | the files above |
| `prepare_<name>_data.py` | builds `dataset/` from the raw source; cuad and sustainability keep theirs in `dataset/source/`, copied unchanged from docetl, so they need no other checkout |
| `scoring.py` | the scorer `quality_evaluator.py` calls (cuad, summary_haystack; SemBench uses SemBench's own evaluators) |
| `baseline_plans.yaml` | one hand-written plan per query, run by [`run_baseline.py`](run_baseline.py) |
| `dataset/`, `datasubset_opt/`, `ground_truth/` | the full data, the optimization subsets (ids only), the per-query ground truth (SemBench reads its ground truth from the SemBench checkout); see "State of the data" below |

cuad and sustainability also hold the scripts that build their ground truth
(`ground_truth/cleaning/`, `ground_truth/llm_grading/`) and a `queries.py` binding query ids to
questions.

## benchmark.yaml

Read by [`run_opt.py`](run_opt.py) and [`config.py`](config.py); the analysis scripts resolve the
same templates through `config.results_prefix`, so moving a results path moves reader and writer
together.

### Top level

| key | required | read by | meaning |
|---|---|---|---|
| `name` | yes | nothing — a label | must match this benchmark's key in `config._BENCHMARK_DIRS`, which is what lets `analysis/` resolve it by name (see step 5 below) |
| `quality_adapter` | yes | `run_opt._load_adapter` | dotted module path exporting `QualityEvaluator` |
| `results_prefix` | yes | `run_opt`, `analysis/` | root of everything a run writes |
| `external_repo` | no | `external_repo.activate` | a checkout to read from and put on `sys.path` |
| `sampling_dir` | for `--opt-subsample` | `run_opt` | embedding cache |
| `sampling_results_path` | for `--opt-subsample` | `run_opt`, `DataSampler` | JSON record of what each sampling run selected |
| `use_cases` | yes | `run_opt` | map of use-case name → the block below |

### Per use case

| key | required | meaning |
|---|---|---|
| `query_source.type` | yes | `text` or `toml` — where **modalities** are declared, not where the prompt lives |
| `query_source.modalities` | `text` only | e.g. `[text]`, `[image]`, `[text, image]` |
| `query_source.path` | `toml` only | the toml whose `metadata.modalities` is read |
| `task_prompts` | yes | `{query id: prompt}` — **always** where the prompt comes from |
| `query_metrics` | yes | `{query id: metric name}`; must match `task_prompts` key for key |
| `dataset.directory` | yes | where the query's source CSV lives |
| `dataset.source_csv` | yes | filename, templated |
| `dataset.id_col` | no (`idx`) | the column identifying a row |
| `dataset.image_subdir` | no (`images`) | images are `<directory>/<image_subdir>/<id_col>.jpg` |
| `subset_path` | yes | the optimization subset (see below) |
| `ground_truth_path` | yes | the benchmark's real ground truth |

### Path templates

Every path value is `str.format`-ed against: `{benchmark_dir}` (this directory), `{results_dir}`
(`results`), `{repo_root}` (the `external_repo` checkout, `""` if none),
`{use_case}`, `{scale_factor}`, `{query_id}`, `{runner}` (the `--agent_type` you pass).

Use only the axes a benchmark actually has. `results_prefix` is the clearest example — SemBench
keys on use case, scale factor and query id, CUAD on query id only:

```yaml
results_prefix: "{results_dir}/SemBench/{use_case}/sf_{scale_factor}/q{query_id}/{runner}"
results_prefix: "{results_dir}/CUAD/q{query_id}/{runner}"
```

If **any** template mentions `{scale_factor}`, `run_opt` requires `--scale-factor` and errors
without it, rather than resolving a literal `sf_None` directory.

`{runner}` is the axis you compare across: one name per optimizer variant
(`execute_optimizer_random`, `execute_optimizer_agentic`, …), which is why the sampling method is
usually part of the name. Artifacts that are **not** runner-specific belong beside `{runner}`
rather than under it — `sampling_dir` and `sampling_results_path` are the existing examples.

### A detail that looks like a bug and is not

- **`subset_path` never exists as written.** `mode_subset_path` rewrites it per sampling method and
  subset id, so `Q2_subset.csv` is stored as `Q2_subset_one-shot-1.csv`, `Q2_subset_random-1.csv`, …
  That lets sampling methods be compared on one query without overwriting each other, and it means
  `--sample-method` and `--subset-id` also pick *which* subset a reusing run optimizes against.

## The adapter

Subclass [`PlanQualityEvaluator`](../raptor/plan_quality_evaluator.py). The base class owns
everything benchmark-independent: loading the ground truth once per run and building the
`QualityResult`.

**Three hooks you must implement.** All three are declared with `*args` in the base class on
purpose — the benchmarks' scorers take genuinely different inputs, so every form is offered on
every call and each adapter names only what it uses.

| hook | called with | returns |
|---|---|---|
| `score_plan` | `(plan_output_df, ground_truth_df, ground_truth_path, population_path)` | a 0–1 float |
| `has_ground_truth` | `(ground_truth_df, ground_truth_path)` | bool — asked of whichever form your scorer consumes |
| `write_scoring_input` | `(plan_output_df, path)` | none; a debug artifact of exactly what the scorer was handed |

`population_path` is the dataset the plan actually ran over — the subset during search, the full
dataset at final evaluation. A scorer that measures over a document population takes it from there
rather than from the plan's own output, so a document the plan dropped is still counted against it.

**Constructor.** The engine passes one fixed keyword set (`query_id`, `use_case`,
`scale_factor`, `subset_path`, `ground_truth_path`). End your signature with
`**_ignored` so a new engine-side keyword does not break your benchmark, and pass through to
`super().__init__` with:

- `normalize_df` — reshapes a plan's raw output into what your scorer reads. Optional; the default
  is identity. Bind whatever it needs (SemBench binds the use case and query id). Everything that
  produces a frame to be scored goes through `evaluator.normalize`.
- `ground_truth_loader` — only if your scorer wants the ground truth as a DataFrame (SemBench's
  regenerates gold SQL over the subset). A file-based scorer needs none: `ground_truth_path` is
  handed to it in place.

## Where a run writes

Everything under `results_prefix`, keyed `Q{query_id}_{runcount}`:

```
trajectory/Q2_1.csv          every step: reasoning, code, observation (sub-agents interleaved)
metrics/Q2_1_results.csv     one row per plan execution; final_selected marks the chosen plan
metrics/final_eval.json      one entry per run: cost, latency, run config, full-dataset runK blocks
final_answer/Q2.json         the agent's answer, keyed by runcount
opt_results/Q2_1/            per-plan debug artifacts
raw_results/Q2_1_1.csv       full-dataset output of the chosen plan, per final-eval run
exploration_check/Q2_1.json  what the exploration checker saw and asked for
```

A failed run still writes these. An interrupted one (Ctrl-C) deliberately does not.

## Adding a benchmark

1. **Prepare the data.** A script writing one CSV per query, under `{benchmark_dir}/dataset/…`.
   Give every row a stable id column; if there are images, name them `<id>.jpg`.
2. **Write `benchmark.yaml`.** Copy the closest existing one —
   [`cuad/benchmark.yaml`](cuad/benchmark.yaml) for a per-query benchmark with
   local ground truth, [`SemBench/benchmark.yaml`](SemBench/benchmark.yaml) for scale factors and
   an external checkout.
3. **Write the adapter**: the four hooks, plus `normalize_df` if your scorer needs a different shape
   than the plan emits.
4. **Add your metric's description** to `_QUALITY_METRIC_DOCS` in
   [`raptor/prompts.py`](../raptor/prompts.py), keyed by the name you used in
   `query_metrics` — this is what tells the agent how the score it is optimizing is computed. A
   missing entry silently falls back to a generic sentence.
5. **Register the name** in `_BENCHMARK_DIRS` ([`config.py`](config.py)) as
   `"<your name:>": "<your directory>"`. Only needed so the `analysis/` scripts can resolve the
   benchmark by name — they call `results_prefix("SemBench", …)` rather than passing a yaml path.
   `run_opt.py` takes `--config` and never needs it.
6. **Check it resolves** before spending money: run with `--no-final-eval` and a small
   `--sample-size`.

## State of the data in this folder

Everything a run reads is tracked, except subsets the experiments did not run on:

- `dataset/` -- each benchmark's full data, built by its `prepare_<name>_data.py`. cuad and
  sustainability keep their raw source in `dataset/source/`. SemBench is the exception: its
  `dataset/ecomm/sf_2000_shared/` CSVs are tracked but its images are not; run
  `python -m benchmarks.SemBench.prepare_ecomm_data --scale-factor 2000` (it
  needs the SemBench checkout, which SemBench's scoring needs anyway). `sf_2000/` beside it is the
  per-run copy holding only the current query's CSV (`config.expose_query_data`).
- `datasubset_opt/` -- the optimization subsets, `Q{id}_subset_{mode}-{k}.csv` for the one-shot,
  random and baseline samplers, and the `sampling_config.json` that records how each was drawn. Each
  subset holds only the dataset's id column; `raptor.subsets.load_subset` rebuilds the rows
  from `dataset/`. Only the one-shot subsets the experiments ran on are tracked; without a subset a
  run stops with "Optimization subset not found" unless given `--opt-subsample` to draw a new one.
- `ground_truth/` -- what each query is scored against. SemBench's is its own (in the SemBench
  checkout); its `ground_truth/` holds the ground truth regenerated over each subset, written by
  `quality_evaluator.py` for inspection (tracked only for the tracked subsets). sustainability's
  `ground_truth/llm_grading/` (how its labels were built) is not tracked.
