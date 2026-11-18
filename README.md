# Cost-Model Agent
## What it does

Given a natural-language query task and a source dataset, `RAPTOR.run()` runs a
bounded tool-loop LLM agent that:

1. explores the dataset (schema, samples, images, aggregate stats),
2. writes a candidate **physical plan** — a tree of operators, some exact/Python, some
   semantic/LLM-backed — as Palimpzest-style pipeline-builder code,
3. **executes** that plan on a small, fixed data subset and observes real cost ($),
   latency (s), token usage, and **quality** (0–1, scored against the benchmark's ground
   truth — see below),
4. repeats steps 2–3, each new plan embodying a different optimization idea (cheaper
   model, narrower `depends_on`, reordered/pruned operators, ...), using the growing
   table of observed plan/operator results to decide what to try next,
5. emits a final JSON answer naming the best plan and why, then re-runs *that* plan
   on the **full** dataset against real ground truth for an out-of-sample readout.

## Layout

The repo is three layers, with one rule: **the engine names no benchmark, and a benchmark
never re-derives a path its config already declares.**

| path | what it is |
|------|------------|
| `raptor/` | **the engine** — benchmark-agnostic. Knows nothing about SemBench, CUAD, or MOAR. |
| `raptor/core_agent.py` | the agent: system prompt, tools, sandboxed step loop, final evaluation |
| `raptor/physical_pipeline/` | `PhysicalPipeline` — the plan IR/executor: PZ's operators, a replica of its execution engine, and the patches and model registration PZ needs (see `PHYSICAL_PIPELINE_DESIGN.md`) |
| `raptor/local_python_executor.py` | sandboxed Python executor the agent's tool calls run in |
| `sampling/` | builds the optimization datasubset: BM25 + keyphrase + embedding scoring, in five modes (`random` uniform draw, `baseline` no LLM, `one-shot` one planning call, `one-shot-gradient` the same but spreading each stratum across its ranking, `agentic` with optional data exploration plus feedback-driven revisions). The LLM modes may split the subset into **strata** — several specs with stated row allocations — when one ranking cannot represent everything a query needs. |
| `raptor/plan_quality_evaluator.py` | benchmark-agnostic quality scoring against the benchmark's ground truth |
| `paths.py` | this repo's own locations — five constants, no benchmark names |
| `benchmarks/run_opt.py` | CLI entry point: wires a benchmark config to one `RAPTOR.run()` call |
| `benchmarks/config.py` | loads a `benchmark.yaml` and resolves its path templates (used by the runner *and* by `analysis/`, so they can't drift) |
| `benchmarks/external_repo.py` | resolves the external checkouts and puts them on `sys.path` |
| `benchmarks/SemBench/`, `benchmarks/cuad/` | **the benchmark adapters** — `benchmark.yaml`, `quality_evaluator.py`, `paths.py`, data prep, datasets, datasubsets |
| `results/` | everything any run produces (see below) |
| `analysis/` | result-inspection scripts/notebooks over `results/` |

`benchmarks/*/dataset/`, `benchmarks/*/datasubset*/`, `benchmarks/*/ground_truth/`, and
`results/` are gitignored (generated/large artifacts, created by the scripts below).

### External benchmark checkouts

SemBench and docetl are used **in place, never vendored**, so their own evaluators stay the
single source of truth for how a benchmark is scored. Neither is importable as an installed
package for our purposes: SemBench ships no packaging metadata and its modules import each
other as top-level packages rooted at `src/`; docetl is packaged but its build excludes
`benchmarks/**`, which is where the CUAD scorer lives. Each benchmark therefore declares its
checkout in its own `benchmark.yaml`:

```yaml
external_repo:
  env_var: SEMBENCH_ROOT     # override the location
  default: ../SemBench       # relative to this repo's root
  python_path: src           # subdir to place on sys.path
```

`benchmarks/external_repo.py` is the only code that puts an external repo on `sys.path`.
Override a location with its env var:

```bash
export SEMBENCH_ROOT=/absolute/path/to/SemBench
export DOCETL_ROOT=/absolute/path/to/docetl
```

### Where results go

**External checkouts are read-only inputs; everything written lands in `results/`.** Each
benchmark declares the prefix, so the path carries only the axes that benchmark actually has
(CUAD has neither a use case nor a scale factor):

```
results/SemBench/{use_case}/sf_{scale_factor}/{runner}/...
results/CUAD/{runner}/...
```

`{runner}` names one optimizer or variant — `execute_optimizer_agentic`,
`execute_optimizer_random`, … alongside external ones like `MOAR`. It is the axis you
compare across. **Everything below the prefix is that runner's own layout**, whatever it
natively produces: `trajectory/`, `metrics/`, `final_answer/`, `opt_results/`, `llm_judge/`
for this agent; `outputs/` for MOAR (point `run_moar.py --output_dir` at
`results/CUAD/MOAR/outputs`, and nothing is written into the docetl checkout).

Artifacts shared by *every* runner sit beside `{runner}`, not under it, so they aren't
duplicated per runner — sampling builds the datasubset all runners consume:

```
results/SemBench/{use_case}/_sampling/cache/      # embedding cache, keyed by model alone
results/SemBench/{use_case}/sf_{sf}/_sampling/    # per-scale-factor sampling output
results/_analysis/                                # cross-runner comparison plots
```

## Run it

```bash
# a real optimization run, via the SemBench-specific CLI wrapper
# keys per raptor/routing.py: OpenAI models use OPENAI_API_KEY, the rest OPENROUTER_API_KEY
export OPENAI_API_KEY=sk-... OPENROUTER_API_KEY=sk-or-...
python3 -m benchmarks.run_opt \
    --config benchmarks/SemBench/benchmark.yaml \
    --use-case movie --query-id 7 --scale-factor 2000 \
    --agent_type execute_optimizer_agentic \
    --opt-subsample --sample-size 10   # first run only: builds the optimization datasubset
```

`run_opt.py` reads `benchmark.yaml` for query/data/evaluation details (task prompt, quality
metric, dataset paths, ground-truth location, and the `results_prefix` template) and owns the
optimization *policy* in code (model, `max_steps`, subset size/seed) so the same
runner works for another benchmark by pointing `--config` elsewhere. `--agent_type` is the
runner name that lands in the results path; it is required on every run.

### Building the datasubset separately

`--opt-subsample --sample-method {random,baseline,one-shot,agentic} --sample-size N` builds the
subset inline. `--sample-size` is required rather than defaulted: the subset size is an
experiment parameter, and a silent default makes two runs look comparable when they are not.
To build it once and optimize against it repeatedly — so sampling and optimization are paid
for and iterated on separately — use the sampler's own CLI with the **same** `benchmark.yaml`,
then run `run_opt.py` *without* `--opt-subsample`:

```bash
python3 -m raptor.sampling.cli \
    --config benchmarks/SemBench/benchmark.yaml \
    --use-case ecomm --query-id 11 --scale-factor 500 \
    --mode agentic --sample-size 10 --embedding-model qwen/qwen3-embedding-8b

python3 -m benchmarks.run_opt \
    --config benchmarks/SemBench/benchmark.yaml \
    --use-case ecomm --query-id 11 --scale-factor 500 \
    --agent_type execute_optimizer_agentic   # reuses the subset above
```

`--config` makes the CLI resolve every path the way the runner does, so the subset lands
exactly where `run_opt.py` looks for it. The second command prints which mode built the subset
it picked up, and what that sampling cost.

### Gradient selection (`one-shot-gradient`)

Every other mode fills a stratum with its **top-m** rows. That subset sits at one end of the
query's decision surface: plan quality saturates, candidate plans stop separating, and the
subset says nothing about what the query *excludes*. `one-shot-gradient` writes the same spec
but spreads each stratum's allocation across its own ranking:

| band | share of `m` | anchored at |
|---|---|---|
| `top` | half | rank 1 |
| `middle` | a quarter | p50 of the ranking |
| `negative` | a quarter | p90 of the ranking |

Half the budget still goes to the rows being targeted — those are what give plan quality a
gradient to move along — and half makes the subset representative. The negative band sits at
p90 rather than just below the top cutoff deliberately: hard negatives are the most
discriminating rows and also the most distorting, since avoiding them is where plan score moves,
which pushes the optimizer toward precision at the cost of recall.

Selection walks **down** from each anchor, skipping rows an earlier stratum claimed. Where the
anchor lands in a *flat run* — rows whose scores agree to a thousandth, which is what a scorer
that either matches or does not produces — the band is drawn uniformly at random from across
that whole run instead, since inside a flat run the ranking is only source-CSV order. That
samples the region honestly; it does not create resolution the score lacks. When two anchors
land in the same flat run their bands coincide, and the per-stratum `bands` diagnostic in
`sampling_results.json` records each band's realized rank and score range so that shows up as a
measurement rather than a surprise.

Every selected row carries its `band` and `rank` in the results record and in the scores CSV.
Because subsets are mode-suffixed, `one-shot` and `one-shot-gradient` can be built for the same
query and compared side by side.

Each sampling run writes four artifacts (paths are echoed at the end of the run, and recorded
under `artifacts` in the results JSON):

| artifact | contents |
|---|---|
| the benchmark's `subset_path` | the selected rows — the only artifact the optimizer reads |
| `_sampling/sampling_results.json` | per-round specs, score summaries, and the cost breakdown, nested `{query_id: {mode: record}}` |
| `_sampling/scores/Q{id}_{mode}_scores.csv` | one row per scored row per round **per stratum**: raw BM25 and cosine, their normalized forms, keyphrase hits, `sparse`, `score`, rank, whether it was selected, and the spec that produced them |
| `_sampling/rounds/Q{id}_{mode}.json` | full prompts and replies (one-shot / agentic only) |

Row embeddings are fetched concurrently — `--embed-workers` (default 10, on both the sampler
CLI and `run_opt.py`). Only the network calls are parallel; every cache mutation happens in the
consuming thread, so the accounting and the selected rows are identical to a serial run. Two
latency figures are recorded because they diverge by roughly the worker count:
`embedding_wallclock_s` is elapsed time and `embedding_iterative_s` is the summed per-call time
a serial run would have spent. `embed_workers` is recorded alongside them, since neither number
is comparable across runs without it. `embedding_sunk_s` is the serial time the cache saved —
what the reused vectors took when first computed — so `iterative + sunk` is what a cold serial
run would have spent and is the figure that compares across warm and cold caches. `total_s` is measured start-to-end for the whole run, so
it exceeds the sum of the parts — it includes cache load/save, LLM retries, and image
description, none of which the component timers cover.

Sampling cost is reported split by what it was spent on. `total_usd_paid_now` is what *this*
run paid; `embedding_rows_usd_sunk` is what the row embeddings it reused had cost when an
earlier run first computed them. Those vectors are cached per (use case, embedding model,
column set) and shared across every query of a use case, so on a warm cache a run pays only for
its LLM calls and a couple of query embeddings.

## The optimization loop

`RAPTOR.run()` is a bounded loop (`max_steps`, default 40). Each step the LLM
emits **exactly one** fenced block:

- a ` ```python ` block — one tool call, executed in a `LocalPythonExecutor` sandbox;
  its stdout/return value becomes next step's observation, or
- a ` ```json ` block — the final answer (parsed as data, not executed), which ends the loop.

The suggested/enforced workflow (see `RAPTOR.briefing`) is:

1. **Explore** — `list_files()`, `explore_schema(f)`, `explore_sample(f)` for a quick look;
   `explore_data(f)` for a full-CSV pandas DataFrame to compute aggregate statistics
   (keyword frequency, `value_counts()`, class balance) that inform filter selectivity,
   join sizes, and class distribution — used to *design* plans, never to hardcode an
   answer. `explore_images(ids, question=None)` shows up to 5 images (via a cheap vision
   model, which returns a text description — pixels never reach the main agent) to gauge
   what a vision operator would see; `question` adds a natural-language question the
   describer must address for every image, on top of its default description.
2. **Write** a plan — `write_plan(code, name, description="")`. `code` is a Python string,
   run in a *separate*, minimal sandbox (only `PhysicalPipeline`, `load_data`,
   `add_image_data`, `pz` — no results stores, no `explore_data`, so plan code can never
   depend on directly-scanned data) that builds and returns a `PhysicalPipeline`.
   Hardcoding row IDs/values found while exploring is explicitly disallowed — plans must
   stay general.
3. **Execute** — `execute_plan(name)` runs the plan on a reproducible data subset (same
   records across every plan in the run), appends one row to `plan_results` (cost, latency,
   tokens, `quality`) and one row per operator invocation to `op_results`.
4. **Evaluate** — read `plan_results.df` / `op_results.df`. `quality` is the score against
   the benchmark's ground truth (the agent is told not to try to reverse-engineer it), and
   `get_op_samples(plan_name, op_name)` shows one operator's input/output pairs.
5. **Iterate** — write one new plan per step embodying a different idea (or a combination
   of ideas that worked), guided by the observed trade-offs. The agent is told not to
   micro-tune a single knob (the subset is small; over-fitting to it is a known failure
   mode) and to stop once gains over its current best plan are marginal or steps run low.

On a final answer (or on running out of steps, which forces one terminal turn), the agent
saves the trajectory/results tables and then runs `_run_final_evaluation`: it rebuilds the
chosen plan from its saved source code and re-executes it — fresh, `num_final_eval_runs` times
— on the **full** dataset against the benchmark's real ground truth CSV, independent of
whatever quality mode scored it during search. This is the only place true, full-dataset,
non-subsampled numbers get produced.

That full-dataset execution is the expensive half of a run, so it is optional:
`query_info["run_final_eval"]` (default `True`, exposed as `run_opt.py --final-eval /
--no-final-eval`) turns it off. With it off the search still writes its `metrics/final_eval.json`
entry — agent cost/latency, plans written/executed, run config — just with no `run1`/`run2`
blocks and no `raw_results/` CSVs.

### Cost-aware search (`--cost-aware-opt`)

By default the search stops on the agent's own sense that gains have gone marginal, with no idea
what the search itself is costing — so it will happily spend more finding a saving than the
saving is ever worth. `run_opt.py --cost-aware-opt` (off by default) changes that:

- every step closes with an `[optimization budget]` line giving what that step cost, what the
  search has spent in total, and the subset size next to the full dataset size. It is the last
  thing the agent sees before its next turn, which is also what bills the exploration checker
  to the step that triggered it rather than to the next one;
- the exploration checker gets the same numbers as of the moment it was called (it cannot be
  shown its own cost, which has not been spent yet);
- both are told to compare the cost of **one more** plan (spend so far is sunk) against the
  saving it would buy on the full dataset — and that the size ratio is only an upper bound on
  the scale factor, since the subset is relevance-sampled, so the agent should count surviving
  rows with `explore_data` instead of multiplying by it.

Quality is explicitly exempt: a plan that might raise quality is still worth executing, and
mapping the cheap/expensive extremes early is still worth paying for. `step_cost_usd` and
`total_opt_cost_usd` are written to the trajectory CSV in **both** modes, so a cost-aware run
and a default run can be compared step by step; `metrics/final_eval.json` records which mode a
run used under `run_config.core_agent.cost_aware_opt`.

### Per-operator feedback ablation (`--no-per-operator-info`)

`--no-per-operator-info` leaves the agent with plan-level results only (`quality`,
`quality_counts`, cost, latency, tokens): no `op_summary` in the `execute_plan` result, no
`op_results` store, and no `get_op_samples` tool. It is only supported in `execute` mode. The
per-operator numbers are still written to each plan's `opt_results/<run>/<plan>/plan_result.json`
for later analysis. `metrics/final_eval.json` records it under `core_agent.per_operator_info`.

## The search space

A plan is a `PhysicalPipeline`: a linear (with an optional join side-branch) sequence of
operators built by chaining methods, e.g.:

```python
p = PhysicalPipeline(plan_name, "reviews", load_data("Reviews.csv"))
p.sem_filter("this review expresses positive sentiment", model=pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE)
p.project(["reviewId"])
p.limit(10)
p
```

**Semantic operators** (LLM-backed, cost real $/latency):

| operator | signature |
|---|---|
| `sem_filter` | `(condition: str, model: pz.Model, depends_on=None)` — keeps rows where the natural-language condition holds |
| `sem_map` | `(cols: list[dict], model: pz.Model, depends_on=None)` — adds LLM-derived columns, `cols=[{"name","type","description"}, ...]` |
| `sem_join` | `(other: PhysicalPipeline, condition: str, model: pz.Model, depends_on=None)` — keeps pairs where the natural-language condition holds; self-join when `other is pipeline` |

**Non-semantic operators** (exact/Python, free):

| operator | signature |
|---|---|
| `filter` | `(fn: Callable[[dict], bool])` |
| `map` | `(udf: Callable[[dict], dict], cols: list[dict])` |
| `join` | `(other, condition_fn: Callable[[dict, dict], bool])` — right-side name collisions get `_right` suffixed |
| `add_col_suffix` | `(suffix: str)` — used ahead of a join to disambiguate column names |
| `project` | `(cols: list[str])` |
| `limit` | `(n: int)` |
| `groupby` | `(group_by_fields, agg_funcs, agg_fields)` |

The model catalog (`available_models_openai.txt`, injected into the system prompt) lists every
`pz.Model` the agent can pick per-operator, grouped by cost/speed tier with $/1M-token
input+output prices and per-output-token latency — this, plus `depends_on` (restrict which
input columns a semantic op actually needs, shrinking its prompt) and operator ordering/
pruning, is the lever set the agent searches over. There is no separate plan-enumeration
algorithm — the LLM itself proposes each next candidate plan in code, informed by the
accumulating `plan_results`/`op_results` tables.

The default catalog is OpenAI-only, because those models go through litellm and so through
OpenRouter (OpenAI models go to OpenAI directly; see `raptor/routing.py` for which key each model uses). `available_models.txt` is the
full catalog (Gemini and Llama as well); palimpzest hands Gemini models a direct `google-genai`
client that bypasses litellm entirely, so those require `GEMINI_API_KEY` too and bill Google
rather than OpenRouter. To use it: `AGENT_MODEL_CATALOG=available_models.txt`.

## The plan quality evaluator

`PlanQualityEvaluator` (subclassed per benchmark — see `benchmarks/*/quality_evaluator.py`) is
benchmark-agnostic and produces a `QualityResult(quality, quality_note, quality_counts,
quality_judge_cost_usd)` for every `execute_plan` call: the plan's output vs. the benchmark's
ground truth, using the benchmark's own metric (F1, relative error → `1/(1+err)`, Spearman
correlation, or a generic accuracy score, selected by result type).

A benchmark whose scorer reads a ground-truth FILE (CUAD, sustainability, SummHay) is handed
benchmark.yaml's `ground_truth_path`. One whose scorer wants a DataFrame (SemBench) supplies a
`ground_truth_loader`, loaded once per run. SemBench's **regenerates** the ground truth by running
the benchmark's own gold query — the movie use case's `gold_sql/Q{id}.sql` via DuckDB over named
domain tables, or ecomm's per-query `[definition].ground_truth` SQL embedded in
`queries/q{id}.toml` over its `styles_details.parquet` — restricted to exactly the rows the
optimization datasubset sampled (every full-dataset table is joined against the datasubset's row
ids; a table with no matching id column comes back empty rather than left unrestricted, so a query
that doesn't touch the sampled table at all correctly regenerates an empty ground truth instead of
silently scoring against unsampled data). It is regenerated on every optimization run and written
next to the datasubset CSV as `datasubset_opt/{use_case}/sf_{sf}/{subset stem}_gt.csv` — e.g.
`Q7_subset_agentic_gt.csv`, so the `--sample-method` subsets of a query don't collide on one
filename. It falls back to the benchmark's precomputed full-dataset ground-truth CSV when no gold
query is available for that use case/query.

## Wiring your own LLM / benchmark

- `LLMClient` is a one-method protocol (`generate(system, messages) -> str`);
  `RoutedClient` (in `llm_client.py`) is the default implementation, sent to the API `routing.py`
  assigns each model, used for the main agent, the exploration checker, and the cheap image
  describer.
- A new benchmark is a new directory under `benchmarks/`, and needs nothing changed in
  `raptor/`:
  1. `benchmark.yaml` — query source, metrics, task prompts, dataset/ground-truth paths,
     `external_repo` (if it reads from a checkout), and `results_prefix` (which axes its
     results path carries).
  2. `quality_evaluator.py` — a `QualityEvaluator` subclass of `PlanQualityEvaluator`
     implementing three hooks: `score_plan`, `has_ground_truth` and `write_scoring_input`. It
     takes `subset_path` / `ground_truth_path` from the config rather than re-deriving them, plus
     `normalize_df` (how a plan's output is reshaped for your scorer) and, if your scorer wants a
     DataFrame, `ground_truth_loader`.
  3. `paths.py` — any file locations only that adapter reads.
  4. a data-prep script producing the per-query source CSV `run_opt.py` expects.

  **See [`benchmarks/README.md`](benchmarks/README.md)** for every `benchmark.yaml` key and
  who reads it, what each adapter hook is called with, the path-template variables, where a run
  writes, and a worked example.
