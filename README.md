# Cost-Model Agent
## What it does

Given a natural-language query task and a source dataset, `CostModelAgent.run()` runs a
bounded tool-loop LLM agent that:

1. explores the dataset (schema, samples, images, aggregate stats),
2. writes a candidate **physical plan** — a tree of operators, some exact/Python, some
   semantic/LLM-backed — as Palimpzest-style pipeline-builder code,
3. **executes** that plan on a small, fixed data subset and observes real cost ($),
   latency (s), token usage, and **quality** (0–1, scored against an oracle or real
   ground truth — see below),
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
| `opt_agent/` | **the engine** — benchmark-agnostic. Knows nothing about SemBench, CUAD, or MOAR. |
| `opt_agent/cost_model_agent.py` | the agent: system prompt, tools, sandboxed step loop, final evaluation |
| `opt_agent/physical_pipeline/` | `PhysicalPipeline` — the plan IR/executor: operators, subset sampling, oracle-copy construction |
| `opt_agent/local_python_executor.py` | sandboxed Python executor the agent's tool calls run in |
| `opt_agent/llm_sampler.py` | builds the optimization datasubset (random or top-k / embedding-based sampling) |
| `oracle_quality_evaluator.py` | benchmark-agnostic quality scoring (oracle- or ground-truth-based) |
| `sample_based_cost_model.py` | optional cost model used only by the legacy `sampleCost` mode |
| `paths.py` | this repo's own locations — five constants, no benchmark names |
| `experiments/run_opt.py` | CLI entry point: wires a benchmark config to one `CostModelAgent.run()` call |
| `experiments/config.py` | loads a `benchmark.yaml` and resolves its path templates (used by the runner *and* by `analysis/`, so they can't drift) |
| `experiments/external_repo.py` | resolves the external checkouts and puts them on `sys.path` |
| `experiments/SemBench/`, `experiments/cuad/` | **the benchmark adapters** — `benchmark.yaml`, `quality_evaluator.py`, `paths.py`, data prep, datasets, datasubsets |
| `results/` | everything any run produces (see below) |
| `analysis/` | result-inspection scripts/notebooks over `results/` |

`experiments/*/dataset/`, `experiments/*/datasubset*/`, `experiments/*/ground_truth/`, and
`results/` are gitignored (generated/large artifacts, created by the scripts below).

### External benchmark checkouts

SemBench and docetl are used **in place, never vendored**, so their own evaluators stay the
single source of truth for how a benchmark is scored. Neither is importable as an installed
package for our purposes: SemBench ships no packaging metadata and its modules import each
other as top-level packages rooted at `src/`; docetl is packaged but its build excludes
`experiments/**`, which is where the CUAD scorer lives. Each benchmark therefore declares its
checkout in its own `benchmark.yaml`:

```yaml
external_repo:
  env_var: SEMBENCH_ROOT     # override the location
  default: ../SemBench       # relative to this repo's root
  python_path: src           # subdir to place on sys.path
```

`experiments/external_repo.py` is the only code that puts an external repo on `sys.path`.
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

`{runner}` names one optimizer or variant — `execute_oracle_sampler`, `customCost`,
`execute_oracle`, `sampleCost`, … alongside external ones like `MOAR`. It is the axis you
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
# offline smoke test — exercises the sandbox + tools with a scripted LLM (no network calls,
# but OPENROUTER_API_KEY must still be set: the oracle client is constructed unconditionally)
OPENROUTER_API_KEY=sk-or-dummy python3 -m agent_cost_model.experiments.SemBench.demo --offline

# a real optimization run, via the SemBench-specific CLI wrapper
export OPENROUTER_API_KEY=sk-or-...
python3 -m agent_cost_model.experiments.run_opt \
    --config agent_cost_model/experiments/SemBench/benchmark.yaml \
    --use-case movie --query-id 7 --scale-factor 2000 \
    --opt-subsample   # first run only: builds the optimization datasubset
```

`run_opt.py` reads `benchmark.yaml` for query/data/evaluation details (task prompt, quality
metric, dataset paths, ground-truth location, and the `results_prefix` template) and owns the
optimization *policy* in code (model, `max_steps`, agent mode, subset size/seed) so the same
runner works for another benchmark by pointing `--config` elsewhere. `AGENT_TYPE` is the
runner name that lands in the results path.

## The optimization loop

`CostModelAgent.run()` is a bounded loop (`max_steps`, default 40). Each step the LLM
emits **exactly one** fenced block:

- a ` ```python ` block — one tool call, executed in a `LocalPythonExecutor` sandbox;
  its stdout/return value becomes next step's observation, or
- a ` ```json ` block — the final answer (parsed as data, not executed), which ends the loop.

The suggested/enforced workflow (see the `execute_briefing` system-prompt text) is:

1. **Explore** — `list_files()`, `explore_schema(f)`, `explore_sample(f)` for a quick look;
   `explore_data(f)` for a full-CSV pandas DataFrame to compute aggregate statistics
   (keyword frequency, `value_counts()`, class balance) that inform filter selectivity,
   join sizes, and class distribution — used to *design* plans, never to hardcode an
   answer. `explore_images(ids)` shows up to 5 images (via a cheap vision model, which
   returns a text description — pixels never reach the main agent) to gauge what a
   vision operator would see.
2. **Write** a plan — `write_plan(code, name, description="")`. `code` is a Python string,
   run in a *separate*, minimal sandbox (only `PhysicalPipeline`, `load_data`,
   `add_image_data`, `pz` — no results stores, no `explore_data`, so plan code can never
   depend on directly-scanned data) that builds and returns a `PhysicalPipeline`.
   Hardcoding row IDs/values found while exploring is explicitly disallowed — plans must
   stay general.
3. **Execute** — `execute_plan(name)` runs the plan on a reproducible data subset (same
   records across every plan in the run), appends one row to `plan_results` (cost, latency,
   tokens, `quality`, `per_sem_op_quality`) and one row per operator invocation to
   `op_results`.
4. **Evaluate** — read `plan_results.df` / `op_results.df`. `quality` is the oracle/
   ground-truth score (treated as ground truth — the agent is told not to try to
   reverse-engineer it); `per_sem_op_quality` isolates which operator is the bottleneck,
   with `get_op_samples(plan_name)` to inspect actual input/output pairs.
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

The model catalog (`available_models.txt`, injected into the system prompt) lists every
`pz.Model` the agent can pick per-operator, grouped by cost/speed tier with $/1M-token
input+output prices and per-output-token latency — this, plus `depends_on` (restrict which
input columns a semantic op actually needs, shrinking its prompt) and operator ordering/
pruning, is the lever set the agent searches over. There is no separate plan-enumeration
algorithm — the LLM itself proposes each next candidate plan in code, informed by the
accumulating `plan_results`/`op_results` tables.

### Modes

`run(..., mode=...)` selects the system-prompt briefing and tool set:

- **`execute*`** (e.g. `execute_oracle_sampler`, used by `run_opt.py`) — the primary mode.
  No cost-model tools at all; the agent just writes, executes, and compares plans directly
  against observed cost/latency/quality.
- **`sampleCost*`** — a `SampleBasedCostModel` is pre-installed and re-fits on every
  `estimate_plan_cost` call, fit from `op_results`' running average per (op, model); the
  agent is told to estimate before executing.
- **`customCost*`** — cost-model authoring is fully delegated to a separate
  `CostHelperAgent` (its own small LLM loop) that fits/refits a `CostModel` after each
  execution batch and returns a relative cost/latency estimate table via `review_plans()`.
  The main agent never writes or sees cost-model code in this mode.

The latter two exist for research comparisons against the cost-model-free approach; they
are not what `run_opt.py` uses today.

## The oracle quality evaluator

`OracleQualityEvaluator` (subclassed per benchmark — see
`experiments/SemBench/quality_evaluator.py`) is benchmark-agnostic and produces a
`QualityResult(quality, per_sem_op_quality, quality_note)` for every `execute_plan` call.
It scores two independent things:

**1. Per-operator quality** — always computed the same way, regardless of ground-truth
mode, from nothing but the plan's *own* execution samples (`plan_context.per_sem_op_info`,
populated by `PhysicalPipeline.run_subset` — every semantic op's actual (input, output)
pairs from this run):

- `sem_map` — one batched oracle call per operator: shown each sampled (input, output)
  pair plus the operator's output field names, the oracle scores each field 1/0 correct;
  the per-op score is the mean over fields and records. Image-valued input fields are
  attached as vision inputs so the oracle can verify vision-derived outputs.
- `sem_filter` / `sem_join` — one batched oracle call per operator: shown each sampled
  input (or joined pair) plus the operator's natural-language condition, the oracle
  independently decides true/false for each; the per-op score is the agreement rate
  between the oracle's decisions and the plan's own pass/fail decisions. `sem_join`'s raw
  sample list keeps *every* candidate pair from the subset (up to n², unlike filter, which
  is capped upstream), so it's randomly downsampled to `max_pairs` before the judge call.

**2. Plan-level quality** — the plan's output DataFrame vs. a ground-truth DataFrame,
using the benchmark evaluator's own metrics (F1, relative error → `1/(1+err)`, Spearman
correlation, or a generic accuracy score, selected by result type). The ground truth comes
from one of two sources, chosen by `use_oracle_ground_truth` (constructor arg on
`OracleQualityEvaluator`/`CostModelAgent`, default `True`):

- **oracle-generated** (`True`) — `plan.make_oracle_copy(oracle_model)` rebuilds the exact
  same plan with every semantic operator's model swapped for a stronger oracle model, runs
  it once on the same data subset, and uses its output as ground truth. Cached to disk per
  `plan_name` (`llm_judge_dir/Q{id}_{plan}_gt.csv`) so repeat evaluations of the same plan
  are free, and the **first** plan's oracle output becomes the shared "canonical" ground
  truth reused for every later plan's scoring. On top of that, individual oracle LLM calls
  are memoized *across plans* by (model, condition, input content) — two plans sharing a
  semantic operator over identical inputs pay for that operator's oracle judgment only
  once (`_MemoizingGenerator`).
- **directly supplied** (`False`) — skips running the oracle-substituted plan entirely and instead
  loads real, already-known ground truth via a `ground_truth_loader` callable. For
  SemBench this **regenerates** ground truth by running the benchmark's own gold
  query — the movie use case's `gold_sql/Q{id}.sql` via DuckDB over named domain tables,
  or ecomm's per-query `[definition].ground_truth` SQL embedded in `queries/q{id}.toml`
  over its `styles_details.parquet` — restricted to exactly the rows the optimization
  datasubset sampled (every full-dataset table is joined against the datasubset's row ids;
  a table with no matching id column comes back empty rather than left unrestricted, so a
  query that doesn't touch the sampled table at all correctly regenerates an empty ground
  truth instead of silently scoring against unsampled data). Regenerated ground truth is
  cached as `datasubset_opt/{use_case}/sf_{sf}/Q{id}_gt.csv`, next to the datasubset CSV
  itself, and reused across the whole run. Falls back to the benchmark's precomputed
  full-dataset ground-truth CSV when no gold query is available for that use case/query.

Either way, `total_oracle_cost_usd` on the evaluator accumulates every real oracle LLM
call made (full pipeline run, per-op judge calls, or both) and rolls up into the agent's
`oracle_cost_usd` after each run.

## Wiring your own LLM / benchmark

- `LLMClient` is a one-method protocol (`generate(system, messages) -> str`);
  `OpenRouterClient` (in `cost_model_agent.py`) is the default OpenRouter-backed
  implementation used for the main agent, the oracle, and the cheap image describer.
- A new benchmark is a new directory under `experiments/`, and needs nothing changed in
  `opt_agent/`:
  1. `benchmark.yaml` — query source, metrics, task prompts, dataset/ground-truth paths,
     `external_repo` (if it reads from a checkout), and `results_prefix` (which axes its
     results path carries). See `experiments/SemBench/benchmark.yaml`.
  2. `quality_evaluator.py` — a `QualityEvaluator` subclass of `OracleQualityEvaluator`,
     taking `subset_path`/`ground_truth_dir` from the config rather than re-deriving them,
     plus `normalize_df`, `evaluator_factory`, and optionally `ground_truth_loader` for
     direct-ground-truth mode.
  3. `paths.py` — any file locations only that adapter reads.
  4. a data-prep script producing the per-query source CSV `run_opt.py` expects.
