# Cost-Model Agent (minimal)

A small tool-loop agent for research on whether agents can use **cost models** to
reason about query plans. The agent (1) **applies** a cost model to a physical
(sub)plan to estimate dollar cost / latency / quality, and (2) **updates** the
cost model — by authoring a new `CostModel` class — based on observed execution
behavior.

It's a stripped-down version of our `MultiTurnAgent` / `SearchAgent`: same
bounded loop + sandboxed-python tool calls, but **no** retrieval, and **no**
`TextBlock`/`ChunkBlock` trajectory (messages are plain `{"role", "content"}`
dicts).

## Layout

| path | what it is |
|------|------------|
| `cost_model_agent.py`, `local_python_executor.py`, `physical_pipeline.py`, `llm_sampler.py` | reusable cost-model implementation |
| `PHYSICAL_PIPELINE_DESIGN.md`, `available_models.txt` | model documentation and configuration |
| `oracle_quality_evaluator.py` | reusable oracle execution, memoization, and per-operator quality scoring |
| `experiments/SemBench/` | SemBench-specific drivers, output normalization, evaluator setup, datasets, and subsets |
| `results/` | generated cost models, final answers, metrics, sampling cache, and trajectories |
| `analysis/` | result-inspection scripts, notebook, and CSV analysis inputs |

`experiments/*/dataset/`, `experiments/*/datasubset*/`, and `results/` are ignored
by Git because they contain generated or large experiment artifacts. They are created
automatically by the corresponding scripts.

SemBench-dependent commands resolve the sibling `SemBench/` checkout by default.
Override its location when needed:

```bash
export SEMBENCH_ROOT=/absolute/path/to/SemBench
```

## Run it

```bash
# 1. from the SemPlan_Opt workspace root
# 2a. offline smoke test -- no API key or network required:
python3 -m agent_cost_model.experiments.SemBench.demo --offline

# 2b. real run via OpenRouter:
pip install openai
export OPENROUTER_API_KEY=sk-or-...
python3 -m agent_cost_model.experiments.SemBench.demo --model openai/gpt-5
```

The offline path is the fastest way to confirm your `local_python_executor.py`
works with the harness (it exercises sandbox class-authoring + every tool).

## How the loop works

Each step the LLM emits exactly one fenced block:

- a ` ```python ` block — a tool call, executed in the `LocalPythonExecutor`;
  its stdout/return value becomes the next observation; or
- a ` ```json ` block — the final answer (parsed as data, not executed).

The agent runs for up to `max_steps`, keeps a flat message trajectory
(`agent.messages`), and returns the parsed JSON final answer.

## Tools (available inside the python sandbox)

| tool | purpose |
|------|---------|
| `inspect_plan(plan)` | introspect a plan's operators (`op_id`, `op_type`, `model`, attrs) |
| `estimate_plan_cost(plan)` | apply the **currently installed** cost model → `PlanCostEstimate` |
| `update_cost_model(cost_model, notes="")` | install a cost-model **class or instance** the agent wrote |
| `execute_subplan(plan, n=5, seed=0)` | **stub** partial execution → appends observed per-op stats to `results` |

Also injected into the sandbox (no import needed): `plans`, `results`,
`PlanCostEstimate`, `CostModel`, and the helpers `iter_operators`,
`get_op_type`, `get_op_id`, `get_op_model`, `describe_operator`.

## The cost-model contract

The agent writes a class with one required method:

```python
class MyCostModel:                 # subclassing CostModel is optional
    def __init__(self, results):   # results store is passed in if __init__ accepts it
        ...                        # fit coefficients from results.rows / results.df
    def estimate_plan(self, plan) -> PlanCostEstimate:
        cost = time = 0.0
        for op in iter_operators(plan):
            ...                    # use get_op_type(op), get_op_model(op), etc.
        return PlanCostEstimate(cost=cost, time=time, quality=1.0)

update_cost_model(MyCostModel, notes="v1")
```

`update_cost_model` validates the `estimate_plan` method, instantiates the class
(passing the `results` store when the constructor accepts it), and version-stamps
it. `estimate_plan_cost` then applies the latest version. This is the
estimate → observe (`execute_subplan`) → update loop the project is about.

## The observed-results store

`results` (a `ResultsStore`) is an append-only log of **per-operator-invocation**
stats — one row per `(operator, single input)`. Canonical columns:

```
op_id, op_type, model, input_id, output,
cost_usd, latency_s, input_tokens, output_tokens
```

Read it as `results.rows` (list of dicts) or `results.df` (pandas, if installed).
`execute_subplan` appends new rows; in the demo it's seeded with a few real-ish
semantic-filter / map invocations.

## Swapping in real palimpzest plans

`cost_model_agent.py` does a **guarded import** of palimpzest
(`PhysicalPlan`, `PhysicalOperator`, `OperatorCostEstimates`, `PlanCost`) — it
uses them when present but never requires them. The harness only relies on a
plan being **iterable over its operators** (real `PhysicalPlan.__iter__` already
is) plus a few operator attributes, so to use real plans you just put them in the
`plans` dict passed to `agent.run(...)` — no agent changes needed. The
duck-typed `DemoPlan` / `DemoOp` in `demo.py` exist only so the demo runs without
a full palimpzest dataset.

`execute_subplan` is a **stub** that fabricates plausible stats so you can
develop the update logic offline; replace its body with real palimpzest
execution when you're ready.

## Wiring your own LLM

Implement the `LLMClient` protocol (one method):

```python
class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict]) -> str: ...
```

`OpenRouterClient` (in `cost_model_agent.py`) is the default, via the OpenAI-
compatible OpenRouter endpoint. `ScriptedLLM` (in `demo.py`) shows the offline
stand-in.
