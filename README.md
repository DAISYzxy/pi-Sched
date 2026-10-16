# π-Sched: Multi-Agent Workflow Scheduling

This release contains the three method stages of π-Sched, a framework for scheduling natural-language agent tasks while preserving dependencies and minimizing execution latency.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Test](#test)
- [Evaluation](#evaluation)
- [Data](#data)

## Introduction

The code in `src/` implements the following pipeline:

1. **Task operationalization** (`parse_task.py`) uses an LLM to turn public agent requests into structured inputs, outputs, resource reads and writes, and side effects.
2. **Acyclic dependency construction** (`acyclic_dependency_build.py`) infers precedence and non-parallel constraints from those operations, then resolves cycles in the directed dependencies.
3. **Time-weighted scheduling** (`execution_evaluate.py`) assigns completion times that satisfy the constraints and minimize latency using each agent's `time` value. It also evaluates the resulting schedules.

## Installation

Use Python 3.10 or later. Clone this repository and install the dependency:

```bash
cd pi-Sched
python -m pip install -r requirements.txt
```

## Test

The release includes ten pre-parsed SGD-based examples. Run the dependency and scheduling stages without an API key:

```bash
python src/acyclic_dependency_build.py --input data/sgd_micro_10_parsed.json
python src/execution_evaluate.py run
```

To run the complete pipeline from public natural-language requests, set `LLM_API_KEY` or `OPENAI_API_KEY` in your environment, then run:

```bash
python src/parse_task.py run
python src/acyclic_dependency_build.py
python src/execution_evaluate.py run
```

The parser defaults to an OpenAI-compatible Chat Completions endpoint. Use `--model` and `--base-url` with `parse_task.py run` to select another compatible model and endpoint. The `--base-url` value must be the full Chat Completions URL.

## Evaluation

Both test workflows write these files to `results/`:

| File | Contents |
| --- | --- |
| `dependency.json` | Inferred mandatory edges and non-parallel pairs |
| `schedule.json` | Detailed schedules and per-case validation |
| `predictions.jsonl` | Predicted completion times for each case |
| `evaluation.json` | Aggregate evaluation metrics |

The complete pipeline also writes `parsed.json`, containing the structured task representations. To evaluate an existing prediction file against the included reference data, run:

```bash
python src/execution_evaluate.py evaluate --pred results/predictions.jsonl
```

The evaluation reports `edge_accuracy` (the fraction of reference precedence edges satisfied), `case_accuracy` (the fraction of cases with a valid schedule satisfying every reference edge), `avg_latency` (mean completion time of valid schedules), and `avg_running_time_ms` (mean scheduler/evaluator runtime per case). These numbers describe the included ten-case example set, not the full paper benchmark.

## Data

`data/sgd_micro_10_public.jsonl` contains ten public cases, each with natural-language agent requests and positive integer `time` values. `data/sgd_micro_10_full.jsonl` contains the corresponding reference answers used for evaluation. The examples are synthesized from the schemas of [Google's Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue); they are not original dialogue records. The source dataset's license is included in [LICENSE-SGD.txt](LICENSE-SGD.txt).
