# Conversational Consumer Selection V1

`conversational_consumer_selection_v1` is a benchmark module for platform-side
guided selection over a small candidate set.

Within this repository, it serves as a cleaner benchmark for studying
platform-side control when user intent is only partially observable and
decisions unfold over multiple turns. For the overall project context, see the
root [`README.md`](../../README.md).

## Purpose

This benchmark is designed for settings where:

- user intent is incomplete at the beginning of the interaction
- the platform must act over multiple turns
- success depends on both language behavior and decision quality

It is intended as a controlled testbed for studying guided selection under
different intent-visibility regimes rather than as a replacement for real
marketplace logs or online A/B testing.

## Core Contract

The benchmark keeps the platform interface intentionally small:

- input: `Observation`
- output: `SelectionAction`

`Observation` includes:

- `revealed_context`
- `available_clarification_slots`
- `offers`
- `history`
- `turn_index`
- `max_turns`
- `remaining_turns`

`SelectionAction` supports:

- `ask_clarification(slot)`
- `compare_options(offer_id, comparison_offer_id)`
- `recommend_option(offer_id)`
- `commit_selection(offer_id)`
- `escalate()`

Anything richer than that, such as belief state or parser traces, is agent
internal rather than part of the benchmark contract.

## Modes

- `v0_structured`
  no dialogue in the contract; the platform acts on structured observations
- `v1_direct_intent`
  dialogue is present and the platform also receives full structured intent
- `v1_partial_intent`
  dialogue is present and the platform receives only partial structured intent
- `v2_hidden_intent`
  dialogue is primary and the environment judges against latent structured
  consumer preferences

## Task Structure

Each task separates:

- `CategorySchema`
  available constraints and preferences for the category
- `UserProfile`
  how the interaction starts
- `LatentConsumerModel`
  the hidden consumer model used by the simulator and the judge

The default examples use a `headphones` category with multiple offers at
different prices and attribute profiles.

## Quick Start

Run the structured demo:

```bash
cd benchmarks/conversational_consumer_selection_v1
PYTHONPATH=src python -m conversational_consumer_selection.single_agent_demo --backend demo
```

Run the dialogue demo with a model backend:

```bash
cd benchmarks/conversational_consumer_selection_v1
PYTHONPATH=src python -m conversational_consumer_selection.dialogue_demo \
  --backend openai \
  --model gpt-5.4-mini \
  --reasoning-effort none \
  --mode v1_partial_intent
```

Add `--debug-actions` if you want to inspect the structured control actions.

## Important Files

- [`src/conversational_consumer_selection/schemas.py`](src/conversational_consumer_selection/schemas.py)
  benchmark I/O contract and task structure
- [`src/conversational_consumer_selection/tasks.py`](src/conversational_consumer_selection/tasks.py)
  example task builders
- [`src/conversational_consumer_selection/env.py`](src/conversational_consumer_selection/env.py)
  environment transition and judgment logic
- [`src/conversational_consumer_selection/simulator.py`](src/conversational_consumer_selection/simulator.py)
  rule-based user simulator
- [`src/conversational_consumer_selection/agents/single_agent.py`](src/conversational_consumer_selection/agents/single_agent.py)
  minimal platform agent
- [`src/conversational_consumer_selection/single_agent_demo.py`](src/conversational_consumer_selection/single_agent_demo.py)
  `v0_structured` demo entry point
- [`src/conversational_consumer_selection/dialogue_demo.py`](src/conversational_consumer_selection/dialogue_demo.py)
  dialogue demo entry point

## Testing

From the repository root:

```bash
pytest benchmarks/conversational_consumer_selection_v1/tests/test_benchmark.py
```
