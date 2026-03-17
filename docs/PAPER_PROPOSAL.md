# Paper Proposal

## Working Title

`Agent Infrastructure Matters: Organizational Control for Reliable Multi-Agent Economic Systems`

## Motivation

LLM agents are increasingly expected to handle economic workflows such as negotiation, transaction, fulfillment, and after-sales support. In long-horizon and high-constraint settings, their failure mode is often not just weak reasoning, but weak organizational control.

AgenticPay already provides a negotiation-oriented benchmark substrate with private values, multi-turn bargaining, and outcome metrics such as feasibility, efficiency, and welfare. This project builds on top of that substrate and asks a different question:

Can an explicit organizational control layer improve reliability, compliance, and economic efficiency in multi-agent economic systems?

## Core Idea

We introduce an Organizational Control Layer (OCL) between language agents and the underlying environment.

The OCL does not change the base LLM itself. Instead, it changes system behavior through:

- role decomposition
- permission and constraint checks
- risk gating
- audit trails
- escalation and replanning
- attribution and incentives

In implementation terms, the OCL is the missing control plane between raw language actions and executable economic actions.

## Formalization Target

The intended formalization follows the proposal structure:

- Economic Multi-Agent Task:
  - state
  - agent actions
  - observations
  - constraints
  - utilities
- Organizational Control Layer:
  - `pi_role`
  - `pi_gate`
  - `pi_audit`
  - `pi_assign`
  - `pi_escalate`
- Control mapping:
  - `g_Pi: (m_1:t, a_raw_1:t) -> a_exec_1:t`

The codebase should eventually expose these objects explicitly rather than hiding them inside prompts.

## Hypotheses

- `H1`: OCL improves task success and feasibility.
- `H2`: OCL reduces policy and constraint violations.
- `H3`: OCL improves cost-adjusted economic return.

## Method Components

The project is intended to implement the following method components:

1. Role decomposition
2. Risk gating
3. Audit trail generation
4. Escalation and replanning
5. Attribution and incentive assignment

## Experimental Structure

The intended core offline setup is:

1. `Single-agent`
2. `Flat multi-agent`
3. `OCL multi-agent`

Primary metrics:

- success or feasibility
- constraint satisfaction
- turns and latency
- token cost
- welfare
- cost-adjusted welfare
- violation rate

Additional experiment families:

- ablations over OCL components
- adversarial robustness
- repeated interactions with reputation and credit assignment
- later offline and online evaluation for AiMai

## Repository Implications

This repository is structured as a control-layer project, not as a fork-level rewrite of AgenticPay.

That means:

- AgenticPay remains the lower-level substrate
- `aimai_ocl` contains the control plane and evaluation harness
- AgenticPay integration should happen through adapters and wrapped environments
- AgenticPay is expected to be importable in the active Python environment
- new logic should favor explicit schemas and event traces over prompt-only conventions

## Build Roadmap

### Phase 1: Foundations

- action schemas
- constraint schemas
- audit schemas
- trace schemas
- AgenticPay adapters

### Phase 2: OCL Control Plane

- role policy
- risk gate
- audit logger
- escalation manager
- coordinator

### Phase 3: Evaluation

- benchmark runner
- single-agent baseline wrapper
- flat multi-agent baseline wrapper
- OCL multi-agent wrapper
- ablation runner

### Phase 4: Attribution and Incentives

- contribution records
- trajectory-level credit assignment
- heuristic incentive coordination
- repeated-interaction support

## Current Status

Current repository status:

- sibling-project scaffold completed
- AgenticPay runtime import boundary completed
- proposal-aligned directory structure started
- v1 execution scope currently limited to single-product negotiation runs
- full OCL implementation not started yet
