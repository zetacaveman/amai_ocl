# AiMai OCL

`aimai_ocl` is an independent project that builds an Organizational Control Layer (OCL) for multi-agent economic systems.

The project follows the direction of the paper:

`Agent Infrastructure Matters: Organizational Control for Reliable Multi-Agent Economic Systems`

The core idea is to use AgenticPay as an external benchmark substrate, while `aimai_ocl` serves as the control-layer implementation and research workspace for:

- organizational control
- role decomposition
- risk gating
- audit trails
- escalation and replanning
- attribution and incentives
- reproducible benchmarks

## Project Positioning

This repository is independent from AgenticPay and from the group that developed AgenticPay.

In this project, AgenticPay is used as:

- a task and environment library
- a negotiation state machine
- a baseline evaluation substrate

And it adds the missing control plane required by the paper proposal:

- raw action to executable action mapping
- explicit constraints and risk checks
- standardized audit traces
- escalation logic
- attribution and incentive signals

## Proposal Alignment

The project structure is designed to match the paper proposal and experiment plan:

- `Single-agent` baseline
- `Flat multi-agent` baseline
- `OCL multi-agent` system
- ablations over `role`, `gate`, `audit`, `escalation`, and `attribution`
- adversarial and long-horizon evaluation
- later mapping to AiMai platform experiments

The full planning document lives in [`docs/PAPER_PROPOSAL.md`](docs/PAPER_PROPOSAL.md).

## Core Algorithms (Paper-Critical)

### 1) Role Decomposition

- Algorithm IDs:
  - `role_v1_rule`
  - `role_v1_seller_only`
  - `role_v2_state_machine`
- Core formulation:

$$
\begin{aligned}
\mathrm{phase}_t &= f_{\mathrm{phase}}(\mathrm{round}_t, \mathrm{buyer}_t, \mathrm{max\_rounds}) \\
\mathrm{decision\_role}_t &= \pi_{\mathrm{role}}(\mathrm{phase}_t) \\
\mathrm{execution\_role}_t &= A_s \\
\mathrm{escalation\_role}_t &= A_p
\end{aligned}
$$

`role_v2_state_machine` uses explicit phases:
`discovery / recommendation / negotiation / closing / escalation`.

- Implementations:
  - [`aimai_ocl/controllers/coordinator.py`](aimai_ocl/controllers/coordinator.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

### 2) Risk Gating as a Control Barrier

- Algorithm IDs:
  - `gate_v1_default`
  - `gate_v1_strict`
  - `gate_v1_lenient`
  - `gate_v2_barrier`
  - `gate_v2_barrier_strict`
- Core formulation:

$$
\rho_t = f_{\mathrm{risk}}(a_t^{\mathrm{raw}}, s_t, h_t), \quad \rho_t \in [0, 1]
$$

$$
\mathrm{action}_t =
\begin{cases}
\mathrm{approve}, & \rho_t < \tau_{\mathrm{rewrite}} \\
\mathrm{rewrite/confirm}, & \tau_{\mathrm{rewrite}} \le \rho_t < \tau_{\mathrm{block}} \\
\mathrm{block/escalate}, & \rho_t \ge \tau_{\mathrm{block}}
\end{cases}
$$

The barrier formulation includes an explicit miss-probability proxy `epsilon_miss`
for high-risk false-pass analysis.

- Implementations:
  - [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)
  - [`aimai_ocl/controllers/ocl_controller.py`](aimai_ocl/controllers/ocl_controller.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

### 3) Attribution and Incentives

- Algorithm IDs:
  - `shapley_v1_exact`
  - `shapley_v1_reward_only`
  - `counterfactual_v1`
- Core formulation:

$$
\begin{aligned}
V(\mathrm{trace}) &= w_s \cdot \mathrm{success}
+ w_r \cdot \mathrm{seller\_reward}
+ w_g \cdot \mathrm{global\_score} \\
&\quad - w_v \cdot \mathrm{violations}
- w_e \cdot \mathrm{escalations}
- w_t \cdot \mathrm{rounds}
\end{aligned}
$$

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}\left(V(S \cup \{i\}) - V(S)\right)
$$

$$
w_i = \frac{\max(\phi_i, 0)}{\sum_j \max(\phi_j, 0)}
$$

`counterfactual_v1` supports sparse coalition values with Monte-Carlo Shapley
estimation.

- Implementations:
  - [`aimai_ocl/attribution_shapley.py`](aimai_ocl/attribution_shapley.py)
  - [`aimai_ocl/attribution_counterfactual.py`](aimai_ocl/attribution_counterfactual.py)
  - [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

## AgenticPay Dependency

`aimai_ocl` is not a downstream or affiliated AgenticPay project. It only
depends on AgenticPay as an external benchmark/runtime package.

This project assumes AgenticPay is importable in the active Python
environment. The recommended setup is:

```bash
pip install "git+https://github.com/SafeRL-Lab/AgenticPay.git"
```

After that, `aimai_ocl` uses standard `import agenticpay` imports with no local
path bootstrap layer.

## Model Runtime Configuration

Model selection is centralized in
[`aimai_ocl/model_runtime.py`](aimai_ocl/model_runtime.py). The current
runtime is intentionally OpenAI-only and uses AgenticPay's
`agenticpay.models.openai_llm.OpenAILLM` directly.

Run-level model fields are defined in
[`RunConfig`](aimai_ocl/experiment_config.py):

- `model`

Environment defaults (optional):

- `AIMAI_MODEL`
- `OPENAI_MODEL`
- `OPENAI_API_KEY` (required)

CLI overrides are available in:

- [`scripts/run_demo.py`](scripts/run_demo.py)
- [`scripts/run_batch_eval.py`](scripts/run_batch_eval.py)
- [`scripts/run_shapley_min.py`](scripts/run_shapley_min.py)

## AgenticPay Data Contract (What Is Passed)

`aimai_ocl` treats AgenticPay as a strict input/output boundary. The runtime
currently forwards and consumes the following payloads:

1. `agenticpay.make(env_id, **env_kwargs)`
- `env_id` selects the task (for example `Task1_basic_price_negotiation-v0`).
- `env_kwargs` are forwarded without remapping by the adapter. Common fields:
  `buyer_agent`, `seller_agent`, `max_rounds`, `initial_seller_price`,
  `buyer_max_price`, `seller_min_price`.

2. `env.reset(**reset_kwargs)`
- `reset_kwargs` are forwarded after `enforce_single_product_scenario`.
- v1 requires exactly one product:
  `product_info` dict, or `products` with exactly one item (normalized).
- Typical reset fields:
  `user_requirement`, `product_info.name`, `product_info.price`, `user_profile`.

3. Per-round actor calls and `env.step(...)`
- Runner reads `observation["current_round"]` and
  `observation["conversation_history"]`.
- Buyer path:
  `buyer_agent.respond(conversation_history, current_state)` -> text ->
  pass-through `buyer_action`.
- Seller path (`single`):
  seller text is audited and then passed through.
- Seller path (`ocl_full`):
  seller text -> `RawAction` -> role/constraint/risk/escalation ->
  `ExecutableAction.final_text`.
- Final call into AgenticPay each round:
  `env.step(buyer_action=<str|None>, seller_action=<str|None>)`.

4. Terminal episode info (`info` from the final step)
- `aimai_ocl` currently consumes:
  `status`, `round`, `termination_reason`, `agreed_price`,
  `buyer_price`, `seller_price`, `buyer_reward`, `seller_reward`,
  `global_score`, `buyer_score`, `seller_score`.
- These are copied into `EpisodeTrace.final_metrics` and CLI result records.

## Where To Edit (Without Reading AgenticPay Internals)

If you want to change what is passed, checked, or controlled, use these files
as stable integration points:

- Environment boundary and API forwarding:
  [`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py),
  [`aimai_ocl/adapters/agenticpay_env.py`](aimai_ocl/adapters/agenticpay_env.py)
- Text-to-action mapping (`RawAction` parsing, intent/price extraction):
  [`aimai_ocl/adapters/agenticpay_actions.py`](aimai_ocl/adapters/agenticpay_actions.py)
- Scenario payload contract (`reset_kwargs` validation/normalization):
  [`aimai_ocl/runners/scenario_validation.py`](aimai_ocl/runners/scenario_validation.py)
- Buyer/seller data flow and control insertion point:
  [`aimai_ocl/runners/single_episode.py`](aimai_ocl/runners/single_episode.py),
  [`aimai_ocl/runners/ocl_episode.py`](aimai_ocl/runners/ocl_episode.py)
- Control behavior (pass/rewrite/block/escalate):
  [`aimai_ocl/controllers/constraint_engine.py`](aimai_ocl/controllers/constraint_engine.py),
  [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py),
  [`aimai_ocl/controllers/escalation_manager.py`](aimai_ocl/controllers/escalation_manager.py)
- CLI argument wiring to runtime payloads:
  [`scripts/run_demo.py`](scripts/run_demo.py)

If AgenticPay updates its observation/info schema, these are the first files
to adjust.

## Current Status

The repository now includes a runnable experiment harness, not only scaffold code:

- baseline arms: `single`, `flat_multi`, `ocl_full`
- pluggable algorithms: `role`, `gate`, `escalation`, `attribution`
- algorithm bundles: `v1_default`, `v1_role_ablation`, `v2_research`
- protocol outputs: `main`, `ablation`, `adversarial`, `repeated`, `roi`
- deterministic tests for controller contracts and algorithm bodies

## Minimal AgenticPay Usage

The current minimal integration path is:

```python
from aimai_ocl import run_single_negotiation_episode

trace, final_info = run_single_negotiation_episode(
    env_id="Task1_basic_price_negotiation-v0",
    buyer_agent=buyer_agent,
    seller_agent=seller_agent,
    env_kwargs={
        "max_rounds": 10,
        "initial_seller_price": 180.0,
        "buyer_max_price": 120.0,
        "seller_min_price": 90.0,
    },
    reset_kwargs={
        "user_requirement": "I need a winter jacket",
        "product_info": {"name": "Winter Jacket", "price": 180.0},
        "user_profile": "Budget-conscious and compares options before buying.",
    },
)
```

At this stage, the runner is intentionally thin. Its job is to make the
AgenticPay benchmark call path explicit while reusing existing AgenticPay
buyers and sellers such as `SellerAgent` or `CollaborativeSellerAgent`.

## Quick Demo Script

Current v1 scope supports single-product negotiation only.

You can run one baseline episode from this repo directly:

```bash
export AIMAI_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
python scripts/run_demo.py --arm single --seed 42
```

Run OCL-controlled mode:

```bash
python scripts/run_demo.py --arm ocl_full --seed 42
```

Run flat multi-agent baseline (collaborative seller, no OCL control):

```bash
python scripts/run_demo.py --arm flat_multi --seed 42
```

In `ocl_full`, buyer actions are treated as external user-simulator inputs
and passed directly to the environment, while seller actions are passed through
OCL control.

### How `ocl_full` Works (for reporting)

`ocl_full` and `flat_multi` differ along two independent axes:

- Control architecture (`runner_mode`):
  - `single`: baseline pass-through runner
  - `ocl`: OCL runner with role planning + risk gate + escalation + audit
- Env-facing seller implementation (`env_seller_impl`):
  - `single`: AgenticPay `SellerAgent`
  - `collaborative`: AgenticPay `CollaborativeSellerAgent` (if available)

Therefore:

- `flat_multi` = `runner_mode=single` + `env_seller_impl=collaborative`
  (AgenticPay-native collaborative baseline, no OCL control layer)
- `ocl_full` = `runner_mode=ocl` + `env_seller_impl=single`
  (OCL multi-role collaboration/control layer on top of a single env-facing seller)

Why `ocl_full` keeps `env_seller_impl=single`:

- It isolates OCL effects (role decomposition, gating, escalation, attribution)
  without mixing in AgenticPay's internal collaborative seller orchestration.
- This keeps the ablation comparison cleaner (`single` vs `flat_multi` vs `ocl_full`).

Compare baseline vs OCL:

```bash
python scripts/run_demo.py --arm single --seed 42
python scripts/run_demo.py --arm ocl_full --seed 42
```

`--arm` fields are treated as ablation contracts; some toggles are scaffold
flags and will be progressively wired into behavior in later steps.

Preview resolved config without running model calls:

```bash
python scripts/run_demo.py --arm ocl_full --seed 42 --dry-run
```

Export full episode audit trace to JSON:

```bash
python scripts/run_demo.py --arm ocl_full --seed 42 --trace-json outputs/trace.json
```

Select explicit algorithm/protocol bundles (step-9/10 modular slots):

```bash
python scripts/run_demo.py \
  --arm ocl_full \
  --algorithm-bundle v2_research \
  --experiment-protocol offline_v1 \
  --seed 42
```

Override only one algorithm component (for ablation/debug):

```bash
python scripts/run_demo.py \
  --arm ocl_full \
  --algorithm-bundle v1_default \
  --role-algorithm role_v1_seller_only \
  --gate-algorithm gate_v1_strict \
  --attribution-algorithm shapley_v1_reward_only \
  --seed 42
```

Run paired batch evaluation (`single` vs `ocl_full`) with CSV/JSON reports:

```bash
python scripts/run_batch_eval.py --episodes-per-arm 5 --seed-base 42
```

Run all three main offline arms (`single`, `flat_multi`, `ocl_full`):

```bash
python scripts/run_batch_eval.py --arms single,flat_multi,ocl_full --episodes-per-arm 5
```

Override bundle/protocol ids for the whole batch:

```bash
python scripts/run_batch_eval.py \
  --arms single,flat_multi,ocl_full \
  --episodes-per-arm 5 \
  --algorithm-bundle v2_research \
  --experiment-protocol offline_v1
```

Component-level overrides are also supported in batch mode:

```bash
python scripts/run_batch_eval.py \
  --arms single,flat_multi,ocl_full \
  --episodes-per-arm 5 \
  --algorithm-bundle v1_default \
  --role-algorithm role_v1_seller_only \
  --gate-algorithm gate_v1_strict \
  --escalation-algorithm escalation_v1_no_replan \
  --attribution-algorithm shapley_v1_reward_only
```

Choose output location and save per-run trace files:

```bash
python scripts/run_batch_eval.py \
  --episodes-per-arm 10 \
  --output-dir outputs/batch_eval_v1 \
  --save-traces
```

Batch artifacts now include `protocol_outputs.json` with `main/ablation/adversarial/repeated/roi` protocol outputs.
`main` now also reports `paired_statistics.ocl_vs_single`, pairing runs by
`(episode_index, seed)` and emitting `mean_delta`, `delta_ci95`, and
`sign_flip_pvalues` (`p_two_sided` included) for
`success/has_violation/round/seller_reward/latency_sec`.

Control statistical sampling with:

```bash
python scripts/run_batch_eval.py \
  --arms single,ocl_full \
  --episodes-per-arm 20 \
  --bootstrap-samples 5000 \
  --permutation-samples 50000
```

Run the default ablation matrix (E0~E5) and automatically summarize
`delta_default - delta_ablation`:

```bash
python scripts/run_ablation_matrix.py \
  --episodes-per-arm 20 \
  --seed-base 42 \
  --bootstrap-samples 1000 \
  --permutation-samples 20000 \
  --output-root outputs/ablation_matrix_v1
```

Artifacts:
- `ablation_summary.json`
- `ablation_metrics.csv`
- `ablation_contributions.csv`

Common custom arguments:

```bash
python scripts/run_demo.py \
  --arm single \
  --seed 123 \
  --max-rounds 12 \
  --buyer-max-price 125 \
  --seller-min-price 95 \
  --product-name "Premium Winter Jacket" \
  --product-price 185
```

## Current Code Map

Core runtime and AgenticPay bridge:

- [`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py)
- [`aimai_ocl/adapters/agenticpay_env.py`](aimai_ocl/adapters/agenticpay_env.py)
- [`aimai_ocl/adapters/agenticpay_actions.py`](aimai_ocl/adapters/agenticpay_actions.py)

Minimal OCL formal interfaces:

- [`aimai_ocl/schemas/actions.py`](aimai_ocl/schemas/actions.py)
- [`aimai_ocl/schemas/constraints.py`](aimai_ocl/schemas/constraints.py)
- [`aimai_ocl/schemas/audit.py`](aimai_ocl/schemas/audit.py)

OCL controller and algorithms:

- [`aimai_ocl/controllers/role_policy.py`](aimai_ocl/controllers/role_policy.py)
- [`aimai_ocl/controllers/risk_gate.py`](aimai_ocl/controllers/risk_gate.py)
- [`aimai_ocl/controllers/ocl_controller.py`](aimai_ocl/controllers/ocl_controller.py)
- [`aimai_ocl/controllers/coordinator.py`](aimai_ocl/controllers/coordinator.py)
- [`aimai_ocl/controllers/escalation_manager.py`](aimai_ocl/controllers/escalation_manager.py)

Attribution algorithms:

- [`aimai_ocl/attribution_shapley.py`](aimai_ocl/attribution_shapley.py)
- [`aimai_ocl/attribution_counterfactual.py`](aimai_ocl/attribution_counterfactual.py)

Plugin and experiment protocol registry:

- [`aimai_ocl/plugin_registry.py`](aimai_ocl/plugin_registry.py)

Episode execution:

- [`aimai_ocl/runners/single_episode.py`](aimai_ocl/runners/single_episode.py)
- [`aimai_ocl/runners/ocl_episode.py`](aimai_ocl/runners/ocl_episode.py)

## Minimal OCL Interface

The first formal OCL objects now live under `aimai_ocl.schemas`:

- `RawAction`
- `ExecutableAction`
- `ConstraintCheck`
- `AuditEvent`
- `EpisodeTrace`

These are the minimal objects needed to eventually implement:

- `g_Pi: (m_1:t, a_raw_1:t) -> a_exec_1:t`
- explicit constraint evaluation
- structured audit traces
- attribution and incentive logic later

## OCL Controller Stack

Current controller stack (v1 + v2 research variants):

- `RolePolicy` validates `role -> intent` permission
- `ConstraintEngine` applies deterministic hard checks (format, budget/floor, privacy)
- `RiskGate` and `BarrierRiskGate` provide two gating families
- `OCLController` returns `OCLControlResult` with:
  - `ExecutableAction`
  - collected `ConstraintCheck` records
  - generated `AuditEvent` records

Role decomposition currently supports:

- `Coordinator` (rule-based)
- `SellerOnlyCoordinator` (role ablation)
- `StateMachineCoordinator` (algorithmic role state machine)

Escalation layer is wired via `EscalationManager`:

- blocked/high-risk seller actions trigger `ESCALATION_TRIGGERED`
- recoverable price violations perform one deterministic `REPLAN_APPLIED`
- infeasible/conflicting cases fall back to human handoff recommendation

Decision/violation contract:

- `ControlDecision`: `approve / rewrite / block / escalate`
- `ViolationType`: canonical taxonomy labels for failed or notable checks

## Roadmap Source of Truth

Roadmap and completion state are maintained in:

- [`TODO.md`](TODO.md)

## Documentation

- Proposal: [`docs/PAPER_PROPOSAL.md`](docs/PAPER_PROPOSAL.md)
- Next-layer handoff: [`docs/NEXT_LAYER_HANDOFF.md`](docs/NEXT_LAYER_HANDOFF.md)
- AgenticPay loader: [`aimai_ocl/agenticpay_runtime.py`](aimai_ocl/agenticpay_runtime.py)
