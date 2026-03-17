# Next-Layer Handoff

This document is a short implementation handoff for the next development round.

## Current Baseline

`aimai_ocl` currently has:

- AgenticPay runtime import boundary
- AgenticPay environment adapter
- AgenticPay text-action adapter
- minimal OCL schemas:
  - `RawAction`
  - `ExecutableAction`
  - `ConstraintCheck`
  - `AuditEvent`
  - `EpisodeTrace`
- minimal OCL controllers:
  - `RolePolicy`
  - `RiskGate`
  - `OCLController`
- minimal runner:
  - `run_single_negotiation_episode`

The runner currently records action events, but it does not yet use
`OCLController` as the actual control transform.

## Required Dependency Assumption

AgenticPay must be importable from the active Python environment:

```bash
pip install "git+https://github.com/SafeRL-Lab/AgenticPay.git"
```

## Next Implementation Target

Integrate the controller into the execution path so that seller-side actions
actually perform:

`RawAction -> OCLController -> ExecutableAction -> env.step(...)`

instead of the current pass-through transform. Buyer-side actions should remain
external user-simulator inputs in the current scope.

## Concrete Tasks

1. Update `aimai_ocl/runners/single_episode.py`
2. Replace seller-side action internals with `OCLController.apply(...)`
3. Append controller-produced `AuditEvent` records to `EpisodeTrace`
4. Send seller-side `ExecutableAction.final_text` to AgenticPay `env.step(...)`
5. Preserve current compatibility with AgenticPay `SellerAgent` and
   `CollaborativeSellerAgent`

## Suggested Acceptance Checks

1. One episode runs end-to-end with `SellerAgent`
2. One episode runs end-to-end with `CollaborativeSellerAgent`
3. `EpisodeTrace.events` includes:
   - `RAW_ACTION_RECEIVED`
   - `CONSTRAINT_EVALUATED`
   - `ACTION_EXECUTED`
   - `EPISODE_FINISHED`
4. `final_info` remains populated with AgenticPay terminal fields

## Known Constraints

- The local environment previously showed OMP runtime issues when importing
  heavy model stacks.
- Keep tests lightweight and avoid forcing heavyweight model backends.
- Prefer syntax and interface checks unless runtime dependencies are guaranteed.

## One-Line Next Prompt

Use this in the next chat to continue directly:

"Integrate `OCLController` into seller-side action flow so actions are truly
transformed from `RawAction` to `ExecutableAction`, while keeping buyer-side
inputs external and preserving AgenticPay SellerAgent/CollaborativeSellerAgent compatibility."
