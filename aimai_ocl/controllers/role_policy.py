"""Role policy for the minimal OCL controller skeleton.

This module answers the first control question:

"Given the actor role and the proposed intent, is this actor allowed to make
this kind of move?"

The current implementation is intentionally simple. It exists to establish the
interface boundary and a default policy that later experiments can replace.


中文翻译：Role policy for the minimal OCL controller skeleton。"""

from __future__ import annotations

from dataclasses import dataclass, field

from aimai_ocl.schemas.actions import ActionIntent, ActionRole, RawAction
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


DEFAULT_ROLE_POLICY: dict[ActionRole, set[ActionIntent]] = {
    ActionRole.BUYER: {
        ActionIntent.NEGOTIATE_PRICE,
        ActionIntent.ACCEPT_DEAL,
        ActionIntent.REJECT_DEAL,
        ActionIntent.REQUEST_INFO,
        ActionIntent.OTHER,
    },
    ActionRole.SELLER: {
        ActionIntent.NEGOTIATE_PRICE,
        ActionIntent.ACCEPT_DEAL,
        ActionIntent.REJECT_DEAL,
        ActionIntent.REQUEST_INFO,
        ActionIntent.EXPLAIN_POLICY,
        ActionIntent.OTHER,
    },
    ActionRole.PLATFORM: {
        ActionIntent.EXPLAIN_POLICY,
        ActionIntent.ESCALATE,
        ActionIntent.TOOL_CALL,
        ActionIntent.OTHER,
    },
    ActionRole.EXPERT: {
        ActionIntent.REQUEST_INFO,
        ActionIntent.EXPLAIN_POLICY,
        ActionIntent.OTHER,
    },
    ActionRole.USER: {
        ActionIntent.REQUEST_INFO,
        ActionIntent.OTHER,
    },
    ActionRole.UNKNOWN: {ActionIntent.OTHER},
}


@dataclass(slots=True)
class RolePolicy:
    """Minimal role policy for validating whether a role may express an intent.

    Inputs:
        allowed_intents: Mapping from actor role to the set of intents this role
            may issue under the current control policy.

    Outputs:
        A callable policy object whose main job is to turn role-permission
        checks into normalized ``ConstraintCheck`` records.
    

    中文翻译：Minimal role policy for validating whether a role may express an intent。"""

    allowed_intents: dict[ActionRole, set[ActionIntent]] = field(
        default_factory=lambda: {role: set(intents) for role, intents in DEFAULT_ROLE_POLICY.items()}
    )

    def evaluate(self, raw_action: RawAction) -> ConstraintCheck:
        """Evaluate whether the actor role may issue the proposed intent.

        Args:
            raw_action: The untrusted action proposed by a buyer, seller, or
                other system role.

        Returns:
            A ``ConstraintCheck`` describing whether the role-intent pair is
            permitted by the current policy.
        

        中文翻译：Evaluate whether the actor role may issue the proposed intent。"""
        allowed = self.allowed_intents.get(raw_action.actor_role, set())
        passed = raw_action.intent in allowed

        if passed:
            return ConstraintCheck(
                constraint_id="role_policy",
                passed=True,
                severity=ConstraintSeverity.INFO,
                reason=(
                    f"Role `{raw_action.actor_role.value}` is allowed to emit "
                    f"intent `{raw_action.intent.value}`."
                ),
                checked_fields=["actor_role", "intent"],
            )

        return ConstraintCheck(
            constraint_id="role_policy",
            passed=False,
            severity=ConstraintSeverity.ERROR,
            reason=(
                f"Role `{raw_action.actor_role.value}` is not allowed to emit "
                f"intent `{raw_action.intent.value}`."
            ),
            violation_type=ViolationType.ROLE_PERMISSION,
            checked_fields=["actor_role", "intent"],
        )
