"""Environment for Best-Offer Selection.

The environment is the only component allowed to mutate true episode state. The
platform produces one structured action, the simulator produces one structured
response, and the environment turns that pair into the next observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from conversational_consumer_selection.schemas import (
    ActionType,
    apply_clarification_to_revealed_context,
    BenchmarkLevel,
    HistoryEntry,
    Observation,
    Offer,
    SelectionAction,
    SelectionTask,
    TerminationReason,
)
from conversational_consumer_selection.simulator import (
    EXPLICIT_VIOLATION_REASONS,
    RuleBasedUserSimulator,
)
from conversational_consumer_selection.surfaces import render_buyer_opening, render_buyer_response
from conversational_consumer_selection.tasks import make_default_task


@dataclass
class EpisodeState:
    """Mutable episode state held by the environment."""

    task: SelectionTask
    revealed_context: dict[str, Any]
    turn_index: int = 0
    history: list[HistoryEntry] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    committed_offer_id: str | None = None
    escalated: bool = False
    clarification_count: int = 0
    transient_violation_count: int = 0
    executed_violation_count: int = 0
    unrecovered_violation: bool = False
    termination_reason: TerminationReason | None = None
    last_response: dict[str, Any] = field(default_factory=dict)
    cumulative_controller_payoff: float = 0.0
    user_utterance_history: list[str] = field(default_factory=list)


class BestOfferSelectionEnv:
    """Minimal Gym-like environment for conversational consumer selection."""

    def __init__(
        self,
        *,
        simulator: RuleBasedUserSimulator | None = None,
        default_task: SelectionTask | None = None,
    ) -> None:
        self._simulator = simulator or RuleBasedUserSimulator()
        self._default_task = default_task
        self._state: EpisodeState | None = None

    @property
    def state(self) -> EpisodeState:
        """Return the current internal state."""

        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    def reset(
        self,
        *,
        task: SelectionTask | None = None,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        del seed
        active_task = task or self._default_task or make_default_task()
        user_utterance_history: list[str] = []
        if active_task.level in {BenchmarkLevel.PARTIAL_INTENT, BenchmarkLevel.HIDDEN_INTENT}:
            user_utterance_history.append(render_buyer_opening(active_task))

        # Reset starts from the explicit opening contract, not from the full
        # latent user model. This is the boundary that makes elicitation matter.
        self._state = EpisodeState(
            task=active_task,
            revealed_context={
                "category": active_task.initial_revealed_context["category"],
                "budget_max": active_task.initial_revealed_context["budget_max"],
                "must_have": dict(active_task.initial_revealed_context["must_have"]),
                "preference_weights": dict(active_task.initial_revealed_context["preference_weights"]),
            },
            user_utterance_history=user_utterance_history,
        )
        observation = self._build_observation()
        return observation, self.episode_summary()

    def step(
        self,
        action: SelectionAction,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        state = self.state
        if state.terminated:
            raise RuntimeError("environment is already terminated")

        state.turn_index += 1
        step_payoff = 0.0
        transient_violation = False
        executed_violation = False

        # Each platform action is adjudicated against the latent simulator
        # before any dialogue rendering happens. The dialogue layer is secondary.
        if action.action_type is ActionType.ASK_CLARIFICATION:
            response = self._step_clarification(action=action)
            if response["status"] == "clarified":
                state.clarification_count += 1
            else:
                transient_violation = True
                step_payoff = -0.25
        elif action.action_type is ActionType.COMPARE_OPTIONS:
            response = self._step_compare(action=action)
            if response["status"] != "compared":
                transient_violation = True
                step_payoff = -0.25
        elif action.action_type is ActionType.RECOMMEND_OPTION:
            response = self._step_recommend(action=action)
            if response.get("reason") in EXPLICIT_VIOLATION_REASONS:
                transient_violation = True
                step_payoff = -0.25
        elif action.action_type is ActionType.COMMIT_SELECTION:
            response = self._step_commit(action=action)
            executed_violation = bool(response.get("executed_violation"))
            if response.get("accepted"):
                step_payoff = float(response.get("consumer_utility", response["utility"]))
                state.committed_offer_id = action.offer_id
                state.terminated = True
                state.termination_reason = TerminationReason.COMMIT_ACCEPTED
            elif executed_violation:
                step_payoff = -1.0
                state.terminated = True
                state.termination_reason = TerminationReason.COMMIT_REJECTED_VIOLATION
            elif response.get("outside_option_preferred"):
                state.terminated = True
                state.termination_reason = TerminationReason.OUTSIDE_OPTION
            else:
                state.terminated = True
                state.termination_reason = TerminationReason.COMMIT_REJECTED
        else:
            response = {"status": "escalated"}
            state.escalated = True
            state.terminated = True
            state.termination_reason = TerminationReason.ESCALATED

        if transient_violation:
            state.transient_violation_count += 1
        if executed_violation:
            state.executed_violation_count += 1

        state.cumulative_controller_payoff += step_payoff
        state.last_response = dict(response)
        state.history.append(
            HistoryEntry(
                turn_index=state.turn_index,
                action=action,
                response=response,
            )
        )
        if state.task.level in {BenchmarkLevel.PARTIAL_INTENT, BenchmarkLevel.HIDDEN_INTENT}:
            state.user_utterance_history.append(
                render_buyer_response(action, response, offers=state.task.offers)
            )

        if not state.terminated and state.turn_index >= state.task.max_turns:
            state.terminated = True
            state.truncated = True
            state.termination_reason = TerminationReason.TIMEOUT

        if state.terminated:
            state.unrecovered_violation = self._compute_unrecovered_violation()

        observation = self._build_observation()
        info = self.episode_summary()
        return observation, step_payoff, state.terminated, state.truncated, info

    def episode_summary(self) -> dict[str, Any]:
        """Return structured metrics for the current episode."""

        state = self.state
        selected_offer = self._lookup_offer(state.committed_offer_id)
        realized_consumer_utility = state.task.outside_option_threshold
        if selected_offer is not None:
            realized_consumer_utility = self._simulator.utility(
                state.task,
                selected_offer,
                state.turn_index,
            )
        elif "consumer_utility" in state.last_response:
            realized_consumer_utility = float(state.last_response["consumer_utility"])

        best_offer = self._simulator.best_offer(state.task, state.turn_index)
        best_offer_id = best_offer.offer_id if best_offer is not None else None
        optimal_consumer_utility = self._simulator.best_utility(state.task, state.turn_index)
        commit_success = (
            state.committed_offer_id is not None and state.executed_violation_count == 0
        )
        consumer_regret = max(0.0, optimal_consumer_utility - realized_consumer_utility)
        return {
            "task_id": state.task.task_id,
            "level": state.task.level.value,
            "commit_success": commit_success,
            "selection_success": commit_success,
            "consumer_utility": realized_consumer_utility,
            "utility": realized_consumer_utility,
            "optimal_consumer_utility": optimal_consumer_utility,
            "optimal_utility": optimal_consumer_utility,
            "consumer_regret": consumer_regret,
            "regret": consumer_regret,
            "controller_payoff": state.cumulative_controller_payoff,
            "rounds": state.turn_index,
            "clarification_count": state.clarification_count,
            "transient_violation_count": state.transient_violation_count,
            "executed_violation_count": state.executed_violation_count,
            "has_transient_violation": state.transient_violation_count > 0,
            "has_executed_violation": state.executed_violation_count > 0,
            "has_unrecovered_violation": state.unrecovered_violation,
            "transient_violation_rate": float(state.transient_violation_count > 0),
            "executed_violation_rate": float(state.executed_violation_count > 0),
            "unrecovered_violation_rate": float(state.unrecovered_violation),
            "escalated": state.escalated,
            "escalation_rate": float(state.escalated),
            "termination_reason": (
                state.termination_reason.value if state.termination_reason is not None else None
            ),
            "committed_offer_id": state.committed_offer_id,
            "best_offer_id": best_offer_id,
            "last_response": dict(state.last_response),
        }

    def _step_clarification(self, action: SelectionAction) -> dict[str, Any]:
        state = self.state
        response = self._simulator.ask_clarification(state.task, action.slot or "")
        if response["status"] == "clarified":
            apply_clarification_to_revealed_context(
                state.revealed_context,
                slot=str(response["slot"]),
                value=response["value"],
            )
        return response

    def _step_compare(self, action: SelectionAction) -> dict[str, Any]:
        state = self.state
        return self._simulator.compare_options(
            state.task,
            self._lookup_offer(action.offer_id),
            self._lookup_offer(action.comparison_offer_id),
            state.turn_index,
        )

    def _step_recommend(self, action: SelectionAction) -> dict[str, Any]:
        state = self.state
        return self._simulator.recommend_option(
            state.task,
            self._lookup_offer(action.offer_id),
            state.turn_index,
        )

    def _step_commit(self, action: SelectionAction) -> dict[str, Any]:
        state = self.state
        return self._simulator.commit_selection(
            state.task,
            self._lookup_offer(action.offer_id),
            state.turn_index,
        )

    def _build_observation(self) -> Observation:
        state = self.state
        # Observations deliberately expose only the public contract and the
        # structured interaction trace. Internal beliefs stay inside the agent.
        return Observation(
            level=state.task.level,
            revealed_context={
                "category": state.revealed_context["category"],
                "budget_max": state.revealed_context["budget_max"],
                "must_have": dict(state.revealed_context["must_have"]),
                "preference_weights": dict(state.revealed_context["preference_weights"]),
            },
            available_clarification_slots=state.task.available_clarification_slots,
            offers=state.task.offers,
            turn_index=state.turn_index,
            max_turns=state.task.max_turns,
            remaining_turns=max(0, state.task.max_turns - state.turn_index),
            history=tuple(state.history),
            user_utterance_history=tuple(state.user_utterance_history),
            committed_offer_id=state.committed_offer_id,
            escalated=state.escalated,
            terminated=state.terminated,
        )

    def _compute_unrecovered_violation(self) -> bool:
        state = self.state
        if state.executed_violation_count > 0:
            return True
        if state.termination_reason is TerminationReason.ESCALATED:
            return False
        if state.committed_offer_id is not None:
            return False
        if state.termination_reason is TerminationReason.OUTSIDE_OPTION:
            return False
        return state.transient_violation_count > 0

    def _lookup_offer(self, offer_id: str | None) -> Offer | None:
        if offer_id is None:
            return None
        state = self.state
        for offer in state.task.offers:
            if offer.offer_id == offer_id:
                return offer
        return None
