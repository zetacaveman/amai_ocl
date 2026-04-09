"""Core schemas for the benchmark.

The benchmark is organized around three layers:

- `CategorySchema`: what a product family can talk about.
- `LatentConsumerModel`: the user's hidden decision rule on top of that schema.
- `SelectionTask`: one concrete episode, including how much of that latent state
  is visible at reset time and what the user's first request looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


CLARIFICATION_BUDGET_MAX = "budget.max"
CLARIFICATION_MUST_HAVE_PREFIX = "must_have."
CLARIFICATION_PREFERENCE_PREFIX = "preference."


class BenchmarkLevel(str, Enum):
    """Supported benchmark settings."""

    ORACLE_DEBUG = "oracle_debug"
    DIRECT_INTENT = "direct_intent"
    PARTIAL_INTENT = "partial_intent"
    HIDDEN_INTENT = "hidden_intent"
    ORACLE_GOAL = "oracle_debug"
    PARTIAL_GOAL = "partial_intent"

    @classmethod
    def from_str(cls, value: str) -> "BenchmarkLevel":
        """Parse new and legacy setting names."""

        legacy_map = {
            "oracle_goal": cls.ORACLE_DEBUG,
            "partial_goal": cls.PARTIAL_INTENT,
        }
        if value in legacy_map:
            return legacy_map[value]
        return cls(value)


class ActionType(str, Enum):
    """Structured action types exposed by the environment."""

    ASK_CLARIFICATION = "ask_clarification"
    COMPARE_OPTIONS = "compare_options"
    RECOMMEND_OPTION = "recommend_option"
    COMMIT_SELECTION = "commit_selection"
    ESCALATE = "escalate"


class RecommendationLabel(str, Enum):
    """Recommendation feedback labels emitted by the simulator."""

    ACCEPT = "accept"
    HESITATE = "hesitate"
    REJECT = "reject"


class TerminationReason(str, Enum):
    """Terminal statuses for one episode."""

    COMMIT_ACCEPTED = "commit_accepted"
    COMMIT_REJECTED = "commit_rejected"
    COMMIT_REJECTED_VIOLATION = "commit_rejected_violation"
    ESCALATED = "escalated"
    OUTSIDE_OPTION = "outside_option"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Offer:
    """One candidate offer presented to the system."""

    offer_id: str
    category: str
    price: float
    features: Mapping[str, Any] = field(default_factory=dict)
    attribute_values: Mapping[str, float] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        if self.price < 0.0:
            raise ValueError("offer price must be non-negative")
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(
            self,
            "attribute_values",
            {str(name): float(value) for name, value in self.attribute_values.items()},
        )


@dataclass(frozen=True)
class UserGoal:
    """Explicit user-facing goal fields."""

    category: str
    budget_max: float
    must_have: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget_max < 0.0:
            raise ValueError("budget_max must be non-negative")
        object.__setattr__(self, "must_have", dict(self.must_have))


@dataclass(frozen=True)
class CategorySchema:
    """Product/category schema that defines which modules exist for one category."""

    category: str
    constraint_slots: tuple[str, ...] = ()
    preference_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_slots", tuple(self.constraint_slots))
        object.__setattr__(self, "preference_slots", tuple(self.preference_slots))


@dataclass(frozen=True)
class UserProfile:
    """User-profile metadata that affects how a user enters the interaction."""

    profile_id: str = "default"
    initial_request_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_request_payload", dict(self.initial_request_payload))


@dataclass(frozen=True)
class LatentConsumerModel:
    """Latent consumer model instantiated on top of a category schema."""

    budget_max: float
    must_have: Mapping[str, Any] = field(default_factory=dict)
    preference_weights: Mapping[str, float] = field(default_factory=dict)
    price_sensitivity: float = 0.0
    outside_option_threshold: float = 0.0
    turn_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.budget_max < 0.0:
            raise ValueError("budget_max must be non-negative")
        if self.price_sensitivity < 0.0:
            raise ValueError("price_sensitivity must be non-negative")
        if self.turn_penalty < 0.0:
            raise ValueError("turn_penalty must be non-negative")
        object.__setattr__(self, "must_have", dict(self.must_have))
        object.__setattr__(
            self,
            "preference_weights",
            {str(name): float(value) for name, value in self.preference_weights.items()},
        )


@dataclass(frozen=True)
class SelectionTask:
    """One benchmark episode specification."""

    task_id: str
    level: BenchmarkLevel
    user_goal: UserGoal
    offers: tuple[Offer, ...]
    preference_weights: Mapping[str, float]
    price_sensitivity: float
    outside_option_threshold: float
    turn_penalty: float = 0.0
    hidden_preference_slots: tuple[str, ...] = ()
    max_turns: int = 6
    initial_intent_reveal_ratio: float = 1.0
    initial_request_payload: Mapping[str, Any] = field(default_factory=dict)
    initial_revealed_context: Mapping[str, Any] = field(default_factory=dict)
    category_schema: CategorySchema | None = None
    user_profile: UserProfile = field(default_factory=UserProfile)
    latent_consumer_model: LatentConsumerModel | None = None

    def __post_init__(self) -> None:
        offers = tuple(self.offers)
        weights = {str(name): float(value) for name, value in self.preference_weights.items()}
        hidden_slots = tuple(self.hidden_preference_slots)
        if len(offers) < 2 or len(offers) > 4:
            raise ValueError("V1 requires 2 to 4 offers per task")
        if not weights:
            raise ValueError("preference_weights must not be empty")
        if self.price_sensitivity < 0.0:
            raise ValueError("price_sensitivity must be non-negative")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.turn_penalty < 0.0:
            raise ValueError("turn_penalty must be non-negative")
        if not (0.0 <= self.initial_intent_reveal_ratio <= 1.0):
            raise ValueError("initial_intent_reveal_ratio must be in [0.0, 1.0]")
        if self.level in {BenchmarkLevel.ORACLE_DEBUG, BenchmarkLevel.DIRECT_INTENT} and hidden_slots:
            raise ValueError(f"{self.level.value} tasks must not hide preference slots")
        if self.level is BenchmarkLevel.PARTIAL_INTENT and not (1 <= len(hidden_slots) <= 2):
            raise ValueError("partial_intent tasks must hide 1 to 2 preference slots")
        if self.level is BenchmarkLevel.HIDDEN_INTENT and not hidden_slots:
            raise ValueError("hidden_intent tasks must hide at least one preference slot")
        missing_slots = [slot for slot in hidden_slots if slot not in weights]
        if missing_slots:
            raise ValueError(f"hidden preference slots missing from weights: {missing_slots}")
        # First canonicalize the latent problem: category modules, latent
        # consumer model, and the first request act all need to agree.
        category_schema = self.category_schema or CategorySchema(
            category=self.user_goal.category,
            constraint_slots=tuple(sorted(self.user_goal.must_have)),
            preference_slots=tuple(sorted(weights)),
        )
        latent_consumer_model = self.latent_consumer_model or LatentConsumerModel(
            budget_max=self.user_goal.budget_max,
            must_have=dict(self.user_goal.must_have),
            preference_weights=weights,
            price_sensitivity=self.price_sensitivity,
            outside_option_threshold=self.outside_option_threshold,
            turn_penalty=self.turn_penalty,
        )
        if category_schema.category != self.user_goal.category:
            raise ValueError("category_schema.category must match user_goal.category")
        request_payload = dict(self.user_profile.initial_request_payload)
        request_payload.update(dict(self.initial_request_payload))
        if "category" in request_payload and request_payload["category"] != category_schema.category:
            raise ValueError("initial_request_payload category must match user_goal.category")
        # The opening contract is intentionally separate from the latent model.
        # A user may know far more than the platform sees at reset time.
        initial_revealed_context = {
            "category": category_schema.category,
            "budget_max": latent_consumer_model.budget_max,
            "must_have": {},
            "preference_weights": {},
        }
        initial_revealed_context["must_have"].update(dict(self.user_goal.must_have))
        initial_revealed_context["preference_weights"].update(
            {
                name: value
                for name, value in latent_consumer_model.preference_weights.items()
                if name not in hidden_slots
            }
        )
        override_revealed_context = dict(self.initial_revealed_context)
        if "category" in override_revealed_context:
            if override_revealed_context["category"] != category_schema.category:
                raise ValueError(
                    "initial_revealed_context category must match user_goal.category"
                )
        initial_revealed_context.update(
            {
                "category": override_revealed_context.get("category", category_schema.category),
                "budget_max": override_revealed_context.get(
                    "budget_max", initial_revealed_context["budget_max"]
                ),
                "must_have": dict(
                    override_revealed_context.get("must_have", initial_revealed_context["must_have"])
                ),
                "preference_weights": dict(
                    override_revealed_context.get(
                        "preference_weights",
                        initial_revealed_context["preference_weights"],
                    )
                ),
            }
        )
        canonical_user_goal = UserGoal(
            category=category_schema.category,
            budget_max=latent_consumer_model.budget_max,
            must_have=dict(latent_consumer_model.must_have),
        )
        object.__setattr__(self, "offers", offers)
        object.__setattr__(self, "user_goal", canonical_user_goal)
        object.__setattr__(self, "preference_weights", dict(latent_consumer_model.preference_weights))
        object.__setattr__(self, "price_sensitivity", latent_consumer_model.price_sensitivity)
        object.__setattr__(
            self,
            "outside_option_threshold",
            latent_consumer_model.outside_option_threshold,
        )
        object.__setattr__(self, "turn_penalty", latent_consumer_model.turn_penalty)
        object.__setattr__(self, "hidden_preference_slots", hidden_slots)
        object.__setattr__(self, "initial_request_payload", request_payload)
        object.__setattr__(self, "initial_revealed_context", initial_revealed_context)
        object.__setattr__(self, "category_schema", category_schema)
        object.__setattr__(self, "latent_consumer_model", latent_consumer_model)

    @property
    def revealed_preference_weights(self) -> dict[str, float]:
        """Return the preference weights visible at reset time."""

        return {
            name: value
            for name, value in self.preference_weights.items()
            if name not in self.hidden_preference_slots
        }

    @property
    def initial_public_goal(self) -> dict[str, Any]:
        """Return the subset of explicit goal fields the buyer reveals at opening."""

        candidates: list[tuple[str, Any]] = [("budget_max", self.user_goal.budget_max)]
        for slot in sorted(self.user_goal.must_have):
            candidates.append((f"must_have.{slot}", self.user_goal.must_have[slot]))

        reveal_count = int(math.ceil(self.initial_intent_reveal_ratio * len(candidates)))
        reveal_count = max(0, min(len(candidates), reveal_count))

        public_goal: dict[str, Any] = {"category": self.user_goal.category, "must_have": {}}
        for key, value in candidates[:reveal_count]:
            if key == "budget_max":
                public_goal["budget_max"] = value
            else:
                slot = key.split(".", maxsplit=1)[1]
                public_goal["must_have"][slot] = value
        return public_goal

    @property
    def initial_user_request(self) -> dict[str, Any]:
        """Return the first-turn structured request act emitted by the user."""

        if self.initial_request_payload:
            return dict(self.initial_request_payload)
        return {"category": self.user_goal.category}

    @property
    def available_clarification_slots(self) -> tuple[str, ...]:
        """Return the product-agnostic clarification interface for this task."""

        slots = [CLARIFICATION_BUDGET_MAX]
        slots.extend(
            f"{CLARIFICATION_MUST_HAVE_PREFIX}{slot}"
            for slot in sorted(self.category_schema.constraint_slots)
        )
        slots.extend(
            f"{CLARIFICATION_PREFERENCE_PREFIX}{slot}"
            for slot in sorted(self.category_schema.preference_slots)
        )
        return tuple(slots)


@dataclass(frozen=True)
class SelectionAction:
    """Structured action emitted by a system policy."""

    action_type: ActionType
    slot: str | None = None
    offer_id: str | None = None
    comparison_offer_id: str | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.action_type is ActionType.ASK_CLARIFICATION and not self.slot:
            raise ValueError("ask_clarification requires slot")
        if self.action_type in {ActionType.RECOMMEND_OPTION, ActionType.COMMIT_SELECTION}:
            if not self.offer_id:
                raise ValueError(f"{self.action_type.value} requires offer_id")
        if self.action_type is ActionType.COMPARE_OPTIONS:
            if not self.offer_id or not self.comparison_offer_id:
                raise ValueError("compare_options requires two offer ids")
            if self.offer_id == self.comparison_offer_id:
                raise ValueError("compare_options requires distinct offer ids")
        if self.action_type is ActionType.ESCALATE:
            if self.slot or self.offer_id or self.comparison_offer_id:
                raise ValueError("escalate does not accept slot or offer ids")

    @classmethod
    def ask_clarification(cls, slot: str, explanation: str = "") -> "SelectionAction":
        return cls(action_type=ActionType.ASK_CLARIFICATION, slot=slot, explanation=explanation)

    @classmethod
    def compare_options(
        cls,
        offer_id: str,
        comparison_offer_id: str,
        explanation: str = "",
    ) -> "SelectionAction":
        return cls(
            action_type=ActionType.COMPARE_OPTIONS,
            offer_id=offer_id,
            comparison_offer_id=comparison_offer_id,
            explanation=explanation,
        )

    @classmethod
    def recommend_option(cls, offer_id: str, explanation: str = "") -> "SelectionAction":
        return cls(
            action_type=ActionType.RECOMMEND_OPTION,
            offer_id=offer_id,
            explanation=explanation,
        )

    @classmethod
    def commit_selection(cls, offer_id: str, explanation: str = "") -> "SelectionAction":
        return cls(
            action_type=ActionType.COMMIT_SELECTION,
            offer_id=offer_id,
            explanation=explanation,
        )

    @classmethod
    def escalate(cls, explanation: str = "") -> "SelectionAction":
        return cls(action_type=ActionType.ESCALATE, explanation=explanation)


@dataclass(frozen=True)
class HistoryEntry:
    """One structured dialogue step."""

    turn_index: int
    action: SelectionAction
    response: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", dict(self.response))


@dataclass(frozen=True)
class Observation:
    """Minimal benchmark observation contract exposed to a controller.

    `Observation` is the public contract for controllers. Internal beliefs,
    parser state, or other model-specific traces do not belong here.
    """

    level: BenchmarkLevel
    revealed_context: Mapping[str, Any]
    available_clarification_slots: tuple[str, ...]
    offers: tuple[Offer, ...]
    turn_index: int
    max_turns: int
    remaining_turns: int
    history: tuple[HistoryEntry, ...]
    user_utterance_history: tuple[str, ...] = ()
    committed_offer_id: str | None = None
    escalated: bool = False
    terminated: bool = False

    def __post_init__(self) -> None:
        revealed_context = dict(self.revealed_context)
        if "must_have" in revealed_context:
            revealed_context["must_have"] = dict(revealed_context["must_have"])
        if "preference_weights" in revealed_context:
            revealed_context["preference_weights"] = dict(
                revealed_context["preference_weights"]
            )
        object.__setattr__(self, "revealed_context", revealed_context)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.remaining_turns < 0:
            raise ValueError("remaining_turns must be non-negative")
        object.__setattr__(
            self,
            "available_clarification_slots",
            tuple(self.available_clarification_slots),
        )
        object.__setattr__(self, "offers", tuple(self.offers))
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "user_utterance_history", tuple(self.user_utterance_history))

    @property
    def available_preference_slots(self) -> tuple[str, ...]:
        """Backward-compatible view of preference-only clarification slots."""

        return tuple(
            slot
            for slot in self.available_clarification_slots
            if slot.startswith(CLARIFICATION_PREFERENCE_PREFIX)
        )


def normalize_clarification_slot(slot: str) -> str:
    """Normalize one clarification slot into the benchmark namespace."""

    if slot == CLARIFICATION_BUDGET_MAX:
        return CLARIFICATION_BUDGET_MAX
    if slot == "budget_max":
        return CLARIFICATION_BUDGET_MAX
    if slot.startswith("preference_weights."):
        return f"{CLARIFICATION_PREFERENCE_PREFIX}{slot.split('.', maxsplit=1)[1]}"
    if slot.startswith(CLARIFICATION_PREFERENCE_PREFIX):
        return slot
    if slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
        return slot
    return f"{CLARIFICATION_PREFERENCE_PREFIX}{slot}"


def clarification_slot_is_revealed(slot: str, revealed_context: Mapping[str, Any]) -> bool:
    """Return whether the normalized clarification slot is already visible."""

    normalized_slot = normalize_clarification_slot(slot)
    if normalized_slot == CLARIFICATION_BUDGET_MAX:
        return revealed_context.get("budget_max") is not None
    if normalized_slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
        key = normalized_slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
        return key in dict(revealed_context.get("must_have", {}))
    if normalized_slot.startswith(CLARIFICATION_PREFERENCE_PREFIX):
        key = normalized_slot[len(CLARIFICATION_PREFERENCE_PREFIX) :]
        return key in dict(revealed_context.get("preference_weights", {}))
    return False


def apply_clarification_to_revealed_context(
    revealed_context: dict[str, Any],
    *,
    slot: str,
    value: Any,
) -> None:
    """Apply one clarified slot value to the mutable revealed context."""

    normalized_slot = normalize_clarification_slot(slot)
    if normalized_slot == CLARIFICATION_BUDGET_MAX:
        revealed_context["budget_max"] = float(value)
        return
    if normalized_slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
        key = normalized_slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
        revealed_context.setdefault("must_have", {})[key] = value
        return
    if normalized_slot.startswith(CLARIFICATION_PREFERENCE_PREFIX):
        key = normalized_slot[len(CLARIFICATION_PREFERENCE_PREFIX) :]
        revealed_context.setdefault("preference_weights", {})[key] = float(value)
        return
    raise ValueError(f"unknown clarification slot: {slot}")
