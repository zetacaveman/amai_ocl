"""Task builders for the benchmark.

The benchmark now has one stable structured base (`v0`) and two dialogue layers
(`v1` and `v2`) built on top of it. The builders in this file keep those roles
separate:

- `make_default_task(...)` is the canonical benchmark task factory used by tests
  and batch runs.
- `make_v0_demo_task()` fixes the opening contract to the structured-only demo.
- `make_v1_direct_intent_task()`, `make_v1_partial_intent_task()`, and
  `make_v2_hidden_intent_task()` reuse the same latent headphone market while
  changing what is visible at reset time.
"""

from __future__ import annotations

from conversational_consumer_selection.schemas import (
    BenchmarkLevel,
    CategorySchema,
    LatentConsumerModel,
    Offer,
    SelectionTask,
    UserGoal,
    UserProfile,
)


def make_default_task(level: BenchmarkLevel = BenchmarkLevel.DIRECT_INTENT) -> SelectionTask:
    """Build a deterministic V1 task for smoke tests and examples."""

    category_schema = CategorySchema(
        category="headphones",
        constraint_slots=("noise_cancellation", "wireless", "foldable"),
        preference_slots=("battery", "comfort", "portability"),
    )
    user_profile = UserProfile(
        profile_id="default_headphones_user",
        initial_request_payload={"category": "headphones"},
    )
    latent_consumer_model = LatentConsumerModel(
        budget_max=100.0,
        must_have={"noise_cancellation": True, "wireless": True},
        preference_weights={"battery": 0.80, "comfort": 1.10, "portability": 0.60},
        price_sensitivity=0.015,
        outside_option_threshold=0.40,
        turn_penalty=0.03,
    )

    offers = (
        Offer(
            offer_id="offer_budget",
            title="StudioLite ANC",
            category=category_schema.category,
            price=79.0,
            features={"noise_cancellation": True, "wireless": True, "foldable": True},
            attribute_values={"battery": 0.70, "comfort": 0.90, "portability": 0.60},
        ),
        Offer(
            offer_id="offer_premium",
            title="StudioMax Pro",
            category=category_schema.category,
            price=109.0,
            features={"noise_cancellation": True, "wireless": True, "foldable": True},
            attribute_values={"battery": 0.95, "comfort": 0.95, "portability": 0.78},
        ),
        Offer(
            offer_id="offer_travel",
            title="AirMove Flex",
            category=category_schema.category,
            price=92.0,
            features={"noise_cancellation": True, "wireless": True, "foldable": True},
            attribute_values={"battery": 0.62, "comfort": 0.82, "portability": 0.96},
        ),
    )
    goal = UserGoal(
        category=category_schema.category,
        budget_max=latent_consumer_model.budget_max,
        must_have=dict(latent_consumer_model.must_have),
    )
    hidden_slots: tuple[str, ...] = ()
    initial_intent_reveal_ratio = 0.67
    if level is BenchmarkLevel.PARTIAL_INTENT:
        hidden_slots = ("comfort", "portability")
        initial_intent_reveal_ratio = 0.34
    elif level is BenchmarkLevel.HIDDEN_INTENT:
        hidden_slots = tuple(sorted(("battery", "comfort", "portability")))
        initial_intent_reveal_ratio = 0.0
    elif level is BenchmarkLevel.ORACLE_DEBUG:
        initial_intent_reveal_ratio = 1.0
    return SelectionTask(
        task_id=f"default_{level.value}",
        level=level,
        user_goal=goal,
        offers=offers,
        preference_weights=dict(latent_consumer_model.preference_weights),
        price_sensitivity=latent_consumer_model.price_sensitivity,
        outside_option_threshold=latent_consumer_model.outside_option_threshold,
        turn_penalty=latent_consumer_model.turn_penalty,
        hidden_preference_slots=hidden_slots,
        max_turns=12,
        initial_intent_reveal_ratio=initial_intent_reveal_ratio,
        category_schema=category_schema,
        user_profile=user_profile,
        latent_consumer_model=latent_consumer_model,
    )


def _clone_task_with_visibility(
    task: SelectionTask,
    *,
    task_id: str,
    initial_request_payload: dict[str, object],
    initial_revealed_context: dict[str, object],
) -> SelectionTask:
    """Clone one latent task while overriding only the visible opening contract.

    The latent consumer model and offer set stay fixed. Only the first-turn
    request and reset-time revealed context change, which is exactly the degree
    of freedom needed to define v0/v1/v2 on a common base.
    """

    return SelectionTask(
        task_id=task_id,
        level=task.level,
        user_goal=task.user_goal,
        offers=task.offers,
        preference_weights=task.preference_weights,
        price_sensitivity=task.price_sensitivity,
        outside_option_threshold=task.outside_option_threshold,
        turn_penalty=task.turn_penalty,
        hidden_preference_slots=task.hidden_preference_slots,
        max_turns=task.max_turns,
        initial_intent_reveal_ratio=task.initial_intent_reveal_ratio,
        initial_request_payload=initial_request_payload,
        initial_revealed_context=initial_revealed_context,
        category_schema=task.category_schema,
        user_profile=task.user_profile,
        latent_consumer_model=task.latent_consumer_model,
    )


def make_v0_demo_task() -> SelectionTask:
    """Build the fixed V0 structured-demo task.

    V0 is a structured protocol demo rather than an intent-visibility benchmark.
    Internally it reuses the partial-intent task shape because the platform still
    needs to clarify missing preference slots, but the public demo entrypoint does
    not expose intent-setting labels.
    """

    task = make_default_task(level=BenchmarkLevel.PARTIAL_INTENT)
    return _clone_task_with_visibility(
        task,
        task_id="default_v0_structured",
        initial_request_payload={"category": task.user_goal.category},
        initial_revealed_context={
            "category": task.user_goal.category,
            "budget_max": None,
            "must_have": {},
            "preference_weights": {},
        },
    )


def make_v1_direct_intent_task() -> SelectionTask:
    """Build the dialogue-layer task with direct structured intent access.

    This is the easier `v1` mode: dialogue exists, but the platform also gets
    the full structured intent state from reset time onward.
    """

    task = make_default_task(level=BenchmarkLevel.DIRECT_INTENT)
    return _clone_task_with_visibility(
        task,
        task_id="default_v1_direct_intent",
        initial_request_payload={"category": task.user_goal.category},
        initial_revealed_context={
            "category": task.user_goal.category,
            "budget_max": task.user_goal.budget_max,
            "must_have": dict(task.user_goal.must_have),
            "preference_weights": dict(task.preference_weights),
        },
    )


def make_v1_partial_intent_task() -> SelectionTask:
    """Build the dialogue-layer task that sits between v0 and hidden intent.

    `v1_partial_intent` keeps the same latent market as `v0`, but the platform
    now sees the user's natural-language request history in addition to the
    structured contract. Budget is revealed structurally; must-haves and
    preferences still need recovery.
    """

    task = make_default_task(level=BenchmarkLevel.PARTIAL_INTENT)
    return _clone_task_with_visibility(
        task,
        task_id="default_v1_partial_intent",
        initial_request_payload={"category": task.user_goal.category},
        initial_revealed_context={
            "category": task.user_goal.category,
            "budget_max": task.user_goal.budget_max,
            "must_have": {"noise_cancellation": True},
            "preference_weights": {},
        },
    )


def make_v1_dialogue_task() -> SelectionTask:
    """Backward-compatible alias for the partial-intent dialogue task."""

    return make_v1_partial_intent_task()


def make_v2_hidden_intent_task() -> SelectionTask:
    """Build the hidden-intent dialogue task.

    `v2` keeps the same opening request style as `v1`, but the structured
    contract falls back to the minimal v0 reset state. The platform therefore
    has to lean much more heavily on user dialogue and history.
    """

    task = make_default_task(level=BenchmarkLevel.HIDDEN_INTENT)
    return _clone_task_with_visibility(
        task,
        task_id="default_v2_hidden_intent",
        initial_request_payload={"category": task.user_goal.category},
        initial_revealed_context={
            "category": task.user_goal.category,
            "budget_max": None,
            "must_have": {},
            "preference_weights": {},
        },
    )
