"""Natural-language surface renderers for the structured benchmark core."""

from conversational_consumer_selection.surfaces.dialogue_decorator import (
    DialogueDecorator,
    DialogueModel,
    OpenAIDialogueModel,
)
from conversational_consumer_selection.surfaces.buyer_surface import render_buyer_response
from conversational_consumer_selection.surfaces.history_renderer import (
    render_history_transcript,
    render_platform_action_signal_tag,
    render_platform_action_surface,
)
from conversational_consumer_selection.surfaces.opening_renderer import (
    build_platform_opening_stage,
    build_user_initial_request,
    render_buyer_opening,
    render_platform_opening,
)

__all__ = [
    "build_platform_opening_stage",
    "build_user_initial_request",
    "DialogueDecorator",
    "DialogueModel",
    "OpenAIDialogueModel",
    "render_buyer_opening",
    "render_buyer_response",
    "render_history_transcript",
    "render_platform_opening",
    "render_platform_action_signal_tag",
    "render_platform_action_surface",
]
