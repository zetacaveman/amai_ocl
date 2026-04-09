"""Opening-turn helpers for demos."""

from __future__ import annotations

from typing import Any

from conversational_consumer_selection.schemas import SelectionTask


def build_platform_opening_stage(task: SelectionTask) -> dict[str, Any]:
    del task
    return {
        "stage_type": "platform_opening",
        "act_type": "welcome",
    }


def build_user_initial_request(task: SelectionTask) -> dict[str, Any]:
    return {
        "stage_type": "user_request",
        "act_type": "request",
        "request": task.initial_user_request,
    }


def render_platform_opening(task: SelectionTask) -> str:
    del task
    return "Welcome. I can help you narrow down the best option today."


def render_buyer_opening(task: SelectionTask) -> str:
    request = task.initial_user_request
    category = str(request["category"])
    must_have = list(dict(request.get("must_have", {})).keys())

    category_phrase = _category_phrase(category)
    must_have_phrase = _must_have_phrase(must_have)
    sentence = f"Hi, I am looking for {category_phrase}{must_have_phrase}"
    budget = request.get("budget_max")
    if budget is not None:
        sentence += f", and I would like to stay under ${float(budget):.0f}"
    sentence += "."
    return sentence


def _category_phrase(category: str) -> str:
    lowered = category.replace("_", " ")
    if lowered.endswith("s"):
        return lowered
    return f"a {lowered}"


def _must_have_phrase(slots: list[str]) -> str:
    readable = [slot.replace("_", " ") for slot in slots]
    if not readable:
        return ""
    if len(readable) == 1:
        return f" with {readable[0]}"
    if len(readable) == 2:
        return f" with {readable[0]} and {readable[1]}"
    return f" with {', '.join(readable[:-1])}, and {readable[-1]}"
