"""Helpers for importing and using AgenticPay.

The functions here provide the first stable integration boundary between
``aimai_ocl`` and AgenticPay. Higher-level adapters and controllers should call
these helpers instead of importing ``agenticpay`` directly all over the code
base.


中文翻译：Helpers for importing and using AgenticPay。"""

from __future__ import annotations

from typing import Any


def load_agenticpay() -> Any:
    """Import the AgenticPay package from the active Python environment.

    Returns:
        The imported ``agenticpay`` module object.

    Raises:
        ModuleNotFoundError: If AgenticPay is not importable in the current
            Python environment.
    

    中文翻译：导入 the AgenticPay package from the active Python environment。"""
    import agenticpay  # noqa: PLC0415

    return agenticpay


def make_env(env_id: str, **kwargs: Any) -> Any:
    """Create an AgenticPay environment via ``agenticpay.make``.

    Args:
        env_id: Registered AgenticPay environment id, for example
            ``"Task1_basic_price_negotiation-v0"``.
        **kwargs: Keyword arguments forwarded directly to ``agenticpay.make``.

    Returns:
        The environment instance returned by ``agenticpay.make``. The concrete
        type depends on the selected AgenticPay task.
    

    中文翻译：创建 an AgenticPay environment via ``agenticpay.make``。"""
    agenticpay = load_agenticpay()
    return agenticpay.make(env_id, **kwargs)
