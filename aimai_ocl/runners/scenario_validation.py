"""Scenario validation helpers for runner entry points.

中文翻译：Scenario validation helpers for runner entry points。"""

from __future__ import annotations

from typing import Any


def enforce_single_product_scenario(reset_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize single-product scenario inputs.

    Current v1 scope only supports a single product. The function accepts:

    - ``product_info`` as one dict
    - or ``products`` as a one-element list (normalized to ``product_info``)

    Args:
        reset_kwargs: Raw reset payload from caller.

    Returns:
        A normalized reset payload guaranteed to contain exactly one
        ``product_info`` dict.

    Raises:
        ValueError: If reset payload is missing product information or contains
            an unsupported multi-product structure.
    

    中文翻译：校验 and normalize single-product scenario inputs。"""
    normalized = dict(reset_kwargs)

    if "products" in normalized:
        products = normalized["products"]
        if not isinstance(products, list):
            raise ValueError("`products` must be a list when provided.")
        if len(products) != 1:
            raise ValueError(
                "Current scope only supports one product. "
                "Received multiple products in `products`.",
            )
        only_product = products[0]
        if not isinstance(only_product, dict):
            raise ValueError("`products[0]` must be a dict.")
        normalized.setdefault("product_info", only_product)
        normalized.pop("products", None)

    product_info = normalized.get("product_info")
    if product_info is None:
        raise ValueError(
            "Single-product scope requires `product_info` "
            "(or `products` with exactly one item).",
        )
    if not isinstance(product_info, dict):
        raise ValueError("`product_info` must be a dict for single-product scope.")

    return normalized
