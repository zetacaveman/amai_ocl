"""YAML config loader with layered precedence: defaults < YAML < env vars < CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aimai_ocl.experiment import RunConfig

# Environment variable overrides.
_ENV_MAP: dict[str, str] = {
    "AIMAI_MODEL": "model",
    "OPENAI_MODEL": "model",
    "AIMAI_PROVIDER": "provider",
    "AIMAI_ENV_ID": "env_id",
    "AIMAI_SEED": "seed",
    "AIMAI_MAX_ROUNDS": "max_rounds",
    "AIMAI_API_KEY_ENV": "api_key_env",
}


def load_config(config_path: str | Path | None = None, *, cli_overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load RunConfig with 4-layer precedence: defaults < YAML < env < CLI."""
    values = RunConfig().to_dict()

    if config_path:
        yaml_values = _flatten_yaml(_load_yaml(Path(config_path)))
        values.update({k: v for k, v in yaml_values.items() if k in values})

    for env_var, field_name in _ENV_MAP.items():
        env_val = os.getenv(env_var)
        if env_val is not None:
            values[field_name] = env_val

    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None and key in values:
                values[key] = val

    return _to_run_config(values)


def load_experiment_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load a full experiment YAML (mode, arms, episodes, etc.)."""
    raw = _load_yaml(Path(config_path))

    # Resolve inheritance
    inherit = raw.get("inherit")
    if inherit:
        inherit_path = Path(config_path).parent / inherit
        base = _load_yaml(inherit_path)
        base.update(raw)
        raw = base

    return raw


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML required: pip install pyyaml") from exc
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_yaml(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML into flat key-value pairs."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result.update(_flatten_yaml(value, prefix=""))
        else:
            result[key] = value
    return result


def _to_run_config(values: dict[str, Any]) -> RunConfig:
    """Coerce flat dict into a RunConfig."""
    def _get(key: str, typ: type, default: Any) -> Any:
        val = values.get(key, default)
        if val is None:
            return default
        try:
            return typ(val)
        except (TypeError, ValueError):
            return default

    return RunConfig(
        env_id=_get("env_id", str, "Task1_basic_price_negotiation-v0"),
        model=_get("model", str, "gpt-4o-mini"),
        provider=_get("provider", str, "openai"),
        api_key_env=_get("api_key_env", str, "OPENAI_API_KEY"),
        seed=_get("seed", int, 42),
        max_rounds=_get("max_rounds", int, 10),
        initial_seller_price=_get("initial_seller_price", float, 180.0),
        buyer_max_price=_get("buyer_max_price", float, 120.0),
        seller_min_price=_get("seller_min_price", float, 90.0),
        user_requirement=_get("user_requirement", str, "I need a winter jacket"),
        product_name=_get("product_name", str, "Winter Jacket"),
        product_price=_get("product_price", float, 180.0),
        user_profile=_get("user_profile", str, "Budget-conscious and compares options before buying."),
    )
