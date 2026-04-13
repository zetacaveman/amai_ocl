"""Shared utilities for experiment scripts and CLI."""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_repo_path(script_file: str) -> Path:
    """Add repo root to sys.path and return it."""
    repo_root = Path(script_file).resolve().parent.parent
    vendored = repo_root / "agenticpay" / "agenticpay" / "__init__.py"
    if vendored.exists():
        vendored_root = str(repo_root / "agenticpay")
        if vendored_root not in sys.path:
            sys.path.insert(0, vendored_root)
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def apply_seed(seed: int) -> None:
    """Apply deterministic seed to Python and NumPy."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def resolve_output_dir(base: str | Path, prefix: str = "output") -> Path:
    """Create a timestamped output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass/object to JSON-serializable form."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "value"):
        return obj.value
    return obj


def write_json(path: Path, data: Any) -> str:
    """Write data to JSON file, return the path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=True, sort_keys=True, indent=2)
    return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write list of dicts to CSV, return the path string."""
    if not rows:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
