"""Batch benchmark demo for quick structured smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversational_consumer_selection.policies import GreedySelectionPolicy
from conversational_consumer_selection.runner import run_benchmark
from conversational_consumer_selection.schemas import BenchmarkLevel
from conversational_consumer_selection.tasks import (
    make_default_task,
    make_v0_demo_task,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small batch smoke test over the structured benchmark variants."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/demo",
        help="Directory used for records.json / summary.json / summary.csv.",
    )
    args = parser.parse_args()

    tasks = [
        make_v0_demo_task(),
        make_default_task(level=BenchmarkLevel.DIRECT_INTENT),
        make_default_task(level=BenchmarkLevel.PARTIAL_INTENT),
        make_default_task(level=BenchmarkLevel.HIDDEN_INTENT),
    ]
    policies = {
        "single": GreedySelectionPolicy(clarify_missing_preferences=False),
        "clarify_then_commit": GreedySelectionPolicy(clarify_missing_preferences=True),
    }
    records, summaries = run_benchmark(tasks, policies, output_dir=args.output_dir)

    print("Episode records")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print()
    print("Summary")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote outputs to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
