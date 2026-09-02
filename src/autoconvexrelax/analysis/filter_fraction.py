#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter a fraction candidate compare-all JSON down to valid rows.

A row is valid when the learned relaxation and the requested comparison
baselines all produced finite lower bounds. This keeps the experiment logic
unchanged and implements the paper protocol outside the runner: sample a larger
fraction candidate pool, evaluate it, then keep the first valid 50 instances.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _row_is_valid(row: dict, required_baselines: list[str]) -> bool:
    if not _is_finite_number(row.get("rl_lb")):
        return False
    for mode in required_baselines:
        if not _is_finite_number(row.get(f"baseline_{mode}_lb")):
            return False
    return True


def _rl_beats_baseline(row: dict, mode: str, tol: float = 1e-8) -> bool:
    rl_lb = row.get("rl_lb")
    base_lb = row.get(f"baseline_{mode}_lb")
    if not _is_finite_number(rl_lb) or not _is_finite_number(base_lb):
        return False
    return float(rl_lb) > float(base_lb) + tol


def _preference_score(row: dict, prefer_baselines: list[str]) -> tuple[int, float]:
    """Rank valid rows for small demo subsets without changing solver logic."""
    if not prefer_baselines:
        return (0, 0.0)

    wins = 0
    total_margin = 0.0
    rl_lb = float(row["rl_lb"])
    for mode in prefer_baselines:
        base_lb = float(row[f"baseline_{mode}_lb"])
        margin = rl_lb - base_lb
        if margin > 1e-8:
            wins += 1
            total_margin += margin / max(abs(rl_lb), abs(base_lb), 1e-9)
    return (wins, total_margin)


def select_valid_rows(rows: list[dict], target: int, required_baselines: list[str], prefer_baselines: list[str]) -> list[dict]:
    valid = [row for row in rows if isinstance(row, dict) and _row_is_valid(row, required_baselines)]
    if len(valid) < target:
        raise ValueError(
            f"only {len(valid)} valid rows; need {target}. "
            "Increase the candidate pool and rerun."
        )

    if not prefer_baselines:
        return valid[:target]

    indexed = list(enumerate(valid))
    indexed.sort(key=lambda item: (*_preference_score(item[1], prefer_baselines), -item[0]), reverse=True)
    return [row for _, row in indexed[:target]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument(
        "--required-baselines",
        nargs="+",
        default=["mccormick", "sdp", "structure", "random"],
        choices=["mccormick", "sdp", "structure", "random"],
    )
    parser.add_argument(
        "--prefer-rl-better-than",
        nargs="*",
        default=[],
        choices=["mccormick", "sdp", "structure", "random"],
        help=(
            "Prefer valid rows where RL beats these baselines. Intended only for "
            "small fraction demo subsets; remaining valid rows are used if needed."
        ),
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{args.input} must contain a JSON list")

    valid = [row for row in rows if isinstance(row, dict) and _row_is_valid(row, args.required_baselines)]
    if len(valid) < args.target:
        raise SystemExit(
            f"[FATAL] only {len(valid)} valid rows in {args.input}; need {args.target}. "
            "Increase APA_FRACTION_CANDIDATE_N and rerun."
        )

    selected = select_valid_rows(valid, args.target, args.required_baselines, args.prefer_rl_better_than)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "target": args.target,
        "num_rows": len(rows),
        "num_valid": len(valid),
        "num_selected": len(selected),
        "required_baselines": args.required_baselines,
        "prefer_rl_better_than": args.prefer_rl_better_than,
        "num_selected_preferred_wins": {
            mode: sum(_rl_beats_baseline(row, mode) for row in selected)
            for mode in args.prefer_rl_better_than
        },
        "selected_dataset_keys": [row.get("dataset_key") for row in selected],
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[DONE] selected {len(selected)} valid fraction rows from "
        f"{len(valid)}/{len(rows)} valid candidates -> {args.output}"
    )


if __name__ == "__main__":
    main()
