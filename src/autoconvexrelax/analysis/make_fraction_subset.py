#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a small held-out subset of true fractional QCQP instances.

The full fraction finetuning set is intentionally larger than needed for the
paper extension experiment. This helper filters to genuinely fractional
families and samples a deterministic subset that can be passed directly to
run/evaluate.py with --no_split.
"""

from __future__ import annotations

import argparse
import pickle
import random
import re
from pathlib import Path


TRUE_FRACTION_RE = re.compile(r"^(FR[1-7]|NR4)(?:_|$)")


def _flatten_groups(groups):
    if not isinstance(groups, list):
        raise ValueError("input pickle must contain a list of problem groups")
    if not groups:
        return []
    if all(isinstance(group, list) for group in groups):
        return [prob for group in groups for prob in group]
    return list(groups)


def is_true_fraction_problem(prob) -> bool:
    """Return True for the FR1-FR7 and NR4 families used in the paper subset."""
    return bool(TRUE_FRACTION_RE.match(str(getattr(prob, "name", ""))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("vector_finetune_fraction.pkl"))
    parser.add_argument("--output", type=Path, default=Path("vector_fraction_eval_subset_50_seed42.pkl"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")

    with args.input.open("rb") as f:
        groups = pickle.load(f)

    problems = _flatten_groups(groups)
    candidates = [prob for prob in problems if is_true_fraction_problem(prob)]
    if len(candidates) < args.n:
        raise ValueError(
            f"requested {args.n} true fractional problems, but only found {len(candidates)} "
            f"in {args.input}"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump([selected], f)

    print(
        f"[DONE] wrote {args.output} with {len(selected)} true fractional problems "
        f"sampled from {len(candidates)} candidates (seed={args.seed})"
    )
    family_counts = {}
    for prob in selected:
        match = TRUE_FRACTION_RE.match(str(getattr(prob, "name", "")))
        family = match.group(1) if match else "OTHER"
        family_counts[family] = family_counts.get(family, 0) + 1
    for family in sorted(family_counts):
        print(f"  {family}: {family_counts[family]}")


if __name__ == "__main__":
    main()
