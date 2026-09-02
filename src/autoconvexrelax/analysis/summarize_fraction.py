#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize fraction compare-all JSON in the paper table format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DISPLAY_NAMES = {
    "mccormick": "McCormick",
    "sdp": "SDP",
    "structure": "Heuristic",
    "random": "Random",
}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _normalized_lb_change(rl_lb: float, base_lb: float) -> float:
    denom = max(abs(rl_lb), abs(base_lb), 1e-9)
    return 100.0 * (rl_lb - base_lb) / denom


def _size_change(rl_size: float, base_size: float) -> float | None:
    if abs(base_size) <= 1e-12:
        return None
    return 100.0 * (rl_size / base_size - 1.0)


def summarize_rows(rows: list[dict], baselines: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for mode in baselines:
        lb_changes = []
        var_changes = []
        con_changes = []
        nnz_changes = []
        better = tie = worse = 0

        for row in rows:
            rl_lb = row.get("rl_lb")
            base_lb = row.get(f"baseline_{mode}_lb")
            if not (_finite_number(rl_lb) and _finite_number(base_lb)):
                continue

            rl_lb = float(rl_lb)
            base_lb = float(base_lb)
            diff = rl_lb - base_lb
            lb_changes.append(_normalized_lb_change(rl_lb, base_lb))
            if diff > 1e-8:
                better += 1
            elif diff < -1e-8:
                worse += 1
            else:
                tie += 1

            for suffix, target in (
                ("vars", var_changes),
                ("cons", con_changes),
                ("nnz", nnz_changes),
            ):
                rl_size = row.get(f"rl_added_{suffix}")
                base_size = row.get(f"baseline_{mode}_added_{suffix}")
                if _finite_number(rl_size) and _finite_number(base_size):
                    change = _size_change(float(rl_size), float(base_size))
                    if change is not None:
                        target.append(change)

        total = better + tie + worse
        if total:
            btw = [100.0 * better / total, 100.0 * tie / total, 100.0 * worse / total]
        else:
            btw = [None, None, None]

        summary[mode] = {
            "num_valid": total,
            "mean_lb_improvement_pct": _mean(lb_changes),
            "better_tie_worse_pct": btw,
            "delta_vars_pct": _mean(var_changes),
            "delta_cons_pct": _mean(con_changes),
            "delta_nnz_pct": _mean(nnz_changes),
        }
    return summary


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.2f}"


def _fmt_plain(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.1f}"


def latex_rows(summary: dict[str, dict], baselines: list[str]) -> str:
    lines = []
    for mode in baselines:
        item = summary[mode]
        btw = item["better_tie_worse_pct"]
        lines.append(
            f"{DISPLAY_NAMES.get(mode, mode)} & "
            f"{_fmt(item['mean_lb_improvement_pct'])} & "
            f"{_fmt_plain(btw[0])} / {_fmt_plain(btw[1])} / {_fmt_plain(btw[2])} & "
            f"{_fmt(item['delta_vars_pct'])} & "
            f"{_fmt(item['delta_cons_pct'])} & "
            f"{_fmt(item['delta_nnz_pct'])} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["mccormick", "sdp", "structure", "random"],
        choices=["mccormick", "sdp", "structure", "random"],
    )
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{args.input} must contain a JSON list")

    summary = summarize_rows(rows, args.baselines)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.latex.write_text(latex_rows(summary, args.baselines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {args.summary_json}")
    print(f"[DONE] wrote {args.latex}")


if __name__ == "__main__":
    main()
