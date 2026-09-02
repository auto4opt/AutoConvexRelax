#!/usr/bin/env python3
"""Summarize checkpoint-sweep evaluation outputs.

The script expects directories produced by sbatch_eval_checkpoint_sweep.sh:

  checkpoint_sweep/
    seed_42/
      checkpoint_100/
        eval_compare_all.json
        runner_stdout.log

It writes one CSV row per seed/checkpoint with root-bound, size, runtime, and
effective action-frequency statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ACTION_NAMES = [
    "sdp_relaxation",
    "mccormick_relaxation",
    "relax_integrality",
    "qcr",
    "bound_tightening",
]


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values) -> float:
    xs = [v for v in values if math.isfinite(v)]
    return sum(xs) / len(xs) if xs else math.nan


def pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


def parse_meta(path: Path) -> dict[str, str]:
    meta = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def parse_effective_actions(path: Path) -> Counter:
    """Count accepted actions only, identified by a following '[Changed] True'."""
    counts = Counter()
    if not path.exists():
        return counts
    pending = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Action: ([a-zA-Z0-9_]+) @", line)
        if match:
            pending = match.group(1)
            continue
        if "[Changed] True" in line and pending:
            counts[pending] += 1
            pending = None
        elif "[Changed] False" in line:
            pending = None
    return counts


def summarize_eval_json(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON: {path}")

    out = {
        "n_rows": len(rows),
        "mean_lb_improve_pct_gurobi": mean([safe_float(r.get("lb_improve_pct")) for r in rows]),
        "mean_lb_improve_pct_scip": mean([safe_float(r.get("lb_improve_scip_pct")) for r in rows]),
        "mean_rl_added_vars": mean([safe_float(r.get("rl_added_vars")) for r in rows]),
        "mean_rl_added_cons": mean([safe_float(r.get("rl_added_cons")) for r in rows]),
        "mean_rl_added_nnz": mean([safe_float(r.get("rl_added_nnz")) for r in rows]),
        "mean_rl_added_psd_size": mean([safe_float(r.get("rl_added_psd_size")) for r in rows]),
        "mean_rl_pipeline_time_sec": mean([safe_float(r.get("rl_pipeline_time_sec")) for r in rows]),
        "mean_rl_solve_time_sec": mean([safe_float(r.get("rl_solve_time_sec")) for r in rows]),
        "mean_sdp_added_nnz": mean([safe_float(r.get("baseline_sdp_added_nnz")) for r in rows]),
        "mean_mccormick_added_nnz": mean([safe_float(r.get("baseline_mccormick_added_nnz")) for r in rows]),
        "mean_sdp_pipeline_time_sec": mean([safe_float(r.get("baseline_sdp_pipeline_time_sec")) for r in rows]),
        "mean_mccormick_pipeline_time_sec": mean([safe_float(r.get("baseline_mccormick_pipeline_time_sec")) for r in rows]),
    }

    sdp_nnz_ratios = []
    mcc_nnz_ratios = []
    for row in rows:
        rl_nnz = safe_float(row.get("rl_added_nnz"))
        sdp_nnz = safe_float(row.get("baseline_sdp_added_nnz"))
        mcc_nnz = safe_float(row.get("baseline_mccormick_added_nnz"))
        if math.isfinite(rl_nnz) and math.isfinite(sdp_nnz) and sdp_nnz > 0:
            sdp_nnz_ratios.append(rl_nnz / sdp_nnz)
        if math.isfinite(rl_nnz) and math.isfinite(mcc_nnz) and mcc_nnz > 0:
            mcc_nnz_ratios.append(rl_nnz / mcc_nnz)
    out["mean_rl_nnz_over_sdp_nnz"] = mean(sdp_nnz_ratios)
    out["mean_rl_nnz_over_mccormick_nnz"] = mean(mcc_nnz_ratios)
    return out


def iter_run_dirs(root: Path):
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        for ckpt_dir in sorted(seed_dir.iterdir()):
            if ckpt_dir.is_dir():
                yield ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or (args.root / "checkpoint_sweep_summary.csv")
    rows = []
    for run_dir in iter_run_dirs(args.root):
        meta = parse_meta(run_dir / "meta.txt")
        seed = meta.get("seed") or run_dir.parent.name.replace("seed_", "")
        ckpt_tag = meta.get("ckpt_tag") or f"{run_dir.name}.pt"
        eval_stats = summarize_eval_json(run_dir / "eval_compare_all.json")
        action_counts = parse_effective_actions(run_dir / "runner_stdout.log")
        total_actions = sum(action_counts.values())

        row = {
            "seed": seed,
            "ckpt_tag": ckpt_tag,
            "run_dir": str(run_dir),
            **eval_stats,
            "effective_actions_total": total_actions,
        }
        for action in ACTION_NAMES:
            row[f"count_{action}"] = action_counts[action]
            row[f"pct_{action}"] = pct(action_counts[action], total_actions)
        rows.append(row)

    fieldnames = [
        "seed",
        "ckpt_tag",
        "n_rows",
        "mean_lb_improve_pct_gurobi",
        "mean_lb_improve_pct_scip",
        "mean_rl_added_nnz",
        "mean_rl_added_psd_size",
        "mean_rl_pipeline_time_sec",
        "mean_rl_nnz_over_sdp_nnz",
        "mean_rl_nnz_over_mccormick_nnz",
        "effective_actions_total",
    ]
    for action in ACTION_NAMES:
        fieldnames.extend([f"count_{action}", f"pct_{action}"])
    fieldnames.append("run_dir")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
