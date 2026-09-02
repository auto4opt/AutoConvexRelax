#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate compare-all evaluation outputs across multiple seeds.

Expected layout:
  ROOT/
    seed_42/
      summary.csv
      eval_compare_all.json
    seed_52/
      summary.csv
      eval_compare_all.json

Outputs:
  - multiseed_seed_summary.csv
  - multiseed_problem_summary.csv
  - multiseed_overall_summary.csv
  - multiseed_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_PAIR_METRICS = [
    "lb_improve",
    "lb_improve_pct",
    "lb_improve_scip",
    "lb_improve_scip_pct",
    "rl_minus_mccormick",
    "rl_minus_mccormick_pct",
    "rl_minus_sdp",
    "rl_minus_sdp_pct",
    "mccormick_minus_root",
    "sdp_minus_root",
    "rl_pipeline_time_sec",
    "baseline_mccormick_pipeline_time_sec",
    "baseline_sdp_pipeline_time_sec",
    "rl_added_vars",
    "rl_added_cons",
    "rl_added_nnz",
    "rl_added_vars_pct_of_orig",
    "rl_added_cons_pct_of_orig",
    "rl_added_nnz_pct_of_orig",
    "baseline_mccormick_added_vars",
    "baseline_mccormick_added_cons",
    "baseline_mccormick_added_nnz",
    "baseline_mccormick_added_vars_pct_of_orig",
    "baseline_mccormick_added_cons_pct_of_orig",
    "baseline_mccormick_added_nnz_pct_of_orig",
    "baseline_sdp_added_vars",
    "baseline_sdp_added_cons",
    "baseline_sdp_added_nnz",
    "rl_vs_mccormick_added_vars_pct",
    "rl_vs_mccormick_added_cons_pct",
    "rl_vs_mccormick_added_nnz_pct",
    "rl_vs_sdp_added_vars_pct",
    "rl_vs_sdp_added_cons_pct",
    "rl_vs_sdp_added_nnz_pct",
    "baseline_sdp_added_vars_pct_of_orig",
    "baseline_sdp_added_cons_pct_of_orig",
    "baseline_sdp_added_nnz_pct_of_orig",
    "rl_time_minus_mccormick_time",
    "rl_time_minus_sdp_time",
    "rl_added_vars_minus_mccormick",
    "rl_added_cons_minus_mccormick",
    "rl_added_nnz_minus_mccormick",
    "rl_added_vars_minus_mccormick_pct_of_orig",
    "rl_added_cons_minus_mccormick_pct_of_orig",
    "rl_added_nnz_minus_mccormick_pct_of_orig",
    "rl_added_psd_size_minus_mccormick",
    "rl_added_vars_minus_sdp",
    "rl_added_cons_minus_sdp",
    "rl_added_nnz_minus_sdp",
    "rl_added_vars_minus_sdp_pct_of_orig",
    "rl_added_cons_minus_sdp_pct_of_orig",
    "rl_added_nnz_minus_sdp_pct_of_orig",
    "rl_added_psd_size_minus_sdp",
]


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def pct_improve(new_v, base_v, eps=1e-9):
    if new_v is None or base_v is None:
        return None
    denom = max(abs(base_v), abs(new_v), eps)
    return (new_v - base_v) / denom * 100.0


def pct_of_orig(added_v, orig_v, eps=1e-9):
    if added_v is None or orig_v is None:
        return None
    if abs(orig_v) <= eps:
        return None
    denom = abs(orig_v)
    return float(added_v) / denom * 100.0


def pct_vs_baseline(new_v, base_v, eps=1e-9):
    if new_v is None or base_v is None:
        return None
    if abs(base_v) <= eps:
        return None
    return (float(new_v) / float(base_v) - 1.0) * 100.0


def parse_seed_name(name: str) -> Optional[int]:
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def discover_seed_dirs(root_dir: str, seeds: Optional[Sequence[int]]) -> List[Tuple[int, str]]:
    found = []
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        seed = parse_seed_name(entry.name)
        if seed is None:
            continue
        if seeds is not None and seed not in seeds:
            continue
        found.append((seed, entry.path))
    found.sort(key=lambda x: x[0])
    return found


def normalize_row(raw: dict, seed: int) -> dict:
    row = dict(raw)
    row["seed"] = seed
    row["dataset_key"] = row.get("dataset_key") or row.get("name")

    root_bound = safe_float(row.get("gurobi_root_bound"))
    scip_root_bound = safe_float(row.get("scip_root_bound"))
    rl_lb = safe_float(row.get("rl_lb"))
    mcc_lb = safe_float(row.get("baseline_mccormick_lb"))
    sdp_lb = safe_float(row.get("baseline_sdp_lb"))
    rl_cost = safe_float(row.get("rl_cost"))
    mcc_cost = safe_float(row.get("baseline_mccormick_cost"))
    sdp_cost = safe_float(row.get("baseline_sdp_cost"))
    rl_time = safe_float(row.get("rl_pipeline_time_sec"))
    mcc_time = safe_float(row.get("baseline_mccormick_pipeline_time_sec"))
    sdp_time = safe_float(row.get("baseline_sdp_pipeline_time_sec"))
    if mcc_time is None:
        mcc_time = safe_float(row.get("baseline_pipeline_time_sec"))
    rl_added_vars = safe_float(row.get("rl_added_vars"))
    rl_added_cons = safe_float(row.get("rl_added_cons"))
    rl_added_nnz = safe_float(row.get("rl_added_nnz"))
    rl_added_psd = safe_float(row.get("rl_added_psd_size"))
    mcc_added_vars = safe_float(row.get("baseline_mccormick_added_vars"))
    mcc_added_cons = safe_float(row.get("baseline_mccormick_added_cons"))
    mcc_added_nnz = safe_float(row.get("baseline_mccormick_added_nnz"))
    mcc_added_psd = safe_float(row.get("baseline_mccormick_added_psd_size"))
    sdp_added_vars = safe_float(row.get("baseline_sdp_added_vars"))
    sdp_added_cons = safe_float(row.get("baseline_sdp_added_cons"))
    sdp_added_nnz = safe_float(row.get("baseline_sdp_added_nnz"))
    sdp_added_psd = safe_float(row.get("baseline_sdp_added_psd_size"))
    orig_vars_total = safe_float(row.get("orig_num_vars_total"))
    orig_cons_total = safe_float(row.get("orig_num_constraints_total"))
    orig_nnz_total = safe_float(row.get("orig_nnz_terms_total"))

    row["gurobi_root_bound"] = root_bound
    row["scip_root_bound"] = scip_root_bound
    row["rl_lb"] = rl_lb
    row["baseline_mccormick_lb"] = mcc_lb
    row["baseline_sdp_lb"] = sdp_lb
    row["rl_cost"] = rl_cost
    row["baseline_mccormick_cost"] = mcc_cost
    row["baseline_sdp_cost"] = sdp_cost
    row["rl_pipeline_time_sec"] = rl_time
    row["baseline_mccormick_pipeline_time_sec"] = mcc_time
    row["baseline_sdp_pipeline_time_sec"] = sdp_time
    row["rl_added_vars"] = rl_added_vars
    row["rl_added_cons"] = rl_added_cons
    row["rl_added_nnz"] = rl_added_nnz
    row["rl_added_psd_size"] = rl_added_psd
    row["baseline_mccormick_added_vars"] = mcc_added_vars
    row["baseline_mccormick_added_cons"] = mcc_added_cons
    row["baseline_mccormick_added_nnz"] = mcc_added_nnz
    row["baseline_mccormick_added_psd_size"] = mcc_added_psd
    row["baseline_sdp_added_vars"] = sdp_added_vars
    row["baseline_sdp_added_cons"] = sdp_added_cons
    row["baseline_sdp_added_nnz"] = sdp_added_nnz
    row["baseline_sdp_added_psd_size"] = sdp_added_psd

    derived = {
        "lb_improve": safe_float(row.get("lb_improve")),
        "lb_improve_pct": safe_float(row.get("lb_improve_pct")),
        "lb_improve_scip": safe_float(row.get("lb_improve_scip")),
        "lb_improve_scip_pct": safe_float(row.get("lb_improve_scip_pct")),
        "rl_minus_mccormick": safe_float(row.get("rl_minus_mccormick")),
        "rl_minus_mccormick_pct": safe_float(row.get("rl_minus_mccormick_pct")),
        "rl_minus_sdp": safe_float(row.get("rl_minus_sdp")),
        "rl_minus_sdp_pct": safe_float(row.get("rl_minus_sdp_pct")),
        "mccormick_minus_root": safe_float(row.get("mccormick_minus_root")),
        "sdp_minus_root": safe_float(row.get("sdp_minus_root")),
        "rl_time_minus_mccormick_time": None,
        "rl_time_minus_sdp_time": None,
        "rl_added_vars_minus_mccormick": None,
        "rl_added_cons_minus_mccormick": None,
        "rl_added_nnz_minus_mccormick": None,
        "rl_vs_mccormick_added_vars_pct": None,
        "rl_vs_mccormick_added_cons_pct": None,
        "rl_vs_mccormick_added_nnz_pct": None,
        "rl_added_vars_minus_mccormick_pct_of_orig": None,
        "rl_added_cons_minus_mccormick_pct_of_orig": None,
        "rl_added_nnz_minus_mccormick_pct_of_orig": None,
        "rl_added_psd_size_minus_mccormick": None,
        "rl_added_vars_minus_sdp": None,
        "rl_added_cons_minus_sdp": None,
        "rl_added_nnz_minus_sdp": None,
        "rl_vs_sdp_added_vars_pct": None,
        "rl_vs_sdp_added_cons_pct": None,
        "rl_vs_sdp_added_nnz_pct": None,
        "rl_added_vars_minus_sdp_pct_of_orig": None,
        "rl_added_cons_minus_sdp_pct_of_orig": None,
        "rl_added_nnz_minus_sdp_pct_of_orig": None,
        "rl_added_psd_size_minus_sdp": None,
        "rl_added_vars_pct_of_orig": None,
        "rl_added_cons_pct_of_orig": None,
        "rl_added_nnz_pct_of_orig": None,
        "baseline_mccormick_added_vars_pct_of_orig": None,
        "baseline_mccormick_added_cons_pct_of_orig": None,
        "baseline_mccormick_added_nnz_pct_of_orig": None,
        "baseline_sdp_added_vars_pct_of_orig": None,
        "baseline_sdp_added_cons_pct_of_orig": None,
        "baseline_sdp_added_nnz_pct_of_orig": None,
    }

    if derived["lb_improve"] is None and rl_lb is not None and root_bound is not None:
        derived["lb_improve"] = rl_lb - root_bound
    if derived["lb_improve_pct"] is None:
        derived["lb_improve_pct"] = pct_improve(rl_lb, root_bound)
    if derived["lb_improve_scip"] is None and rl_lb is not None and scip_root_bound is not None:
        derived["lb_improve_scip"] = rl_lb - scip_root_bound
    if derived["lb_improve_scip_pct"] is None:
        derived["lb_improve_scip_pct"] = pct_improve(rl_lb, scip_root_bound)
    if derived["rl_minus_mccormick"] is None and rl_lb is not None and mcc_lb is not None:
        derived["rl_minus_mccormick"] = rl_lb - mcc_lb
    if derived["rl_minus_mccormick_pct"] is None:
        derived["rl_minus_mccormick_pct"] = pct_improve(rl_lb, mcc_lb)
    if derived["rl_minus_sdp"] is None and rl_lb is not None and sdp_lb is not None:
        derived["rl_minus_sdp"] = rl_lb - sdp_lb
    if derived["rl_minus_sdp_pct"] is None:
        derived["rl_minus_sdp_pct"] = pct_improve(rl_lb, sdp_lb)
    if derived["mccormick_minus_root"] is None and mcc_lb is not None and root_bound is not None:
        derived["mccormick_minus_root"] = mcc_lb - root_bound
    if derived["sdp_minus_root"] is None and sdp_lb is not None and root_bound is not None:
        derived["sdp_minus_root"] = sdp_lb - root_bound
    if rl_cost is None and rl_lb is not None:
        rl_cost = rl_lb
        row["rl_cost"] = rl_cost
    if mcc_cost is None and mcc_lb is not None:
        mcc_cost = mcc_lb
        row["baseline_mccormick_cost"] = mcc_cost
    if sdp_cost is None and sdp_lb is not None:
        sdp_cost = sdp_lb
        row["baseline_sdp_cost"] = sdp_cost
    if rl_time is not None and mcc_time is not None:
        derived["rl_time_minus_mccormick_time"] = rl_time - mcc_time
    if rl_time is not None and sdp_time is not None:
        derived["rl_time_minus_sdp_time"] = rl_time - sdp_time
    if rl_added_vars is not None and mcc_added_vars is not None:
        derived["rl_added_vars_minus_mccormick"] = rl_added_vars - mcc_added_vars
    if rl_added_cons is not None and mcc_added_cons is not None:
        derived["rl_added_cons_minus_mccormick"] = rl_added_cons - mcc_added_cons
    if rl_added_nnz is not None and mcc_added_nnz is not None:
        derived["rl_added_nnz_minus_mccormick"] = rl_added_nnz - mcc_added_nnz
    if rl_added_psd is not None and mcc_added_psd is not None:
        derived["rl_added_psd_size_minus_mccormick"] = rl_added_psd - mcc_added_psd
    if rl_added_vars is not None and sdp_added_vars is not None:
        derived["rl_added_vars_minus_sdp"] = rl_added_vars - sdp_added_vars
    if rl_added_cons is not None and sdp_added_cons is not None:
        derived["rl_added_cons_minus_sdp"] = rl_added_cons - sdp_added_cons
    if rl_added_nnz is not None and sdp_added_nnz is not None:
        derived["rl_added_nnz_minus_sdp"] = rl_added_nnz - sdp_added_nnz
    if rl_added_psd is not None and sdp_added_psd is not None:
        derived["rl_added_psd_size_minus_sdp"] = rl_added_psd - sdp_added_psd

    derived["rl_vs_mccormick_added_vars_pct"] = pct_vs_baseline(rl_added_vars, mcc_added_vars)
    derived["rl_vs_mccormick_added_cons_pct"] = pct_vs_baseline(rl_added_cons, mcc_added_cons)
    derived["rl_vs_mccormick_added_nnz_pct"] = pct_vs_baseline(rl_added_nnz, mcc_added_nnz)
    derived["rl_vs_sdp_added_vars_pct"] = pct_vs_baseline(rl_added_vars, sdp_added_vars)
    derived["rl_vs_sdp_added_cons_pct"] = pct_vs_baseline(rl_added_cons, sdp_added_cons)
    derived["rl_vs_sdp_added_nnz_pct"] = pct_vs_baseline(rl_added_nnz, sdp_added_nnz)

    derived["rl_added_vars_pct_of_orig"] = pct_of_orig(rl_added_vars, orig_vars_total)
    derived["rl_added_cons_pct_of_orig"] = pct_of_orig(rl_added_cons, orig_cons_total)
    derived["rl_added_nnz_pct_of_orig"] = pct_of_orig(rl_added_nnz, orig_nnz_total)
    derived["baseline_mccormick_added_vars_pct_of_orig"] = pct_of_orig(mcc_added_vars, orig_vars_total)
    derived["baseline_mccormick_added_cons_pct_of_orig"] = pct_of_orig(mcc_added_cons, orig_cons_total)
    derived["baseline_mccormick_added_nnz_pct_of_orig"] = pct_of_orig(mcc_added_nnz, orig_nnz_total)
    derived["baseline_sdp_added_vars_pct_of_orig"] = pct_of_orig(sdp_added_vars, orig_vars_total)
    derived["baseline_sdp_added_cons_pct_of_orig"] = pct_of_orig(sdp_added_cons, orig_cons_total)
    derived["baseline_sdp_added_nnz_pct_of_orig"] = pct_of_orig(sdp_added_nnz, orig_nnz_total)

    if (
        derived["rl_added_vars_pct_of_orig"] is not None
        and derived["baseline_mccormick_added_vars_pct_of_orig"] is not None
    ):
        derived["rl_added_vars_minus_mccormick_pct_of_orig"] = (
            derived["rl_added_vars_pct_of_orig"] - derived["baseline_mccormick_added_vars_pct_of_orig"]
        )
    if (
        derived["rl_added_cons_pct_of_orig"] is not None
        and derived["baseline_mccormick_added_cons_pct_of_orig"] is not None
    ):
        derived["rl_added_cons_minus_mccormick_pct_of_orig"] = (
            derived["rl_added_cons_pct_of_orig"] - derived["baseline_mccormick_added_cons_pct_of_orig"]
        )
    if (
        derived["rl_added_nnz_pct_of_orig"] is not None
        and derived["baseline_mccormick_added_nnz_pct_of_orig"] is not None
    ):
        derived["rl_added_nnz_minus_mccormick_pct_of_orig"] = (
            derived["rl_added_nnz_pct_of_orig"] - derived["baseline_mccormick_added_nnz_pct_of_orig"]
        )
    if (
        derived["rl_added_vars_pct_of_orig"] is not None
        and derived["baseline_sdp_added_vars_pct_of_orig"] is not None
    ):
        derived["rl_added_vars_minus_sdp_pct_of_orig"] = (
            derived["rl_added_vars_pct_of_orig"] - derived["baseline_sdp_added_vars_pct_of_orig"]
        )
    if (
        derived["rl_added_cons_pct_of_orig"] is not None
        and derived["baseline_sdp_added_cons_pct_of_orig"] is not None
    ):
        derived["rl_added_cons_minus_sdp_pct_of_orig"] = (
            derived["rl_added_cons_pct_of_orig"] - derived["baseline_sdp_added_cons_pct_of_orig"]
        )
    if (
        derived["rl_added_nnz_pct_of_orig"] is not None
        and derived["baseline_sdp_added_nnz_pct_of_orig"] is not None
    ):
        derived["rl_added_nnz_minus_sdp_pct_of_orig"] = (
            derived["rl_added_nnz_pct_of_orig"] - derived["baseline_sdp_added_nnz_pct_of_orig"]
        )

    for key, val in derived.items():
        row[key] = val

    for key in DEFAULT_PAIR_METRICS:
        row[key] = safe_float(row.get(key))

    row["name"] = str(row.get("name", "noname"))
    return row


def load_rows_from_csv(path: str, seed: int) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(normalize_row(raw, seed))
    return rows


def load_rows_from_json(path: str, seed: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data)}")
    return [normalize_row(raw, seed) for raw in data if isinstance(raw, dict)]


def load_seed_rows(seed_dir: str, seed: int) -> List[dict]:
    csv_path = os.path.join(seed_dir, "summary.csv")
    json_path = os.path.join(seed_dir, "eval_compare_all.json")
    if os.path.exists(json_path):
        return load_rows_from_json(json_path, seed)
    if os.path.exists(csv_path):
        return load_rows_from_csv(csv_path, seed)
    raise FileNotFoundError(f"Neither summary.csv nor eval_compare_all.json exists in {seed_dir}")


def clean_values(rows: Iterable[dict], key: str) -> List[float]:
    xs = []
    for row in rows:
        v = safe_float(row.get(key))
        if v is not None:
            xs.append(v)
    return xs


def summarize_values(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    out = {
        "count": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "std": stdev(xs) if len(xs) >= 2 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }
    return out


def add_win_tie_loss(
    out: dict,
    rows: Sequence[dict],
    value_key: str,
    prefix: str,
    eps: float,
) -> None:
    vals = clean_values(rows, value_key)
    out[f"{prefix}_valid"] = len(vals)
    out[f"{prefix}_win"] = sum(1 for v in vals if v > eps)
    out[f"{prefix}_tie"] = sum(1 for v in vals if abs(v) <= eps)
    out[f"{prefix}_loss"] = sum(1 for v in vals if v < -eps)
    denom = len(vals) or 1
    out[f"{prefix}_win_rate"] = out[f"{prefix}_win"] / denom
    out[f"{prefix}_tie_rate"] = out[f"{prefix}_tie"] / denom
    out[f"{prefix}_loss_rate"] = out[f"{prefix}_loss"] / denom


def add_win_tie_loss_minimize(
    out: dict,
    rows: Sequence[dict],
    value_key: str,
    prefix: str,
    eps: float,
) -> None:
    vals = clean_values(rows, value_key)
    out[f"{prefix}_valid"] = len(vals)
    out[f"{prefix}_win"] = sum(1 for v in vals if v < -eps)
    out[f"{prefix}_tie"] = sum(1 for v in vals if abs(v) <= eps)
    out[f"{prefix}_loss"] = sum(1 for v in vals if v > eps)
    denom = len(vals) or 1
    out[f"{prefix}_win_rate"] = out[f"{prefix}_win"] / denom
    out[f"{prefix}_tie_rate"] = out[f"{prefix}_tie"] / denom
    out[f"{prefix}_loss_rate"] = out[f"{prefix}_loss"] / denom


def summarize_seed(seed: int, rows: Sequence[dict], metrics: Sequence[str], eps: float) -> dict:
    out = {"seed": seed, "n_cases": len(rows)}
    for metric in metrics:
        stats = summarize_values(clean_values(rows, metric))
        for suffix, val in stats.items():
            out[f"{metric}_{suffix}"] = val

    add_win_tie_loss(out, rows, "lb_improve", "rl_vs_root", eps)
    add_win_tie_loss(out, rows, "lb_improve_scip", "rl_vs_scip", eps)
    add_win_tie_loss(out, rows, "rl_minus_mccormick", "rl_vs_mccormick", eps)
    add_win_tie_loss(out, rows, "rl_minus_sdp", "rl_vs_sdp", eps)
    return out


def summarize_problem(rows: Sequence[dict], metrics: Sequence[str], eps: float) -> dict:
    first = rows[0]
    out = {
        "dataset_key": first.get("dataset_key"),
        "name": first.get("name"),
        "n_seeds": len(rows),
        "seed_list": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])),
    }
    for metric in metrics:
        stats = summarize_values(clean_values(rows, metric))
        for suffix, val in stats.items():
            out[f"{metric}_{suffix}"] = val

    add_win_tie_loss(out, rows, "lb_improve", "rl_vs_root", eps)
    add_win_tie_loss(out, rows, "lb_improve_scip", "rl_vs_scip", eps)
    add_win_tie_loss(out, rows, "rl_minus_mccormick", "rl_vs_mccormick", eps)
    add_win_tie_loss(out, rows, "rl_minus_sdp", "rl_vs_sdp", eps)
    return out


def summarize_overall(rows: Sequence[dict], metrics: Sequence[str], eps: float) -> List[dict]:
    records = []
    for metric in metrics:
        stats = summarize_values(clean_values(rows, metric))
        records.append({"metric": metric, **stats})

    def add_cmp(metric: str, label: str):
        vals = clean_values(rows, metric)
        record = {
            "metric": label,
            "count": len(vals),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "win": sum(1 for v in vals if v > eps),
            "tie": sum(1 for v in vals if abs(v) <= eps),
            "loss": sum(1 for v in vals if v < -eps),
        }
        denom = len(vals) or 1
        record["win_rate"] = record["win"] / denom
        record["tie_rate"] = record["tie"] / denom
        record["loss_rate"] = record["loss"] / denom
        records.append(record)

    add_cmp("lb_improve", "cmp_rl_vs_root")
    add_cmp("lb_improve_scip", "cmp_rl_vs_scip")
    add_cmp("rl_minus_mccormick", "cmp_rl_vs_mccormick")
    add_cmp("rl_minus_sdp", "cmp_rl_vs_sdp")
    return records


def format_triplet(win: Optional[float], tie: Optional[float], loss: Optional[float], digits: int = 1) -> str:
    if win is None or tie is None or loss is None:
        return "NA"
    return f"{win * 100:.{digits}f}/{tie * 100:.{digits}f}/{loss * 100:.{digits}f}"


def build_lower_bound_tables(
    seed_summaries: Sequence[dict],
    overall_rows: Sequence[dict],
    all_rows: Sequence[dict],
    eps: float,
) -> Tuple[List[dict], List[dict]]:
    per_seed = []
    for row in seed_summaries:
        one = {
            "seed": row["seed"],
            "n_cases": row["n_cases"],
            "lb_improve_mean": row.get("lb_improve_mean"),
            "lb_improve_median": row.get("lb_improve_median"),
            "lb_improve_pct_mean": row.get("lb_improve_pct_mean"),
            "lb_improve_pct_median": row.get("lb_improve_pct_median"),
            "scip_lb_improve_mean": row.get("lb_improve_scip_mean"),
            "scip_lb_improve_median": row.get("lb_improve_scip_median"),
            "scip_lb_improve_pct_mean": row.get("lb_improve_scip_pct_mean"),
            "scip_lb_improve_pct_median": row.get("lb_improve_scip_pct_median"),
            "root_better": row.get("rl_vs_root_win"),
            "root_tie": row.get("rl_vs_root_tie"),
            "root_worse": row.get("rl_vs_root_loss"),
            "root_better_rate": row.get("rl_vs_root_win_rate"),
            "root_tie_rate": row.get("rl_vs_root_tie_rate"),
            "root_worse_rate": row.get("rl_vs_root_loss_rate"),
            "root_better_tie_worse_pct": format_triplet(
                row.get("rl_vs_root_win_rate"),
                row.get("rl_vs_root_tie_rate"),
                row.get("rl_vs_root_loss_rate"),
            ),
            "scip_better": row.get("rl_vs_scip_win"),
            "scip_tie": row.get("rl_vs_scip_tie"),
            "scip_worse": row.get("rl_vs_scip_loss"),
            "scip_better_rate": row.get("rl_vs_scip_win_rate"),
            "scip_tie_rate": row.get("rl_vs_scip_tie_rate"),
            "scip_worse_rate": row.get("rl_vs_scip_loss_rate"),
            "scip_better_tie_worse_pct": format_triplet(
                row.get("rl_vs_scip_win_rate"),
                row.get("rl_vs_scip_tie_rate"),
                row.get("rl_vs_scip_loss_rate"),
            ),
            "rl_pipeline_time_sec_mean": row.get("rl_pipeline_time_sec_mean"),
        }
        per_seed.append(one)

    overall_by_metric = {row["metric"]: row for row in overall_rows}
    root_cmp = overall_by_metric.get("cmp_rl_vs_root", {})
    scip_cmp = overall_by_metric.get("cmp_rl_vs_scip", {})
    lb_stats = overall_by_metric.get("lb_improve", {})
    lb_pct_stats = overall_by_metric.get("lb_improve_pct", {})
    scip_stats = overall_by_metric.get("lb_improve_scip", {})
    scip_pct_stats = overall_by_metric.get("lb_improve_scip_pct", {})
    time_stats = overall_by_metric.get("rl_pipeline_time_sec", {})

    overall = [
        {
            "scope": "overall",
            "n_seed_problem_pairs": len(all_rows),
            "lb_improve_mean": lb_stats.get("mean"),
            "lb_improve_median": lb_stats.get("median"),
            "lb_improve_pct_mean": lb_pct_stats.get("mean"),
            "lb_improve_pct_median": lb_pct_stats.get("median"),
            "root_better": root_cmp.get("win"),
            "root_tie": root_cmp.get("tie"),
            "root_worse": root_cmp.get("loss"),
            "root_better_rate": root_cmp.get("win_rate"),
            "root_tie_rate": root_cmp.get("tie_rate"),
            "root_worse_rate": root_cmp.get("loss_rate"),
            "root_better_tie_worse_pct": format_triplet(
                root_cmp.get("win_rate"),
                root_cmp.get("tie_rate"),
                root_cmp.get("loss_rate"),
            ),
            "scip_lb_improve_mean": scip_stats.get("mean"),
            "scip_lb_improve_median": scip_stats.get("median"),
            "scip_lb_improve_pct_mean": scip_pct_stats.get("mean"),
            "scip_lb_improve_pct_median": scip_pct_stats.get("median"),
            "scip_better": scip_cmp.get("win"),
            "scip_tie": scip_cmp.get("tie"),
            "scip_worse": scip_cmp.get("loss"),
            "scip_better_rate": scip_cmp.get("win_rate"),
            "scip_tie_rate": scip_cmp.get("tie_rate"),
            "scip_worse_rate": scip_cmp.get("loss_rate"),
            "scip_better_tie_worse_pct": format_triplet(
                scip_cmp.get("win_rate"),
                scip_cmp.get("tie_rate"),
                scip_cmp.get("loss_rate"),
            ),
            "rl_pipeline_time_sec_mean": time_stats.get("mean"),
            "eps": eps,
        }
    ]
    return per_seed, overall


def build_baseline_tables(
    seed_summaries: Sequence[dict],
    overall_rows: Sequence[dict],
    all_rows: Sequence[dict],
    eps: float,
) -> Tuple[List[dict], List[dict]]:
    per_seed = []
    for row in seed_summaries:
        one = {
            "seed": row["seed"],
            "n_cases": row["n_cases"],
            "rl_minus_mccormick_mean": row.get("rl_minus_mccormick_mean"),
            "rl_minus_mccormick_median": row.get("rl_minus_mccormick_median"),
            "mccormick_better": row.get("rl_vs_mccormick_win"),
            "mccormick_tie": row.get("rl_vs_mccormick_tie"),
            "mccormick_worse": row.get("rl_vs_mccormick_loss"),
            "mccormick_better_rate": row.get("rl_vs_mccormick_win_rate"),
            "mccormick_tie_rate": row.get("rl_vs_mccormick_tie_rate"),
            "mccormick_worse_rate": row.get("rl_vs_mccormick_loss_rate"),
            "mccormick_better_tie_worse_pct": format_triplet(
                row.get("rl_vs_mccormick_win_rate"),
                row.get("rl_vs_mccormick_tie_rate"),
                row.get("rl_vs_mccormick_loss_rate"),
            ),
            "rl_minus_sdp_mean": row.get("rl_minus_sdp_mean"),
            "rl_minus_sdp_median": row.get("rl_minus_sdp_median"),
            "sdp_better": row.get("rl_vs_sdp_win"),
            "sdp_tie": row.get("rl_vs_sdp_tie"),
            "sdp_worse": row.get("rl_vs_sdp_loss"),
            "sdp_better_rate": row.get("rl_vs_sdp_win_rate"),
            "sdp_tie_rate": row.get("rl_vs_sdp_tie_rate"),
            "sdp_worse_rate": row.get("rl_vs_sdp_loss_rate"),
            "sdp_better_tie_worse_pct": format_triplet(
                row.get("rl_vs_sdp_win_rate"),
                row.get("rl_vs_sdp_tie_rate"),
                row.get("rl_vs_sdp_loss_rate"),
            ),
            "rl_pipeline_time_sec_mean": row.get("rl_pipeline_time_sec_mean"),
        }
        per_seed.append(one)

    overall_by_metric = {row["metric"]: row for row in overall_rows}
    mcc_stats = overall_by_metric.get("rl_minus_mccormick", {})
    sdp_stats = overall_by_metric.get("rl_minus_sdp", {})
    mcc_cmp = overall_by_metric.get("cmp_rl_vs_mccormick", {})
    sdp_cmp = overall_by_metric.get("cmp_rl_vs_sdp", {})
    time_stats = overall_by_metric.get("rl_pipeline_time_sec", {})

    overall = [
        {
            "scope": "overall",
            "n_seed_problem_pairs": len(all_rows),
            "rl_minus_mccormick_mean": mcc_stats.get("mean"),
            "rl_minus_mccormick_median": mcc_stats.get("median"),
            "mccormick_better": mcc_cmp.get("win"),
            "mccormick_tie": mcc_cmp.get("tie"),
            "mccormick_worse": mcc_cmp.get("loss"),
            "mccormick_better_rate": mcc_cmp.get("win_rate"),
            "mccormick_tie_rate": mcc_cmp.get("tie_rate"),
            "mccormick_worse_rate": mcc_cmp.get("loss_rate"),
            "mccormick_better_tie_worse_pct": format_triplet(
                mcc_cmp.get("win_rate"),
                mcc_cmp.get("tie_rate"),
                mcc_cmp.get("loss_rate"),
            ),
            "rl_minus_sdp_mean": sdp_stats.get("mean"),
            "rl_minus_sdp_median": sdp_stats.get("median"),
            "sdp_better": sdp_cmp.get("win"),
            "sdp_tie": sdp_cmp.get("tie"),
            "sdp_worse": sdp_cmp.get("loss"),
            "sdp_better_rate": sdp_cmp.get("win_rate"),
            "sdp_tie_rate": sdp_cmp.get("tie_rate"),
            "sdp_worse_rate": sdp_cmp.get("loss_rate"),
            "sdp_better_tie_worse_pct": format_triplet(
                sdp_cmp.get("win_rate"),
                sdp_cmp.get("tie_rate"),
                sdp_cmp.get("loss_rate"),
            ),
            "rl_pipeline_time_sec_mean": time_stats.get("mean"),
            "eps": eps,
        }
    ]
    return per_seed, overall


def build_baseline_lb_tables(
    seed_summaries: Sequence[dict],
    overall_rows: Sequence[dict],
    all_rows: Sequence[dict],
    eps: float,
) -> Tuple[List[dict], List[dict]]:
    def fallback_pct(new_v: Optional[float], base_v: Optional[float]) -> Optional[float]:
        return pct_improve(new_v, base_v)

    per_seed = []
    for row in seed_summaries:
        mcc_pct_mean = row.get("rl_minus_mccormick_pct_mean")
        if mcc_pct_mean is None:
            mcc_pct_mean = fallback_pct(row.get("lb_improve_mean"), row.get("mccormick_minus_root_mean"))
        mcc_pct_median = row.get("rl_minus_mccormick_pct_median")
        if mcc_pct_median is None:
            mcc_pct_median = fallback_pct(row.get("lb_improve_median"), row.get("mccormick_minus_root_median"))

        sdp_pct_mean = row.get("rl_minus_sdp_pct_mean")
        if sdp_pct_mean is None:
            sdp_pct_mean = fallback_pct(row.get("lb_improve_mean"), row.get("sdp_minus_root_mean"))
        sdp_pct_median = row.get("rl_minus_sdp_pct_median")
        if sdp_pct_median is None:
            sdp_pct_median = fallback_pct(row.get("lb_improve_median"), row.get("sdp_minus_root_median"))

        per_seed.append(
            {
                "seed": row["seed"],
                "n_cases": row["n_cases"],
                "rl_minus_mccormick_lb_mean": row.get("rl_minus_mccormick_mean"),
                "rl_minus_mccormick_lb_median": row.get("rl_minus_mccormick_median"),
                "rl_minus_mccormick_lb_pct_mean": mcc_pct_mean,
                "rl_minus_mccormick_lb_pct_median": mcc_pct_median,
                "mccormick_lb_better": row.get("rl_vs_mccormick_win"),
                "mccormick_lb_tie": row.get("rl_vs_mccormick_tie"),
                "mccormick_lb_worse": row.get("rl_vs_mccormick_loss"),
                "mccormick_lb_better_rate": row.get("rl_vs_mccormick_win_rate"),
                "mccormick_lb_tie_rate": row.get("rl_vs_mccormick_tie_rate"),
                "mccormick_lb_worse_rate": row.get("rl_vs_mccormick_loss_rate"),
                "mccormick_lb_better_tie_worse_pct": format_triplet(
                    row.get("rl_vs_mccormick_win_rate"),
                    row.get("rl_vs_mccormick_tie_rate"),
                    row.get("rl_vs_mccormick_loss_rate"),
                ),
                "rl_minus_sdp_lb_mean": row.get("rl_minus_sdp_mean"),
                "rl_minus_sdp_lb_median": row.get("rl_minus_sdp_median"),
                "rl_minus_sdp_lb_pct_mean": sdp_pct_mean,
                "rl_minus_sdp_lb_pct_median": sdp_pct_median,
                "sdp_lb_better": row.get("rl_vs_sdp_win"),
                "sdp_lb_tie": row.get("rl_vs_sdp_tie"),
                "sdp_lb_worse": row.get("rl_vs_sdp_loss"),
                "sdp_lb_better_rate": row.get("rl_vs_sdp_win_rate"),
                "sdp_lb_tie_rate": row.get("rl_vs_sdp_tie_rate"),
                "sdp_lb_worse_rate": row.get("rl_vs_sdp_loss_rate"),
                "sdp_lb_better_tie_worse_pct": format_triplet(
                    row.get("rl_vs_sdp_win_rate"),
                    row.get("rl_vs_sdp_tie_rate"),
                    row.get("rl_vs_sdp_loss_rate"),
                ),
            }
        )

    overall_by_metric = {row["metric"]: row for row in overall_rows}
    mcc_stats = overall_by_metric.get("rl_minus_mccormick", {})
    mcc_pct_stats = overall_by_metric.get("rl_minus_mccormick_pct", {})
    sdp_stats = overall_by_metric.get("rl_minus_sdp", {})
    sdp_pct_stats = overall_by_metric.get("rl_minus_sdp_pct", {})
    lb_stats = overall_by_metric.get("lb_improve", {})
    mcc_root_stats = overall_by_metric.get("mccormick_minus_root", {})
    sdp_root_stats = overall_by_metric.get("sdp_minus_root", {})
    mcc_cmp = overall_by_metric.get("cmp_rl_vs_mccormick", {})
    sdp_cmp = overall_by_metric.get("cmp_rl_vs_sdp", {})

    mcc_pct_mean = mcc_pct_stats.get("mean")
    if mcc_pct_mean is None:
        mcc_pct_mean = fallback_pct(lb_stats.get("mean"), mcc_root_stats.get("mean"))
    mcc_pct_median = mcc_pct_stats.get("median")
    if mcc_pct_median is None:
        mcc_pct_median = fallback_pct(lb_stats.get("median"), mcc_root_stats.get("median"))

    sdp_pct_mean = sdp_pct_stats.get("mean")
    if sdp_pct_mean is None:
        sdp_pct_mean = fallback_pct(lb_stats.get("mean"), sdp_root_stats.get("mean"))
    sdp_pct_median = sdp_pct_stats.get("median")
    if sdp_pct_median is None:
        sdp_pct_median = fallback_pct(lb_stats.get("median"), sdp_root_stats.get("median"))

    overall = [
        {
            "scope": "overall",
            "n_seed_problem_pairs": len(all_rows),
            "rl_minus_mccormick_lb_mean": mcc_stats.get("mean"),
            "rl_minus_mccormick_lb_median": mcc_stats.get("median"),
            "rl_minus_mccormick_lb_pct_mean": mcc_pct_mean,
            "rl_minus_mccormick_lb_pct_median": mcc_pct_median,
            "mccormick_lb_better": mcc_cmp.get("win"),
            "mccormick_lb_tie": mcc_cmp.get("tie"),
            "mccormick_lb_worse": mcc_cmp.get("loss"),
            "mccormick_lb_better_rate": mcc_cmp.get("win_rate"),
            "mccormick_lb_tie_rate": mcc_cmp.get("tie_rate"),
            "mccormick_lb_worse_rate": mcc_cmp.get("loss_rate"),
            "mccormick_lb_better_tie_worse_pct": format_triplet(
                mcc_cmp.get("win_rate"),
                mcc_cmp.get("tie_rate"),
                mcc_cmp.get("loss_rate"),
            ),
            "rl_minus_sdp_lb_mean": sdp_stats.get("mean"),
            "rl_minus_sdp_lb_median": sdp_stats.get("median"),
            "rl_minus_sdp_lb_pct_mean": sdp_pct_mean,
            "rl_minus_sdp_lb_pct_median": sdp_pct_median,
            "sdp_lb_better": sdp_cmp.get("win"),
            "sdp_lb_tie": sdp_cmp.get("tie"),
            "sdp_lb_worse": sdp_cmp.get("loss"),
            "sdp_lb_better_rate": sdp_cmp.get("win_rate"),
            "sdp_lb_tie_rate": sdp_cmp.get("tie_rate"),
            "sdp_lb_worse_rate": sdp_cmp.get("loss_rate"),
            "sdp_lb_better_tie_worse_pct": format_triplet(
                sdp_cmp.get("win_rate"),
                sdp_cmp.get("tie_rate"),
                sdp_cmp.get("loss_rate"),
            ),
            "eps": eps,
        }
    ]
    return per_seed, overall


def build_baseline_cost_tables(
    seed_rows_map: Dict[int, Sequence[dict]],
    all_rows: Sequence[dict],
    eps: float,
) -> Tuple[List[dict], List[dict]]:
    def summarize_cost_rows(rows: Sequence[dict]) -> dict:
        mcc_time_vals = clean_values(rows, "rl_time_minus_mccormick_time")
        sdp_time_vals = clean_values(rows, "rl_time_minus_sdp_time")
        mcc_vars_vals = clean_values(rows, "rl_added_vars_minus_mccormick")
        mcc_cons_vals = clean_values(rows, "rl_added_cons_minus_mccormick")
        mcc_nnz_vals = clean_values(rows, "rl_added_nnz_minus_mccormick")
        sdp_vars_vals = clean_values(rows, "rl_added_vars_minus_sdp")
        sdp_cons_vals = clean_values(rows, "rl_added_cons_minus_sdp")
        sdp_nnz_vals = clean_values(rows, "rl_added_nnz_minus_sdp")
        mcc_vars_rel_pct_vals = clean_values(rows, "rl_vs_mccormick_added_vars_pct")
        mcc_cons_rel_pct_vals = clean_values(rows, "rl_vs_mccormick_added_cons_pct")
        mcc_nnz_rel_pct_vals = clean_values(rows, "rl_vs_mccormick_added_nnz_pct")
        sdp_vars_rel_pct_vals = clean_values(rows, "rl_vs_sdp_added_vars_pct")
        sdp_cons_rel_pct_vals = clean_values(rows, "rl_vs_sdp_added_cons_pct")
        sdp_nnz_rel_pct_vals = clean_values(rows, "rl_vs_sdp_added_nnz_pct")
        orig_vars_vals = clean_values(rows, "orig_num_vars_total")
        orig_cons_vals = clean_values(rows, "orig_num_constraints_total")
        orig_nnz_vals = clean_values(rows, "orig_nnz_terms_total")
        rl_vars_pct_vals = clean_values(rows, "rl_added_vars_pct_of_orig")
        rl_cons_pct_vals = clean_values(rows, "rl_added_cons_pct_of_orig")
        rl_nnz_pct_vals = clean_values(rows, "rl_added_nnz_pct_of_orig")
        mcc_vars_pct_vals = clean_values(rows, "baseline_mccormick_added_vars_pct_of_orig")
        mcc_cons_pct_vals = clean_values(rows, "baseline_mccormick_added_cons_pct_of_orig")
        mcc_nnz_pct_vals = clean_values(rows, "baseline_mccormick_added_nnz_pct_of_orig")
        sdp_vars_pct_vals = clean_values(rows, "baseline_sdp_added_vars_pct_of_orig")
        sdp_cons_pct_vals = clean_values(rows, "baseline_sdp_added_cons_pct_of_orig")
        sdp_nnz_pct_vals = clean_values(rows, "baseline_sdp_added_nnz_pct_of_orig")
        d_mcc_vars_pct_vals = clean_values(rows, "rl_added_vars_minus_mccormick_pct_of_orig")
        d_mcc_cons_pct_vals = clean_values(rows, "rl_added_cons_minus_mccormick_pct_of_orig")
        d_mcc_nnz_pct_vals = clean_values(rows, "rl_added_nnz_minus_mccormick_pct_of_orig")
        d_sdp_vars_pct_vals = clean_values(rows, "rl_added_vars_minus_sdp_pct_of_orig")
        d_sdp_cons_pct_vals = clean_values(rows, "rl_added_cons_minus_sdp_pct_of_orig")
        d_sdp_nnz_pct_vals = clean_values(rows, "rl_added_nnz_minus_sdp_pct_of_orig")
        mcc_time_stats = summarize_values(mcc_time_vals)
        sdp_time_stats = summarize_values(sdp_time_vals)
        mcc_vars_stats = summarize_values(mcc_vars_vals)
        mcc_cons_stats = summarize_values(mcc_cons_vals)
        mcc_nnz_stats = summarize_values(mcc_nnz_vals)
        sdp_vars_stats = summarize_values(sdp_vars_vals)
        sdp_cons_stats = summarize_values(sdp_cons_vals)
        sdp_nnz_stats = summarize_values(sdp_nnz_vals)
        mcc_vars_rel_pct_stats = summarize_values(mcc_vars_rel_pct_vals)
        mcc_cons_rel_pct_stats = summarize_values(mcc_cons_rel_pct_vals)
        mcc_nnz_rel_pct_stats = summarize_values(mcc_nnz_rel_pct_vals)
        sdp_vars_rel_pct_stats = summarize_values(sdp_vars_rel_pct_vals)
        sdp_cons_rel_pct_stats = summarize_values(sdp_cons_rel_pct_vals)
        sdp_nnz_rel_pct_stats = summarize_values(sdp_nnz_rel_pct_vals)
        orig_vars_stats = summarize_values(orig_vars_vals)
        orig_cons_stats = summarize_values(orig_cons_vals)
        orig_nnz_stats = summarize_values(orig_nnz_vals)
        rl_vars_pct_stats = summarize_values(rl_vars_pct_vals)
        rl_cons_pct_stats = summarize_values(rl_cons_pct_vals)
        rl_nnz_pct_stats = summarize_values(rl_nnz_pct_vals)
        mcc_vars_pct_stats = summarize_values(mcc_vars_pct_vals)
        mcc_cons_pct_stats = summarize_values(mcc_cons_pct_vals)
        mcc_nnz_pct_stats = summarize_values(mcc_nnz_pct_vals)
        sdp_vars_pct_stats = summarize_values(sdp_vars_pct_vals)
        sdp_cons_pct_stats = summarize_values(sdp_cons_pct_vals)
        sdp_nnz_pct_stats = summarize_values(sdp_nnz_pct_vals)
        d_mcc_vars_pct_stats = summarize_values(d_mcc_vars_pct_vals)
        d_mcc_cons_pct_stats = summarize_values(d_mcc_cons_pct_vals)
        d_mcc_nnz_pct_stats = summarize_values(d_mcc_nnz_pct_vals)
        d_sdp_vars_pct_stats = summarize_values(d_sdp_vars_pct_vals)
        d_sdp_cons_pct_stats = summarize_values(d_sdp_cons_pct_vals)
        d_sdp_nnz_pct_stats = summarize_values(d_sdp_nnz_pct_vals)
        out = {
            "n_cases": len(rows),
            "rl_minus_mccormick_time_mean": mcc_time_stats["mean"],
            "rl_minus_mccormick_time_median": mcc_time_stats["median"],
            "rl_minus_sdp_time_mean": sdp_time_stats["mean"],
            "rl_minus_sdp_time_median": sdp_time_stats["median"],
            "rl_minus_mccormick_added_vars_mean": mcc_vars_stats["mean"],
            "rl_minus_mccormick_added_cons_mean": mcc_cons_stats["mean"],
            "rl_minus_mccormick_added_nnz_mean": mcc_nnz_stats["mean"],
            "rl_minus_sdp_added_vars_mean": sdp_vars_stats["mean"],
            "rl_minus_sdp_added_cons_mean": sdp_cons_stats["mean"],
            "rl_minus_sdp_added_nnz_mean": sdp_nnz_stats["mean"],
            "rl_vs_mccormick_added_vars_pct_mean": mcc_vars_rel_pct_stats["mean"],
            "rl_vs_mccormick_added_cons_pct_mean": mcc_cons_rel_pct_stats["mean"],
            "rl_vs_mccormick_added_nnz_pct_mean": mcc_nnz_rel_pct_stats["mean"],
            "rl_vs_sdp_added_vars_pct_mean": sdp_vars_rel_pct_stats["mean"],
            "rl_vs_sdp_added_cons_pct_mean": sdp_cons_rel_pct_stats["mean"],
            "rl_vs_sdp_added_nnz_pct_mean": sdp_nnz_rel_pct_stats["mean"],
            "orig_num_vars_total_mean": orig_vars_stats["mean"],
            "orig_num_constraints_total_mean": orig_cons_stats["mean"],
            "orig_nnz_terms_total_mean": orig_nnz_stats["mean"],
            "rl_added_vars_pct_of_orig_mean": rl_vars_pct_stats["mean"],
            "rl_added_cons_pct_of_orig_mean": rl_cons_pct_stats["mean"],
            "rl_added_nnz_pct_of_orig_mean": rl_nnz_pct_stats["mean"],
            "baseline_mccormick_added_vars_pct_of_orig_mean": mcc_vars_pct_stats["mean"],
            "baseline_mccormick_added_cons_pct_of_orig_mean": mcc_cons_pct_stats["mean"],
            "baseline_mccormick_added_nnz_pct_of_orig_mean": mcc_nnz_pct_stats["mean"],
            "baseline_sdp_added_vars_pct_of_orig_mean": sdp_vars_pct_stats["mean"],
            "baseline_sdp_added_cons_pct_of_orig_mean": sdp_cons_pct_stats["mean"],
            "baseline_sdp_added_nnz_pct_of_orig_mean": sdp_nnz_pct_stats["mean"],
            "rl_minus_mccormick_added_vars_pct_of_orig_mean": d_mcc_vars_pct_stats["mean"],
            "rl_minus_mccormick_added_cons_pct_of_orig_mean": d_mcc_cons_pct_stats["mean"],
            "rl_minus_mccormick_added_nnz_pct_of_orig_mean": d_mcc_nnz_pct_stats["mean"],
            "rl_minus_sdp_added_vars_pct_of_orig_mean": d_sdp_vars_pct_stats["mean"],
            "rl_minus_sdp_added_cons_pct_of_orig_mean": d_sdp_cons_pct_stats["mean"],
            "rl_minus_sdp_added_nnz_pct_of_orig_mean": d_sdp_nnz_pct_stats["mean"],
        }
        add_win_tie_loss_minimize(out, rows, "rl_time_minus_mccormick_time", "mccormick_time", eps)
        add_win_tie_loss_minimize(out, rows, "rl_time_minus_sdp_time", "sdp_time", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_vars_minus_mccormick", "mccormick_added_vars", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_cons_minus_mccormick", "mccormick_added_cons", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_nnz_minus_mccormick", "mccormick_added_nnz", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_vars_minus_sdp", "sdp_added_vars", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_cons_minus_sdp", "sdp_added_cons", eps)
        add_win_tie_loss_minimize(out, rows, "rl_added_nnz_minus_sdp", "sdp_added_nnz", eps)
        out["mccormick_time_better_tie_worse_pct"] = format_triplet(
            out.get("mccormick_time_win_rate"),
            out.get("mccormick_time_tie_rate"),
            out.get("mccormick_time_loss_rate"),
        )
        out["sdp_time_better_tie_worse_pct"] = format_triplet(
            out.get("sdp_time_win_rate"),
            out.get("sdp_time_tie_rate"),
            out.get("sdp_time_loss_rate"),
        )
        return out

    per_seed = []
    for seed in sorted(seed_rows_map.keys()):
        one = {"seed": seed}
        one.update(summarize_cost_rows(seed_rows_map[seed]))
        one["mccormick_time_better"] = one.pop("mccormick_time_win")
        one["mccormick_time_tie"] = one.pop("mccormick_time_tie")
        one["mccormick_time_worse"] = one.pop("mccormick_time_loss")
        one["mccormick_time_better_rate"] = one.pop("mccormick_time_win_rate")
        one["mccormick_time_tie_rate"] = one.pop("mccormick_time_tie_rate")
        one["mccormick_time_worse_rate"] = one.pop("mccormick_time_loss_rate")
        one["sdp_time_better"] = one.pop("sdp_time_win")
        one["sdp_time_tie"] = one.pop("sdp_time_tie")
        one["sdp_time_worse"] = one.pop("sdp_time_loss")
        one["sdp_time_better_rate"] = one.pop("sdp_time_win_rate")
        one["sdp_time_tie_rate"] = one.pop("sdp_time_tie_rate")
        one["sdp_time_worse_rate"] = one.pop("sdp_time_loss_rate")
        per_seed.append(one)

    overall = [{"scope": "overall", "n_seed_problem_pairs": len(all_rows), **summarize_cost_rows(all_rows), "eps": eps}]
    overall[0]["mccormick_time_better"] = overall[0].pop("mccormick_time_win")
    overall[0]["mccormick_time_tie"] = overall[0].pop("mccormick_time_tie")
    overall[0]["mccormick_time_worse"] = overall[0].pop("mccormick_time_loss")
    overall[0]["mccormick_time_better_rate"] = overall[0].pop("mccormick_time_win_rate")
    overall[0]["mccormick_time_tie_rate"] = overall[0].pop("mccormick_time_tie_rate")
    overall[0]["mccormick_time_worse_rate"] = overall[0].pop("mccormick_time_loss_rate")
    overall[0]["sdp_time_better"] = overall[0].pop("sdp_time_win")
    overall[0]["sdp_time_tie"] = overall[0].pop("sdp_time_tie")
    overall[0]["sdp_time_worse"] = overall[0].pop("sdp_time_loss")
    overall[0]["sdp_time_better_rate"] = overall[0].pop("sdp_time_win_rate")
    overall[0]["sdp_time_tie_rate"] = overall[0].pop("sdp_time_tie_rate")
    overall[0]["sdp_time_worse_rate"] = overall[0].pop("sdp_time_loss_rate")
    return per_seed, overall


def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def build_report(
    root_dir: str,
    all_rows: Sequence[dict],
    seed_summaries: Sequence[dict],
    problem_summaries: Sequence[dict],
    overall_rows: Sequence[dict],
) -> str:
    lines = []
    lines.append("# Multi-seed Compare-All Summary")
    lines.append("")
    lines.append(f"- root_dir: `{root_dir}`")
    lines.append(f"- seeds_found: {len(seed_summaries)}")
    lines.append(f"- total_seed_problem_pairs: {len(all_rows)}")
    lines.append(f"- unique_problems: {len(problem_summaries)}")
    lines.append("")

    lines.append("## Per-seed snapshot")
    lines.append("")
    lines.append(
        "| seed | n_cases | rl_vs_root win rate | mean lb_improve | mean lb_improve_pct | "
        "mean lb_improve_scip | mean lb_improve_scip_pct | mean rl_minus_mccormick | mean rl_minus_sdp |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in seed_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt(row.get("n_cases")),
                    fmt(row.get("rl_vs_root_win_rate")),
                    fmt(row.get("lb_improve_mean")),
                    fmt(row.get("lb_improve_pct_mean")),
                    fmt(row.get("lb_improve_scip_mean")),
                    fmt(row.get("lb_improve_scip_pct_mean")),
                    fmt(row.get("rl_minus_mccormick_mean")),
                    fmt(row.get("rl_minus_sdp_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| metric | count | mean | median | std | min | max | win | tie | loss |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("metric")),
                    fmt(row.get("count")),
                    fmt(row.get("mean")),
                    fmt(row.get("median")),
                    fmt(row.get("std")),
                    fmt(row.get("min")),
                    fmt(row.get("max")),
                    fmt(row.get("win")),
                    fmt(row.get("tie")),
                    fmt(row.get("loss")),
                ]
            )
            + " |"
        )
    lines.append("")

    top_stable = sorted(
        [r for r in problem_summaries if r.get("lb_improve_mean") is not None],
        key=lambda x: (-(x.get("lb_improve_mean") or -1e18), -(x.get("rl_vs_root_win_rate") or -1e18)),
    )[:10]
    lines.append("## Top-10 problems by mean RL gain over root")
    lines.append("")
    lines.append("| dataset_key | n_seeds | mean lb_improve | std lb_improve | win rate | mean rl_minus_mccormick | mean rl_minus_sdp |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in top_stable:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset_key")),
                    fmt(row.get("n_seeds")),
                    fmt(row.get("lb_improve_mean")),
                    fmt(row.get("lb_improve_std")),
                    fmt(row.get("rl_vs_root_win_rate")),
                    fmt(row.get("rl_minus_mccormick_mean")),
                    fmt(row.get("rl_minus_sdp_mean")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_focused_report(
    lower_seed_rows: Sequence[dict],
    lower_overall_rows: Sequence[dict],
    baseline_lb_seed_rows: Sequence[dict],
    baseline_lb_overall_rows: Sequence[dict],
    baseline_cost_seed_rows: Sequence[dict],
    baseline_cost_overall_rows: Sequence[dict],
) -> str:
    lines = []
    lines.append("# Focused Tables")
    lines.append("")

    lines.append("## Lower-bound Table")
    lines.append("")
    lines.append(
        "| seed | n_cases | mean lb_improve | median lb_improve | mean lb_improve_pct | "
        "mean scip_lb_improve | mean scip_lb_improve_pct | root better/tie/worse (%) | "
        "scip better/tie/worse (%) | mean rl time (s) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |")
    for row in lower_seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt(row.get("n_cases")),
                    fmt(row.get("lb_improve_mean")),
                    fmt(row.get("lb_improve_median")),
                    fmt(row.get("lb_improve_pct_mean")),
                    fmt(row.get("scip_lb_improve_mean")),
                    fmt(row.get("scip_lb_improve_pct_mean")),
                    str(row.get("root_better_tie_worse_pct")),
                    str(row.get("scip_better_tie_worse_pct")),
                    fmt(row.get("rl_pipeline_time_sec_mean")),
                ]
            )
            + " |"
        )
    if lower_overall_rows:
        row = lower_overall_rows[0]
        lines.append(
            "| overall | "
            + " | ".join(
                [
                    fmt(row.get("n_seed_problem_pairs")),
                    fmt(row.get("lb_improve_mean")),
                    fmt(row.get("lb_improve_median")),
                    fmt(row.get("lb_improve_pct_mean")),
                    fmt(row.get("scip_lb_improve_mean")),
                    fmt(row.get("scip_lb_improve_pct_mean")),
                    str(row.get("root_better_tie_worse_pct")),
                    str(row.get("scip_better_tie_worse_pct")),
                    fmt(row.get("rl_pipeline_time_sec_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Baseline Lower-Bound Table")
    lines.append("")
    lines.append(
        "| seed | n_cases | mean rl-mccormick_lb | mean rl-mccormick_lb(%) | "
        "median rl-mccormick_lb | median rl-mccormick_lb(%) | mcc better/tie/worse (%) | "
        "mean rl-sdp_lb | mean rl-sdp_lb(%) | median rl-sdp_lb | median rl-sdp_lb(%) | sdp better/tie/worse (%) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for row in baseline_lb_seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt(row.get("n_cases")),
                    fmt(row.get("rl_minus_mccormick_lb_mean")),
                    fmt(row.get("rl_minus_mccormick_lb_pct_mean")),
                    fmt(row.get("rl_minus_mccormick_lb_median")),
                    fmt(row.get("rl_minus_mccormick_lb_pct_median")),
                    str(row.get("mccormick_lb_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_sdp_lb_mean")),
                    fmt(row.get("rl_minus_sdp_lb_pct_mean")),
                    fmt(row.get("rl_minus_sdp_lb_median")),
                    fmt(row.get("rl_minus_sdp_lb_pct_median")),
                    str(row.get("sdp_lb_better_tie_worse_pct")),
                ]
            )
            + " |"
        )
    if baseline_lb_overall_rows:
        row = baseline_lb_overall_rows[0]
        lines.append(
            "| overall | "
            + " | ".join(
                [
                    fmt(row.get("n_seed_problem_pairs")),
                    fmt(row.get("rl_minus_mccormick_lb_mean")),
                    fmt(row.get("rl_minus_mccormick_lb_pct_mean")),
                    fmt(row.get("rl_minus_mccormick_lb_median")),
                    fmt(row.get("rl_minus_mccormick_lb_pct_median")),
                    str(row.get("mccormick_lb_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_sdp_lb_mean")),
                    fmt(row.get("rl_minus_sdp_lb_pct_mean")),
                    fmt(row.get("rl_minus_sdp_lb_median")),
                    fmt(row.get("rl_minus_sdp_lb_pct_median")),
                    str(row.get("sdp_lb_better_tie_worse_pct")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Baseline Cost Table")
    lines.append("")
    lines.append(
        "| seed | n_cases | orig vars | orig cons | orig nnz | mean rl-mcc_time(s) | mcc time better/tie/worse (%) | "
        "mean rl-sdp_time(s) | sdp time better/tie/worse (%) | mean delta vars vs mcc | mean delta cons vs mcc | "
        "mean delta nnz vs mcc | mean rl/mcc vars growth (%) | mean rl/mcc cons growth (%) | mean rl/mcc nnz growth (%) | "
        "mean delta vars vs sdp | mean delta cons vs sdp | mean delta nnz vs sdp | mean rl/sdp vars growth (%) | "
        "mean rl/sdp cons growth (%) | mean rl/sdp nnz growth (%) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in baseline_cost_seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt(row.get("n_cases")),
                    fmt(row.get("orig_num_vars_total_mean")),
                    fmt(row.get("orig_num_constraints_total_mean")),
                    fmt(row.get("orig_nnz_terms_total_mean")),
                    fmt(row.get("rl_minus_mccormick_time_mean")),
                    str(row.get("mccormick_time_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_sdp_time_mean")),
                    str(row.get("sdp_time_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_mccormick_added_vars_mean")),
                    fmt(row.get("rl_minus_mccormick_added_cons_mean")),
                    fmt(row.get("rl_minus_mccormick_added_nnz_mean")),
                    fmt(row.get("rl_vs_mccormick_added_vars_pct_mean")),
                    fmt(row.get("rl_vs_mccormick_added_cons_pct_mean")),
                    fmt(row.get("rl_vs_mccormick_added_nnz_pct_mean")),
                    fmt(row.get("rl_minus_sdp_added_vars_mean")),
                    fmt(row.get("rl_minus_sdp_added_cons_mean")),
                    fmt(row.get("rl_minus_sdp_added_nnz_mean")),
                    fmt(row.get("rl_vs_sdp_added_vars_pct_mean")),
                    fmt(row.get("rl_vs_sdp_added_cons_pct_mean")),
                    fmt(row.get("rl_vs_sdp_added_nnz_pct_mean")),
                ]
            )
            + " |"
        )
    if baseline_cost_overall_rows:
        row = baseline_cost_overall_rows[0]
        lines.append(
            "| overall | "
            + " | ".join(
                [
                    fmt(row.get("n_seed_problem_pairs")),
                    fmt(row.get("orig_num_vars_total_mean")),
                    fmt(row.get("orig_num_constraints_total_mean")),
                    fmt(row.get("orig_nnz_terms_total_mean")),
                    fmt(row.get("rl_minus_mccormick_time_mean")),
                    str(row.get("mccormick_time_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_sdp_time_mean")),
                    str(row.get("sdp_time_better_tie_worse_pct")),
                    fmt(row.get("rl_minus_mccormick_added_vars_mean")),
                    fmt(row.get("rl_minus_mccormick_added_cons_mean")),
                    fmt(row.get("rl_minus_mccormick_added_nnz_mean")),
                    fmt(row.get("rl_vs_mccormick_added_vars_pct_mean")),
                    fmt(row.get("rl_vs_mccormick_added_cons_pct_mean")),
                    fmt(row.get("rl_vs_mccormick_added_nnz_pct_mean")),
                    fmt(row.get("rl_minus_sdp_added_vars_mean")),
                    fmt(row.get("rl_minus_sdp_added_cons_mean")),
                    fmt(row.get("rl_minus_sdp_added_nnz_mean")),
                    fmt(row.get("rl_vs_sdp_added_vars_pct_mean")),
                    fmt(row.get("rl_vs_sdp_added_cons_pct_mean")),
                    fmt(row.get("rl_vs_sdp_added_nnz_pct_mean")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=str,
        default=os.path.join("outputs", "logs", "multiseed_eval", "compare_all"),
        help="Directory containing seed_xx subdirectories.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed filter. Default: discover all seed_xx directories.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: root_dir",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Tolerance for win/tie/loss counts.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    out_dir = args.out_dir or root_dir
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = discover_seed_dirs(root_dir, args.seeds)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_xx directories found under: {root_dir}")

    all_rows: List[dict] = []
    seed_summaries: List[dict] = []
    rows_by_problem: Dict[str, List[dict]] = defaultdict(list)
    rows_by_seed: Dict[int, List[dict]] = defaultdict(list)

    for seed, seed_dir in seed_dirs:
        rows = load_seed_rows(seed_dir, seed)
        if not rows:
            continue
        all_rows.extend(rows)
        rows_by_seed[seed].extend(rows)
        seed_summaries.append(summarize_seed(seed, rows, DEFAULT_PAIR_METRICS, args.eps))
        for row in rows:
            rows_by_problem[str(row["dataset_key"])].append(row)

    if not all_rows:
        raise RuntimeError("No rows loaded from any seed directory.")

    problem_summaries = [
        summarize_problem(rows, DEFAULT_PAIR_METRICS, args.eps)
        for _, rows in sorted(rows_by_problem.items(), key=lambda x: x[0])
    ]
    overall_rows = summarize_overall(all_rows, DEFAULT_PAIR_METRICS, args.eps)

    seed_summaries.sort(key=lambda x: x["seed"])
    problem_summaries.sort(
        key=lambda x: (
            -(x.get("lb_improve_mean") if x.get("lb_improve_mean") is not None else -1e18),
            x.get("dataset_key"),
        )
    )

    write_csv(os.path.join(out_dir, "multiseed_seed_summary.csv"), seed_summaries)
    write_csv(os.path.join(out_dir, "multiseed_problem_summary.csv"), problem_summaries)
    write_csv(os.path.join(out_dir, "multiseed_overall_summary.csv"), overall_rows)

    lower_seed_rows, lower_overall_rows = build_lower_bound_tables(
        seed_summaries, overall_rows, all_rows, args.eps
    )
    baseline_lb_seed_rows, baseline_lb_overall_rows = build_baseline_lb_tables(
        seed_summaries, overall_rows, all_rows, args.eps
    )
    baseline_cost_seed_rows, baseline_cost_overall_rows = build_baseline_cost_tables(
        rows_by_seed, all_rows, args.eps
    )
    write_csv(os.path.join(out_dir, "table_lb_vs_gurobi_scip_by_seed.csv"), lower_seed_rows)
    write_csv(os.path.join(out_dir, "table_lb_vs_gurobi_scip_overall.csv"), lower_overall_rows)
    write_csv(os.path.join(out_dir, "table_lb_vs_baselines_by_seed.csv"), baseline_lb_seed_rows)
    write_csv(os.path.join(out_dir, "table_lb_vs_baselines_overall.csv"), baseline_lb_overall_rows)
    write_csv(os.path.join(out_dir, "table_cost_vs_baselines_by_seed.csv"), baseline_cost_seed_rows)
    write_csv(os.path.join(out_dir, "table_cost_vs_baselines_overall.csv"), baseline_cost_overall_rows)

    report = build_report(root_dir, all_rows, seed_summaries, problem_summaries, overall_rows)
    report_path = os.path.join(out_dir, "multiseed_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    focused_report = build_focused_report(
        lower_seed_rows,
        lower_overall_rows,
        baseline_lb_seed_rows,
        baseline_lb_overall_rows,
        baseline_cost_seed_rows,
        baseline_cost_overall_rows,
    )
    focused_report_path = os.path.join(out_dir, "focused_tables.md")
    with open(focused_report_path, "w", encoding="utf-8") as f:
        f.write(focused_report)

    print(f"[INFO] root_dir={root_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] seeds={[seed for seed, _ in seed_dirs]}")
    print(f"[INFO] total_seed_problem_pairs={len(all_rows)}")
    print(f"[INFO] unique_problems={len(problem_summaries)}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'multiseed_seed_summary.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'multiseed_problem_summary.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'multiseed_overall_summary.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_lb_vs_gurobi_scip_by_seed.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_lb_vs_gurobi_scip_overall.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_lb_vs_baselines_by_seed.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_lb_vs_baselines_overall.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_cost_vs_baselines_by_seed.csv')}")
    print(f"[INFO] wrote {os.path.join(out_dir, 'table_cost_vs_baselines_overall.csv')}")
    print(f"[INFO] wrote {report_path}")
    print(f"[INFO] wrote {focused_report_path}")


if __name__ == "__main__":
    main()
