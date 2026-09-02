# -*- coding: utf-8 -*-
"""
Runner: one-pass RL + multi-baseline comparison.

Key property:
  - RL relaxation is computed exactly once per problem.
  - In the same pass, compare against Gurobi/SCIP root bounds and handcrafted/non-learned baselines.
  - Persist explicit cost fields for RL and every baseline.
"""

import os
import json
import pickle
import argparse
import re
import time
import copy
import numpy as np
import sympy as sp

import torch

from autoconvexrelax.evaluation.solvers.mosek import solve_convex_relax_mosek
from autoconvexrelax.evaluation.baselines import apply_heuristic_relaxation, derive_random_baseline_seed
try:
    from autoconvexrelax.evaluation.solvers.scip import solve_nonconvex_qcqp_scip
except Exception:
    solve_nonconvex_qcqp_scip = None

import autoconvexrelax.training.stage2 as train_mod
from autoconvexrelax.evaluation.runner import (
    load_eval_groups,
    load_split_indices,
    resolve_split_json,
    collect_split_problems,
    build_model,
    apply_policy_until_convex,
    canonicalize_problem,
    set_seed,
)


DEFAULT_BASELINE_MODES = ["mccormick", "sdp", "structure", "random"]
RANDOM_BASELINE_POLICY_VERSION = "single_action_v2"


def _matrix_dim(matrix_expr) -> int:
    shape = getattr(matrix_expr, "shape", None)
    if shape is None:
        try:
            shape = sp.Matrix(matrix_expr).shape
        except Exception:
            return 0
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        return 0
    try:
        nrow = int(shape[0])
        ncol = int(shape[1])
    except Exception:
        return 0
    if nrow <= 0 or ncol <= 0:
        return 0
    return nrow if nrow == ncol else min(nrow, ncol)


def _fallback_symbolic_nnz(prob) -> int:
    def _count_expr(expr) -> int:
        if expr is None:
            return 0
        try:
            shape = getattr(expr, "shape", None)
            if isinstance(shape, (tuple, list)) and len(shape) == 2:
                r, c = int(shape[0]), int(shape[1])
                if r == 1 and c == 1:
                    expr = expr[0, 0]
                else:
                    mat = sp.Matrix(expr)
                    return sum(_count_expr(mat[i, j]) for i in range(mat.rows) for j in range(mat.cols))
        except Exception:
            pass

        try:
            expanded = sp.expand(expr, mul=True, power_exp=False, power_base=False, log=False)
        except Exception:
            expanded = expr

        cnt = 0
        for t in sp.Add.make_args(expanded):
            if t is None:
                continue
            try:
                if t.is_zero is True:
                    continue
            except Exception:
                pass
            free_syms = getattr(t, "free_symbols", None)
            if free_syms is not None and len(free_syms) == 0:
                continue
            cnt += 1
        return cnt

    nnz = _count_expr(getattr(prob, "obj_expr", None))
    for c in getattr(prob, "constraints", []):
        nnz += _count_expr(getattr(c, "expr", None))
        nnz += _count_expr(getattr(c, "rhs", None))
    for psd in getattr(prob, "psd_constraints", []):
        nnz += _count_expr(getattr(psd, "matrix_expr", None))
    return int(nnz)


def _problem_structure_stats(prob) -> dict:
    scalar_vars = len(getattr(prob, "variables", {}))
    matrix_vars = list(getattr(prob, "matrix_variables", {}).values())
    matrix_var_entries = 0
    for mv in matrix_vars:
        try:
            matrix_var_entries += int(mv.rows) * int(mv.cols)
        except Exception:
            continue

    n_cons = len(getattr(prob, "constraints", []))
    psd_cons = getattr(prob, "psd_constraints", [])
    n_psd_cons = len(psd_cons)

    psd_dims = []
    for psd in psd_cons:
        d = _matrix_dim(getattr(psd, "matrix_expr", None))
        if d > 0:
            psd_dims.append(d)

    nnz_terms_total = None
    try:
        prob.map_all_terms(update_problem=False)
        nnz_terms_total = int(getattr(prob, "counter", 0))
    except Exception:
        nnz_terms_total = _fallback_symbolic_nnz(prob)

    return {
        "num_scalar_vars": int(scalar_vars),
        "num_matrix_vars": int(len(matrix_vars)),
        "num_matrix_var_entries": int(matrix_var_entries),
        "num_vars_total": int(scalar_vars + matrix_var_entries),
        "num_constraints": int(n_cons),
        "num_psd_constraints": int(n_psd_cons),
        "num_constraints_total": int(n_cons + n_psd_cons),
        "nnz_terms_total": int(nnz_terms_total),
        "psd_dim_sum": int(sum(psd_dims)),
        "psd_dim_max": int(max(psd_dims) if psd_dims else 0),
        "psd_entry_sum": int(sum(d * d for d in psd_dims)),
    }


def _flatten_stats(prefix: str, stats: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _safe_ratio(num: float, den: float):
    if den is None or den <= 0:
        return None
    return float(num) / float(den)


def _growth_stats(prefix: str, base: dict, new: dict) -> dict:
    add_vars = int(new["num_vars_total"] - base["num_vars_total"])
    add_cons = int(new["num_constraints_total"] - base["num_constraints_total"])
    add_nnz = int(new["nnz_terms_total"] - base["nnz_terms_total"])
    add_psd_size = int(new["psd_entry_sum"] - base["psd_entry_sum"])
    add_psd_blocks = int(new["num_psd_constraints"] - base["num_psd_constraints"])

    return {
        f"{prefix}_added_vars": add_vars,
        f"{prefix}_added_cons": add_cons,
        f"{prefix}_added_nnz": add_nnz,
        f"{prefix}_added_psd_size": add_psd_size,
        f"{prefix}_added_psd_blocks": add_psd_blocks,
        f"{prefix}_var_growth_ratio": _safe_ratio(new["num_vars_total"], base["num_vars_total"]),
        f"{prefix}_cons_growth_ratio": _safe_ratio(new["num_constraints_total"], base["num_constraints_total"]),
        f"{prefix}_nnz_growth_ratio": _safe_ratio(new["nnz_terms_total"], base["nnz_terms_total"]),
        f"{prefix}_psd_size_growth_ratio": _safe_ratio(new["psd_entry_sum"], base["psd_entry_sum"]),
    }


def _pct_improve(new_v, base_v, eps=1e-9):
    if new_v is None or base_v is None:
        return None
    denom = max(abs(base_v), abs(new_v), eps)
    return (new_v - base_v) / denom * 100.0


def _make_baseline_cache_record(
    mode: str,
    growth_prefix: str,
    *,
    lb,
    relax_time: float,
    canon_time: float,
    solve_time: float,
    per_solver_budget: float,
    stats: dict,
    orig_stats: dict,
):
    return {
        "mode": mode,
        "lb": lb,
        "cost": lb,
        "relax_time_sec": relax_time,
        "canonicalize_time_sec": canon_time,
        "solve_time_sec": solve_time,
        "pipeline_time_sec": relax_time + canon_time + solve_time,
        "per_solver_time_limit_sec": per_solver_budget,
        "stats": stats,
        "growth": _growth_stats(growth_prefix, orig_stats, stats),
    }


def _mean_numeric_dict(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = set()
    for dct in dicts:
        keys.update(dct.keys())
    out = {}
    for key in keys:
        vals = [dct.get(key) for dct in dicts]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and np.isfinite(float(v))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _std(values) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(float(v))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _baseline_cache_mode_name(mode: str, args) -> str:
    if mode == "structure":
        return f"structure_k{args.structure_k_min}_tau{args.structure_tau_density:g}"
    if mode == "random":
        qcr_flag = "qcr" if args.random_include_qcr else "noqcr"
        return f"random_{RANDOM_BASELINE_POLICY_VERSION}_r{args.random_rollouts}_seed{args.seed}_{qcr_flag}"
    return mode


def _is_number_or_none(value) -> bool:
    return value is None or isinstance(value, (int, float))


def _row_has_requested_results(row: dict, requested_modes: list[str]) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("rl_lb", None), (int, float)):
        return False
    for mode in requested_modes:
        if mode == "random" and row.get("baseline_random_policy_version") != RANDOM_BASELINE_POLICY_VERSION:
            return False
        if not _is_number_or_none(row.get(f"baseline_{mode}_lb", None)):
            return False
        if f"baseline_{mode}_lb" not in row:
            return False
    return True


def _numeric_or_none(row: dict, key: str):
    value = row.get(key, None)
    return value if _is_number_or_none(value) else None


def _extract_cached_growth(row: dict, prefix: str) -> dict:
    suffixes = (
        "added_vars",
        "added_cons",
        "added_nnz",
        "added_psd_size",
        "added_psd_blocks",
        "var_growth_ratio",
        "cons_growth_ratio",
        "nnz_growth_ratio",
        "psd_size_growth_ratio",
    )
    return {f"{prefix}_{suffix}": row[f"{prefix}_{suffix}"] for suffix in suffixes if f"{prefix}_{suffix}" in row}


def _extract_cached_stats(row: dict, prefix: str) -> dict:
    stat_keys = (
        "num_scalar_vars",
        "num_matrix_vars",
        "num_matrix_var_entries",
        "num_vars_total",
        "num_constraints",
        "num_psd_constraints",
        "num_constraints_total",
        "nnz_terms_total",
        "psd_dim_sum",
        "psd_dim_max",
        "psd_entry_sum",
    )
    return {key: row[f"{prefix}_{key}"] for key in stat_keys if f"{prefix}_{key}" in row}


def _cached_baseline_payload(row: dict, mode: str) -> dict:
    prefix = f"baseline_{mode}"
    lb = _numeric_or_none(row, f"{prefix}_lb")
    return {
        "raw": None,
        "solver": None,
        "stats": _extract_cached_stats(row, prefix),
        "growth": _extract_cached_growth(row, prefix),
        "relax_time": float(row.get(f"{prefix}_relax_time_sec", 0.0) or 0.0),
        "canon_time": float(row.get(f"{prefix}_canonicalize_time_sec", 0.0) or 0.0),
        "solve_time": float(row.get(f"{prefix}_solve_time_sec", 0.0) or 0.0),
        "pipeline_time": float(row.get(f"{prefix}_pipeline_time_sec", 0.0) or 0.0),
        "lb": lb,
        "cost": _numeric_or_none(row, f"{prefix}_cost") if f"{prefix}_cost" in row else lb,
        "cache_hit": True,
        "lb_std": float(row.get(f"{prefix}_lb_std", 0.0) or 0.0),
        "n_rollouts": int(row.get(f"{prefix}_n_rollouts", 1) or 1),
        "error": row.get(f"{prefix}_error", None),
    }


def _can_resume_baseline_from_row(row: dict, mode: str) -> bool:
    if not isinstance(row, dict):
        return False
    if mode == "random" and row.get("baseline_random_policy_version") != RANDOM_BASELINE_POLICY_VERSION:
        return False
    mode_lb_key = f"baseline_{mode}_lb"
    return mode_lb_key in row and _is_number_or_none(row.get(mode_lb_key))


def _run_one_baseline(prob, mode: str, args, per_solver_budget: float, orig_stats: dict):
    prob_base_raw = copy.deepcopy(prob)
    t0 = time.perf_counter()
    prob_base_raw = apply_heuristic_relaxation(
        prob_base_raw,
        mode=mode,
        k_min=args.structure_k_min,
        tau_density=args.structure_tau_density,
        random_seed=args.seed,
        random_include_qcr=args.random_include_qcr,
    )
    base_relax_time = time.perf_counter() - t0

    base_stats = _problem_structure_stats(prob_base_raw)
    error = None
    prob_base_solver = None
    base_lb = None

    t0 = time.perf_counter()
    try:
        prob_base_solver = canonicalize_problem(copy.deepcopy(prob_base_raw))
    except Exception as e:
        if not args.skip_failed_baselines:
            raise
        error = f"canonicalize failed: {type(e).__name__}: {e}"
        print(f"  [BASELINE-{mode}] {error}")
    base_canon_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if error is None:
        try:
            base_lb = solve_convex_relax_mosek(prob_base_solver, time_limit=per_solver_budget, verbose=True)
        except Exception as e:
            if not args.skip_failed_baselines:
                raise
            error = f"solve failed: {type(e).__name__}: {e}"
            print(f"  [BASELINE-{mode}] {error}")
    base_solve_time = time.perf_counter() - t0

    growth = _growth_stats(f"baseline_{mode}", orig_stats, base_stats)
    return {
        "raw": prob_base_raw,
        "solver": prob_base_solver,
        "stats": base_stats,
        "growth": growth,
        "relax_time": base_relax_time,
        "canon_time": base_canon_time,
        "solve_time": base_solve_time,
        "pipeline_time": base_relax_time + base_canon_time + base_solve_time,
        "lb": base_lb,
        "cost": base_lb,
        "cache_hit": False,
        "lb_std": 0.0,
        "error": error,
    }


def _run_random_baseline(prob, args, per_solver_budget: float, orig_stats: dict):
    payloads = []
    errors = []
    for rollout in range(max(1, int(args.random_rollouts))):
        prob_base_raw = copy.deepcopy(prob)
        t0 = time.perf_counter()
        prob_base_raw = apply_heuristic_relaxation(
            prob_base_raw,
            mode="random",
            k_min=args.structure_k_min,
            tau_density=args.structure_tau_density,
            random_seed=derive_random_baseline_seed(args.seed, rollout, getattr(prob, "name", "")),
            random_include_qcr=args.random_include_qcr,
        )
        relax_time = time.perf_counter() - t0

        stats = _problem_structure_stats(prob_base_raw)
        error = None
        prob_base_solver = None
        lb = None

        t0 = time.perf_counter()
        try:
            prob_base_solver = canonicalize_problem(copy.deepcopy(prob_base_raw))
        except Exception as e:
            if not args.skip_failed_baselines:
                raise
            error = f"rollout {rollout}: canonicalize failed: {type(e).__name__}: {e}"
            print(f"  [BASELINE-random] {error}")
        canon_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        if error is None:
            try:
                lb = solve_convex_relax_mosek(prob_base_solver, time_limit=per_solver_budget, verbose=True)
            except Exception as e:
                if not args.skip_failed_baselines:
                    raise
                error = f"rollout {rollout}: solve failed: {type(e).__name__}: {e}"
                print(f"  [BASELINE-random] {error}")
        solve_time = time.perf_counter() - t0
        if error is not None:
            errors.append(error)

        growth = _growth_stats("baseline_random", orig_stats, stats)
        payloads.append(
            {
                "raw": prob_base_raw,
                "solver": prob_base_solver,
                "stats": stats,
                "growth": growth,
                "relax_time": relax_time,
                "canon_time": canon_time,
                "solve_time": solve_time,
                "pipeline_time": relax_time + canon_time + solve_time,
                "lb": lb,
                "cost": lb,
                "cache_hit": False,
                "error": error,
            }
        )

    lbs = [p["lb"] for p in payloads]
    finite_lbs = [float(v) for v in lbs if isinstance(v, (int, float)) and np.isfinite(float(v))]
    mean_lb = float(np.mean(finite_lbs)) if finite_lbs else None
    return {
        "raw": None,
        "solver": None,
        "stats": _mean_numeric_dict([p["stats"] for p in payloads]),
        "growth": _mean_numeric_dict([p["growth"] for p in payloads]),
        "relax_time": float(np.mean([p["relax_time"] for p in payloads])),
        "canon_time": float(np.mean([p["canon_time"] for p in payloads])),
        "solve_time": float(np.mean([p["solve_time"] for p in payloads])),
        "pipeline_time": float(np.mean([p["pipeline_time"] for p in payloads])),
        "lb": mean_lb,
        "cost": mean_lb,
        "cache_hit": False,
        "lb_std": _std(finite_lbs),
        "n_rollouts": len(payloads),
        "policy_version": RANDOM_BASELINE_POLICY_VERSION,
        "error": "; ".join(errors) if errors else None,
    }


def main():
    # Keep evaluation behavior aligned across all comparison modes.
    train_mod.RELAX_ENGINE_BT_WARMUP = False
    train_mod.RELAX_ENGINE_BT_BEFORE_GLOBAL_CUT = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="trained PolicyNet checkpoint")
    parser.add_argument("--data", type=str, default="outputs/data/vector_finetune_HYBRID_MIX_1200.pkl")
    parser.add_argument("--split_json", type=str, default="", help="path to *_split_indices.json")
    parser.add_argument("--split_use", type=str, default="infer", choices=["infer", "train"])
    parser.add_argument("--no_split", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time_limit", type=float, default=60.0, help="per-problem total time budget for solver stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_problems", type=int, default=50, help="-1 means all")
    parser.add_argument("--sample", type=str, default="head", choices=["head", "random"])
    parser.add_argument("--root_lb_cache", type=str, default=None, help="precomputed Gurobi root bound cache json")
    parser.add_argument("--scip_root_cache", type=str, default=None, help="precomputed SCIP root bound cache json")
    parser.add_argument("--dataset_tag", type=str, default="HYBRID_MIX_1200")
    parser.add_argument("--no_gurobi", action="store_true", help="disable gurobi root compare")
    parser.add_argument("--scip_root", action="store_true", help="compute scip root online if cache miss")
    parser.add_argument("--scip_time_limit", type=float, default=60.0)
    parser.add_argument("--save_dir", type=str, default=os.path.join("outputs", "logs"))
    parser.add_argument("--out_json_name", type=str, default="eval_compare_all.json")
    parser.add_argument("--no_resume_results", action="store_true", help="do not reuse rows already present in out_json")
    parser.add_argument("--no_case_pkl", action="store_true", help="do not write per-problem pickle packs")
    parser.add_argument(
        "--baseline_modes",
        nargs="+",
        default=DEFAULT_BASELINE_MODES,
        choices=["mccormick", "sdp", "structure", "random"],
        help="non-learned baselines to evaluate in addition to RL",
    )
    parser.add_argument("--structure_k_min", type=int, default=4)
    parser.add_argument("--structure_tau_density", type=float, default=0.75)
    parser.add_argument("--random_rollouts", type=int, default=1)
    parser.add_argument("--random_include_qcr", action="store_true", default=True)
    parser.add_argument("--random_no_qcr", dest="random_include_qcr", action="store_false")
    parser.add_argument(
        "--baseline_cache_json",
        type=str,
        default=None,
        help="shared baseline cache json; hit entries will skip baseline recomputation",
    )
    parser.add_argument(
        "--skip_failed_baselines",
        action="store_true",
        help="record failed baseline solves as null instead of aborting; intended for fraction candidate pools",
    )
    parser.add_argument(
        "--skip_failed_instances",
        action="store_true",
        help="record failed RL instances as null instead of aborting; intended for fraction candidate pools",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] SAVE_DIR = {save_dir}")

    set_seed(args.seed, deterministic=True)
    print(f"[INFO] Use device = {args.device}")

    groups = load_eval_groups(args.data)
    if args.no_split:
        problems = groups[0]
        orig_indices = list(range(len(problems)))
        print(f"[INFO] #Problems = {len(problems)} (no split)")
    else:
        split_path = resolve_split_json(args.split_json)
        if not split_path or (not os.path.exists(split_path)):
            raise ValueError("split file not found; pass --split_json or use --no_split")
        split_data = load_split_indices(split_path)
        problems, orig_indices = collect_split_problems(groups, split_data, use=args.split_use)
        print(f"[INFO] #Problems = {len(problems)} (split={args.split_use})")

    if args.n_problems == -1:
        args.n_problems = len(problems)
    if args.n_problems > len(problems):
        raise ValueError(f"n_problems={args.n_problems} > total={len(problems)}")

    if args.sample == "head":
        problems = problems[: args.n_problems]
        orig_indices = orig_indices[: args.n_problems]
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(problems), size=args.n_problems, replace=False)
        idx = sorted(idx.tolist())
        problems = [problems[i] for i in idx]
        orig_indices = [orig_indices[i] for i in idx]

    root_cache = None
    if (not args.no_gurobi) and args.root_lb_cache:
        with open(args.root_lb_cache, "r", encoding="utf-8") as f:
            root_cache = json.load(f)
        print(f"[INFO] Loaded root_lb_cache: {args.root_lb_cache}, #keys={len(root_cache)}")
    elif args.no_gurobi:
        print("[INFO] --no_gurobi set: skip gurobi root compare.")
    else:
        print("[WARN] No --root_lb_cache provided. gurobi_root_bound will be None.")

    scip_root_cache = None
    if args.scip_root_cache:
        with open(args.scip_root_cache, "r", encoding="utf-8") as f:
            scip_root_cache = json.load(f)
        print(f"[INFO] Loaded scip_root_cache: {args.scip_root_cache}, #keys={len(scip_root_cache)}")

    model = build_model(args.ckpt, args.device, problems[0])

    out_path = os.path.join(save_dir, args.out_json_name)
    baseline_cache_path = args.baseline_cache_json
    baseline_cache = {}
    if baseline_cache_path:
        if os.path.exists(baseline_cache_path):
            with open(baseline_cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                baseline_cache = loaded
            else:
                baseline_cache = {}
            print(f"[INFO] Loaded baseline cache: {baseline_cache_path}, #keys={len(baseline_cache)}")
        else:
            print(f"[INFO] baseline cache not found, will create: {baseline_cache_path}")

    existing_result_cache = {}
    if (not args.no_resume_results) and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                existing_result_cache = {
                    row.get("dataset_key"): row
                    for row in existing_results
                    if isinstance(row, dict) and row.get("dataset_key") is not None
                }
                print(f"[INFO] Loaded result resume cache: {out_path}, #rows={len(existing_result_cache)}")
        except Exception as e:
            print(f"[WARN] Failed to load result resume cache {out_path}: {e}")

    results = []
    flush_every = 5

    def _flush():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def _flush_baseline_cache():
        if not baseline_cache_path:
            return
        cache_dir = os.path.dirname(baseline_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(baseline_cache_path, "w", encoding="utf-8") as f:
            json.dump(baseline_cache, f, indent=2)

    def make_key(dataset_tag: str, global_idx: int, name: str) -> str:
        name = str(name).replace(":", "_").replace("/", "_")
        return f"{dataset_tag}:{global_idx}:{name}"

    for idx, prob in enumerate(problems):
        print(f"\n=== Problem {idx+1}/{len(problems)}: {prob.name} ===")
        global_idx = orig_indices[idx]
        key = make_key(args.dataset_tag, global_idx, prob.name)

        cached_row = existing_result_cache.get(key)
        if args.skip_failed_instances and isinstance(cached_row, dict) and cached_row.get("rl_error"):
            results.append(cached_row)
            print(f"  [RESULT] resume failed-instance hit -> reused {key}")
            if (idx + 1) % flush_every == 0:
                _flush()
            continue
        if _row_has_requested_results(cached_row, list(args.baseline_modes)):
            results.append(cached_row)
            print(f"  [RESULT] resume hit -> reused {key}")
            if (idx + 1) % flush_every == 0:
                _flush()
            continue

        prob_orig_raw = copy.deepcopy(prob)
        orig_stats = _problem_structure_stats(prob_orig_raw)

        # RL: compute exactly once
        rl_error = None
        resume_rl = isinstance((cached_row or {}).get("rl_lb", None), (int, float))
        if resume_rl:
            prob_rl_raw = None
            prob_rl_solver = None
            rl_stats = _extract_cached_stats(cached_row, "rl")
            rl_relax_time = float(cached_row.get("rl_relax_time_sec", 0.0) or 0.0)
            rl_canon_time = float(cached_row.get("rl_canonicalize_time_sec", 0.0) or 0.0)
            rl_solve_time = float(cached_row.get("rl_solve_time_sec", 0.0) or 0.0)
            rl_lb = float(cached_row["rl_lb"])
            rl_error = cached_row.get("rl_error", None)
            print("  [RL] resume hit -> reused learned relaxation result")
        else:
            prob_rl_raw = copy.deepcopy(prob)
            t0 = time.perf_counter()
            try:
                prob_rl_raw = apply_policy_until_convex(prob_rl_raw, model, max_steps=50, device=args.device)
            except Exception as e:
                if not args.skip_failed_instances:
                    raise
                rl_relax_time = time.perf_counter() - t0
                rl_error = f"policy failed: {type(e).__name__}: {e}"
                print(f"  [RL] {rl_error}")
                one = {
                    "name": prob.name,
                    "dataset_key": key,
                    "rl_lb": None,
                    "rl_cost": None,
                    "rl_error": rl_error,
                    "rl_relax_time_sec": rl_relax_time,
                    "rl_canonicalize_time_sec": None,
                    "rl_solve_time_sec": None,
                    "rl_pipeline_time_sec": rl_relax_time,
                }
                one.update(_flatten_stats("orig", orig_stats))
                results.append(one)
                if (idx + 1) % flush_every == 0:
                    _flush()
                continue
            rl_relax_time = time.perf_counter() - t0
            rl_stats = _problem_structure_stats(prob_rl_raw)

            t0 = time.perf_counter()
            try:
                prob_rl_solver = canonicalize_problem(copy.deepcopy(prob_rl_raw))
            except Exception as e:
                if not args.skip_failed_instances:
                    raise
                prob_rl_solver = None
                rl_error = f"canonicalize failed: {type(e).__name__}: {e}"
                print(f"  [RL] {rl_error}")
            rl_canon_time = time.perf_counter() - t0

        # baselines (reuse cache when available)
        baseline_payload = {}
        per_solver_budget = max(args.time_limit / 3.0, 1e-3)
        for mode in args.baseline_modes:
            if _can_resume_baseline_from_row(cached_row, mode):
                baseline_payload[mode] = _cached_baseline_payload(cached_row, mode)
                print(f"  [BASELINE-{mode}] result resume hit")
                continue

            per_case_cache = baseline_cache.get(key, {}) if isinstance(baseline_cache.get(key, {}), dict) else {}
            cache_mode = _baseline_cache_mode_name(mode, args)
            mode_cache = per_case_cache.get(cache_mode, {}) if isinstance(per_case_cache.get(cache_mode, {}), dict) else {}

            cached_budget = mode_cache.get("per_solver_time_limit_sec", None)
            cache_hit = (
                _is_number_or_none(mode_cache.get("lb", None))
                and isinstance(mode_cache.get("stats", None), dict)
                and isinstance(mode_cache.get("growth", None), dict)
                and isinstance(cached_budget, (int, float))
                and abs(float(cached_budget) - float(per_solver_budget)) <= 1e-12
            )

            if cache_hit:
                baseline_payload[mode] = {
                    "raw": None,
                    "solver": None,
                    "stats": mode_cache["stats"],
                    "growth": mode_cache["growth"],
                    "relax_time": float(mode_cache.get("relax_time_sec", 0.0)),
                    "canon_time": float(mode_cache.get("canonicalize_time_sec", 0.0)),
                    "solve_time": float(mode_cache.get("solve_time_sec", 0.0)),
                    "pipeline_time": float(mode_cache.get("pipeline_time_sec", 0.0)),
                    "lb": _numeric_or_none(mode_cache, "lb"),
                    "cost": _numeric_or_none(mode_cache, "cost") if "cost" in mode_cache else _numeric_or_none(mode_cache, "lb"),
                    "cache_hit": True,
                    "lb_std": float(mode_cache.get("lb_std", 0.0)),
                    "n_rollouts": int(mode_cache.get("n_rollouts", 1)),
                    "error": mode_cache.get("error", None),
                }
                print(f"  [BASELINE-{mode}] cache hit")
                continue

            if mode == "random":
                baseline_payload[mode] = _run_random_baseline(prob, args, per_solver_budget, orig_stats)
            else:
                baseline_payload[mode] = _run_one_baseline(prob, mode, args, per_solver_budget, orig_stats)

            cache_record = _make_baseline_cache_record(
                mode,
                growth_prefix=f"baseline_{mode}",
                lb=baseline_payload[mode]["lb"],
                relax_time=baseline_payload[mode]["relax_time"],
                canon_time=baseline_payload[mode]["canon_time"],
                solve_time=baseline_payload[mode]["solve_time"],
                per_solver_budget=per_solver_budget,
                stats=baseline_payload[mode]["stats"],
                orig_stats=orig_stats,
            )
            cache_record["lb_std"] = baseline_payload[mode].get("lb_std", 0.0)
            cache_record["n_rollouts"] = baseline_payload[mode].get("n_rollouts", 1)
            cache_record["error"] = baseline_payload[mode].get("error", None)
            if mode == "random":
                cache_record["policy_version"] = RANDOM_BASELINE_POLICY_VERSION
            if key not in baseline_cache or not isinstance(baseline_cache.get(key), dict):
                baseline_cache[key] = {}
            baseline_cache[key][cache_mode] = cache_record
            baseline_payload[mode]["growth"] = cache_record["growth"]
            print(f"  [BASELINE-{mode}] cache miss -> computed and stored")

        # root bounds
        root_bound = None
        scip_root_bound = None
        if root_cache is not None:
            v = root_cache.get(key, None)
            root_bound = float(v) if isinstance(v, (int, float)) else None
        if scip_root_cache is not None:
            v = scip_root_cache.get(key, None)
            scip_root_bound = float(v) if isinstance(v, (int, float)) else None

        if scip_root_bound is None and args.scip_root:
            if solve_nonconvex_qcqp_scip is None:
                print("  [SCIP] solver_interface_scip not available.")
            else:
                try:
                    prob_orig_solver = canonicalize_problem(copy.deepcopy(prob_orig_raw))
                    _obj, _bnd, scip_root_bound = solve_nonconvex_qcqp_scip(
                        prob_orig_solver, time_limit=args.scip_time_limit, root_only=True
                    )
                except Exception as e:
                    print(f"  [SCIP] root solve failed: {e}")
                    scip_root_bound = None

        # solve RL relaxed convex problem
        if not resume_rl:
            rl_lb = None
            t0 = time.perf_counter()
            if rl_error is None:
                try:
                    rl_lb = solve_convex_relax_mosek(prob_rl_solver, time_limit=per_solver_budget, verbose=True)
                except Exception as e:
                    if not args.skip_failed_instances:
                        raise
                    rl_error = f"solve failed: {type(e).__name__}: {e}"
                    print(f"  [RL] {rl_error}")
            rl_solve_time = time.perf_counter() - t0
        if cached_row:
            if root_bound is None:
                root_bound = _numeric_or_none(cached_row, "gurobi_root_bound")
            if scip_root_bound is None:
                scip_root_bound = _numeric_or_none(cached_row, "scip_root_bound")

        mcc = baseline_payload.get("mccormick", {})
        sdp = baseline_payload.get("sdp", {})
        structure = baseline_payload.get("structure", {})
        random_base = baseline_payload.get("random", {})
        mcc_lb = mcc.get("lb", None)
        sdp_lb = sdp.get("lb", None)
        structure_lb = structure.get("lb", None)
        random_lb = random_base.get("lb", None)
        mcc_solve_time = mcc.get("solve_time", None)
        sdp_solve_time = sdp.get("solve_time", None)
        structure_solve_time = structure.get("solve_time", None)
        random_solve_time = random_base.get("solve_time", None)

        # costs: keep explicit aliases requested by user
        rl_cost = rl_lb
        baseline_mccormick_cost = mcc.get("cost", mcc_lb)
        baseline_sdp_cost = sdp.get("cost", sdp_lb)
        baseline_structure_cost = structure.get("cost", structure_lb)
        baseline_random_cost = random_base.get("cost", random_lb)

        one = {
            "name": prob.name,
            "dataset_key": key,
            "gurobi_root_bound": root_bound,
            "scip_root_bound": scip_root_bound,
            "rl_lb": rl_lb,
            "rl_error": rl_error,
            "baseline_mccormick_lb": mcc_lb,
            "baseline_sdp_lb": sdp_lb,
            "baseline_structure_lb": structure_lb,
            "baseline_random_lb": random_lb,
            "baseline_structure_k_min": args.structure_k_min,
            "baseline_structure_tau_density": args.structure_tau_density,
            "baseline_random_include_qcr": bool(args.random_include_qcr),
            "baseline_random_policy_version": RANDOM_BASELINE_POLICY_VERSION,
            "baseline_random_lb_std": random_base.get("lb_std", 0.0),
            "baseline_random_n_rollouts": random_base.get("n_rollouts", args.random_rollouts),
            "rl_cost": rl_cost,
            "baseline_mccormick_cost": baseline_mccormick_cost,
            "baseline_sdp_cost": baseline_sdp_cost,
            "baseline_structure_cost": baseline_structure_cost,
            "baseline_random_cost": baseline_random_cost,
            "gurobi_root_cost": root_bound,
            "scip_root_cost": scip_root_bound,
            "lb_improve": None if (root_bound is None or rl_lb is None) else (rl_lb - root_bound),
            "lb_improve_pct": _pct_improve(rl_lb, root_bound),
            "lb_improve_scip": None if (scip_root_bound is None or rl_lb is None) else (rl_lb - scip_root_bound),
            "lb_improve_scip_pct": _pct_improve(rl_lb, scip_root_bound),
            "mccormick_minus_rl": None if (mcc_lb is None or rl_lb is None) else (mcc_lb - rl_lb),
            "sdp_minus_rl": None if (sdp_lb is None or rl_lb is None) else (sdp_lb - rl_lb),
            "rl_minus_mccormick": None if (mcc_lb is None or rl_lb is None) else (rl_lb - mcc_lb),
            "rl_minus_sdp": None if (sdp_lb is None or rl_lb is None) else (rl_lb - sdp_lb),
            "rl_minus_structure": None if (structure_lb is None or rl_lb is None) else (rl_lb - structure_lb),
            "rl_minus_random": None if (random_lb is None or rl_lb is None) else (rl_lb - random_lb),
            "mccormick_minus_root": None if (root_bound is None or mcc_lb is None) else (mcc_lb - root_bound),
            "sdp_minus_root": None if (root_bound is None or sdp_lb is None) else (sdp_lb - root_bound),
            "structure_minus_root": None if (root_bound is None or structure_lb is None) else (structure_lb - root_bound),
            "random_minus_root": None if (root_bound is None or random_lb is None) else (random_lb - root_bound),
            "mccormick_minus_scip": None if (scip_root_bound is None or mcc_lb is None) else (mcc_lb - scip_root_bound),
            "sdp_minus_scip": None if (scip_root_bound is None or sdp_lb is None) else (sdp_lb - scip_root_bound),
            "structure_minus_scip": None if (scip_root_bound is None or structure_lb is None) else (structure_lb - scip_root_bound),
            "random_minus_scip": None if (scip_root_bound is None or random_lb is None) else (random_lb - scip_root_bound),
            "rl_relax_time_sec": rl_relax_time,
            "rl_canonicalize_time_sec": rl_canon_time,
            "rl_solve_time_sec": rl_solve_time,
            "rl_pipeline_time_sec": rl_relax_time + rl_canon_time + rl_solve_time,
            "baseline_mccormick_relax_time_sec": mcc.get("relax_time", None),
            "baseline_mccormick_canonicalize_time_sec": mcc.get("canon_time", None),
            "baseline_mccormick_solve_time_sec": mcc_solve_time,
            "baseline_mccormick_pipeline_time_sec": mcc.get("pipeline_time", None),
            "baseline_sdp_relax_time_sec": sdp.get("relax_time", None),
            "baseline_sdp_canonicalize_time_sec": sdp.get("canon_time", None),
            "baseline_sdp_solve_time_sec": sdp_solve_time,
            "baseline_sdp_pipeline_time_sec": sdp.get("pipeline_time", None),
            "baseline_structure_relax_time_sec": structure.get("relax_time", None),
            "baseline_structure_canonicalize_time_sec": structure.get("canon_time", None),
            "baseline_structure_solve_time_sec": structure_solve_time,
            "baseline_structure_pipeline_time_sec": structure.get("pipeline_time", None),
            "baseline_random_relax_time_sec": random_base.get("relax_time", None),
            "baseline_random_canonicalize_time_sec": random_base.get("canon_time", None),
            "baseline_random_solve_time_sec": random_solve_time,
            "baseline_random_pipeline_time_sec": random_base.get("pipeline_time", None),
            "baseline_mccormick_cache_hit": bool(mcc.get("cache_hit", False)),
            "baseline_sdp_cache_hit": bool(sdp.get("cache_hit", False)),
            "baseline_structure_cache_hit": bool(structure.get("cache_hit", False)),
            "baseline_random_cache_hit": bool(random_base.get("cache_hit", False)),
            "baseline_mccormick_error": mcc.get("error", None),
            "baseline_sdp_error": sdp.get("error", None),
            "baseline_structure_error": structure.get("error", None),
            "baseline_random_error": random_base.get("error", None),
            # backward-compat fields used by summarize.py
            "baseline_lb": mcc_lb,
            "lb_improve_baseline": None if (root_bound is None or mcc_lb is None) else (mcc_lb - root_bound),
            "lb_improve_baseline_scip": None if (scip_root_bound is None or mcc_lb is None) else (mcc_lb - scip_root_bound),
            "rl_minus_baseline": None if (rl_lb is None or mcc_lb is None) else (rl_lb - mcc_lb),
            "baseline_relax_time_sec": mcc.get("relax_time", None),
            "baseline_solve_time_sec": mcc_solve_time,
            "baseline_pipeline_time_sec": mcc.get("pipeline_time", None),
            "baseline_mode": "mccormick",
        }
        one.update(_flatten_stats("orig", orig_stats))
        one.update(_flatten_stats("rl", rl_stats))
        if mcc:
            one.update(_flatten_stats("baseline_mccormick", mcc["stats"]))
        if sdp:
            one.update(_flatten_stats("baseline_sdp", sdp["stats"]))
        if structure:
            one.update(_flatten_stats("baseline_structure", structure["stats"]))
        if random_base:
            one.update(_flatten_stats("baseline_random", random_base["stats"]))
        if all(k in rl_stats for k in ("num_vars_total", "num_constraints_total", "nnz_terms_total", "psd_entry_sum", "num_psd_constraints")):
            one.update(_growth_stats("rl", orig_stats, rl_stats))
        elif cached_row:
            one.update(_extract_cached_growth(cached_row, "rl"))
        if mcc:
            one.update(mcc["growth"])
        if sdp:
            one.update(sdp["growth"])
        if structure:
            one.update(structure["growth"])
        if random_base:
            one.update(random_base["growth"])

        results.append(one)

        if not args.no_case_pkl:
            safe_name = re.sub(r"[^0-9a-zA-Z._\-+=]", "_", re.sub(r"\s+", "_", str(prob.name)))
            case_name = f"{idx:04d}__{safe_name}"
            pack_path = os.path.join(save_dir, f"{case_name}.pkl")
            pack = {
                "name": prob.name,
                "prob_orig_raw": prob_orig_raw,
                "prob_rl_raw": prob_rl_raw,
                "prob_baseline_mccormick_raw": mcc.get("raw", None),
                "prob_baseline_sdp_raw": sdp.get("raw", None),
                "prob_baseline_structure_raw": structure.get("raw", None),
                "prob_baseline_random_raw": random_base.get("raw", None),
                "result": one,
            }
            with open(pack_path, "wb") as f:
                pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            "  [Result] "
            f"root={root_bound}, scip={scip_root_bound}, "
            f"rl_cost={rl_cost}, mcc_cost={baseline_mccormick_cost}, sdp_cost={baseline_sdp_cost}, "
            f"structure_cost={baseline_structure_cost}, random_cost={baseline_random_cost}"
        )

        if (idx + 1) % flush_every == 0:
            _flush()
            _flush_baseline_cache()
            print(f"  [SAVE] flushed results -> {out_path}")

    _flush()
    _flush_baseline_cache()
    print(f"\n[INFO] Saved results to {out_path}")
    if baseline_cache_path:
        print(f"[INFO] Saved baseline cache to {baseline_cache_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
