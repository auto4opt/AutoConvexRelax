# -*- coding: utf-8 -*-
import os
import json
import csv
import pickle
import argparse
import math
import subprocess
import sys
from typing import Any

from autoconvexrelax.evaluation.solvers.scip import solve_nonconvex_qcqp_scip


# ---- copied from train.py (one-time preprocessing) ----
def preprocess_problem_once(prob):
    # 1) canonicalize constraint senses (once)
    if not getattr(prob, "_sense_canon_done", False):
        if hasattr(prob, "_canonicalize_constraint_senses_inplace"):
            prob._canonicalize_constraint_senses_inplace()
        prob._sense_canon_done = True

    # 2) term mapping/expansion (once)
    if not getattr(prob, "_cons_preproc_done", False):
        if hasattr(prob, "map_all_terms"):
            prob.map_all_terms(update_problem=True)
        prob._cons_preproc_done = True

    return prob


def preprocess_any(obj: Any):
    if hasattr(obj, "_canonicalize_constraint_senses_inplace"):
        return preprocess_problem_once(obj)
    if isinstance(obj, list):
        return [preprocess_any(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(preprocess_any(x) for x in obj)
    if isinstance(obj, dict):
        return {k: preprocess_any(v) for k, v in obj.items()}
    return obj


def make_key(dataset_tag: str, global_idx: int, prob) -> str:
    name = getattr(prob, "name", "noname")
    name = str(name).replace(":", "_").replace("/", "_")
    return f"{dataset_tag}:{global_idx}:{name}"


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _is_missing_value(v) -> bool:
    if v is None:
        return True
    try:
        fv = float(v)
    except Exception:
        return True
    if math.isnan(fv) or math.isinf(fv):
        return True
    return False


def _run_single_in_subprocess(args, idx: int, key: str, total: int, fail_writer):
    tmp_out = f"{args.out}.tmp_idx{idx}.json"
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--data", args.data,
        "--out", args.out,
        "--time_limit", str(args.time_limit),
        "--dataset_tag", args.dataset_tag,
        "--single_idx", str(idx),
        "--single_out", tmp_out,
    ]
    env = os.environ.copy()
    # reduce chance of native crashes
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        res = subprocess.run(cmd, env=env)
    except Exception as e:
        if fail_writer is not None:
            fail_writer.writerow([idx, "", key, f"subprocess_error: {repr(e)}"])
        print(f"[FAIL] idx={idx}/{total} key={key} err=subprocess_error:{repr(e)}", flush=True)
        return None

    if res.returncode != 0 or (not os.path.exists(tmp_out)):
        if fail_writer is not None:
            fail_writer.writerow([idx, "", key, f"subprocess_rc={res.returncode}"])
        print(f"[FAIL] idx={idx}/{total} key={key} err=subprocess_rc={res.returncode}", flush=True)
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        return None

    try:
        with open(tmp_out, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("value", None)
    finally:
        try:
            os.remove(tmp_out)
        except Exception:
            pass


def _run_block_in_subprocess(args, indices, total: int, fail_writer):
    tmp_idx = f"{args.out}.tmp_idxlist_{indices[0]}_{indices[-1]}.json"
    tmp_out = f"{args.out}.tmp_block_{indices[0]}_{indices[-1]}.json"

    with open(tmp_idx, "w", encoding="utf-8") as f:
        json.dump(indices, f, ensure_ascii=False)

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--data", args.data,
        "--time_limit", str(args.time_limit),
        "--dataset_tag", args.dataset_tag,
        "--idx_list", tmp_idx,
        "--block_out", tmp_out,
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        res = subprocess.run(cmd, env=env)
    except Exception as e:
        if fail_writer is not None:
            for idx in indices:
                fail_writer.writerow([idx, "", "", f"subprocess_error: {repr(e)}"])
        print(f"[FAIL] block={indices[0]}-{indices[-1]} err=subprocess_error:{repr(e)}", flush=True)
        try:
            if os.path.exists(tmp_idx):
                os.remove(tmp_idx)
        except Exception:
            pass
        return None

    if res.returncode != 0 or (not os.path.exists(tmp_out)):
        if fail_writer is not None:
            for idx in indices:
                fail_writer.writerow([idx, "", "", f"subprocess_rc={res.returncode}"])
        print(f"[FAIL] block={indices[0]}-{indices[-1]} err=subprocess_rc={res.returncode}", flush=True)
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            if os.path.exists(tmp_idx):
                os.remove(tmp_idx)
        except Exception:
            pass
        return None

    try:
        with open(tmp_out, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    finally:
        for p in (tmp_out, tmp_idx):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pkl path, e.g., outputs/data/vector_finetune_HYBRID_MIX_1200.pkl")
    ap.add_argument("--out", required=False, default=None, help="output cache json path")
    ap.add_argument("--fail_csv", default=None, help="failure log csv (optional)")
    ap.add_argument("--time_limit", type=float, default=60.0, help="SCIP TimeLimit per problem (s)")
    ap.add_argument("--dataset_tag", default="HYBRID_MIX_1200", help="cache key prefix")
    ap.add_argument("--save_every", type=int, default=10, help="save every N problems")
    ap.add_argument("--start_idx", type=int, default=0, help="start index (inclusive)")
    ap.add_argument("--end_idx", type=int, default=-1, help="end index (exclusive), -1 means all")
    ap.add_argument("--block_size", type=int, default=1, help="subprocess block size (only when not --direct)")
    ap.add_argument("--direct", action="store_true", help="run in-process (faster but may segfault)")
    ap.add_argument("--max_new", type=int, default=0, help="max number of missing entries to compute (0 = no limit)")
    # internal: single problem mode (used by subprocess for safety)
    ap.add_argument("--single_idx", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--single_out", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--idx_list", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--block_out", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # validate required args for normal mode
    if args.single_idx is None and args.idx_list is None and not args.out:
        ap.error("the following arguments are required: --out")

    # 1) load dataset
    with open(args.data, "rb") as f:
        problem_set = pickle.load(f)

    # expect [[prob, prob, ...]]
    problem_set = preprocess_any(problem_set)
    assert isinstance(problem_set, list) and len(problem_set) == 1, f"expect 1 group, got {len(problem_set)}"
    problems = problem_set[0]
    assert isinstance(problems, list), "group[0] should be a list of problems"

    # list-of-indices mode (subprocess target)
    if args.idx_list is not None:
        if args.block_out is None:
            raise ValueError("--block_out is required when --idx_list is set")
        with open(args.idx_list, "r", encoding="utf-8") as f:
            indices = json.load(f)
        payload = {"values": {}, "errors": []}
        for idx in indices:
            if idx < 0 or idx >= len(problems):
                continue
            prob = problems[idx]
            key = make_key(args.dataset_tag, idx, prob)
            try:
                _obj, _best_bound, root_lb = solve_nonconvex_qcqp_scip(
                    prob,
                    time_limit=args.time_limit,
                    root_only=True
                )
                val = float(root_lb) if root_lb is not None else None
                payload["values"][key] = val
            except Exception as e:
                payload["values"][key] = None
                payload["errors"].append([idx, getattr(prob, "name", "noname"), key, repr(e)])
        with open(args.block_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return

    # single-problem mode (subprocess target)
    if args.single_idx is not None:
        if args.single_out is None:
            raise ValueError("--single_out is required when --single_idx is set")
        idx = int(args.single_idx)
        if idx < 0 or idx >= len(problems):
            raise IndexError(f"single_idx out of range: {idx}")
        prob = problems[idx]
        key = make_key(args.dataset_tag, idx, prob)
        try:
            _obj, _best_bound, root_lb = solve_nonconvex_qcqp_scip(
                prob,
                time_limit=args.time_limit,
                root_only=True
            )
            val = float(root_lb) if root_lb is not None else None
            payload = {"key": key, "value": val}
        except Exception as e:
            payload = {"key": key, "value": None, "error": repr(e)}
        with open(args.single_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return

    # 2) load existing cache
    cache = load_cache(args.out)

    # 3) failure log
    fail_writer = None
    fail_f = None
    if args.fail_csv:
        fail_f = open(args.fail_csv, "a", newline="", encoding="utf-8")
        fail_writer = csv.writer(fail_f)
        if fail_f.tell() == 0:
            fail_writer.writerow(["global_idx", "name", "key", "error"])

    total = len(problems)
    start_idx = max(0, int(args.start_idx))
    end_idx = total if args.end_idx is None or int(args.end_idx) < 0 else min(total, int(args.end_idx))
    if start_idx >= end_idx:
        print(f"[DONE] empty range: start_idx={start_idx}, end_idx={end_idx}, total={total}", flush=True)
        return
    indices_to_process = []
    for idx in range(start_idx, end_idx):
        prob = problems[idx]
        key = make_key(args.dataset_tag, idx, prob)
        if key in cache and (not _is_missing_value(cache[key])):
            continue
        indices_to_process.append(idx)

    if args.max_new > 0:
        indices_to_process = indices_to_process[: int(args.max_new)]

    processed = 0
    last_saved = 0

    if args.direct or args.block_size <= 1:
        for idx in indices_to_process:
            prob = problems[idx]
            key = make_key(args.dataset_tag, idx, prob)
            try:
                if args.direct:
                    _obj, _best_bound, root_lb = solve_nonconvex_qcqp_scip(
                        prob,
                        time_limit=args.time_limit,
                        root_only=True
                    )
                    cache[key] = float(root_lb) if root_lb is not None else None
                else:
                    val = _run_single_in_subprocess(args, idx, key, total, fail_writer)
                    cache[key] = val
            except Exception as e:
                cache[key] = None
                if fail_writer is not None:
                    fail_writer.writerow([idx, getattr(prob, "name", "noname"), key, repr(e)])
                print(f"[FAIL] idx={idx}/{total} name={getattr(prob,'name','noname')} err={repr(e)}", flush=True)

            processed += 1
            if (processed - last_saved) >= args.save_every:
                save_cache(args.out, cache)
                last_saved = processed
                print(f"[SAVE] {processed}/{total} saved to {args.out}", flush=True)
    else:
        bs = max(1, int(args.block_size))
        for i in range(0, len(indices_to_process), bs):
            block = indices_to_process[i:i + bs]
            payload = _run_block_in_subprocess(args, block, total, fail_writer)
            if payload is None:
                # mark all in block failed
                for idx in block:
                    prob = problems[idx]
                    key = make_key(args.dataset_tag, idx, prob)
                    cache[key] = None
                processed += len(block)
            else:
                vals = payload.get("values", {})
                errs = payload.get("errors", [])
                for key, val in vals.items():
                    cache[key] = val
                if fail_writer is not None:
                    for row in errs:
                        fail_writer.writerow(row)
                processed += len(block)

            if (processed - last_saved) >= args.save_every:
                save_cache(args.out, cache)
                last_saved = processed
                print(f"[SAVE] {processed}/{total} saved to {args.out}", flush=True)

    save_cache(args.out, cache)
    if fail_f is not None:
        fail_f.close()

    ok = sum(1 for v in cache.values() if isinstance(v, (int, float)))
    bad = sum(1 for v in cache.values() if v is None)
    print(f"[DONE] total={total}, ok={ok}, failed={bad}, cache={args.out}", flush=True)


if __name__ == "__main__":
    main()
