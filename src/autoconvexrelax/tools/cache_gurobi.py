# -*- coding: utf-8 -*-
import os
import json
import csv
import pickle
import argparse
import copy
from typing import Any
from autoconvexrelax.evaluation.solvers.gurobi import solve_nonconvex_qcqp  # 你现有实现

# ---- 从 train.py 里抄过来的 “一次性预处理”逻辑（避免 sense/term_map 不一致）----
def preprocess_problem_once(prob):
    # 1) 约束 sense 归一化（只做一次）
    if not getattr(prob, "_sense_canon_done", False):
        if hasattr(prob, "_canonicalize_constraint_senses_inplace"):
            prob._canonicalize_constraint_senses_inplace()
        prob._sense_canon_done = True

    # 2) 对原始约束做一次 term 映射/展开（只做一次）
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pkl 路径，例如 outputs/data/vector_finetune_HYBRID_MIX_1200.pkl")
    ap.add_argument("--out", required=True, help="输出 cache json 路径")
    ap.add_argument("--fail_csv", default=None, help="失败记录 csv（可选）")
    ap.add_argument("--time_limit", type=float, default=60.0, help="每个问题 Gurobi root bound 计算的 TimeLimit（秒）")
    ap.add_argument("--dataset_tag", default="HYBRID_MIX_1200", help="key 前缀（建议与数据集绑定）")
    ap.add_argument("--save_every", type=int, default=10, help="每算多少个问题落盘一次")
    args = ap.parse_args()

    # 1) load dataset
    with open(args.data, "rb") as f:
        problem_set = pickle.load(f)

    # 你的 train.py 假设是“单组”：[[prob, prob, ...]]
    problem_set = preprocess_any(problem_set)
    assert isinstance(problem_set, list) and len(problem_set) == 1, f"expect 1 group, got {len(problem_set)}"
    problems = problem_set[0]
    assert isinstance(problems, list), "group[0] should be a list of problems"

    # 2) load existing cache (支持断点续算)
    cache = load_cache(args.out)

    # 3) prepare failure log
    fail_writer = None
    fail_f = None
    if args.fail_csv:
        fail_f = open(args.fail_csv, "a", newline="", encoding="utf-8")
        fail_writer = csv.writer(fail_f)
        if fail_f.tell() == 0:
            fail_writer.writerow(["global_idx", "name", "key", "error"])

    total = len(problems)
    done = 0

    for idx, prob in enumerate(problems):
        key = make_key(args.dataset_tag, idx, prob)
        if key in cache and cache[key] is not None:
            done += 1
            continue

        try:
            # 注意：solve_nonconvex_qcqp 内部会 build Gurobi 模型；
            # 如果问题里含分式/不支持的 Trace 形态，会抛异常。
            _obj, _best_bound, root_lb = solve_nonconvex_qcqp(
                prob,
                time_limit=args.time_limit,
                root_only=True
            )

            cache[key] = float(root_lb)

        except Exception as e:
            cache[key] = None
            if fail_writer is not None:
                fail_writer.writerow([idx, getattr(prob, "name", "noname"), key, repr(e)])
            print(f"[FAIL] idx={idx}/{total} name={getattr(prob,'name','noname')} err={repr(e)}", flush=True)

        done += 1
        if done % args.save_every == 0:
            save_cache(args.out, cache)
            print(f"[SAVE] {done}/{total} saved to {args.out}", flush=True)

    save_cache(args.out, cache)
    if fail_f is not None:
        fail_f.close()

    ok = sum(1 for v in cache.values() if isinstance(v, (int, float)))
    bad = sum(1 for v in cache.values() if v is None)
    print(f"[DONE] total={total}, ok={ok}, failed={bad}, cache={args.out}", flush=True)

if __name__ == "__main__":
    main()
