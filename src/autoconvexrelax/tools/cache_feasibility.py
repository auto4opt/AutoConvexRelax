# -*- coding: utf-8 -*-
import os
import json
import csv
import pickle
import argparse
from typing import Any, Dict

from autoconvexrelax.evaluation.solvers.gurobi import solve_nonconvex_qcqp  # 你现有实现


# ---- 从 train.py 里抄过来的“一次性预处理”逻辑（避免 sense/term_map 不一致）----
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


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json_atomic(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pkl 路径，例如 outputs/data/vector_finetune_HYBRID_MIX_1200.pkl")

    # 可行性 cache（单独文件）
    ap.add_argument("--out_feas", required=True, help="输出可行性 cache json 路径（单独文件）")

    # 可选：同时输出 root lb cache（独立文件，方便训练）
    ap.add_argument("--out_rootlb", default=None, help="输出 root lb cache json（可选）")

    ap.add_argument("--fail_csv", default=None, help="失败记录 csv（可选）")
    ap.add_argument("--time_limit", type=float, default=60.0, help="每个问题求解 TimeLimit（秒）")
    ap.add_argument("--dataset_tag", default="DATASET", help="key 前缀（建议与数据集绑定）")
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

    total = len(problems)

    # 2) load existing caches (支持断点续算)
    feas_cache: Dict[str, Any] = load_json(args.out_feas)
    rootlb_cache: Dict[str, Any] = load_json(args.out_rootlb) if args.out_rootlb else {}

    # 3) prepare failure log
    fail_writer = None
    fail_f = None
    if args.fail_csv:
        fail_f = open(args.fail_csv, "a", newline="", encoding="utf-8")
        fail_writer = csv.writer(fail_f)
        if fail_f.tell() == 0:
            fail_writer.writerow(["global_idx", "name", "key", "error"])

    done = 0

    for idx, prob in enumerate(problems):
        key = make_key(args.dataset_tag, idx, prob)

        # 如果已存在并且不是 None，就跳过（断点续跑）
        if key in feas_cache and feas_cache[key] is not None:
            done += 1
            continue

        try:
            # 优先尝试：若 solver_interface 支持 return_status=True，则拿到 status/sol_count
            strict_info = None
            try:
                strict_info = solve_nonconvex_qcqp(prob, time_limit=args.time_limit, return_status=True)
            except TypeError:
                strict_info = None  # 说明你的函数还没加 return_status 参数

            if isinstance(strict_info, dict):
                # 严格判定（推荐）
                sol_count = int(strict_info.get("sol_count", 0) or 0)
                status = strict_info.get("status", None)

                # has_feasible: 找到至少一个 incumbent
                has_feasible = (sol_count > 0)

                feas_cache[key] = {
                    "feasible": bool(has_feasible),
                    "status": status,
                    "sol_count": sol_count,
                    "best_obj": strict_info.get("best_obj", None),
                    "best_bound": strict_info.get("best_bound", None),
                    "root_lb": strict_info.get("root_lb", None),
                }

                if args.out_rootlb:
                    rootlb_cache[key] = strict_info.get("root_lb", None)

            else:
                # 兼容模式：只能用旧返回值做弱判定（不如 sol_count 稳）
                _obj, _best_bound, root_lb = solve_nonconvex_qcqp(prob, time_limit=args.time_limit)

                # 弱判据：只要拿到一个可行 incumbent，Gurobi 一般就能给 ObjVal（你旧函数通常会返回 obj）
                # 注意：TIME_LIMIT 且没找到 incumbent 时 obj 可能是 None，此时 feasible 只能记为 None（未知）
                if _obj is None:
                    has_feasible = None
                else:
                    has_feasible = True

                feas_cache[key] = {
                    "feasible": has_feasible,
                    "status": None,
                    "sol_count": None,
                    "best_obj": _obj,
                    "best_bound": _best_bound,
                    "root_lb": root_lb,
                }

                if args.out_rootlb:
                    rootlb_cache[key] = root_lb

        except Exception as e:
            feas_cache[key] = None
            if args.out_rootlb:
                rootlb_cache[key] = None
            if fail_writer is not None:
                fail_writer.writerow([idx, getattr(prob, "name", "noname"), key, repr(e)])
            print(f"[FAIL] idx={idx}/{total} name={getattr(prob,'name','noname')} err={repr(e)}", flush=True)

        done += 1
        if done % args.save_every == 0:
            save_json_atomic(args.out_feas, feas_cache)
            if args.out_rootlb:
                save_json_atomic(args.out_rootlb, rootlb_cache)
            print(f"[SAVE] {done}/{total} saved: feas={args.out_feas}" +
                  (f", rootlb={args.out_rootlb}" if args.out_rootlb else ""), flush=True)

    save_json_atomic(args.out_feas, feas_cache)
    if args.out_rootlb:
        save_json_atomic(args.out_rootlb, rootlb_cache)

    if fail_f is not None:
        fail_f.close()

    # 汇总统计
    feas_vals = [v for v in feas_cache.values() if isinstance(v, dict)]
    n_true = sum(1 for v in feas_vals if v.get("feasible") is True)
    n_false = sum(1 for v in feas_vals if v.get("feasible") is False)
    n_unk = sum(1 for v in feas_vals if v.get("feasible") is None)
    n_none = sum(1 for v in feas_cache.values() if v is None)

    print(
        f"[DONE] total={total}, feasible={n_true}, infeasible={n_false}, unknown={n_unk}, failed={n_none}\n"
        f"       feas_cache={args.out_feas}" + (f"\n       rootlb_cache={args.out_rootlb}" if args.out_rootlb else ""),
        flush=True
    )


if __name__ == "__main__":
    main()
