# Policy evaluation utilities.
# -*- coding: utf-8 -*-
"""
对比实验 Runner:
    - 加载已经训练好的 PolicyNet（和 train.py 完全一致）
    - 从 vector_finetune_PURE_1000.pkl 加载一批非分式 QCQP
    - 对每个问题:
        * 用 RL policy 反复对单个 term 做松弛，直到没有非凸项
        * 用 Gurobi 对原问题做 nonconvex QCQP 求解 (obj + bound)
        * 用 Gurobi 对 RL 松弛后的凸问题求 root relax (convex LB)
    - 结果保存到 eval_vs_gurobi.json
"""

import os
    
import copy
import json
import glob
import pickle
import argparse
import sympy as sp
import torch
import random
import numpy as np


from autoconvexrelax.core.problem import QCQPProblem
from autoconvexrelax.core.relaxation import RelaxationEngine
from autoconvexrelax.evaluation.expressions import normalize_expr
from autoconvexrelax.evaluation.solvers.gurobi import solve_nonconvex_qcqp
try:
    from autoconvexrelax.evaluation.solvers.scip import solve_nonconvex_qcqp_scip
except Exception:
    solve_nonconvex_qcqp_scip = None
from autoconvexrelax.evaluation.solvers.mosek import solve_convex_relax_mosek

from autoconvexrelax.training.stage2 import PolicyNet, relaxation, ACTION_TYPE
import autoconvexrelax.training.stage2 as train_mod
from autoconvexrelax.reward import get_reward

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 可选：尽量保证 GPU 推理确定性（会略慢，但评估阶段通常可接受）
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def action_changed(last_rewrite) -> bool:
    """根据 RelaxationEngine 的返回判断是否真的修改了表达式/结构。"""
    if not last_rewrite:
        return False
    old = last_rewrite.get('old', None)
    new = last_rewrite.get('new', None)
    if old is None or new is None:
        # 兼容性：如果没有提供 old/new，就当未知处理为“无变化”
        return False
    return str(old) != str(new)


def canonicalize_problem(prob: QCQPProblem) -> QCQPProblem:
    # objective
    prob.obj_expr = normalize_expr(prob.obj_expr)

    # constraints
    for c in prob.constraints:
        c.expr = normalize_expr(c.expr)
        if c.rhs is not None and hasattr(c.rhs, "free_symbols"):
            c.rhs = normalize_expr(c.rhs)

    # 如果你依赖 term mapping，规范化后建议重新 map
    try:
        prob.map_all_terms()
    except Exception:
        pass
    return prob


def load_eval_groups(path: str):
    with open(path, "rb") as f:
        groups = pickle.load(f)
    # ?? [[prob,...]]
    assert isinstance(groups, list)
    assert isinstance(groups[0], list)
    return groups


def load_split_indices(split_path: str) -> dict:
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError("split json missing 'groups'")
    return data


def resolve_split_json(path_hint: str) -> str:
    if path_hint:
        return path_hint
    candidates = sorted(glob.glob(os.path.join("train_logs", "*_split_indices.json")))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError("Multiple split files found in train_logs; pass --split_json explicitly.")
    return ""


def collect_split_problems(groups, split_data: dict, use: str = "infer"):
    split_groups = split_data.get("groups", [])
    if len(split_groups) != len(groups):
        raise ValueError(f"split groups={len(split_groups)} mismatch data groups={len(groups)}")
    key = "infer_indices" if use == "infer" else "train_indices"
    selected = []
    selected_idx = []
    for probs, ginfo in zip(groups, split_groups):
        idxs = ginfo.get(key, [])
        if not isinstance(idxs, list):
            raise ValueError("split indices must be list")
        for idx in idxs:
            if isinstance(idx, int) and 0 <= idx < len(probs):
                selected.append(probs[idx])
                selected_idx.append(idx)
    return selected, selected_idx




def build_model(ckpt_path: str, device: str, example_prob: QCQPProblem):
    """
    按照 train.py 里的配置构建 PolicyNet，然后加载 ckpt["model_state_dict"]。

    关键点：
      - 先用 example_prob 做一遍 state_representation，触发 GNN 初始化
      - 再 load_state_dict，这样就不会有 'Unexpected key(s)' 的问题
    """
    n_actions = len(ACTION_TYPE)

    model = PolicyNet(
        d_model=256,
        d_embed=256,
        n_head=8,
        ffn_hidden=512,
        drop_prob=0.1,
        n_actions=n_actions,
        device=device,
    ).to(device)

    # 先 dummy forward 一次，把 GNN 等 lazy module 建出来
    print("[INFO] Dummy forward to initialize Agent/GNN...")
    with torch.no_grad():
        _ = model.agent.state_representation(example_prob)

    # 再加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 用 strict=False，万一有一点小的不对齐也直接忽略
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys when loading state_dict: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys in state_dict (ignored): {len(incompatible.unexpected_keys)}")

    model.eval()
    return model



@torch.no_grad()
def apply_policy_until_convex(
    prob: QCQPProblem,
    model,
    max_steps: int = 50,
    device: str = "cpu",
    post_convex_budget: int = 6,        # NEW: 变凸后允许继续的最多步数
    stagnation_patience: int = 2,       # NEW: 连续几步不变就停
):
    cur_prob = copy.deepcopy(prob)
    try:
        cur_prob.map_all_terms()
    except Exception:
        pass

    TOPK = 10

    no_change_streak = 0        # NEW
    convex_steps_used = 0       # NEW

    # 你的全局动作 id（按你当前 ACTION_TYPE 表：5/6）
    GLOBAL_AIDS = {5, 6}        # bound_tightening / global_cut_generation

    for step in range(max_steps):
        try:
            cur_prob.map_all_terms()
        except Exception:
            pass

        state_info_raw = model.agent.state_representation(cur_prob)
        seq = state_info_raw["seq"]
        non_convex_indices = state_info_raw["non_convex_indices"]
        non_convex_vtypes = state_info_raw["non_convex_vtypes"]

        is_convex_now = (non_convex_indices.numel() == 0)

        if is_convex_now:
            convex_steps_used += 1
            if convex_steps_used > post_convex_budget:
                print(f"  [step {step}] convex and post_convex_budget exhausted, stop.")
                break
        else:
            convex_steps_used = 0

        state_info = {
            "seq": seq.to(device),
            "problem_type": cur_prob.problem_type,
            "non_convex_indices": non_convex_indices.to(device),
            "non_convex_vtypes": non_convex_vtypes.to(device),
        }

        # ===== 用 train.py 的 Agent 输出 logits，自己实现 “同一位置 Top-K action 回退” =====
        seq_dev = seq.to(device)
        ncv_vtypes_dev = non_convex_vtypes.to(device)

        # 直接调用 agent 拿全量 logits（注意：返回的 lp_id_all / lp_loc_all 是 logits 空间）:contentReference[oaicite:6]{index=6}
        (a_id, a_id_probs,
        a_loc, a_loc_probs,
        lp_id_all, lp_loc_all,
        lp_id, lp_loc) = model.agent(
            seq_dev,
            mode="eval",                          # eval：通常是 argmax（取决于你 AgentLayer 的实现）
            problem_type=cur_prob.problem_type,
            non_convex_vtypes=ncv_vtypes_dev
        )

        # 统一成 1D
        if lp_id_all.dim() > 1:
            lp_id_all = lp_id_all.squeeze(0)
        if lp_loc_all.dim() > 1:
            lp_loc_all = lp_loc_all.squeeze(0)

        # 1) 位置：argmax
        seq_idx = int(torch.argmax(lp_loc_all).item())

        # # 2) 动作：top-k
        # TOPK = min(TOPK, int(lp_id_all.numel()))
        # aid_list = torch.topk(lp_id_all, k=TOPK).indices.tolist()
        
        # 2) 动作：argmax（取消 top-k）
        aid = int(torch.argmax(lp_id_all).item())
        aid_list = [aid]   # 保持后面代码结构不动



        if seq_idx < 0:
            print(f"  [step {step}] invalid a_loc_seq_idx={seq_idx}, stop.")
            break

        # --- 分两种：GLOBAL or term ---
        if seq_idx == 0:
            # GLOBAL：只允许全局动作
            term_id_to_act_on = 0
            aid_list = [int(a) for a in aid_list if int(a) in GLOBAL_AIDS]
            if not aid_list:
                print(f"  [step {step}] GLOBAL chosen but no global aids in topk, count as no-change.")
                no_change_streak += 1
                if no_change_streak >= stagnation_patience:
                    print(f"  [step {step}] stagnation, stop.")
                    break
                continue
        else:
            # term 分支：和你原来一样映射到 term_id
            local_idx = seq_idx - 1
            if local_idx < 0 or local_idx >= non_convex_indices.numel():
                print(f"  [step {step}] local_idx out of range: {local_idx}/{non_convex_indices.numel()}, stop.")
                break
            original_idx = non_convex_indices[local_idx]
            term_id_to_act_on = int(original_idx.item()) + 1

        applied = False
        for rank, aid in enumerate(aid_list, start=1):
            aid = int(aid)

            if aid in (13, 14):
                print(f"    [Skip] disabled action aid={aid}")
                continue

            a_id_tensor = torch.tensor(aid, dtype=torch.long, device=device)
            term_id_tensor = torch.tensor(term_id_to_act_on, dtype=torch.long, device=device)

            print(f"  [step {step}] Try #{rank}: aid={aid} @ loc(seq)={seq_idx} -> term_id={term_id_to_act_on}")
            next_prob, last_rewrite = relaxation(cur_prob, a_id_tensor, term_id_tensor)

            changed = action_changed(last_rewrite)
            r = get_reward(cur_prob, next_prob, aid, last_rewrite=last_rewrite)
            print(f"    [Changed] {changed}   [Reward] {r:.4f}")

            if changed:
                cur_prob = next_prob
                try:
                    cur_prob.map_all_terms()
                except Exception:
                    pass
                applied = True
                no_change_streak = 0   # NEW: 有改变则清零
                break
            else:
                print("    [Skip] no change, try next action id")

        # if not applied:
        #     no_change_streak += 1
        #     print(f"  [step {step}] no applicable change (streak={no_change_streak})")
        #     if no_change_streak >= stagnation_patience:
        #         print(f"  [step {step}] stagnation, stop.")
        #         break
        if not applied:
            print(f"  [step {step}] action ineffective, stop.")
            break


    return cur_prob



def main():

    # RelaxationEngine toggles (no CLI; set here for experiments)
    train_mod.RELAX_ENGINE_BT_WARMUP = True  # or False
    train_mod.RELAX_ENGINE_BT_BEFORE_GLOBAL_CUT = False  # set True if needed
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/checkpoints/latest.pt",
        help="???? PolicyNet checkpoint ??",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/data/vector_finetune_HYBRID_MIX_1200.pkl",
        help="???????????? QCQP ???",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="",
        help="path to *_split_indices.json",
    )
    parser.add_argument(
        "--split_use",
        type=str,
        default="infer",
        choices=["infer", "train"],
        help="use infer or train split",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="ignore split and use full dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda ? cpu",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=60.0,
        help="Gurobi ????????????",
    )
    parser.add_argument(
        "--scip_root",
        action="store_true",
        help="compute SCIP root bound (nonconvex BnB)",
    )
    parser.add_argument(
        "--scip_time_limit",
        type=float,
        default=60.0,
        help="SCIP time limit (s)",
    )
    parser.add_argument(
        "--scip_root_cache",
        type=str,
        default=None,
        help="SCIP root bound cache json (optional, same key format as compute_root_lb_cache_scip.py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="???????random/numpy/torch?",
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=50,
        help="<0 run all problems; otherwise run this many problems (after split/sample) ",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="head",
        choices=["head", "random"],
        help="head=??N??random=?seed???N??????",
    )
    parser.add_argument(
        "--root_lb_cache",
        type=str,
        default=None,
        help="?????? Gurobi root bound cache json?? root_lb_cache_HYBRID_MIX_1200_root_only.json?",
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        default="HYBRID_MIX_1200",
        help="cache key ??????? compute_root_lb_cache.py ? dataset_tag ??",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("outputs", "logs"),
        help="directory for runner outputs",
    )

    args = parser.parse_args()

    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[INFO] SAVE_DIR = {SAVE_DIR}")


    device = args.device
    print(f"[INFO] Use device = {device}")

    print(f"[INFO] Loading eval set from: {args.data}")
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
    
    # problems = problems[:50]
    # print(f"[INFO] Only evaluating first {len(problems)} problems.")
    
    set_seed(args.seed, deterministic=True)

    if args.n_problems <= 0:
        args.n_problems = len(problems)


    if args.sample == "head":
        problems = problems[: args.n_problems]
        orig_indices = orig_indices[: args.n_problems]
        print(f"[INFO] Only evaluating first {len(problems)} problems.")

    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(problems), size=args.n_problems, replace=False)
        idx = sorted(idx.tolist())
        problems = [problems[i] for i in idx]
        orig_indices = [orig_indices[i] for i in idx]
        print(f"[INFO] Only evaluating {len(problems)} problems (random, seed={args.seed}). idx[:10]={idx[:10]}")


    root_cache = None
    if args.root_lb_cache:
        with open(args.root_lb_cache, "r", encoding="utf-8") as f:
            root_cache = json.load(f)
        print(f"[INFO] Loaded root_lb_cache: {args.root_lb_cache}, #keys={len(root_cache)}")
    else:
        print("[WARN] No --root_lb_cache provided. gurobi_root_bound will be None.")

    scip_root_cache = None
    if args.scip_root_cache:
        with open(args.scip_root_cache, "r", encoding="utf-8") as f:
            scip_root_cache = json.load(f)
        print(f"[INFO] Loaded scip_root_cache: {args.scip_root_cache}, #keys={len(scip_root_cache)}")


    print(f"[INFO] Building model from ckpt: {args.ckpt}")
    # 用第一个问题做 dummy forward 初始化 GNN
    model = build_model(args.ckpt, device, problems[0])

    results = []
    out_path = os.path.join(SAVE_DIR, "eval_vs_gurobi.json")
    flush_every = 5

    def _flush_results():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    _flush_results()

    for idx, prob in enumerate(problems):
        print(f"\n=== Problem {idx+1}/{len(problems)}: {prob.name} ===")

        # A) RL 看原问题（不规范化）
        prob_orig_raw = copy.deepcopy(prob)
        prob_rl_raw   = copy.deepcopy(prob)

        print("  [RL] Applying policy to relax problem (RAW problem)...")
        prob_rl_raw = apply_policy_until_convex(prob_rl_raw, model, max_steps=50, device=device)

        # B) solver interface 只看规范化后的副本（分别规范化原问题和松弛后问题）
        prob_orig_solver = canonicalize_problem(copy.deepcopy(prob_orig_raw))
        prob_rl_solver   = canonicalize_problem(copy.deepcopy(prob_rl_raw))

        # print("  [Gurobi] Solving original nonconvex QCQP (canonicalized for solver)...")
        # # best_obj, best_bound, root_bound = solve_nonconvex_qcqp(prob_orig_solver, time_limit=args.time_limit)
        # best_obj, best_bound, root_bound = solve_nonconvex_qcqp(
        #     prob_orig_solver,
        #     time_limit=args.time_limit,
        #     root_only=True
        # )
        
        # --- Gurobi root bound: load from cache (no re-solving) ---
        def make_key(dataset_tag: str, global_idx: int, name: str) -> str:
            name = str(name).replace(":", "_").replace("/", "_")
            return f"{dataset_tag}:{global_idx}:{name}"

        global_idx = orig_indices[idx]
        key = make_key(args.dataset_tag, global_idx, prob.name)

        root_bound = None
        if root_cache is not None:
            v = root_cache.get(key, None)
            root_bound = float(v) if isinstance(v, (int, float)) else None

        best_obj = None
        best_bound = None

        print(f"  [Gurobi-CACHE] key={key} root_bound={root_bound}")

        scip_root_bound = None
        if scip_root_cache is not None:
            v = scip_root_cache.get(key, None)
            scip_root_bound = float(v) if isinstance(v, (int, float)) else None

        if scip_root_bound is None and args.scip_root:
            if solve_nonconvex_qcqp_scip is None:
                print("  [SCIP] solver_interface_scip not available (PySCIPOpt missing).")
            else:
                try:
                    print("  [SCIP] Solving original nonconvex QCQP (root bound)...")
                    _obj, _bnd, scip_root_bound = solve_nonconvex_qcqp_scip(
                        prob_orig_solver,
                        time_limit=args.scip_time_limit,
                        root_only=True,
                    )
                except Exception as e:
                    print(f"  [SCIP] failed: {e}")
                    scip_root_bound = None




        # === 1) 先解：RL 松弛后的“基线”下界 ===
        print("  [MOSEK] Solving RL-relaxed problem (baseline)...")
        rl_lb = solve_convex_relax_mosek(prob_rl_solver, time_limit=args.time_limit / 2, verbose=True)

        # # === 2) 再解：在同一个 prob_rl_solver 上施加“全局 tightening”后的下界 ===
        # # === NEW：全局 tightening 对比 ===
        # engine = RelaxationEngine()

        # prob_rl_tight_raw = copy.deepcopy(prob_rl_raw)

        # # 用标准 apply_action 接口触发全局操作（location/sub_expr 传什么都行）
        # engine.apply_action(
        #     prob_rl_tight_raw,
        #     location="objective",
        #     sub_expr=prob_rl_tight_raw.obj_expr,
        #     action_type="bound_tightening",
        #     extra_args={"max_rounds": 1, "tol": 1e-9},
        # )
        # engine.apply_action(
        #     prob_rl_tight_raw,
        #     location="objective",
        #     sub_expr=prob_rl_tight_raw.obj_expr,
        #     action_type="global_cut_generation",
        #     extra_args={"rlt_budget": 200, "oa_budget": 20, "tol": 1e-9},
        # )

        # prob_rl_tight_solver = canonicalize_problem(copy.deepcopy(prob_rl_tight_raw))

        # print("  [MOSEK] Solving RL-relaxed problem (after global tightening)...")
        # rl_lb_tight = solve_convex_relax_mosek(prob_rl_tight_solver, time_limit=args.time_limit / 2, verbose=True)

        # tight_gain = None if (rl_lb is None or rl_lb_tight is None) else (rl_lb_tight - rl_lb)
        # print(f"  [Tightening] rl_lb={rl_lb}, rl_lb_tight={rl_lb_tight}, gain={tight_gain}")

        def _pct_improve(rl_value, root_value, eps=1e-9):
            if rl_value is None or root_value is None:
                return None
            denom = max(abs(root_value), abs(rl_value), eps)
            return (rl_value - root_value) / denom * 100.0

        lb_improve = None if (root_bound is None or rl_lb is None) else (rl_lb - root_bound)
        lb_improve_pct = _pct_improve(rl_lb, root_bound)

        lb_improve_scip = None if (scip_root_bound is None or rl_lb is None) else (rl_lb - scip_root_bound)
        lb_improve_scip_pct = _pct_improve(rl_lb, scip_root_bound)
        # gap_solver = None if (best_obj is None or best_bound is None) else (best_obj - best_bound)
        # gap_rl     = None if (best_obj is None or rl_lb is None) else (best_obj - rl_lb)
        gap = None if (best_obj is None or rl_lb is None) else (best_obj - rl_lb)


        one = {
            "name": prob.name,
            "gurobi_obj": best_obj,         # 原问题求到的最好可行解（上界）
            "gurobi_bound": best_bound,     # 原问题 B&B / cuts 的 best bound（下界）
            "gurobi_root_bound": root_bound,
            "scip_root_bound": scip_root_bound, # 原问题 root relax 最优值（下界）
            "rl_lb": rl_lb,                 # RL 松弛后凸问题最优值（下界）
            "lb_improve": lb_improve,       # 下界提升
            "lb_improve_pct": lb_improve_pct,   # 下界提升(%) w.r.t gurobi root
            "lb_improve_scip": lb_improve_scip, # 下界提升 w.r.t SCIP root
            "lb_improve_scip_pct": lb_improve_scip_pct, # 下界提升(%) w.r.t SCIP root
            "gap": gap,                     # 用同一上界 best_obj 衡量 RL 下界 gap
            # "rl_lb_tight": rl_lb_tight,     # 全局 tightening 后 RL 松弛后凸问题最优值（下界）
            # "tight_gain": tight_gain,       # 全局 tightening 带来的下界提升
            # "gap_solver": gap_solver,       # 原问题求解 gap（参考）
            # "gap_rl": gap_rl,               # 用同一上界 best_obj 衡量 RL 下界 gap（可选）
        }
        results.append(one)

        import re

        def _safe_name(s: str) -> str:
            s = re.sub(r"\s+", "_", s)
            return re.sub(r"[^0-9a-zA-Z._\-+=]", "_", s)

        case_name = f"{idx:04d}__{_safe_name(prob.name)}"
        pack_path = os.path.join(SAVE_DIR, f"{case_name}.pkl")

        pack = {
            "name": prob.name,
            "prob_orig_raw": prob_orig_raw,           # 原问题（raw）
            # "prob_rl_tight_raw": prob_rl_tight_raw,   # tighten 后的松弛问题（raw）
            "result": one,                            # 该题的结果 dict
            # 可选：如果你也想存 tighten 前 RL 松弛后问题
            # "prob_rl_raw": prob_rl_raw,
        }

        # with open(pack_path, "wb") as f:
        #     pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f"  [SAVE] packed case -> {pack_path}")


        print(
            f"  [Result] gurobi_obj={best_obj}, gurobi_root_bound={root_bound}, "
            f"scip_root_bound={scip_root_bound}, rl_lb={rl_lb}, "
            f"lb_improve={lb_improve}, lb_improve_scip={lb_improve_scip}, gap={gap}"
        )

        if (idx + 1) % flush_every == 0:
            _flush_results()
            print(f"  [SAVE] flushed results -> {out_path}")

    # 4) 保存结果
    _flush_results()

    print(f"\n[INFO] Saved results to {out_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
