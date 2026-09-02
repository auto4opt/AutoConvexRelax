# -*- coding: utf-8 -*-
"""
Plain inference for QCQP relaxation (aligned with train.py):
- No extra mask in inference (all masks are inside AgentLayer)
- No reward
- Just call model -> (action_id, action_loc_seq_idx) -> map to term_id -> apply RelaxationEngine action
"""

from __future__ import annotations

import os
import sys
import json
import time
import copy
import random
import pickle
import argparse
import glob
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import sympy as sp
import numpy as np

# ==== keep consistent with your repo ====
from autoconvexrelax.model.agent import Agent
from autoconvexrelax.core.relaxation import RelaxationEngine


# =========================
# Action space (keep aligned with training)
# =========================
ACTION_TYPE = {
    0: "relax_integrality",
    1: "remove_fraction",
    2: "mccormick_relaxation",
    3: "sdp_relaxation",
    4: "alphaBB",
    5: "bound_tightening",
    6: "global_cut_generation",
}


# =========================
# Model wrapper (MUST match train.py)
# =========================
class PolicyNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_embed: int,
        n_head: int,
        ffn_hidden: int,
        drop_prob: float,
        n_actions: int,
        device: str,
    ):
        super().__init__()
        self.device = device
        self.agent = Agent(
            d_model=d_model,
            d_embed=d_embed,
            d_actions=n_actions,
            n_head=n_head,
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            device=device,
        )

    @torch.no_grad()
    def forward(self, state_info: Dict[str, Any], mode: str = "eval"):
        """
        state_info must be aligned with train.py:
          {
            "seq": Tensor,
            "problem_type": str,
            "non_convex_vtypes": Tensor (optional but your AgentLayer uses it)
          }
        returns: a_id, a_loc_seq_idx(1-based, 0 means GLOBAL), logp, entropy
        """
        seq = state_info["seq"]
        problem_type = state_info["problem_type"]
        non_convex_vtypes = state_info.get("non_convex_vtypes")

        (a_id, _a_id_probs,
         a_loc, _a_loc_probs,
         lp_id_all, lp_loc_all,
         lp_id, lp_loc) = self.agent(
            seq,
            mode=mode,
            problem_type=problem_type,
            non_convex_vtypes=non_convex_vtypes,
        )

        logp = lp_id + lp_loc
        entropy = (
            torch.distributions.Categorical(logits=lp_id_all).entropy()
            + torch.distributions.Categorical(logits=lp_loc_all).entropy()
        ).sum()

        return a_id, a_loc, logp, entropy


def action_changed(last_rewrite) -> bool:
    """Best-effort: judge whether an action actually changed something."""
    if not last_rewrite:
        return False
    old = last_rewrite.get("old", None)
    new = last_rewrite.get("new", None)
    if old is None or new is None:
        return False
    return str(old) != str(new)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Checkpoint loading (keep your surgery loader)
# =========================
def load_checkpoint_with_surgery(model: torch.nn.Module, ckpt_path: str, device: str):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # extract real state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
        print("[INFO] Found 'model_state_dict'.")
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        print("[INFO] Found 'state_dict'.")
    else:
        print("[INFO] Assume raw state_dict.")

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())

    # LayerNorm gamma/beta <-> weight/bias remap (optional compatibility)
    model_uses_gamma_beta = any(k.endswith(".gamma") or k.endswith(".beta") for k in model_keys)
    ckpt_uses_gamma_beta = any(k.endswith(".gamma") or k.endswith(".beta") for k in ckpt_keys)

    if (not ckpt_uses_gamma_beta) and model_uses_gamma_beta:
        remapped = {}
        for k, v in state.items():
            if k.endswith(".weight") and (k[:-7] + ".gamma") in model_keys:
                remapped[k[:-7] + ".gamma"] = v
            elif k.endswith(".bias") and (k[:-5] + ".beta") in model_keys:
                remapped[k[:-5] + ".beta"] = v
            else:
                remapped[k] = v
        state = remapped
        print("[INFO] Remapped LayerNorm: weight/bias -> gamma/beta")
    elif ckpt_uses_gamma_beta and (not model_uses_gamma_beta):
        remapped = {}
        for k, v in state.items():
            if k.endswith(".gamma") and (k[:-6] + ".weight") in model_keys:
                remapped[k[:-6] + ".weight"] = v
            elif k.endswith(".beta") and (k[:-5] + ".bias") in model_keys:
                remapped[k[:-5] + ".bias"] = v
            else:
                remapped[k] = v
        state = remapped
        print("[INFO] Remapped LayerNorm: gamma/beta -> weight/bias")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[INFO] Missing keys:")
        for k in missing:
            print("  -", k)
    if unexpected:
        print("[INFO] Unexpected keys:")
        for k in unexpected:
            print("  -", k)
    print("[INFO] Checkpoint loaded.")


# =========================
# Apply one action (aligned with train.py's engine.apply_action signature)
# =========================
def apply_one_action(problem, action_id: int, term_id: int):
    """
    term_id:
      - 0 means GLOBAL action (bound_tightening / global_cut_generation)
      - otherwise 1-based term_id used by problem.get_term_by_id(term_id)
    """
    new_problem = problem.copy()
    engine = RelaxationEngine()

    a_name = ACTION_TYPE.get(int(action_id), "Unknown")

    if term_id == 0:
        sub_expr = sp.Integer(0)
        location = "GLOBAL"
    else:
        sub_expr, location = new_problem.get_term_by_id(term_id)
        if sub_expr is None:
            raise RuntimeError(f"Invalid term_id={term_id}, mapping failed.")

    last_rewrite = engine.apply_action(
        problem=new_problem,
        sub_expr=sub_expr,
        location=location,
        action_type=a_name,
    )
    return new_problem, last_rewrite


# =========================
# Data loading helpers
# =========================
def load_problem_groups(pkl_path: str) -> List[List[Any]]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # [[prob...]] or [[g1...],[g2...]] or [prob...]
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
        return obj
    if isinstance(obj, list):
        return [obj]
    raise ValueError(f"Unrecognized pickle format: {type(obj)}")


def sample_problems(groups: List[List[Any]], num: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    flat: List[Any] = []
    for g in groups:
        flat.extend(g)
    if num <= 0 or num >= len(flat):
        return flat
    rng.shuffle(flat)
    return flat[:num]


def sample_items(items: List[Any], num: int, seed: int) -> List[Any]:
    if not items:
        return []
    rng = random.Random(seed)
    if num <= 0 or num >= len(items):
        return list(items)
    items = list(items)
    rng.shuffle(items)
    return items[:num]


def load_split_indices(split_path: str) -> Dict[str, Any]:
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
        raise RuntimeError(
            "Multiple split files found in train_logs; please pass --split_json explicitly."
        )
    return ""


def collect_split_problems(
    groups: List[List[Any]],
    split_data: Dict[str, Any],
    use: str = "infer",
) -> List[Any]:
    split_groups = split_data.get("groups", [])
    if len(split_groups) != len(groups):
        raise ValueError(
            f"split groups={len(split_groups)} mismatch data groups={len(groups)}"
        )
    key = "infer_indices" if use == "infer" else "train_indices"
    selected: List[Any] = []
    for gi, (probs, ginfo) in enumerate(zip(groups, split_groups)):
        idxs = ginfo.get(key, [])
        if not isinstance(idxs, list):
            raise ValueError(f"group {gi} {key} is not a list")
        for idx in idxs:
            if not isinstance(idx, int):
                continue
            if 0 <= idx < len(probs):
                selected.append(probs[idx])
    return selected


# =========================
# Plain inference loop
# =========================
def relax_problem_plain(
    model: PolicyNet,
    problem,
    max_steps: int,
    noop_patience: int = 3,
    tighten_budget: int = 3,
    verbose: bool = True,
):
    """
    Loop:
      - state_info = agent.state_representation(problem) -> dict containing seq/non_convex_indices/non_convex_vtypes
      - model -> (a_id, a_loc_seq_idx) where a_loc_seq_idx is 0(GLOBAL) or 1..k
      - map to term_id (0 or 1-based) and apply action
    """
    action_trace = []
    noops = 0
    tighten_left = int(tighten_budget)
    convex_reached = False
    cur = copy.deepcopy(problem)

    for step in range(max_steps):
        cur.map_all_terms()

        is_convex = bool(cur.is_convex()) if hasattr(cur, "is_convex") else False
        if is_convex and (not convex_reached):
            convex_reached = True

        s_info = model.agent.state_representation(cur)
        seq = s_info["seq"]
        non_convex_indices = s_info.get("non_convex_indices")
        non_convex_vtypes = s_info.get("non_convex_vtypes")

        has_nonconvex = (non_convex_indices is not None) and (non_convex_indices.numel() > 0)

        # Phase A: convexification (must have non-convex term tokens)
        # Phase B: post-convex global actions (no term tokens, limited by tighten_budget)
        if (not convex_reached) and (not has_nonconvex):
            if verbose:
                print(f"  [DONE] no non-convex terms found at step={step}")
            break

        if convex_reached and (not has_nonconvex):
            if tighten_left <= 0:
                if verbose:
                    print(f"  [DONE] post-convex global budget exhausted at step={step}")
                break

        forward_info = {
            "seq": seq,
            "problem_type": cur.problem_type,
            "non_convex_vtypes": non_convex_vtypes,
        }

        a_id, a_loc_seq_idx, _, _ = model(forward_info, mode="eval")

        aid = int(a_id.item()) if hasattr(a_id, "item") else int(a_id)
        loc_seq = int(a_loc_seq_idx.item()) if hasattr(a_loc_seq_idx, "item") else int(a_loc_seq_idx)

        # loc_seq == 0 means GLOBAL
        if convex_reached and (not has_nonconvex):
            # Post-convex stage: force GLOBAL regardless of loc_seq (AgentLayer should already enforce this).
            term_id = 0
            loc_seq = 0
        else:
            if loc_seq == 0:
                term_id = 0
            else:
                local_idx = loc_seq - 1  # 0-based inside non_convex_indices
                if local_idx < 0:
                    local_idx = 0
                if local_idx >= non_convex_indices.numel():
                    local_idx = non_convex_indices.numel() - 1

                original_idx = non_convex_indices[local_idx]
                original_idx_int = int(original_idx.item()) if hasattr(original_idx, "item") else int(original_idx)

                # IMPORTANT: convert global 0-based -> term_id (1-based)
                term_id = original_idx_int + 1

        action_name = ACTION_TYPE.get(aid, f"unknown_{aid}")
        if verbose:
            print(f"  [step {step}] action={aid}:{action_name}  loc_seq={loc_seq} -> term_id={term_id}")

        nxt, last_rewrite = apply_one_action(cur, aid, term_id)
        changed = action_changed(last_rewrite)

        action_trace.append(
            {
                "step": step,
                "action_id": aid,
                "action_name": action_name,
                "loc_seq": loc_seq,
                "term_id": term_id,
                "changed": bool(changed),
                "last_rewrite": last_rewrite,
            }
        )

        if convex_reached and term_id == 0:
            # Post-convex global phase: each attempt consumes budget; stop on first no-op.
            tighten_left -= 1
            if not changed:
                if verbose:
                    print("  [STOP] global action no-op in post-convex phase.")
                cur = nxt
                break
            # reset noop counter when we are making progress
            noops = 0
        else:
            if not changed:
                noops += 1
                if noops >= noop_patience:
                    if verbose:
                        print(f"  [STOP] {noop_patience} consecutive no-op actions.")
                    cur = nxt
                    break
            else:
                noops = 0

        cur = nxt

    return cur, action_trace


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/latest.pt")
    parser.add_argument("--data", type=str, default="outputs/data/vector_all_mix_1600.pkl")
    parser.add_argument("--num", type=int, default=10, help="sampled problems")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_json", type=str, default="", help="path to *_split_indices.json")
    parser.add_argument("--split_use", type=str, default="infer", choices=["infer", "train"])
    parser.add_argument("--no_split", action="store_true", help="ignore split and use full dataset")
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[INFO] seed = {args.seed}")

    device = args.device
    if device == "cuda" and (not torch.cuda.is_available()):
        device = "cpu"
    print(f"[INFO] device = {device}")

    if not os.path.exists(args.data):
        print(f"[ERROR] data not found: {args.data}")
        sys.exit(1)
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] ckpt not found: {args.ckpt}")
        sys.exit(1)

    groups = load_problem_groups(args.data)
    if args.no_split:
        problems = sample_problems(groups, args.num, args.seed)
        print(f"[INFO] loaded groups={len(groups)}, sampled problems={len(problems)} (no split)")
    else:
        split_path = resolve_split_json(args.split_json)
        if not split_path or (not os.path.exists(split_path)):
            print("[ERROR] split file not found; pass --split_json or use --no_split.")
            sys.exit(1)
        split_data = load_split_indices(split_path)
        split_problems = collect_split_problems(groups, split_data, use=args.split_use)
        if not split_problems:
            print("[ERROR] split problems empty; check split file.")
            sys.exit(1)
        problems = sample_items(split_problems, args.num, args.seed)
        print(
            f"[INFO] loaded groups={len(groups)}, split={args.split_use}, "
            f"split_problems={len(split_problems)}, sampled problems={len(problems)}"
        )

    n_actions = len(ACTION_TYPE)

    # MUST match train.py hyperparams for checkpoint_400
    model = PolicyNet(
        d_model=256,
        d_embed=256,
        n_head=8,
        ffn_hidden=512,
        drop_prob=0.1,
        n_actions=n_actions,
        device=device,
    ).to(device)

    # Dummy forward init (build GNN etc.) exactly like train
    print("[INFO] Dummy forward to initialize Agent/GNN...")
    dummy_prob = problems[0]
    with torch.no_grad():
        _ = model.agent.state_representation(dummy_prob)
    print("[INFO] Initialized.")

    load_checkpoint_with_surgery(model, args.ckpt, device=device)
    model.eval()

    all_results = []
    t0 = time.time()

    for i, prob in enumerate(problems, 1):
        print(f"\n=== Problem {i}/{len(problems)}: {getattr(prob, 'name', 'NONAME')} ===")
        print(f"  Original problem: {prob}")
        prob.display_mappings()
        relaxed, trace = relax_problem_plain(
            model=model,
            problem=prob,
            max_steps=args.max_steps,
            noop_patience=3,
            verbose=(not args.quiet),
        )
        is_convex = bool(relaxed.is_convex()) if hasattr(relaxed, "is_convex") else None
        all_results.append(
            {
                "name": getattr(relaxed, "name", f"prob_{i}"),
                "is_convex": is_convex,
                "steps": len(trace),
                "trace": trace,
            }
        )
        print(f"  Final relaxed problem: {relaxed}")
        print(f"[RESULT] is_convex={is_convex}, steps={len(trace)}")

    print(f"\n[INFO] done. total_time={time.time() - t0:.2f}s")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] wrote traces to {args.out_json}")


if __name__ == "__main__":
    main()
