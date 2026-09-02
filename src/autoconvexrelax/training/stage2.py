# -*- coding: utf-8 -*-
"""
PPO‑QCQP (baseline 版本)
--------------------------------------------------
*   仅策略网络 (无 value head)
*   Advantage = reward − moving baseline
*   Clip‑PPO 损失 + 可选熵正则
*   支持多环境并行 rollout
*   依赖于：QCQPProblem / RelaxationEngine / Agent 等与原工程一致的模块
"""

from __future__ import annotations

import copy
import math
import os
import random
import csv
import sys
import time
import pickle
from typing import List, Tuple
import json

from autoconvexrelax.evaluation.solvers.mosek import solve_convex_relax_mosek

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp

from autoconvexrelax.core.relaxation import RelaxationEngine

# RelaxationEngine toggles (overridable by runner/baseline)
RELAX_ENGINE_BT_WARMUP = True
RELAX_ENGINE_BT_BEFORE_GLOBAL_CUT = False

from autoconvexrelax.core.problem import *
from autoconvexrelax.reward import get_reward, problem_size_cost
from autoconvexrelax.model.agent import Agent    
import glob

import time
from collections import defaultdict, deque

import time
from collections import defaultdict

_TIMING = defaultdict(float)
_TCOUNT = defaultdict(int)

def _tadd(k, dt):
    _TIMING[k] += dt
    _TCOUNT[k] += 1

def _tavg(k):
    return _TIMING[k] / max(_TCOUNT[k], 1)

def timing_report(prefix="", reset=False):
    keys = sorted(_TIMING.keys(), key=lambda x: -_tavg(x))
    msg = prefix + " | " + ", ".join([f"{k}={_tavg(k):.4f}s" for k in keys])
    print(msg, flush=True)
    if reset:
        _TIMING.clear()
        _TCOUNT.clear()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _env_int_list(name: str, default: list[int]) -> list[int]:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return list(default)
    out = [int(x.strip()) for x in v.split(",") if x.strip()]
    return out if out else list(default)



RESUME      = _env_bool("APA_RESUME", True)                     # 关掉续训就设 False
TRAIN_SEED  = _env_int("APA_TRAIN_SEED", 42)
CKPT_FILE   = _env_str("APA_CKPT_FILE", "latest.pt")     # 手动指定 / 或者自动搜

# -------------------- 日志配置 -------------------- #
LOG_DETAIL_LEVEL = "batch"  # 可选: "step", "batch", "update"
SAVE_DETAILED_LOGS = True   # 是否保存详细日志

# ===== Finetune switches =====
FINETUNE = _env_bool("APA_FINETUNE", False)
LOAD_WEIGHTS_FROM = os.getenv("APA_LOAD_WEIGHTS_FROM") or None  # 从头开始训练
LOG_DIR       = _env_str("APA_LOG_DIR", "outputs/logs/train/stage2")
CKPT_SAVE_DIR = _env_str("APA_CKPT_SAVE_DIR", os.path.join(LOG_DIR, "checkpoints"))
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
FINETUNE_FILE = _env_str("APA_FINETUNE_FILE", "outputs/data/vector_finetune_HYBRID_MIX_1200.pkl")
# 可选：在微调时先冻结 encoder N 个 update，再解冻
FREEZE_ENCODER_UPDATES = 0       # 例如 200；不想冻结就设 0
# ENCODER_LR_SCALE = 0.05           # encoder 学习率缩放（相对 LR）
ENCODER_LR_SCALE = 1.0

# Anchor regularization
# LAMBDA_ANCHOR_ID  = 0.03   # 建议 0.02 ~ 0.05 # 0.03
# LAMBDA_ANCHOR_LOC = 0.01   # 可选：约束位置分布，0~0.02 之间 # 0.01

LAMBDA_ANCHOR_ID  = 0.0 # 微调expand临时关闭
LAMBDA_ANCHOR_LOC = 0.0

# -------------------- 常量 & 超参 -------------------- #
# PROB_DIR      = os.path.join(LOG_DIR, "problems")

ENTROPY_COEF  = 0.1        # 设 0 即关闭探索正则
CLIP_COEF     = 0.15
LR            = 1e-4
BATCH_SIZE    = 128
EPOCHS_PPO    = _env_int("APA_EPOCHS_PPO", 1)
VALUE_LOSS_COEF = _env_float("APA_VALUE_LOSS_COEF", 0.5)
VALUE_CLIP_COEF = _env_float("APA_VALUE_CLIP_COEF", 0.2)
# NUM_STEPS     = 128         # 采样步长 (per env per update)
PROBLEMS_PER_UPDATE = _env_int("APA_PROBLEMS_PER_UPDATE", 32)  # 【新增】每轮 PPO Update 要采样多少个 *不同* 的问题 (推荐 32 或 64)
ROLLOUT_STEPS_PER_PROBLEM = _env_int("APA_ROLLOUT_STEPS_PER_PROBLEM", 16) # 【新增】在每个问题上采样多少步 (推荐 8 或 16)

# ---------- Exp‑moving baseline ---------- #
BASELINE : torch.Tensor | None = None  # 按需在 ppo_update 内更新
BETA_BLS = 0.2                         # 更大 → baseline 跟随更快


### PATCH START: curriculum 配置
# 例如 6 组：前两组各 150 轮，中等 300/500，最后两组 800/1200
GROUP_UPDATES = _env_int_list("APA_GROUP_UPDATES", [10])
TOTAL_UPDATES = sum(GROUP_UPDATES)

import bisect
cum_updates = np.cumsum(GROUP_UPDATES).tolist()




# ===== stage-2 solver reward config =====
ROOT_LB_CACHE_PATH = _env_str("APA_ROOT_LB_CACHE_PATH", "outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json")
DATASET_TAG = _env_str("APA_DATASET_TAG", "HYBRID_MIX_1200")

# ===== dataset split config =====
TRAIN_SPLIT_RATIO = 0.8
SHUFFLE_DATASET = True
DATASET_SPLIT_SEED = _env_int("APA_DATASET_SPLIT_SEED", 42)
SAVE_SPLIT_INDICES = True
SPLIT_INDICES_PATH = ""
USE_EXISTING_SPLIT = True

SOLVER_REWARD_ENABLE = True
SOLVER_REWARD_PROB   = _env_float("APA_SOLVER_REWARD_PROB", 1.0)      # 0.2: 只在 20% episode done 时算一次
SOLVER_REWARD_LAMBDA = 6.0      # solver reward 权重（加大相对提升信号）
SOLVER_REWARD_SCALE  = 0.20     # 更敏感：20% 提升 -> tanh(1)
SOLVER_REWARD_COMPLEXITY_DECAY = _env_float("APA_SOLVER_REWARD_COMPLEXITY_DECAY", 1.0)
SOLVER_REWARD_SIZE_PENALTY = _env_float("APA_SOLVER_REWARD_SIZE_PENALTY", 0.25)
SOLVER_REWARD_SIZE_PREMIUM = _env_float("APA_SOLVER_REWARD_SIZE_PREMIUM", 0.08)
SOLVER_FAIL_PENALTY  = 1.0

GUROBI_ROOT_TL = 30.0           # 预计算 root bound 用
MOSEK_TL       = 10.0           # 训练时求凸问题用
LB_SCALE_FLOOR = 1.0

### PATCH END
# ACTION_TYPE = {
#     0: "expand", ##
#     1: "factor_merge", ##
    
#     2: "cancel", #
    
#     3: "expand_log", ##
#     4: "logcombine", ##
#     5: "remove_log", ##
    
#     6: "relax_integrality",
#     7: "remove_fraction",
#     8: "remove_abs", ##
    
#     9: "trace_transformation", #
    
#     10: "mccormick_relaxation",
#     11: "sdp_relaxation",
#     12: "first_order_taylor", ##
    
#     13: "spectral_psd_projection",
#     14: "diagonal_relaxation"
# }

ACTION_TYPE = {
    0: "relax_integrality",
    1: "remove_fraction",
    2: "mccormick_relaxation",
    3: "sdp_relaxation",
    4: "qcr",
    5: "bound_tightening",
    6: "global_cut_generation",
    # 7: "spectral_psd_projection", 
    # 8: "diagonal_relaxation",
}

from autoconvexrelax.evaluation.expressions import normalize_expr
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

def load_root_lb_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_root_lb_cache(path: str, cache: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

def get_root_lb_from_cache_only(cache: dict, key: str) -> float | None:
    """Cache-only: never calls Gurobi. Returns None if missing or cached as null."""
    if key not in cache:
        return None
    v = cache.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _truncate_csv_to_update(csv_path: str, max_update: int):
    """
    Keep header + rows whose first column(update) <= max_update.
    Prevent out-of-order rows when resuming from an earlier checkpoint.
    """
    if not os.path.isfile(csv_path):
        return
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return

        kept = [rows[0]]
        dropped = 0
        for row in rows[1:]:
            if not row:
                continue
            try:
                upd = int(float(row[0]))
            except Exception:
                kept.append(row)
                continue
            if upd <= int(max_update):
                kept.append(row)
            else:
                dropped += 1

        if dropped > 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(kept)
            print(f"[Resume] Truncated {csv_path}: dropped {dropped} rows > update {max_update}")
    except Exception as e:
        print(f"[Resume] WARN: failed to truncate {csv_path}: {e}")


def read_last_line(csv_file: str) -> list[str] | None:
    "快速读 csv 最后一行（windows/Unix 均可）"
    try:
        with open(csv_file, "rb") as f:
            # 从文件尾部反向找换行
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            return f.readline().decode().strip().split(",")
    except (FileNotFoundError, OSError):
        return None
    
def make_problem_key(dataset_tag: str, idx: int, prob) -> str:
    name = getattr(prob, "name", "noname")
    # 处理一下分隔符，避免 json key 难看
    name = str(name).replace(":", "_").replace("/", "_")
    return f"{dataset_tag}:{idx}:{name}"
# =====================================================
#                    编码  &  动作执行
# =====================================================
def entropy_coef_schedule(step, total_updates=TOTAL_UPDATES,
                          start=0.03, mid=0.02, end=0.005, hold_ratio=0.3):
    # 前 hold_ratio 段保持 start，不变
    hold_steps = int(total_updates * hold_ratio)
    if step <= hold_steps:
        return start
    # 之后线性衰减到 end
    t = (step - hold_steps) / max(total_updates - hold_steps, 1)
    return mid + (end - mid) * t

# def input(problem: QCQPProblem):
#     """SymPy -> LaTeX -> 嵌入，一次性批量化加速"""
#     exprs = [problem.obj_expr] + [c.expr for c in problem.constraints] + problem.get_all_items()
#     latex_list = latex_batch(exprs)
#     embeddings = embed_latex_batch(latex_list)

#     seq_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32)
#     m = len(problem.constraints)
#     from_constraints = list(map(int, problem.get_from_constraints()))
#     # print("problem is ", problem)
#     # print("from_constraints is ", from_constraints)
#     # print("convexity_flags is ", problem.get_convexity_flags())
#     return seq_tensor, m, from_constraints


def relaxation(problem: QCQPProblem, action_id, action_loc):
    new_problem = problem.copy()
    engine = RelaxationEngine()
    engine.enable_bt_warmup = RELAX_ENGINE_BT_WARMUP
    engine.enable_bt_before_global_cut = RELAX_ENGINE_BT_BEFORE_GLOBAL_CUT
    
    # action_loc 现在是 term_id，可以是 tensor 也可以是 int
    if hasattr(action_loc, "item"):
        term_id = int(action_loc.item())
    else:
        term_id = int(action_loc)
        
    # action_id 同理，既支持 tensor 也支持 int
    if hasattr(action_id, "item"):
        a_id_int = int(action_id.item())
    else:
        a_id_int = int(action_id)
    
    print(f"Action: {ACTION_TYPE[a_id_int]} @ term_id={term_id}")

    if term_id == 0:
        # 全局动作：不给具体项也可以（engine.apply_action 对 bound_tightening/global_cut 应当不依赖 sub_expr）
        import sympy as sp
        sub_expr = sp.Integer(0)
        location = "GLOBAL"
    else:
        sub_expr, location = new_problem.get_term_by_id(term_id)
        
        # # DEBUG PRINT
        # print(new_problem)
        # new_problem.display_mappings()
        # print(f"Located sub_expr: {sub_expr} at {location}")
        
        # 防御：避免 None
        if sub_expr is None:
            raise RuntimeError(f"Invalid term_id={term_id}, mapping failed.")

    last_rewrite = engine.apply_action(
        problem=new_problem,
        sub_expr=sub_expr,
        location=location,
        action_type=ACTION_TYPE[a_id_int],   # 这里才是字符串
    )
    return new_problem, last_rewrite


def get_performance(problem_before: QCQPProblem, problem_after: QCQPProblem, a_id: int = -1, last_rewrite=None):
    return get_reward(problem_before, problem_after, a_id, last_rewrite)

# =====================================================
#                 策略网络 (新版本)
# =====================================================
class PolicyNet(nn.Module):
    def __init__(self, d_model: int, d_embed: int, n_head: int,
                 ffn_hidden: int, drop_prob: float, n_actions: int, device: str):
        super().__init__()
        self.device = device
        self.agent = Agent(d_model=d_model, d_embed=d_embed, d_actions=n_actions,
                           n_head=n_head, ffn_hidden=ffn_hidden,
                           drop_prob=drop_prob, device=device)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def _predict_value(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, 1+k, d_model], use global token at position 0.
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        global_token = seq[:, 0, :]
        return self.value_head(global_token).squeeze(-1)

    def forward(self, state_info: dict, mode="train"):
        # obs 现在是一个包含了所有信息的字典
        seq = state_info["seq"]
        problem_type = state_info["problem_type"]
        non_convex_vtypes = state_info.get("non_convex_vtypes") # 使用.get以保持向后兼容
        value_pred = self._predict_value(seq)

        (a_id, a_id_probs,
         a_loc, a_loc_probs,
         lp_id_all, lp_loc_all,
         lp_id, lp_loc) = self.agent(
             seq, 
             mode=mode, 
             problem_type=problem_type,
             non_convex_vtypes=non_convex_vtypes # 传递给Agent
         )
        
        logp = lp_id + lp_loc
        
        # 使用 logits 计算熵更稳定
        entropy = (torch.distributions.Categorical(logits=lp_id_all).entropy() +
                   torch.distributions.Categorical(logits=lp_loc_all).entropy()).sum()
                   
        return a_id, a_loc, logp, entropy, value_pred

    # 评估 ~ get log‑prob & entropy for 旧动作
    def evaluate(self, state_info: dict, a_id: torch.Tensor, a_loc_orig_idx: torch.Tensor):
        seq = state_info["seq"]
        non_convex_indices = state_info["non_convex_indices"]
        problem_type = state_info["problem_type"]
        non_convex_vtypes = state_info.get("non_convex_vtypes")
        value_pred = self._predict_value(seq)

        # a_loc_orig_idx 是 0-based 的“全局索引”，需要映射回当前 seq 的“局部索引”
        a_loc_orig_idx_exp = a_loc_orig_idx.unsqueeze(1)  # [B, 1]
        matches = (non_convex_indices == a_loc_orig_idx_exp).nonzero(as_tuple=False)

        # 默认 0（无效）；匹配到的 +1 => 1-based 的序列索引
        a_loc_seq_idx = torch.zeros_like(a_loc_orig_idx)
        if matches.size(0) > 0:
            a_loc_seq_idx[matches[:, 0]] = matches[:, 1] + 1  # 转 1-based

        # -1 表示全局动作：天然映射到 seq_idx=0，不应报警
        is_global = (a_loc_orig_idx == -1)

        if matches.size(0) != a_loc_orig_idx.size(0):
            missing = a_loc_orig_idx.size(0) - matches.size(0)
            # 若缺失全部来自 global sentinel，则不报警
            if not (is_global.sum().item() == missing):
                print(f"Warning: {missing} actions could not be mapped back in evaluate!")


        # 通过 agent 得到 logits（可能是 [N] 或 [1,N]）
        (_, _, _, _, logp_id_all, logp_loc_all, _, _) = \
            self.agent(
                seq,
                mode="eval",
                problem_type=problem_type,
                non_convex_vtypes=non_convex_vtypes # 传递给Agent
            )

        # ---- 关键：统一张量形状 ----
        # 确保都有 batch 维
        if logp_id_all.dim() == 1:
            logp_id_all = logp_id_all.unsqueeze(0)         # [1, n_actions]
        if logp_loc_all.dim() == 1:
            logp_loc_all = logp_loc_all.unsqueeze(0)       # [1, 1+k]

        B = logp_id_all.shape[0]

        # a_id / a_loc_seq_idx 统一为 [B, 1] 的 long
        # 传入时可能是标量、[1] 或 [B]
        a_id = a_id.view(-1)[:B].view(B, 1).long()
        a_loc_seq_idx = a_loc_seq_idx.view(-1)[:B].view(B, 1).long()

        # 范围保护（避免偶发越界）
        a_id = torch.clamp(a_id, 0, logp_id_all.shape[1] - 1)
        a_loc_seq_idx = torch.clamp(a_loc_seq_idx, 0, logp_loc_all.shape[1] - 1)

        # 按索引取对应 logit（仍在 logits 空间，数值更稳定）
        lp_id = torch.gather(logp_id_all, dim=-1, index=a_id).squeeze(-1)     # [B]
        lp_loc = torch.gather(logp_loc_all, dim=-1, index=a_loc_seq_idx).squeeze(-1)  # [B]
        logp = lp_id + lp_loc

        # 熵：对 batch 取 mean 即可
        ent_id  = torch.distributions.Categorical(logits=logp_id_all).entropy()
        ent_loc = torch.distributions.Categorical(logits=logp_loc_all).entropy()
        entropy = (ent_id + ent_loc).mean()

        return logp, entropy, value_pred
     
# =====================================================
#                       环境封装
# =====================================================
class QCQPEnv:
    def __init__(self, init_problem: QCQPProblem, max_steps: int = 20, root_lb_cache: dict | None = None, problem_key: str | None = None):
        self.init_problem = init_problem
        self.max_steps = max_steps
        self.problem_type = init_problem.problem_type  # 固定下来
        self.root_lb_cache = root_lb_cache if root_lb_cache is not None else {}
        self.problem_key = problem_key  # 新增
        self.root_lb = None
        self.reset()

    def reset(self):
        self.cur_problem = copy.deepcopy(self.init_problem)
        self.t = 0
        # cache last action info for debugging
        self.last_action_id = None
        self.last_term_id = None
        self.last_rewrite = None
        # 不返回 problem_type 了
        
        # ==== 新增：tightening 预算 ====
        self.convex_reached = self.cur_problem.is_convex()
        self.tighten_left = random.randint(2, 4) if self.convex_reached else 0
        
        # 每个 episode 开始时拿到 baseline root_lb（缺失则补算一次）
        if SOLVER_REWARD_ENABLE:
            # 注意：这里改为用 problem_key
            self.root_lb = get_root_lb_from_cache_only(self.root_lb_cache, self.problem_key) if self.problem_key is not None else None
        return
    
    def _solver_reward(self, relaxed_prob: QCQPProblem):
        """
        返回 (r_add, info_dict)
        """
        if (not SOLVER_REWARD_ENABLE) or (self.root_lb is None):
            return 0.0, {}

        if random.random() > SOLVER_REWARD_PROB:
            return 0.0, {"solver_skipped": True}

        p = canonicalize_problem(copy.deepcopy(relaxed_prob))
        try:
            rl_lb = solve_convex_relax_mosek(p, time_limit=MOSEK_TL, verbose=False)  # returns float|None
        except Exception as e:
            # log-only debug (no pickle)
            try:
                print(
                    f"[MOSEK-FAIL] problem={getattr(p,'name','NONAME')} type={getattr(p,'problem_type',None)} "
                    f"vars={len(getattr(p,'variables',{}))} mvars={len(getattr(p,'matrix_variables',{}))} "
                    f"cons={len(getattr(p,'constraints',[]))} psd={len(getattr(p,'psd_constraints',[]))}"
                )
                print(f"[MOSEK-FAIL] last_action_id={self.last_action_id} last_term_id={self.last_term_id}")
                lr = self.last_rewrite if self.last_rewrite is not None else getattr(p, "last_rewrite", None)
                if lr:
                    print(f"[MOSEK-FAIL] last_rewrite={lr}")
            except Exception as dump_e:
                print(f"[MOSEK-FAIL] log failed: {dump_e}")
            raise e
        if rl_lb is None:
            return -SOLVER_FAIL_PENALTY, {"rl_lb": None, "solver_fail": True}

        root_lb = float(self.root_lb)
        denom = max(abs(root_lb), LB_SCALE_FLOOR)
        pct = (float(rl_lb) - root_lb) / denom

        init_size_cost = max(problem_size_cost(self.init_problem), 1.0)
        relaxed_size_cost = problem_size_cost(p)
        size_cost_growth = max(0.0, relaxed_size_cost - init_size_cost)
        size_growth_ratio = size_cost_growth / init_size_cost

        required_lb_premium = SOLVER_REWARD_SIZE_PREMIUM * math.tanh(size_growth_ratio)
        cost_adjusted_pct = pct - required_lb_premium

        lb_shaped = math.tanh(pct / SOLVER_REWARD_SCALE)
        cost_adjusted_lb_shaped = math.tanh(cost_adjusted_pct / SOLVER_REWARD_SCALE)
        complexity_discount = math.exp(-SOLVER_REWARD_COMPLEXITY_DECAY * size_growth_ratio)
        efficient_lb_shaped = lb_shaped
        size_penalty = SOLVER_REWARD_SIZE_PENALTY * math.tanh(size_growth_ratio)
        r_add = SOLVER_REWARD_LAMBDA * lb_shaped

        info = {
            "root_lb": root_lb,
            "rl_lb": float(rl_lb),
            "lb_improve": float(rl_lb) - root_lb,
            "lb_improve_pct": pct,
            "solver_cost_adjusted_lb_pct": cost_adjusted_pct,
            "solver_required_lb_premium": required_lb_premium,
            "solver_lb_shaped": lb_shaped,
            "solver_cost_adjusted_lb_shaped": cost_adjusted_lb_shaped,
            "solver_eff_lb_shaped": efficient_lb_shaped,
            "solver_complexity_discount": complexity_discount,
            "solver_size_cost_init": init_size_cost,
            "solver_size_cost_relaxed": relaxed_size_cost,
            "solver_size_cost_growth": size_cost_growth,
            "solver_size_growth_ratio": size_growth_ratio,
            "solver_size_penalty": size_penalty,
            "solver_r_add": r_add,
        }
        return r_add, info

    def step(self, a_id: torch.Tensor, a_loc: torch.Tensor):
        # 记录 step 前是否已经凸（避免 reward+=2 重复加）
        was_convex = self.convex_reached

        # term_id（你传进来的是 term_id_to_act_on，0 表示全局）
        term_id = int(a_loc.item()) if hasattr(a_loc, "item") else int(a_loc)

        next_problem, last_rewrite = relaxation(self.cur_problem, a_id, a_loc)
        # cache last action info for debug
        try:
            self.last_action_id = int(a_id.item()) if hasattr(a_id, "item") else int(a_id)
        except Exception:
            self.last_action_id = None
        self.last_term_id = term_id
        self.last_rewrite = last_rewrite
        reward = float(get_performance(self.cur_problem, next_problem, int(a_id), last_rewrite))

        is_convex = next_problem.is_convex()

        # ==== 只在“首次变凸”时给 bonus，并开启 tightening 预算 ====
        if (not was_convex) and is_convex:
            reward += 2.0
            self.convex_reached = True
            # self.tighten_left = random.randint(2, 4)
            self.tighten_left = 4

        # ==== 凸后 tightening：只在执行全局动作时扣次数 ====
        if self.convex_reached and is_convex and was_convex and (term_id == 0):
            if self.tighten_left > 0:
                self.tighten_left -= 1

        self.cur_problem = next_problem
        self.t += 1

        # ==== 终止条件：到步数上限，或“凸且 tightening 次数用完” ====
        done = (self.t >= self.max_steps) or (self.convex_reached and is_convex and self.tighten_left <= 0)

        # 原来的“没凸就结束惩罚”保留
        if done and not is_convex:
            flags = next_problem.get_convexity_flags()
            if isinstance(flags, torch.Tensor):
                remain = int((~flags.bool()).sum().item())
            else:
                remain = sum(not f for f in flags)
            reward -= 0.1 * remain

        info = {"is_convex": is_convex, "tighten_left": self.tighten_left}
        
        # >>> 新增：episode 成功(变凸)时，加 solver-based reward <<<
        if done and is_convex:
            r_add, sinfo = self._solver_reward(next_problem)
            reward += r_add
            info.update(sinfo)
            
        return reward, done, info



# =====================================================
#               Rollout (采样) (新版本)
# =====================================================
def rollout(env: QCQPEnv, model: PolicyNet, num_steps: int, device: str,
            csv_writer, global_step: int = 0, save_every: int = 1000):
    storage = []
    ep_done = ep_succ = 0
    ep_ret = 0.0
    ep_len = 0
    ep_returns, ep_lengths = [], []

    env.reset()
    for _ in range(num_steps):
        with torch.no_grad():
            # --- >>> 1. 获取包含所有信息的字典 <<< ---
            # seq, non_convex_indices = model.agent.state_representation(env.cur_problem) # <-- 旧
            t0 = time.perf_counter()
            state_info_dict = model.agent.state_representation(env.cur_problem)
            _tadd("state_rep+build_graph", time.perf_counter() - t0)

            seq = state_info_dict["seq"]
            non_convex_indices = state_info_dict["non_convex_indices"]
            non_convex_vtypes = state_info_dict["non_convex_vtypes"]

        # # 如果没有非凸项，意味着问题已解决或无法继续，结束当前episode
        # if non_convex_indices.numel() == 0:
        #     if ep_len > 0: # 结算上一轮未结束的 trajectory
        #         ep_returns.append(ep_ret)
        #         ep_lengths.append(ep_len)
        #     ep_ret, ep_len = 0.0, 0
        #     ep_done += 1
        #     if env.cur_problem.is_convex(): ep_succ += 1
        #     env.reset()
        #     continue

        # --- >>> 2. 将整个字典传递给模型 <<< ---
        # 准备一个包含所有需要信息的字典
        forward_pass_info = {
            "seq": seq,
            "problem_type": env.problem_type,
            "non_convex_vtypes": non_convex_vtypes
        }
        # a_id, a_loc_seq_idx, logp, entropy = model(seq, mode="train", problem_type=env.problem_type) # <-- 旧
        t0 = time.perf_counter()
        a_id, a_loc_seq_idx, logp, entropy, value_pred = model(forward_pass_info, mode="train")
        _tadd("policy_forward", time.perf_counter() - t0)


        # 3. 将模型输出的“局部序列索引”映射回“全局term_id”
        if int(a_loc_seq_idx.item()) == 0:
            # problem token：用 term_id=0 表示“全局动作”
            original_idx = torch.tensor(-1)      # 0-based global idx 的哨兵值（用于 PPO evaluate 映射）
            term_id_to_act_on = 0               # 传给 env.step / relaxation
        else:
            action_local_idx = int(a_loc_seq_idx.item()) - 1
            original_idx = non_convex_indices[action_local_idx]
            term_id_to_act_on = int(original_idx.item()) + 1

        
        # DEBUG PRINT 更新
        DEBUG = False
        if DEBUG:
            print("\n=== DEBUG ===")
            print("映射结果 (id → term @ loc)")
            env.cur_problem.display_mappings()
            print("Non-convex indices (0-based global):", non_convex_indices.tolist())
            
            # 从缓存中获取对应非凸项的 logits
            logits_row = model.agent.agent.last_action_logits[0].cpu() # shape [1+k]
            # 只取出非凸项对应的 logits (位置 1 到 k)
            non_convex_logits = logits_row[1:] 
            print("Action logits for non-convex terms:", non_convex_logits.numpy())
            print(f"Agent chose seq_idx={a_loc_seq_idx.item()} -> local_idx={action_local_idx} -> original_idx={original_idx.item()} -> term_id={term_id_to_act_on}")
            print("──────────────\n")

        old_problem = copy.deepcopy(env.cur_problem)
        # 4. 将正确的 term_id 传递给环境
        t0 = time.perf_counter()
        reward, done, info = env.step(a_id, torch.tensor(term_id_to_act_on))
        _tadd("env_step_total", time.perf_counter() - t0)

        is_convex = info.get("is_convex", False)
        
        # --- >>> 5. 存储 old_problem 和其他信息 <<< ---
        #    注意：我们不再需要存储完整的 state_info 字典，
        #    因为 ppo_update 会用 old_problem 重新生成它。
        storage.append((
            old_problem, # 【关键】存储修改前的 problem 对象
            a_id.detach().cpu(),
            original_idx.detach().cpu(),
            logp.detach(),
            entropy.detach(),
            reward,
            done,
            is_convex,
            value_pred.detach().cpu(),
        ))

        global_step += 1
        timing_report(prefix=f"[timing @ step {global_step}]", reset=True)
        if done:
            ep_done += 1
            if is_convex: ep_succ += 1
            ep_returns.append(ep_ret + reward)
            ep_lengths.append(ep_len + 1)
            ep_ret, ep_len = 0.0, 0
            env.reset()
        else:
            ep_ret += reward
            ep_len += 1

    if ep_len > 0:
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
    return storage, global_step, ep_done, ep_succ, ep_returns, ep_lengths



# =====================================================
#                  PPO update (新版本)
# =====================================================
def ppo_update(model: PolicyNet, optimizer: optim.Optimizer,
               storage: List[Tuple], device: str, clip_coef: float = 0.2,
               epochs: int = 6, batch_size: int = 32, entropy_coef: float = 0.02,
               anchor_model: PolicyNet = None, lambda_anchor_id: float = 0.0, lambda_anchor_loc: float = 0.0,
               update_idx: int = -1):
    global BASELINE

    # 1. 解包时，a_loc_s 现在是 a_loc_orig_idx_s
    (prob_s, a_id_s, a_loc_orig_idx_s, logp_old_s, ent_old_s, rew_s, done_s, _, value_old_s) = zip(*storage)
    old_logp_t = torch.stack(logp_old_s).to(device).view(-1)
    a_id_t = torch.stack(a_id_s).to(device)
    a_loc_orig_idx_t = torch.stack(a_loc_orig_idx_s).to(device)
    value_old_t = torch.stack(value_old_s).to(device).view(-1)
    
    # Advantage 计算部分不变
    gamma = 0.99
    returns = torch.zeros(len(rew_s), dtype=torch.float32, device=device)
    R = 0.
    for t in reversed(range(len(rew_s))):
        if done_s[t]: R = 0.
        R = rew_s[t] + gamma * R
        returns[t] = R
    with torch.no_grad():
        if BASELINE is None: BASELINE = returns.mean()
        else: BASELINE = (1 - BETA_BLS) * BASELINE + BETA_BLS * returns.mean()
    # Keep critic target scale stable across updates.
    ret_mean = returns.mean()
    ret_std = returns.std(unbiased=False).clamp_min(1e-6)
    returns_tgt = (returns - ret_mean) / ret_std

    advantages = returns_tgt - value_old_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    idxs = list(range(len(storage)))
    update_policy_losses, update_entropy_losses, update_value_losses, update_total_losses, update_clip_fractions = [], [], [], [], []
    nan_skip_batches = 0
    
    # === 新增：记录 KL 统计 ===
    update_kl_id, update_kl_loc = [], []
    
    for epoch in range(epochs):
        random.shuffle(idxs)
        for start in range(0, len(storage), batch_size):
            mb_idx = idxs[start:start + batch_size]
            logp_new_lst, entropy_lst, value_new_lst = [], [], []
            
            valid_indices_in_mb = []
            # 【新增】收集 KL 项
            kl_id_lst, kl_loc_lst = [], []

            for i, storage_idx in enumerate(mb_idx):
                # --- >>> 2. 用存储的 problem 重新生成包含所有信息的状态字典 <<< ---
                prob_i = prob_s[storage_idx]
                # seq_i, non_convex_indices_i = model.agent.state_representation(prob_i) # <-- 旧
                
                # 重新生成完整的 state_info 字典
                state_info_i_dict = model.agent.state_representation(prob_i)
                # 别忘了加上 problem_type，因为 evaluate 需要它
                state_info_i_dict["problem_type"] = prob_i.problem_type
                
                # # 安全检查
                # if state_info_i_dict["non_convex_indices"].numel() == 0:
                #     continue

                # --- >>> 3. 将整个字典传递给 evaluate 方法 <<< ---
                logp_i, ent_i, value_i = model.evaluate(
                    state_info_i_dict, # <-- 新
                    a_id_t[storage_idx].unsqueeze(0),
                    a_loc_orig_idx_t[storage_idx].unsqueeze(0)
                    # problem_type 参数已经包含在字典里，不再需要单独传
                )
                logp_new_lst.append(logp_i)
                entropy_lst.append(ent_i)
                value_new_lst.append(value_i)
                valid_indices_in_mb.append(i)
                
                
                # ===【新增】计算 new / anchor 的 logits 并做 KL ===
                seq_i  = state_info_i_dict["seq"]
                ncv_i  = state_info_i_dict.get("non_convex_vtypes")

                # 当前模型 logits
                (_, _, _, _, new_id_all, new_loc_all, _, _) = model.agent(
                    seq_i, mode="eval",
                    problem_type=prob_i.problem_type,
                    non_convex_vtypes=ncv_i
                )

                # 锚点模型 logits（冻结、无梯度）
                if anchor_model is not None and (lambda_anchor_id > 0 or lambda_anchor_loc > 0):
                    with torch.no_grad():
                        (_, _, _, _, anc_id_all, anc_loc_all, _, _) = anchor_model.agent(
                            seq_i, mode="eval",
                            problem_type=prob_i.problem_type,
                            non_convex_vtypes=ncv_i
                        )

                    # 统一 batch 维度
                    if new_id_all.dim() == 1:  new_id_all  = new_id_all.unsqueeze(0)
                    if anc_id_all.dim() == 1:  anc_id_all  = anc_id_all.unsqueeze(0)
                    if new_loc_all.dim() == 1: new_loc_all = new_loc_all.unsqueeze(0)
                    if anc_loc_all.dim() == 1: anc_loc_all = anc_loc_all.unsqueeze(0)

                    # KL(new || anchor) —— 用 logits 构造 Categorical 更稳定
                    p_new_id = torch.distributions.Categorical(logits=new_id_all)
                    p_old_id = torch.distributions.Categorical(logits=anc_id_all)
                    kl_id = torch.distributions.kl.kl_divergence(p_new_id, p_old_id).mean()
                    kl_id_lst.append(kl_id)

                    if lambda_anchor_loc > 0:
                        p_new_loc = torch.distributions.Categorical(logits=new_loc_all)
                        p_old_loc = torch.distributions.Categorical(logits=anc_loc_all)
                        kl_loc = torch.distributions.kl.kl_divergence(p_new_loc, p_old_loc).mean()
                        kl_loc_lst.append(kl_loc)

            if not logp_new_lst: continue
                
            logp_new = torch.cat(logp_new_lst)
            entropies = torch.stack(entropy_lst)
            values_new = torch.cat(value_new_lst).view(-1)
            
            mb_idx_tensor = torch.tensor(mb_idx, device=device)
            valid_mb_indices = mb_idx_tensor[valid_indices_in_mb]
            if valid_mb_indices.numel() == 0:
                continue

            ratio = torch.exp(logp_new - old_logp_t[valid_mb_indices])
            advantages_mb = advantages[valid_mb_indices]
            pg1 = advantages_mb * ratio
            pg2 = advantages_mb * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            policy_loss = -torch.min(pg1, pg2).mean()
            entropy_loss = -entropy_coef * entropies.mean()

            value_old_mb = value_old_t[valid_mb_indices]
            returns_mb = returns_tgt[valid_mb_indices]
            value_pred_clipped = value_old_mb + torch.clamp(values_new - value_old_mb, -VALUE_CLIP_COEF, VALUE_CLIP_COEF)
            value_losses = (values_new - returns_mb).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_mb).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # === 【新增】KL 正则 ===
            kl_id_reg  = (torch.stack(kl_id_lst).mean()  if kl_id_lst  else torch.tensor(0., device=device))
            kl_loc_reg = (torch.stack(kl_loc_lst).mean() if kl_loc_lst else torch.tensor(0., device=device))
            
            # （可选）夹紧，防止偶发爆炸
            MAX_KL = 10.0
            kl_id_reg  = torch.clamp(kl_id_reg,  max=MAX_KL)
            kl_loc_reg = torch.clamp(kl_loc_reg, max=MAX_KL)

            # 记录到本次 update 的统计
            update_kl_id.append(kl_id_reg.item())
            update_kl_loc.append(kl_loc_reg.item())

            loss = policy_loss + entropy_loss + VALUE_LOSS_COEF * value_loss \
                + lambda_anchor_id  * kl_id_reg \
                + lambda_anchor_loc * kl_loc_reg

            finite_checks = {
                "logp_new": logp_new,
                "entropies": entropies,
                "values_new": values_new,
                "advantages_mb": advantages_mb,
                "returns_mb": returns_mb,
                "ratio": ratio,
                "value_loss": value_loss,
                "loss": loss,
            }
            bad_name = None
            for name, tensor in finite_checks.items():
                if not torch.isfinite(tensor).all():
                    bad_name = name
                    break
            if bad_name is not None:
                nan_skip_batches += 1
                print(
                    f"[WARN] skip non-finite minibatch: update={update_idx} "
                    f"epoch={epoch + 1}/{epochs} mb_start={start} bad={bad_name}",
                    flush=True
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss.backward()
            _tadd("backward", time.perf_counter() - t0)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                nan_skip_batches += 1
                print(
                    f"[WARN] skip non-finite grad_norm: update={update_idx} "
                    f"epoch={epoch + 1}/{epochs} mb_start={start}",
                    flush=True
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            t0 = time.perf_counter()
            optimizer.step()
            _tadd("optim_step", time.perf_counter() - t0)
            
            update_policy_losses.append(policy_loss.item())
            update_entropy_losses.append(entropy_loss.item())
            update_value_losses.append(value_loss.item())
            update_total_losses.append(loss.item())
            clip_frac = ((ratio > 1 + clip_coef) | (ratio < 1 - clip_coef)).float().mean().item()
            update_clip_fractions.append(clip_frac)

    
    return {
        'policy_loss_mean': np.mean(update_policy_losses) if update_policy_losses else 0,
        'policy_loss_std': np.std(update_policy_losses) if update_policy_losses else 0,
        'entropy_loss_mean': np.mean(update_entropy_losses) if update_entropy_losses else 0,
        'entropy_loss_std': np.std(update_entropy_losses) if update_entropy_losses else 0,
        'value_loss_mean': np.mean(update_value_losses) if update_value_losses else 0,
        'value_loss_std': np.std(update_value_losses) if update_value_losses else 0,
        'total_loss_mean': np.mean(update_total_losses) if update_total_losses else 0,
        'total_loss_std': np.std(update_total_losses) if update_total_losses else 0,
        'clip_fraction_mean': np.mean(update_clip_fractions) if update_clip_fractions else 0,
        'clip_fraction_std': np.std(update_clip_fractions) if update_clip_fractions else 0,
        # === 新增：KL 统计 ===
        'kl_id_mean':  np.mean(update_kl_id) if update_kl_id else 0,
        'kl_loc_mean': np.mean(update_kl_loc) if update_kl_loc else 0,
        'nan_skip_batches': nan_skip_batches,
    }, BASELINE
    
# =====================================================
#                     构造示例问题
# =====================================================

def build_qcqp1():
    prob = QCQPProblem(name="nonconvex_qcqp")

    # 添加变量 x1, x2
    x1 = prob.add_variable("x1")
    x2 = prob.add_variable("x2")
    x3 = prob.add_variable("x3", vtype="binary", lb = 0, ub = 1)
    x4 = prob.add_variable("x4", lb = 0)
    x5 = prob.add_variable("x5")

    # 设置目标函数: -x1^2 - x2^2 + x1
    obj_expr = x1*x2 + sp.Abs(x3 * x5 - x4) - (x2 + x5)**2 + (-x1**2 + x4) / x2
    
    # obj_expr = (x1**2 + 1) / x4 + x3 * x5 + (x1**2 + x4) / x2 + (x1**2 + 1) / x5
    prob.set_objective(obj_expr)

    # 添加约束: x1^2 + x2^2 <= 1
    prob.add_constraint(expr=x4, rhs=0, sense=">=")
    prob.add_constraint(expr=x3, rhs=0, sense=">=")
    prob.add_constraint(expr=x3, rhs=1, sense="<=")
    prob.add_constraint(expr=x3, rhs='binary', sense="is")
    prob.map_all_terms()
    print(prob)
    return prob

def build_qcqp2():
    prob = QCQPProblem(name="nonconvex_qcqp")

    # 添加变量 x1, x2
    x1 = prob.add_variable("x1")
    x2 = prob.add_variable("x2")
    x3 = prob.add_variable("x3", vtype="binary", lb = 0, ub = 1)
    x4 = prob.add_variable("x4", lb = 0)
    x5 = prob.add_variable("x5")

    # 设置目标函数: -x1^2 - x2^2 + x1
    obj_expr = x1 * x2 + x3 * x5 - (x2 + x5)**2 + sp.Abs(- x3**2 - x1 * x4) + (x1**2 - x3**2) / (x5 + 2)
    # obj_expr = (x1**2 + x2**2) / (x4 + 1) + (x1**2 + x3**2) / (x5 + 2) - x1**2 / (x2 + 1)
    prob.set_objective(obj_expr)

    # 添加约束: x1^2 + x2^2 <= 1
    prob.add_constraint(expr=x4, rhs=0, sense=">")
    prob.add_constraint(expr=x3, rhs=0, sense=">=")
    prob.add_constraint(expr=x3, rhs=1, sense="<=")
    prob.add_constraint(expr=x3, rhs='binary', sense="is")
    prob.add_constraint(expr=x1**2 + x2**2, rhs=4, sense="<=")
    prob.add_constraint(expr=x3 * x4 - x1, rhs=1, sense="<=")
    prob.add_constraint(expr=sp.Abs(x1 - x5), rhs=2, sense="<=")
    prob.add_constraint(expr=x3 + x4 + x5, rhs=3, sense="==")
    prob.map_all_terms()
    print(prob)
    return prob

def build_qcqp3():
    prob = QCQPProblem(name="nonconvex_qcqp")

    # 添加变量 x1, x2
    x1 = prob.add_variable("x1")
    x2 = prob.add_variable("x2")
    x3 = prob.add_variable("x3", vtype="binary", lb=0, ub=1)
    x4 = prob.add_variable("x4")
    x5 = prob.add_variable("x5", lb = 0)

    # 设置目标函数: -x1^2 - x2^2 + x1
    obj_expr = x1 * x3 + x2 * x5 - (x1**2 + x2**2) + sp.Abs(x2 - x4**2) + (-x3 ** 2 + 2) / x5
    # obj_expr = (x3 ** 2 + 2) / x5 + (x2 ** 2 + 1) / x4 + x4 / (x5 ** 2 + 2)
    prob.set_objective(obj_expr)

    # 添加约束: x1^2 + x2^2 <= 1
    prob.add_constraint(expr=x5, rhs=0, sense=">")
    prob.add_constraint(expr=x3, rhs=0, sense=">=")
    prob.add_constraint(expr=x3, rhs=1, sense="<=")
    prob.add_constraint(expr=x3, rhs='binary', sense="is")
    prob.add_constraint(expr=(x1 + x2 + x3) ** 2, rhs=9, sense="<=")
    prob.add_constraint(expr=x1 * x5 - x4, rhs=2, sense="<=")
    prob.add_constraint(expr=sp.Abs(x3 - x5), rhs=1, sense="<=")
    prob.add_constraint(expr=x3 + x4 + x5, rhs=2, sense="==")
    prob.map_all_terms()
    print(prob)
    return prob

# =====================================================
#                          主函数
# =====================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === 匹配 encoder 参数用的标签（按你的类层级命名来）===
ENCODER_TAGS = (
    "agent.state_representation.gnn",         # GNN 主体
    "agent.state_representation.term_proj",   # term 投影
    "agent.state_representation.global_proj", # global 投影
)

def is_encoder_param(name: str) -> bool:
    # 只要参数名里包含任意一个 tag，就视作 encoder
    return any(tag in name for tag in ENCODER_TAGS)

def preprocess_problem_once(prob):
    # 1) 约束 sense 归一化（只做一次）
    if not getattr(prob, "_sense_canon_done", False):
        prob._canonicalize_constraint_senses_inplace()
        prob._sense_canon_done = True

    # 2) 可选：你如果还有“约束侧预处理”（比如 pre_expand/trace 展开）
    # 强烈建议：只对原始约束做一次，并且加一个 flag
    if not getattr(prob, "_cons_preproc_done", False):
        # 如果你把约束预处理写在 map_constraint_terms / map_all_terms 里：
        # 那就在这里调用一次 map_all_terms(update_problem=True)，然后之后别再用 update_problem=True
        prob.map_all_terms(update_problem=True)
        prob._cons_preproc_done = True

    # 3) 之后训练中如果还需要 term_map/indices，最多 update_problem=False（不再做 sense/展开）
    # prob.map_all_terms(update_problem=False)  # 看你是否依赖实时映射

    return prob

def preprocess_any(obj):
    # 递归处理嵌套结构
    if hasattr(obj, "_canonicalize_constraint_senses_inplace"):
        return preprocess_problem_once(obj)
    if isinstance(obj, list):
        return [preprocess_any(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(preprocess_any(x) for x in obj)
    if isinstance(obj, dict):
        return {k: preprocess_any(v) for k, v in obj.items()}
    return obj

def main():
    set_seed(TRAIN_SEED)
    print(f"[INFO] TRAIN_SEED={TRAIN_SEED}, DATASET_SPLIT_SEED={DATASET_SPLIT_SEED}")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_SAVE_DIR, exist_ok=True) # 确保 finetune 目录也存在
    
    root_cache = load_root_lb_cache(ROOT_LB_CACHE_PATH)

    # 根据配置决定创建哪些日志文件
    csv_writer = None
    STEP_CSV = None
    
    # if SAVE_DETAILED_LOGS and LOG_DETAIL_LEVEL == "step":
    #     STEP_CSV = open(os.path.join(LOG_DIR, "step_log.csv"), "w", newline="")
    #     csv_writer = csv.writer(STEP_CSV)
    #     csv_writer.writerow(["step", "a_id", "a_loc", "reward"])


    device = "cuda" if torch.cuda.is_available() else "cpu"
    anchor_model = None

    # envs = [
    #     QCQPEnv(build_qcqp1(), max_steps=30),
    #     QCQPEnv(build_qcqp2(), max_steps=30),
    #     QCQPEnv(build_qcqp3(), max_steps=30),
    # ]

    # ===== 数据加载：微调集 or 课程集 =====
    # ===== 数据加载：(重构) 无条件加载指定的数据集 =====
    curriculum_problems = None
    
    # 1. （原 FINTUNE_FILE）要加载的数据集
    DATA_FILE_PATH = FINETUNE_FILE 
    
    try:
        with open(DATA_FILE_PATH, "rb") as f:
            # 结构：[[prob, prob, ...]] 单组
            problem_set = pickle.load(f)
            problem_set = [preprocess_any(p) for p in problem_set]
        assert isinstance(problem_set, list) and len(problem_set) == 1, \
            f"Data set should be 1 group, got {len(problem_set)}"
        curriculum_problems = problem_set
    
        # 2. (原 GROUP_UPDATES)
        # 默认单阶段训练 800 个 updates，可由 APA_GROUP_UPDATES 覆盖
        GROUP_UPDATES[:] = _env_int_list("APA_GROUP_UPDATES", [800])
        TOTAL_UPDATES = sum(GROUP_UPDATES)
        cum_updates[:] = np.cumsum(GROUP_UPDATES).tolist()
        
        print(f"[INFO] Loaded data set from {DATA_FILE_PATH} with {len(problem_set[0])} problems.")
    
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load data. {e}")
        sys.exit(1)
        
        # CURRICULUM_FILE = "outputs/data/vector_curriculum.pkl"
        # with open(CURRICULUM_FILE, "rb") as f:
        #     curriculum_problems = pickle.load(f)
        # assert isinstance(curriculum_problems, list) and len(curriculum_problems) == len(GROUP_UPDATES), \
        #     f"curriculum has {len(curriculum_problems)} groups but GROUP_UPDATES has {len(GROUP_UPDATES)}"


    # Build or load train/infer split indices (keep original order for cache keys)
    split_path = SPLIT_INDICES_PATH or os.path.join(LOG_DIR, f"{DATASET_TAG}_split_indices.json")
    train_indices_by_group = []
    infer_indices_by_group = []
    loaded_split = False

    if USE_EXISTING_SPLIT and os.path.exists(split_path):
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            groups = split_data.get("groups", [])
            if len(groups) != len(curriculum_problems):
                raise ValueError("split groups mismatch dataset groups")
            for gi, ginfo in enumerate(groups):
                t = ginfo.get("train_indices", [])
                v = ginfo.get("infer_indices", [])
                if not isinstance(t, list) or not isinstance(v, list):
                    raise ValueError("split indices must be lists")
                train_indices_by_group.append(t)
                infer_indices_by_group.append(v)
                print(f"[INFO] Loaded split group {gi + 1}: train={len(t)}, infer={len(v)}")
            loaded_split = True
            print(f"[INFO] Using existing split indices: {split_path}")
        except Exception as e:
            print(f"[WARN] Failed to load split indices ({split_path}): {e}. Will regenerate.")
            train_indices_by_group = []
            infer_indices_by_group = []

    if not loaded_split:
        rng = random.Random(DATASET_SPLIT_SEED)
        for gi, probs in enumerate(curriculum_problems):
            idxs = list(range(len(probs)))
            if SHUFFLE_DATASET:
                rng.shuffle(idxs)
            split = int(len(idxs) * TRAIN_SPLIT_RATIO)
            if len(idxs) >= 2:
                split = max(1, min(split, len(idxs) - 1))
            else:
                split = len(idxs)
            train_idx = idxs[:split]
            infer_idx = idxs[split:]
            train_indices_by_group.append(train_idx)
            infer_indices_by_group.append(infer_idx)
            print(f"[INFO] Split group {gi + 1}: train={len(train_idx)}, infer={len(infer_idx)} (total={len(idxs)})")

        if SAVE_SPLIT_INDICES:
            os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_tag": DATASET_TAG,
                        "train_ratio": TRAIN_SPLIT_RATIO,
                        "shuffle": SHUFFLE_DATASET,
                        "seed": DATASET_SPLIT_SEED,
                        "groups": [
                            {"train_indices": t, "infer_indices": v}
                            for t, v in zip(train_indices_by_group, infer_indices_by_group)
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[INFO] Saved split indices to {split_path}")

    # 课程学习参数
    # EPISODES_PER_GROUP = 500  # 每组训练500轮
    # TOTAL_GROUPS = 6
    current_group = 0

    model = PolicyNet(d_model=256, d_embed=256, n_head=8,
                      ffn_hidden=512, drop_prob=0.1,
                      n_actions=len(ACTION_TYPE), device=device).to(device)
    
    # —— 确保子模块建好（尤其是 GNN）——
    dummy_problem = curriculum_problems[0][0]
    with torch.no_grad():
        _ = model.agent.state_representation(dummy_problem)
        
    # ===== 分组参数：encoder vs head =====
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if is_encoder_param(name):
            encoder_params.append(p)
        else:
            head_params.append(p)

    # 诊断输出，确保分组不是空的
    print(f"[LR groups] encoder_params={len(encoder_params)}, head_params={len(head_params)}")
    if len(encoder_params) == 0:
        print("Warning: 没有匹配到任何 encoder 参数，请检查 ENCODER_TAGS 是否与实际命名一致。")
    if len(head_params) == 0:
        print("Warning: head 参数为空，这通常不正常，请检查命名。")

    # ===== 初始化优化器：encoder 用缩放学习率，head 用原 LR =====
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': LR * ENCODER_LR_SCALE, 'eps': 1e-8},
        {'params': head_params,   'lr': LR,                    'eps': 1e-8},
    ])

    # ===== 3. [!] 新的加载/续训逻辑 (替换旧代码) =====
    start_update = 1
    global_step = 0
    global BASELINE
    BASELINE = None

    # 检查顶层的 RESUME 标志 (你脚本最上方定义的)
    if RESUME:
        # 续训: 尝试加载 checkpoint
        resume_path = os.path.join(CKPT_SAVE_DIR, CKPT_FILE)
        if os.path.isfile(resume_path):
            print(f"[Resume] 续训模式开启。加载checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            load_info = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if getattr(load_info, "missing_keys", None) or getattr(load_info, "unexpected_keys", None):
                print(f"[Resume] load_state_dict non-strict: missing={len(load_info.missing_keys)} unexpected={len(load_info.unexpected_keys)}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ckpt_update = int(checkpoint.get('update', 1))
            start_update = ckpt_update + 1 # 从下一轮开始
            global_step = checkpoint.get('global_step', 0)
            BASELINE = checkpoint.get('baseline', None)

            # Keep log files consistent with loaded checkpoint before append.
            tag = "train"
            _truncate_csv_to_update(os.path.join(LOG_DIR, f"{tag}_reward_log.csv"), ckpt_update)
            _truncate_csv_to_update(os.path.join(LOG_DIR, f"{tag}_loss_log.csv"), ckpt_update)

            mode_rw = 'a' # [!] 设为 'a' (append)
            
            print(f"       成功加载状态。将从 update={start_update}, global_step={global_step} 继续。")

        else:
            print(f"[Resume] 警告: RESUME=True 但 'finetune_latest.pt' 未找到。")
            print(f"       将检查 LOAD_WEIGHTS_FROM 或从零开始。")
            mode_rw = 'w'

    # 如果不是续训 (RESUME=False) 或 续训失败 (start_update=1)
    if start_update == 1:
        if FINETUNE and LOAD_WEIGHTS_FROM and os.path.isfile(LOAD_WEIGHTS_FROM):
            # 新阶段 Fintune: 加载指定的起点权重 (不加载 optimizer)
            print(f"[Finetune] 开启新阶段。加载起点权重: {LOAD_WEIGHTS_FROM}")
            checkpoint = torch.load(LOAD_WEIGHTS_FROM, map_location=device)
            
            # 兼容旧的 .pt (只有 state_dict) 和新的 .pt (字典)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            print(f"       成功加载模型权重。Optimizer 为全新。")
        
        else:
            # 从零开始 (Cold Start)
            print("[Cold Start] 未指定加载权重，将从随机初始化开始训练。")
            # (确保 encoder 冻结已关闭, LR Scale 为 1.0)
            if FREEZE_ENCODER_UPDATES > 0 or ENCODER_LR_SCALE != 1.0:
                print("Warning: Cold Start 时，请确保 FREEZE_ENCODER_UPDATES=0 且 ENCODER_LR_SCALE=1.0")
        
        mode_rw = 'w'
        
    if FINETUNE and (LAMBDA_ANCHOR_ID > 0 or LAMBDA_ANCHOR_LOC > 0):
        if LOAD_WEIGHTS_FROM and os.path.isfile(LOAD_WEIGHTS_FROM):
            print(f"[Anchor] 正在从 {LOAD_WEIGHTS_FROM} 加载锚点模型...")
            anchor_model = PolicyNet(d_model=256, d_embed=256, n_head=8,
                                     ffn_hidden=512, drop_prob=0.1,
                                     n_actions=len(ACTION_TYPE), device=device).to(device)
            
            # 确保 GNN 也被初始化
            with torch.no_grad():
                _ = anchor_model.agent.state_representation(dummy_problem)

            checkpoint_anchor = torch.load(LOAD_WEIGHTS_FROM, map_location=device)
            
            if 'model_state_dict' in checkpoint_anchor:
                anchor_model.load_state_dict(checkpoint_anchor['model_state_dict'], strict=False)
            else:
                anchor_model.load_state_dict(checkpoint_anchor, strict=False)
            
            anchor_model.eval()
            for p in anchor_model.parameters(): p.requires_grad = False
            print("[Anchor] 锚点模型已加载并冻结。")
        
        elif start_update == 1: # 仅在 cold start 时警告
            print(f"[Anchor] 警告: 开启了 Anchor 正则，但 LOAD_WEIGHTS_FROM "
                  f"({LOAD_WEIGHTS_FROM}) 未设置或未找到。")
            print("[Anchor] 将不使用锚点模型 (anchor_model = None)。")
        
        elif RESUME:
            print(f"[Anchor] 警告: 续训 (RESUME=True) 时无法加载锚点模型，"
                  f"因为 LOAD_WEIGHTS_FROM ({LOAD_WEIGHTS_FROM}) 未设置或未找到。")
            print("[Anchor] 将不使用锚点模型 (anchor_model = None)。")

    # 可选：如果要“直接锁死”encoder（完全不更新），把下行打开即可，相当于 lr=0 + 冻结
    # for p in encoder_params: p.requires_grad = False
    
    # ===== 可选冻结：(重构逻辑) =====
    if FINETUNE and FREEZE_ENCODER_UPDATES > 0:
        if start_update <= FREEZE_ENCODER_UPDATES:
            # 还在冻结期 (或刚开始)
            for name, p in model.named_parameters():
                if is_encoder_param(name):
                    p.requires_grad = False
            # 确保 optimizer (可能来自 cold start) 也是 0
            optimizer.param_groups[0]['lr'] = 0.0
            print(f"[INFO] Encoder is FROZEN (update {start_update} <= {FREEZE_ENCODER_UPDATES}).")
        else:
            # 已经解冻了 (从 > FREEZE_... 续训)
            # 确保 requires_grad = True (模型默认)
            # 确保 optimizer (从 checkpoint 加载) lr 是正确的
            print(f"[INFO] Encoder is UNFROZEN (resuming from update {start_update} > {FREEZE_ENCODER_UPDATES}).")
            # (optimizer.load_state_dict() 已经恢复了正确的 lr)

    
    # ==== 3.  打开日志文件时使用追加模式 (‘a’) ===========================

    tag = "train"
    reward_log_path = os.path.join(LOG_DIR, f"{tag}_reward_log.csv")
    loss_log_path   = os.path.join(LOG_DIR, f"{tag}_loss_log.csv")
    mode_step= 'w'


    reward_csv = open(reward_log_path, mode_rw, newline="")
    reward_writer_simple = csv.writer(reward_csv)
    if not RESUME:
        reward_writer_simple.writerow([
            "update",
            "avg_reward", "reward_std",
            "positive_count", "negative_count",
            "convex_count",
            "succ_rate",
            "episode_return_mean", "episode_return_median",
            "episode_length_mean",
            "baseline"
        ])
        
    loss_csv = open(loss_log_path, mode_rw, newline="")
    loss_writer_simple = csv.writer(loss_csv)
    if not RESUME:
        loss_writer_simple.writerow([
            "update",
            "policy_loss_mean", "policy_loss_std",
            "entropy_loss_mean", "entropy_loss_std",
            "value_loss_mean", "value_loss_std",
            "total_loss_mean",  "total_loss_std",
            "clip_fraction_mean", "clip_fraction_std",
            "kl_id_mean", "kl_loc_mean"
        ])

    for update in range(start_update, TOTAL_UPDATES + 1):
        # ===== 定时解冻 encoder =====
        if FINETUNE and FREEZE_ENCODER_UPDATES > 0 and update == 1 + FREEZE_ENCODER_UPDATES:
            for name, p in model.named_parameters():
                if is_encoder_param(name):
                    p.requires_grad = True
            # 恢复 encoder 组的学习率
            optimizer.param_groups[0]['lr'] = LR * ENCODER_LR_SCALE
            print(f"[INFO] Unfroze encoder at update {update}. Set encoder lr back to {LR * ENCODER_LR_SCALE:.2e}")


        # 课程学习：根据当前进度选择问题组
        # current_group = min(update // EPISODES_PER_GROUP, TOTAL_GROUPS - 1)
        current_group = bisect.bisect_right(cum_updates, update - 1)
        problems = curriculum_problems[current_group]
        train_indices = train_indices_by_group[current_group]
        group_name = f"Group_{current_group + 1}"
        
        storage_all = []
        all_ep_returns, all_ep_lengths = [], []
        ep_done = ep_succ = 0
        # # 从当前组随机选择问题
        # for _ in range(3):  # 每次选择3个问题
        #     problem = random.choice(problems)
        #     env = QCQPEnv(problem, max_steps=20)
        #     traj, global_step, d, s, ep_rets, ep_lens = rollout(env, model, NUM_STEPS, device, csv_writer, global_step)
        #     storage_all.extend(traj)
        #     ep_done += d
        #     ep_succ += s
        #     all_ep_returns.extend(ep_rets)
        #     all_ep_lengths.extend(ep_lens)
        # if not storage_all:
        #     continue
        # 从当前组筛出“root_lb 可用”的问题，再无放回采样 N 个问题（root_lb 为 null 的直接跳过）
        valid_idx_pool = []
        for _idx in train_indices:
            _prob = problems[_idx]
            _key = make_problem_key(DATASET_TAG, _idx, _prob)
            _v = root_cache.get(_key, None)
            if isinstance(_v, (int, float)):
                valid_idx_pool.append(_idx)

        if len(valid_idx_pool) == 0:
            print(f"[WARN] No valid problems with root_lb in cache. DATASET_TAG={DATASET_TAG} cache={ROOT_LB_CACHE_PATH}", flush=True)
            continue

        if len(valid_idx_pool) <= PROBLEMS_PER_UPDATE:
            batch_indices = list(valid_idx_pool)
        else:
            batch_indices = random.sample(valid_idx_pool, k=PROBLEMS_PER_UPDATE)

        for idx in batch_indices:
            problem = problems[idx]
            problem_key = make_problem_key(DATASET_TAG, idx, problem)   # dataset_tag + 全局idx + 名字# 组号 + 组内序号 + 名字（你说的“索引+名字”）
            env = QCQPEnv(problem, max_steps=20, root_lb_cache=root_cache, problem_key=problem_key)
            
            # 替换 'NUM_STEPS'
            traj, global_step, d, s, ep_rets, ep_lens = rollout(
                env, model, ROLLOUT_STEPS_PER_PROBLEM, device, csv_writer, global_step
            )
            
            storage_all.extend(traj)
            ep_done += d
            ep_succ += s
            all_ep_returns.extend(ep_rets)
            all_ep_lengths.extend(ep_lens)
        
        if not storage_all:
            continue
        
        succ_rate = ep_succ / max(ep_done, 1)
        ep_ret_mean = np.mean(all_ep_returns) if all_ep_returns else 0
        ep_ret_median = np.median(all_ep_returns) if all_ep_returns else 0
        ep_len_mean = np.mean(all_ep_lengths) if all_ep_lengths else 0
        
        # 计算当前update的reward统计
        rewards = [s[5] for s in storage_all]  # 第9个元素是reward
        is_convex_list = [s[7] for s in storage_all]  # 第10个元素是is_convex
        avg_reward = np.mean(rewards)
        reward_std = np.std(rewards)
        positive_count = sum(1 for r in rewards if r > 0)
        negative_count = sum(1 for r in rewards if r < 0)
        convex_count = sum(1 for c in is_convex_list if c)  # 统计变凸的数量

        entropy_now = entropy_coef_schedule(update, total_updates=TOTAL_UPDATES)
        
        # PPO更新并获取loss统计
        loss_stats, updated_baseline = ppo_update(model, optimizer, storage_all, device=device,
                                clip_coef=CLIP_COEF, epochs=EPOCHS_PPO,
                                batch_size=BATCH_SIZE, entropy_coef=entropy_now,
                                anchor_model=anchor_model,                     # <--- 新增
                                lambda_anchor_id=LAMBDA_ANCHOR_ID,            # <--- 新增
                                lambda_anchor_loc=LAMBDA_ANCHOR_LOC,          # <--- 新增（可为 0）
                                update_idx=update
                            )
        BASELINE = updated_baseline
        
        # 记录update级别的汇总信息
        baseline_value = BASELINE.item() if BASELINE is not None else 0.0
        reward_writer_simple.writerow([
            update,
            f"{avg_reward:.4f}", f"{reward_std:.4f}",
            positive_count, negative_count,
            convex_count,
            f"{succ_rate:.4f}",
            f"{ep_ret_mean:.4f}", f"{ep_ret_median:.4f}", f"{ep_len_mean:.2f}",
            f"{baseline_value:.4f}"
        ])
        reward_csv.flush()

        # 记录loss统计
        loss_writer_simple.writerow([
            update,
            f"{loss_stats['policy_loss_mean']:.6f}", f"{loss_stats['policy_loss_std']:.6f}",
            f"{loss_stats['entropy_loss_mean']:.6f}", f"{loss_stats['entropy_loss_std']:.6f}",
            f"{loss_stats['value_loss_mean']:.6f}", f"{loss_stats['value_loss_std']:.6f}",
            f"{loss_stats['total_loss_mean']:.6f}", f"{loss_stats['total_loss_std']:.6f}",
            f"{loss_stats['clip_fraction_mean']:.6f}", f"{loss_stats['clip_fraction_std']:.6f}",
            f"{loss_stats['kl_id_mean']:.6f}", f"{loss_stats['kl_loc_mean']:.6f}"
        ])
        loss_csv.flush()
        
        # --- 新增：每 5 轮保存一次 checkpoint ---
        if update % 10 == 0 or update == TOTAL_UPDATES: # 在第5轮和最后一轮保存
            ckpt_save_path = os.path.join(CKPT_SAVE_DIR, f"checkpoint_{update}.pt")
            
            # [!] 完整保存所有状态
            state_to_save = {
                'update': update,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': BASELINE
            }
            
            torch.save(state_to_save, ckpt_save_path)
            print(f"[Finetune] 完整 checkpoint 已保存到 {ckpt_save_path}")

            # [!] 保存一个 'latest' 副本，用于自动续训
            if update % 5 == 0 or update == TOTAL_UPDATES:
                torch.save(state_to_save, os.path.join(CKPT_SAVE_DIR, "latest.pt"))

        baseline_value = BASELINE.item() if BASELINE is not None else 0.0
        
        print(f"[Update {update:03d}] Group: {current_group} ({group_name}) "
            f"samples={len(storage_all)} avg_reward={avg_reward:.4f} baseline={baseline_value:.4f} "
            f"convex_count={convex_count} total_loss={loss_stats['total_loss_mean']:.4f} "
            f"nan_skip={int(loss_stats.get('nan_skip_batches', 0))}")
        # root_lb_cache 为离线预计算结果；训练过程不写回（避免误覆盖）
    
    # 关闭日志文件
    if STEP_CSV:
        STEP_CSV.close()
    reward_csv.close()
    loss_csv.close()


if __name__ == "__main__":
    main()
