# -*- coding: utf-8 -*-
"""
vector_qcqp_all_generators.py
----------------------------------
整合版生成脚本，包含以下几类问题构造：

[A 组] Expand 类型（带括号结构，如 (x - y)^T Q (x - y) 等）
[B-MC 组] McCormick 候选（x^T y, x^T Q y, 对角/稠密二次型等）
[B-SDP 组] SDP 候选（稠密不定二次型、-x^T x 等）
[C 组] Hybrid 混合问题（A + B 组合）
[FR 组] 分式问题（numerator/denominator 各种组合）
[NR 组] Neutral regularizers（防遗忘 McCormick / SDP / 常数分母）

命令行参数说明：
    1) python vector_qcqp_all_generators.py pure
       -> 生成 Phase 1: "vector_finetune_PURE_1000.pkl"

    2) python vector_qcqp_all_generators.py hybrid
       -> 生成 Phase 2: "vector_finetune_HYBRID_MIX_1200.pkl"

    3) python vector_qcqp_all_generators.py fraction
       -> 生成分式微调集: "vector_finetune_fraction.pkl"

    4) python vector_qcqp_all_generators.py all
       -> 生成一个“大杂烩”全集合: "vector_all_mix_1600.pkl"
          里面同时覆盖：
          - x^T y, x^T Q y, x^T Q x, -x^T x
          - 有括号的 x±y, A x + B b 等
          - 各种分式项（分母仿射 / 二次 / 仿射+二次）
"""

import pickle
import sympy as sp
from typing import List
import autoconvexrelax.core.problem as ps
from autoconvexrelax.paths import OUTPUT_ROOT
import random
import numpy as np
import sys

# ====== 配置每次调用产生的样本数（按构造函数设计） ======
EXP_PER_CALL   = 4   # _e_expand_generator 产出 4 个
MC_PER_CALL    = 7   # _e_direct_mccormick_generator 产出 7 个
SDP_PER_CALL   = 5   # _e_sdp_candidate_generator 产出 5 个
HYB_PER_CALL   = 5   # _e_hybrid_generator 产出 5 个
FR_PER_CALL    = 7   # _finetune_fraction 产出 7 个 (FR1–FR7)
NR_PER_CALL    = 4   # _neutral_regularizer 产出 4 个 (NR1–NR3)

# ====== bounds randomized knobs ======
RANDOMIZE_FIXED_BOUNDS = True
CONT_BOUNDS_SCALE = (0.7, 1.6)
INT_BOUNDS_SCALE = (0.8, 1.5)

# ---------- 随机矩阵辅助工具 ----------
def _rand_indefinite_matrix(n: int, scale_pos: float = 3.0, scale_neg: float = -3.0) -> np.ndarray:
    diag = np.random.uniform(scale_neg, scale_pos, n)
    if not np.any(diag > 0):
        diag[random.randint(0, n - 1)] = random.uniform(1.0, scale_pos)
    if not np.any(diag < 0):
        diag[random.randint(0, n - 1)] = random.uniform(scale_neg, -1.0)
    D = np.diag(diag)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    return Q @ D @ Q.T

def _rand_indefinite_diag(n: int, scale_pos: float = 3.0, scale_neg: float = -3.0) -> np.ndarray:
    diag = np.random.uniform(scale_neg, scale_pos, n)
    if not np.any(diag > 0):
        diag[random.randint(0, n - 1)] = random.uniform(1.0, scale_pos)
    if not np.any(diag < 0):
        diag[random.randint(0, n - 1)] = random.uniform(scale_neg, -1.0)
    return np.diag(diag)

def _rand_bounds(vtype: str, dim: int):
    """
    返回 (lb, ub)：
      - continuous: 对称区间 [-R, R]
      - integer:    [-R, R] 且 R 更大
      - binary:     [0, 1]
    """
    if vtype == "binary":
        return 0.0, 1.0
    if vtype == "integer":
        R = random.choice([2, 3, 4, 5])
        return -float(R), float(R)
    # continuous
    R = random.choice([0.5, 1.0, 2.0, 3.0])
    return -float(R), float(R)

def _maybe_randomize_bounds(lb, ub, vtype: str):
    if vtype == "binary":
        return 0.0, 1.0
    if lb is None or ub is None:
        return lb, ub
    if not RANDOMIZE_FIXED_BOUNDS:
        return float(lb), float(ub)

    base_lb, base_ub = float(lb), float(ub)
    if base_lb == base_ub:
        return base_lb, base_ub

    scale = random.uniform(*CONT_BOUNDS_SCALE) if vtype == "continuous" else random.uniform(*INT_BOUNDS_SCALE)
    new_lb = base_lb * scale
    new_ub = base_ub * scale
    if new_lb > new_ub:
        new_lb, new_ub = new_ub, new_lb

    if vtype == "integer":
        new_lb = int(np.floor(new_lb))
        new_ub = int(np.ceil(new_ub))
        if new_lb == new_ub:
            new_ub = new_lb + 1

    return float(new_lb), float(new_ub)

def _vec(prob: ps.QCQPProblem, name: str, dim: int, lb=None, ub=None, vtype="continuous"):
    """创建 dim×1 向量决策变量；lb/ub 若不提供则随机。"""
    if lb is None or ub is None:
        lb0, ub0 = _rand_bounds(vtype, dim)
        lb = lb0 if lb is None else lb
        ub = ub0 if ub is None else ub
    lb, ub = _maybe_randomize_bounds(lb, ub, vtype)
    return prob.add_vector_variable(name, dim, lb=lb, ub=ub, vtype=vtype)


def _const_vec(vals):
    """列向量常数矩阵（shape: n×1）"""
    return sp.Matrix(vals).reshape(len(vals), 1)

def _as_list(var_or_list):
    if var_or_list is None:
        return []
    if isinstance(var_or_list, (list, tuple)):
        return list(var_or_list)
    return [var_or_list]

def _ub_norm2(prob: ps.QCQPProblem, v) -> float:
    lb, ub = _get_var_bounds(prob, v)
    n = int(v.shape[0])
    m = max(abs(lb), abs(ub))
    return float(n) * (m ** 2)

def _lb_norm2(prob: ps.QCQPProblem, v) -> float:
    lb, ub = _get_var_bounds(prob, v)
    if lb <= 0 <= ub:
        return 0.0
    n = int(v.shape[0])
    m = min(abs(lb), abs(ub))
    return float(n) * (m ** 2)

def _ub_sum_norm2(prob: ps.QCQPProblem, vars_list) -> float:
    return float(sum(_ub_norm2(prob, v) for v in _as_list(vars_list)))

def _lin_bounds_trace(prob: ps.QCQPProblem, v, e: sp.Matrix) -> tuple[float, float]:
    """
    bounds of Trace(e.T * v) with v_i in [lb,ub], e constant.
    """
    lb, ub = _get_var_bounds(prob, v)
    if lb > ub:
        lb, ub = ub, lb
    enp = np.array(e.tolist(), dtype=float).reshape(-1)
    min_val = float(np.sum([c * (lb if c >= 0 else ub) for c in enp]))
    max_val = float(np.sum([c * (ub if c >= 0 else lb) for c in enp]))
    return min_val, max_val

def _ub_bilinear(prob: ps.QCQPProblem, x, Q: sp.Matrix, y) -> float:
    lbx, ubx = _get_var_bounds(prob, x)
    lby, uby = _get_var_bounds(prob, y)
    nx = int(x.shape[0])
    ny = int(y.shape[0])
    Mx = max(abs(lbx), abs(ubx))
    My = max(abs(lby), abs(uby))
    Qnp = np.array(Q.tolist(), dtype=float)
    Q_inf = np.max(np.sum(np.abs(Qnp), axis=1))  # max row sum
    return float(nx * Mx * Q_inf * ny * My)

def _ub_bilinear_noq(prob: ps.QCQPProblem, x, y) -> float:
    n = int(x.shape[0])
    Q = sp.eye(n)
    return _ub_bilinear(prob, x, Q, y)

def _ub_quadratic(prob: ps.QCQPProblem, x, Q: sp.Matrix) -> float:
    lb, ub = _get_var_bounds(prob, x)
    n = int(x.shape[0])
    M = max(abs(lb), abs(ub))
    Qnp = np.array(Q.tolist(), dtype=float)
    Q_inf = np.max(np.sum(np.abs(Qnp), axis=1))
    x1 = n * M
    return float(x1 * Q_inf * x1)

def _get_var_bounds(prob: ps.QCQPProblem, v) -> tuple[float, float]:
    """
    v 是向量 MatrixSymbol（例如 x, y, b）。
    注意：QCQPProblem.add_vector_variable 不会生成 prob.variables['x_0'] 之类标量，
    bounds 存在 prob.matrix_variables[v.name].lb/ub。
    """
    try:
        name = getattr(v, "name", None) or str(v)
        mv = getattr(prob, "matrix_variables", {}).get(name, None)
        if mv is not None:
            lb = getattr(mv, "lb", None)
            ub = getattr(mv, "ub", None)
            if lb is not None and ub is not None:
                return float(lb), float(ub)
    except Exception:
        pass
    return -2.0, 2.0

def _safe_rhs_from_ub(ub: float, frac_range=(0.5, 0.95), add_eps=1e-6) -> float:
    """
    从 [frac_range] 中采一个比例 * ub，当作 <= RHS。
    保证 RHS <= ub，从而“至少不会因为这个约束本身”导致不可行。
    """
    lo, hi = frac_range
    frac = random.uniform(lo, hi)
    return float(frac * ub + add_eps)

def _ub_trace_linear(prob: ps.QCQPProblem, v, e: sp.Matrix) -> float:
    """
    ub of Trace(e.T * v) with v_i in [lb,ub], e constant.
    粗上界：sum |e_i| * max(|lb|,|ub|)
    """
    _, ub = _lin_bounds_trace(prob, v, e)
    return float(ub)

def _add_safe_leq(prob: ps.QCQPProblem,
                  lhs,
                  ub_est: float,
                  frac_range=(0.6, 0.95),
                  add_eps=1e-6):
    rhs = _safe_rhs_from_ub(ub_est, frac_range=frac_range, add_eps=add_eps)
    prob.add_constraint(lhs, "<=", rhs)
    return rhs

def _safe_rhs_ratio(ub_num: float, lb_den: float, frac_range=(0.75, 0.97), min_den=1e-6) -> float:
    den = max(float(lb_den), float(min_den))
    ub_ratio = float(ub_num) / den
    return _safe_rhs_from_ub(ub_ratio, frac_range=frac_range)

def _add_safe_geq(prob: ps.QCQPProblem,
                  lhs,
                  lb_est: float,
                  frac_range=(0.6, 0.95),
                  add_eps=1e-6):
    """
    对 >= 约束：用“安全下界”做 RHS。
    最稳的做法：直接取 rhs = lb_est * frac（靠近 lb），保证可行。
    """
    lo, hi = frac_range
    frac = random.uniform(lo, hi)
    rhs = float(lb_est * frac - add_eps)
    prob.add_constraint(lhs, ">=", rhs)
    return rhs

def _add_global_linear_couplings(prob: ps.QCQPProblem,
                                vec_vars,
                                max_cons: int = 3,
                                p_add: float = 0.7,
                                p_eq: float = 0.25,
                                p_link: float = 0.6,
                                p_budget: float = 0.6,
                                p_mutex: float = 0.25):
    """
    在 prob 上随机添加 1~max_cons 条“全局线性耦合约束”，但做可行性保护：
      1) vtype / bounds 从 prob.matrix_variables 读取（而不是 prob.variables）
      2) equality 的 k 会尊重已有的 sum(b) 上界（如 Trace(b.T*b) <= cap）
      3) mutex 与 equality 不同时加（避免把可行域掐死）
      4) budget 会尊重已有的 sum(v) 下界/上界，避免与 “sum>=k_min” 类约束冲突
    """
    import sympy as sp
    vec_vars = _as_list(vec_vars)
    if not vec_vars:
        return
    if random.random() > p_add:
        return

    # ---------------- helpers ----------------
    def _vec_meta(v):
        name = getattr(v, "name", None) or str(v)
        mv = getattr(prob, "matrix_variables", {}).get(name, None)
        vtype = getattr(mv, "vtype", "continuous") if mv is not None else "continuous"
        lb, ub = _get_var_bounds(prob, v)
        return name, vtype, lb, ub

    def _is_trace(expr):
        return isinstance(expr, sp.Trace)

    def _match_sum_expr(expr, v):
        """
        识别 sum(v_i) 的两种常见写法：
          - Trace(e.T * v)   其中 e 是常数向量（Matrix）
          - Trace(v.T * v)   当 v 是 binary 时等价于 sum(v_i)
        返回 ("sum", v) 或 ("bin_norm", v) 或 None
        """
        if not _is_trace(expr):
            return None
        arg = expr.args[0]

        # Trace(v.T * v)
        try:
            if isinstance(arg, sp.MatMul) and len(arg.args) == 2:
                A, B = arg.args
                if A == v.T and B == v:
                    return ("bin_norm", v)
        except Exception:
            pass

        # Trace(e.T * v) : e 是常数 Matrix
        try:
            if isinstance(arg, sp.MatMul) and len(arg.args) == 2:
                A, B = arg.args
                if B == v and isinstance(A, sp.Transpose) and isinstance(A.args[0], sp.MatrixBase):
                    return ("sum", v)
        except Exception:
            pass

        return None

    def _sum_bounds_from_existing(v, vtype):
        """
        从已有约束里抽取关于 sum(v) 的 [lb, ub]（尽量保守）。
        只处理非常简单的模式：Trace(e.T*v) ? k 以及 binary 的 Trace(v.T*v) ? k
        """
        lb = None
        ub = None
        for c in getattr(prob, "constraints", []):
            expr = getattr(c, "expr", None)
            sense = getattr(c, "sense", None)
            rhs = getattr(c, "rhs", None)

            if expr is None or sense is None:
                continue

            m = _match_sum_expr(expr, v)
            if m is None:
                continue

            kind, _ = m
            # 对非 binary：Trace(v.T*v) 不是 sum(v)，跳过
            if kind == "bin_norm" and vtype != "binary":
                continue

            # rhs 必须是常数（int/float）才做界抽取
            if isinstance(rhs, (int, float)):
                k = float(rhs)
            else:
                # 允许 sympy.Number
                try:
                    if isinstance(rhs, sp.Number):
                        k = float(rhs)
                    else:
                        continue
                except Exception:
                    continue

            if sense in ("<=", "<"):
                ub = k if ub is None else min(ub, k)
            elif sense in (">=", ">"):
                lb = k if lb is None else max(lb, k)
            elif sense == "=":
                lb = k if lb is None else max(lb, k)
                ub = k if ub is None else min(ub, k)

        return lb, ub

    # ---------------- split vars ----------------
    cont, disc = [], []
    meta = {}
    for v in vec_vars:
        name, vtype, lb, ub = _vec_meta(v)
        meta[v] = (name, vtype, lb, ub)
        if vtype in ("binary", "integer"):
            disc.append(v)
        else:
            cont.append(v)
    bin_vars = [v for v in disc if meta[v][1] == "binary"]

    n_cons = random.randint(1, max_cons)
    used_eq = False     # mutex 与 equality 不同时加
    used_mutex = False  # 防止先加 mutex 再加 equality

    for _ in range(n_cons):
        r = random.random()

        # ---------- (1) equality: sum(b) = k (binary/integer 常见) ----------
        if (not used_eq) and (not used_mutex) and disc and (random.random() < p_eq):
            b = random.choice(disc)
            name, vtype, lb, ub = meta[b]
            n = int(b.shape[0])
            e = _const_vec([1] * n)

            # 从已有约束里抽取 sum(b) 的 [lb, ub]（对 binary 特别关键）
            s_lb, s_ub = _sum_bounds_from_existing(b, vtype)

            # 自然范围（binary / integer 均由 bounds 决定）
            if vtype == "binary":
                nat_lb, nat_ub = 0.0, float(n)
            else:
                nat_lb, nat_ub = float(n) * float(lb), float(n) * float(ub)

            # 最终可选区间
            lo = nat_lb if s_lb is None else max(nat_lb, float(s_lb))
            hi = nat_ub if s_ub is None else min(nat_ub, float(s_ub))

            if hi < lo + 1e-9:
                # 没有合法 k，跳过 equality
                pass
            else:
                # 选一个整数 k（更稳）
                k_lo = int(np.ceil(lo))
                k_hi = int(np.floor(hi))
                if k_hi >= k_lo:
                    k = random.randint(k_lo, k_hi)
                    prob.add_constraint(sp.Trace(e.T * b), "=", int(k))
                    used_eq = True
                    continue

        # ---------- (2) mutex: b_i + b_j <= 1 ----------
        if (not used_eq) and (not used_mutex) and bin_vars and (random.random() < p_mutex):
            b = random.choice(bin_vars)
            name, vtype, lb, ub = meta[b]
            n = int(b.shape[0])
            if n >= 2:
                i, j = random.sample(range(n), 2)
                prob.add_constraint(sp.Matrix([b[i, 0]])[0] + sp.Matrix([b[j, 0]])[0], "<=", 1)
                used_mutex = True
                continue

        # ---------- (3) linking: x_i <= U*b_i (可选加下界 x_i >= L*b_i) ----------
        if cont and bin_vars and (random.random() < p_link):
            x = random.choice(cont)
            b = random.choice(bin_vars)
            _, x_type, x_lb, x_ub = meta[x]
            _, b_type, b_lb, b_ub = meta[b]
            if x_ub is None or float(x_ub) <= 0:
                continue
            n = min(int(x.shape[0]), int(b.shape[0]))
            for _ in range(random.randint(1, min(3, n))):
                i = random.randint(0, n - 1)
                # big-M 用真实 ub
                U = float(x_ub)
                prob.add_constraint(sp.Matrix([x[i, 0]])[0] - U * sp.Matrix([b[i, 0]])[0], "<=", 0)
                # 下界（更稳：只有当 L>0 才加；L<=0 加了也没意义还可能误伤）
                if x_lb is not None and float(x_lb) > 0 and random.random() < 0.5:
                    L = float(x_lb)
                    prob.add_constraint(sp.Matrix(-[x[i, 0]])[0] + L * sp.Matrix([b[i, 0]])[0], "<=", 0)
            continue

        # ---------- (4) budget: sum(v) <= k ----------
        if r < p_budget:
            v = random.choice(vec_vars)
            name, vtype, lb, ub = meta[v]
            n = int(v.shape[0])
            e = _const_vec([1] * n)

            # 已有 sum(v) 的界
            s_lb, s_ub = _sum_bounds_from_existing(v, vtype)

            # 按 bounds 推一个“肯定可行”的 upper 区间
            # 对 binary：sum ∈ [0, n]
            if vtype == "binary":
                nat_lb, nat_ub = 0.0, float(n)
            else:
                nat_lb, nat_ub = _lin_bounds_trace(prob, v, e)

            lo = nat_lb if s_lb is None else max(nat_lb, float(s_lb))
            hi = nat_ub if s_ub is None else min(nat_ub, float(s_ub))

            # 预算是 <= k：k 至少要 >= lo
            if hi < lo + 1e-9:
                # 当前 sum 已经被夹死了，别再加 budget
                pass
            else:
                # 选一个偏“松”的 k，避免和别的约束冲突
                # binary 用整数；连续用浮点
                if vtype == "binary":
                    k_lo = int(np.ceil(lo))
                    k_hi = int(np.floor(hi))
                    if k_hi >= k_lo:
                        # 取更靠近上界（更安全）
                        k = random.randint(max(k_lo, int(0.6 * k_hi)), k_hi)
                        prob.add_constraint(sp.Trace(e.T * v), "<=", int(k))
                        continue
                else:
                    k = random.uniform(max(lo, 0.6 * hi), hi)
                    prob.add_constraint(sp.Trace(e.T * v), "<=", float(k))
                    continue

        # fallback：不加（避免硬塞导致不可行）
        continue


# ===================================================================
# [A 组] 纯粹的 Expand 问题
# ===================================================================
def _e_expand_generator() -> List[ps.QCQPProblem]:
    """
    [纯粹 A 组] 复合表达式，必须 Expand。
    这里刻意让 ~ 一半问题包含离散变量（binary / integer）。
    [产出: 4 个问题]
    """
    gs = []

    # --- E_EXP_P1: (x - y).T * Q_dense * (x - y)，y 为 binary ---
    n1 = random.randint(3, 6)
    p = ps.QCQPProblem(f"E_EXP_P1_n{n1}_{random.randint(100, 999)}")
    x1 = _vec(p, "x", n1, lb=-2, ub=2, vtype="continuous")
    y1 = _vec(p, "y", n1, lb=0,  ub=1, vtype="binary")      # 改成 binary
    Q1 = sp.Matrix(np.round(_rand_indefinite_matrix(n1), 3))
    expr1 = x1 - y1
    obj1 = sp.Trace(expr1.T * Q1 * expr1) + random.uniform(0.3, 0.7) * sp.Trace(y1.T * y1)
    p.set_objective(obj1, "min")
    _add_safe_leq(p,
                  sp.Trace(x1.T * x1) + sp.Trace(y1.T * y1),
                  _ub_sum_norm2(p, [x1, y1]))
    _add_global_linear_couplings(p, [x1, y1], max_cons=3, p_add=0.8)
    p.map_all_terms(); gs.append(p)

    # --- E_EXP_P2: (x + y).T * Q_diag * (x - z)，z 为 integer 向量 ---
    n2 = random.randint(3, 5)
    p = ps.QCQPProblem(f"E_EXP_P2_n{n2}_{random.randint(100, 999)}")
    x2 = _vec(p, "x", n2, lb=-2, ub=2, vtype="continuous")
    y2 = _vec(p, "y", n2, lb=-2, ub=2, vtype="continuous")
    z2 = _vec(p, "z", n2, lb=-3, ub=3, vtype="integer")     # 新增 integer
    Q2 = sp.Matrix(np.round(_rand_indefinite_diag(n2), 3))
    expr2 = x2 + y2
    obj2 = sp.Trace(expr2.T * Q2 * (x2 - z2))
    p.set_objective(obj2, "min")
    # 用 x,y,z 同时出现在约束中
    _add_safe_leq(p,
                  sp.Trace(x2.T*x2)+sp.Trace(y2.T*y2)+sp.Trace(z2.T*z2),
                  _ub_sum_norm2(p, [x2, y2, z2]))
    _add_global_linear_couplings(p, [x2, y2, z2], max_cons=3, p_add=0.8)
    p.map_all_terms(); gs.append(p)

    # --- E_EXP_P3: (A*x + B*b).T * (A*x + B*b)，b 为 binary（原样保留） ---
    n3 = random.randint(4, 6)
    p = ps.QCQPProblem(f"E_EXP_P3_n{n3}_{random.randint(100, 999)}")
    x3 = _vec(p, "x", n3, lb=-2, ub=2, vtype="continuous")
    b3 = _vec(p, "b", n3, lb=0,  ub=1, vtype="binary")
    A = sp.Matrix(np.random.randint(-2, 3, size=(n3, n3)))
    B = sp.Matrix(np.diag(np.random.choice([-1, 1], n3)))
    expr3 = A * x3 + B * b3
    obj3 = sp.Trace(expr3.T * expr3) - random.uniform(0.3, 0.8) * sp.Trace(x3.T * x3)
    p.set_objective(obj3, "min")
    _add_safe_leq(p, sp.Trace(b3.T * b3), _ub_norm2(p, b3))
    _add_safe_leq(p, sp.Trace(x3.T * x3), _ub_norm2(p, x3))
    _add_global_linear_couplings(p, [x3, b3], max_cons=3, p_add=0.8)
    p.map_all_terms(); gs.append(p)

    # --- E_EXP_P4: (x - y).T * Q_diag * (x - y)，x 连续，y integer ---
    n4 = random.randint(4, 7)
    p = ps.QCQPProblem(f"E_EXP_P4_PURE_n{n4}_{random.randint(100, 999)}")
    x4 = _vec(p, "x", n4, lb=-2, ub=2, vtype="continuous")
    y4 = _vec(p, "y", n4, lb=-2, ub=2, vtype="integer")     # integer
    Qd = sp.Matrix(np.round(_rand_indefinite_diag(n4), 3))
    expr4 = x4 - y4
    obj4 = sp.Trace(expr4.T * Qd * expr4) + random.uniform(0.1, 0.5) * sp.Trace(x4.T * x4)
    p.set_objective(obj4, "min")
    _add_safe_leq(p,
                  sp.Trace(x4.T * x4) + sp.Trace(y4.T * y4),
                  _ub_sum_norm2(p, [x4, y4]))
    _add_global_linear_couplings(p, [x4, y4], max_cons=3, p_add=0.8)
    p.map_all_terms(); gs.append(p)

    return gs


# ===================================================================
# [B 组] 纯粹的 McCormick 问题
# ===================================================================
def _e_direct_mccormick_generator() -> List[ps.QCQPProblem]:
    """
    [纯粹 B 组 - MC] 纯粹的、应被 McCormick 松弛的项。
    增加若干 binary / integer 变量，强调“混合整数 + McCormick”的场景。
    [产出: 7 个问题]
    """
    gs = []
    
    # --- MC_P1: Trace(x.T * Qd * x) (对角二次型，连续) ---
    n1 = random.randint(4, 7)
    p = ps.QCQPProblem(f"E_DIR_MC_P1_n{n1}_{random.randint(100, 999)}")
    x1 = _vec(p, "x", n1)
    Qd = sp.Matrix(np.round(_rand_indefinite_diag(n1, 4.0, -4.0), 3))
    obj1 = sp.Trace(x1.T * Qd * x1) + random.uniform(0.1, 0.5) * sp.Trace(x1.T * x1) 
    p.set_objective(obj1, "min")
    _add_safe_leq(p, sp.Trace(x1.T * x1), _ub_norm2(p, x1))
    _add_global_linear_couplings(p, [x1], max_cons=2, p_add=0.5)

    p.map_all_terms(); gs.append(p)

    # --- MC_P2: Trace(x.T*Q1d*x) + Trace(y.T*Q2d*y)，y 为 binary ---
    n2 = random.randint(3, 5)
    p = ps.QCQPProblem(f"E_DIR_MC_P2_n{n2}_{random.randint(100, 999)}")
    x2 = _vec(p, "x", n2)
    y2 = _vec(p, "y", n2, lb=0, ub=1, vtype="binary")
    Q1d = sp.Matrix(np.round(_rand_indefinite_diag(n2, 3.0, -2.0), 3))
    Q2d = sp.Matrix(np.round(_rand_indefinite_diag(n2, 2.0, -3.0), 3))
    obj2 = sp.Trace(x2.T * Q1d * x2) + sp.Trace(y2.T * Q2d * y2)
    p.set_objective(obj2, "min")
    _add_safe_leq(p,
                  sp.Trace(x2.T * x2) + sp.Trace(y2.T * y2),
                  _ub_sum_norm2(p, [x2, y2]))
    _add_global_linear_couplings(p, [x2, y2], max_cons=2, p_add=0.5)

    p.map_all_terms(); gs.append(p)

    # --- MC_P3: Trace(x.T * Qd * y) (对角双线性，y 为 integer) ---
    n3 = random.randint(3, 6)
    p = ps.QCQPProblem(f"E_DIR_MC_P3_n{n3}_{random.randint(100, 999)}")
    x3 = _vec(p, "x", n3)
    y3 = _vec(p, "y", n3, lb=-2, ub=2, vtype="integer")
    Qd = sp.Matrix(np.round(_rand_indefinite_diag(n3, 2.0, -2.0), 3))
    obj3 = sp.Trace(x3.T * Qd * y3)
    p.set_objective(obj3, "min")
    _add_safe_leq(p,
                  sp.Trace(x3.T * x3) + sp.Trace(y3.T * y3),
                  _ub_sum_norm2(p, [x3, y3]))
    _add_global_linear_couplings(p, [x3, y3], max_cons=2, p_add=0.5)

    p.map_all_terms(); gs.append(p)
    
    # --- MC_P4: Trace(x.T * Q_dense * y) (稠密双线性，全连续) ---
    n4 = random.randint(3, 5)
    p = ps.QCQPProblem(f"E_DIR_MC_P4_n{n4}_{random.randint(100, 999)}")
    x4, y4 = _vec(p, "x", n4), _vec(p, "y", n4)
    Q = sp.Matrix(np.round(_rand_indefinite_matrix(n4, 2.0, -2.0), 3))
    obj4 = sp.Trace(x4.T * Q * y4)
    p.set_objective(obj4, "min")
    _add_safe_leq(p,
                  sp.Trace(x4.T * x4) + sp.Trace(y4.T * y4),
                  _ub_sum_norm2(p, [x4, y4]))
    _add_global_linear_couplings(p, [x4, y4], max_cons=2, p_add=0.5)

    p.map_all_terms(); gs.append(p)
    
    # --- MC_P5: x.T*Q_dense*y (在约束中)，y 为 binary ---
    n5 = random.randint(3, 5)
    p = ps.QCQPProblem(f"E_DIR_MC_P5_n{n5}_{random.randint(100, 999)}")
    x5 = _vec(p, "x", n5)
    y5 = _vec(p, "y", n5, lb=0, ub=1, vtype="binary")
    Q = sp.Matrix(np.round(_rand_indefinite_matrix(n5, 3.0, -3.0), 3))
    p.set_objective(sp.Trace(x5.T * x5) + sp.Trace(y5.T * y5), "min") # 凸目标
    _add_safe_leq(p, sp.Trace(x5.T * Q * y5), _ub_bilinear(p, x5, Q, y5)) # 非凸约束 + binary
    _add_global_linear_couplings(p, [x5, y5], max_cons=2, p_add=0.5)

    p.map_all_terms(); gs.append(p)
    
    # --- MC_P6: 纯 x^T y 出现在目标，y integer ---
    n6 = random.randint(2, 10)
    p = ps.QCQPProblem(f"E_DIR_MC_P6_n{n6}_{random.randint(100, 999)}")
    x6 = _vec(p, "x", n6, lb=-1, ub=1, vtype="continuous")
    y6 = _vec(p, "y", n6, lb=-3, ub=3, vtype="integer")
    obj6 = sp.Trace(x6.T * y6)
    p.set_objective(obj6, "min")
    _add_safe_leq(p,
                  sp.Trace(x6.T * x6) + sp.Trace(y6.T * y6),
                  _ub_sum_norm2(p, [x6, y6]))
    _add_global_linear_couplings(p, [x6, y6], max_cons=2, p_add=0.5)
    p.map_all_terms(); gs.append(p)

    # --- MC_P7: 纯 x^T y 出现在约束，x binary，y 连续 ---
    n7 = random.randint(2, 10)
    p = ps.QCQPProblem(f"E_DIR_MC_P7_n{n7}_{random.randint(100, 999)}")
    x7 = _vec(p, "x", n7, lb=0,  ub=1, vtype="binary")
    y7 = _vec(p, "y", n7, lb=-2, ub=2, vtype="continuous")
    obj7 = sp.Trace(x7.T * x7) + sp.Trace(y7.T * y7)
    p.set_objective(obj7, "min")
    _add_safe_leq(p, sp.Trace(x7.T * y7), _ub_bilinear_noq(p, x7, y7))
    _add_global_linear_couplings(p, [x7, y7], max_cons=2, p_add=0.5)
    p.map_all_terms(); gs.append(p)

    return gs


# ===================================================================
# [B 组] 纯粹的 SDP 问题
# ===================================================================
def _e_sdp_candidate_generator() -> List[ps.QCQPProblem]:
    """
    [纯粹 B 组]  (稠密二次型)。
    增加少量离散变量（integer / binary），突出“PSD + MIP”组合。
    [产出: 5 个问题]
    """
    gs = []
    
    # --- SDP_P1: x.T*Q*x (在目标中，连续) ---
    n1 = random.randint(4, 8)
    p = ps.QCQPProblem(f"E_SDP_P1_n{n1}_{random.randint(100, 999)}")
    x1 = _vec(p, "x", n1)
    Q = sp.Matrix(np.round(_rand_indefinite_matrix(n1, 5.0, -3.0), 3))
    obj1 = sp.Trace(x1.T * Q * x1)
    p.set_objective(obj1, "min")
    _add_safe_leq(p, sp.Trace(x1.T * x1), _ub_norm2(p, x1))
    _add_global_linear_couplings(p, [x1], max_cons=2, p_add=0.7)
    p.map_all_terms(); gs.append(p)

    # --- SDP_P2: x.T*Q*x (在约束中)，连续 ---
    n2 = random.randint(4, 7)
    p = ps.QCQPProblem(f"E_SDP_P2_n{n2}_{random.randint(100, 999)}")
    x2 = _vec(p, "x", n2)
    Q = sp.Matrix(np.round(_rand_indefinite_matrix(n2, 4.0, -2.0), 3))
    obj2 = sp.Trace(x2.T * x2) # 凸目标
    p.set_objective(obj2, "min")
    _add_safe_leq(p, sp.Trace(x2.T * Q * x2), _ub_quadratic(p, x2, Q)) # 非凸约束
    _add_safe_geq(p, sp.Trace(x2.T * x2), _lb_norm2(p, x2))
    _add_global_linear_couplings(p, [x2], max_cons=2, p_add=0.7)
    p.map_all_terms(); gs.append(p)

    # --- SDP_P3: x.T*Q1*x + y.T*Q2*y，y 为 integer 向量 ---
    n3_x, n3_y = random.randint(3, 5), random.randint(3, 5)
    p = ps.QCQPProblem(f"E_SDP_P3_n{n3_x}_{n3_y}_{random.randint(100, 999)}")
    x3 = _vec(p, "x", n3_x, lb=-2, ub=2, vtype="continuous")
    y3 = _vec(p, "y", n3_y, lb=-3, ub=3, vtype="integer")
    Qx = sp.Matrix(np.round(_rand_indefinite_matrix(n3_x, 3.0, -3.0), 3))
    Qy = sp.Matrix(np.round(_rand_indefinite_matrix(n3_y, 2.0, -4.0), 3))
    obj3 = sp.Trace(x3.T * Qx * x3) + sp.Trace(y3.T * Qy * y3)
    p.set_objective(obj3, "min")
    _add_safe_leq(p,
                  sp.Trace(x3.T * x3) + sp.Trace(y3.T * y3),
                  _ub_sum_norm2(p, [x3, y3]))
    _add_global_linear_couplings(p, [x3, y3], max_cons=2, p_add=0.7)
    p.map_all_terms(); gs.append(p)
    
    # --- SDP_P4: 纯凹目标 -x^T x，连续 ---
    n4 = random.randint(3, 6)
    p = ps.QCQPProblem(f"E_SDP_P4_NEG_OBJ_n{n4}_{random.randint(100, 999)}")
    x4 = _vec(p, "x", n4)
    obj4 = - sp.Trace(x4.T * x4)
    p.set_objective(obj4, "min")
    _add_safe_leq(p, sp.Trace(x4.T * x4), _ub_norm2(p, x4))
    _add_global_linear_couplings(p, [x4], max_cons=2, p_add=0.7)
    p.map_all_terms(); gs.append(p)

    # --- SDP_P5: 凹约束 -x^T x <= c，x 为 binary+连续混合向量 (简单化：binary) ---
    n5 = random.randint(3, 6)
    p = ps.QCQPProblem(f"E_SDP_P5_NEG_CONS_n{n5}_{random.randint(100, 999)}")
    x5 = _vec(p, "x", n5, lb=0, ub=1, vtype="binary")
    obj5 = sp.Trace(x5.T * x5)
    p.set_objective(obj5, "min")
    # sum(x_i) >= k_min，基于 bounds 选一个安全 k_min
    k_min = max(1, int(np.floor(_safe_rhs_from_ub(_ub_norm2(p, x5), frac_range=(0.2, 0.6)))))
    rhs5 = -float(k_min)
    p.add_constraint(-sp.Trace(x5.T * x5), "<=", rhs5)
    _add_global_linear_couplings(p, [x5], max_cons=2, p_add=0.7)
    p.map_all_terms(); gs.append(p)

    return gs

# ===================================================================
# [C 组] 混合问题 (用于阶段 2)
# ===================================================================
def _e_hybrid_generator() -> List[ps.QCQPProblem]:
    """
    [混合 C 组] 混合问题，包含 A 组和 B 组的特征。
    同时出现 Expand 项和 McCormick/SDP 候选项。
    [产出: 5 个问题]
    """
    gs = []
    
    # --- H_MIXED_1: (x-y)Q(x-y) [Expand] + x.T*y [McCormick] ---
    n1 = random.randint(4, 7)
    p = ps.QCQPProblem(f"H_MIXED_1_n{n1}_{random.randint(100, 999)}")
    x1, y1 = _vec(p, "x", n1), _vec(p, "y", n1)
    Qd1 = sp.Matrix(np.round(_rand_indefinite_diag(n1, 3.0, -3.0), 3))
    expr1 = x1 - y1
    obj1 = sp.Trace(expr1.T * Qd1 * expr1) + random.uniform(0.1, 0.9) * sp.Trace(x1.T * y1)
    p.set_objective(obj1, "min")
    _add_safe_leq(p,
                  sp.Trace(x1.T * x1) + sp.Trace(y1.T * y1),
                  _ub_sum_norm2(p, [x1, y1]))
    p.map_all_terms(); gs.append(p)

    # --- H_MIXED_2: (x+y)Q(x-z) [Expand] + x.T*Q*y [McCormick] ---
    n2 = random.randint(3, 5)
    p = ps.QCQPProblem(f"H_MIXED_2_n{n2}_{random.randint(100, 999)}")
    x2, y2, z2 = _vec(p, "x", n2), _vec(p, "y", n2), _vec(p, "z", n2)
    Q1 = sp.Matrix(np.round(_rand_indefinite_diag(n2, 2.0, -2.0), 3))
    Q2 = sp.Matrix(np.round(_rand_indefinite_diag(n2, 1.0, -1.0), 3))
    obj2 = sp.Trace((x2 + y2).T * Q1 * (x2 - z2)) + sp.Trace(x2.T * Q2 * y2)
    p.set_objective(obj2, "min")
    _add_safe_leq(p,
                  sp.Trace(x2.T*x2)+sp.Trace(y2.T*y2)+sp.Trace(z2.T*z2),
                  _ub_sum_norm2(p, [x2, y2, z2]))
    p.map_all_terms(); gs.append(p)
    
    # --- H_MIXED_3: (Ax+Bb)T(Ax+Bb) [Expand] + x.T*b [McCormick] ---
    n3 = random.randint(4, 6)
    p = ps.QCQPProblem(f"H_MIXED_3_n{n3}_{random.randint(100, 999)}")
    x3, b3 = _vec(p, "x", n3), _vec(p, "b", n3, lb=0, ub=1, vtype="binary")
    A3 = sp.Matrix(np.random.randint(-2, 3, size=(n3, n3)))
    B3 = sp.Matrix(np.diag(np.random.choice([-1, 1], n3)))
    expr3 = A3 * x3 + B3 * b3
    obj3 = sp.Trace(expr3.T * expr3) + sp.Trace(x3.T * b3) 
    p.set_objective(obj3, "min")
    _add_safe_leq(p, sp.Trace(b3.T * b3), _ub_norm2(p, b3))
    _add_safe_leq(p, sp.Trace(x3.T * x3), _ub_norm2(p, x3))
    _add_global_linear_couplings(p, [x3, b3], max_cons=2, p_add=0.7)

    p.map_all_terms(); gs.append(p)

    # --- H_MIXED_4: (x-y)Q1(x-y) [Expand] + x.T*Q2*x [SDP] ---
    n4 = random.randint(3, 5)
    p = ps.QCQPProblem(f"H_MIXED_4_n{n4}_{random.randint(100, 999)}")
    x4, y4 = _vec(p, "x", n4), _vec(p, "y", n4)
    Q1 = sp.Matrix(np.round(_rand_indefinite_diag(n4, 2.0, -2.0), 3))
    Q2 = sp.Matrix(np.round(_rand_indefinite_matrix(n4, 3.0, -3.0), 3))
    expr4 = x4 - y4
    obj4 = sp.Trace(expr4.T * Q1 * expr4) + sp.Trace(x4.T * Q2 * x4)
    p.set_objective(obj4, "min")
    _add_safe_leq(p,
                  sp.Trace(x4.T * x4) + sp.Trace(y4.T * y4),
                  _ub_sum_norm2(p, [x4, y4]))
    p.map_all_terms(); gs.append(p)
    
    # --- H_MIXED_5: x.T*Q1d*x [McCormick] + y.T*Q2*y [SDP] ---
    n5_x, n5_y = random.randint(3, 5), random.randint(3, 5)
    p = ps.QCQPProblem(f"H_MIXED_5_n{n5_x}_{n5_y}_{random.randint(100, 999)}")
    x5, y5 = _vec(p, "x", n5_x), _vec(p, "y", n5_y)
    Q1d = sp.Matrix(np.round(_rand_indefinite_diag(n5_x, 4.0, -4.0), 3))
    Q2m = sp.Matrix(np.round(_rand_indefinite_matrix(n5_y, 5.0, -3.0), 3))
    obj5 = sp.Trace(x5.T * Q1d * x5) + sp.Trace(y5.T * Q2m * y5)
    p.set_objective(obj5, "min")
    _add_safe_leq(p,
                  sp.Trace(x5.T * x5) + sp.Trace(y5.T * y5),
                  _ub_sum_norm2(p, [x5, y5]))
    p.map_all_terms(); gs.append(p)
    
    return gs

# ===================================================================
# [FR 组] 分式问题（偏向 remove_fraction）
# ===================================================================
def _finetune_fraction() -> List[ps.QCQPProblem]:
    gs = []

    # FR1: 纯双线性 / 线性分母（z 连续）
    p = ps.QCQPProblem("FR1_bilin_over_linear_binary_xy")
    n = 4
    x = _vec(p, "x", n, lb=0, ub=1, vtype="binary")
    y = _vec(p, "y", n, lb=0, ub=1, vtype="binary")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    num = sp.Trace(x.T * y)
    # z_i ∈ [-2,2]，(1/n)*sum z_i ∈ [-2,2]，所以 5 + (1/n)*sum z_i ∈ [3,7] > 0
    e_z = _const_vec([1] * n)
    den = 5 + (1.0 / n) * sp.Trace(e_z.T * z)
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y) + sp.Trace(z.T * z),
                  _ub_sum_norm2(p, [x, y, z]))
    _add_global_linear_couplings(p, [x, y, z], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR2: 不定二次 / 仿射分母 —— x 连续, b integer, 分母含 integer（保持不变，本来就是仿射）
    p = ps.QCQPProblem("FR2_indef_quad_over_affine_with_integer")
    n = 5
    x = _vec(p, "x", n)
    b = _vec(p, "b", n, lb=-2, ub=2, vtype="integer")
    Q = sp.Matrix([[ 2, -1, 0,  1, 0],
                   [-1,-2, 1,  0, 2],
                   [ 0,  1,-1, -2, 0],
                   [ 1,  0,-2,  3,-1],
                   [ 0,  2, 0, -1,-2]])
    num = sp.Trace(x.T * Q * x)
    den = 23 + sp.Trace(b.T * x)      # 仿射 + 常数 > 0
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(b.T * b),
                  _ub_sum_norm2(p, [x, b]))
    _add_global_linear_couplings(p, [x, b], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR3: y^T y / 线性分母（原来是 2 + Trace(x.T*x)）
    p = ps.QCQPProblem("FR3_sum_over_linear_with_binary")
    n = 4
    x = _vec(p, "x", n)
    y = _vec(p, "y", n, lb=0, ub=1, vtype="binary")
    num = sp.Trace(y.T * y)
    # x_i ∈ [-2,2]，(1/n)*sum x_i ∈ [-2,2]，5 + (...) ∈ [3,7] > 0
    e_x = _const_vec([1] * n)
    den = 5 + (1.0 / n) * sp.Trace(e_x.T * x)
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    _add_global_linear_couplings(p, [x, y], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR4: (x−y)^T(x−y) / 线性分母（原来是 1 + Trace(x.T*x)）
    p = ps.QCQPProblem("FR4_diff_square_over_linear_with_integer")
    n = 4
    x = _vec(p, "x", n)
    y = _vec(p, "y", n, lb=-2, ub=2, vtype="integer")
    num = sp.Trace((x - y).T * (x - y))
    # x_i ∈ [-2,2] → (1/n)*sum x_i ∈ [-2,2]，4 + (...) ∈ [2,6] > 0
    e_x = _const_vec([1] * n)
    den = 4 + (1.0 / n) * sp.Trace(e_x.T * x)
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    _add_global_linear_couplings(p, [x, y], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR5: 分母为仿射(binary)，先 relax_integrality 再 remove_fraction（保持不变，本来就是仿射）
    p = ps.QCQPProblem("FR5_bilin_over_affine_with_binary_den")
    n = 5
    x = _vec(p, "x", n, lb=-1, ub=1)
    y = _vec(p, "y", n, lb=0,  ub=1, vtype="binary")
    e = _const_vec([0, 1, 0, -1, 2])
    num = 0.8 * sp.Trace(x.T * y)
    den = 2 + sp.Trace(e.T * y)     # 仿射(binary) + 常数 > 0
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    _add_global_linear_couplings(p, [x, y], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR6: 不定二次 / 线性分母，分母含 binary+continuous 混合
    p = ps.QCQPProblem("FR6_indef_over_linear_with_binary_den")
    n = 4
    x = _vec(p, "x", n)
    y = _vec(p, "y", n, lb=0, ub=1, vtype="binary")
    Q = sp.Matrix([[ 3, -2, 0, 0],
                   [-2, -1, 1, 0],
                   [ 0,  1, 2, 1],
                   [ 0,  0, 1,-2]])
    num = sp.Trace(x.T * Q * x)
    # x_i ∈ [-2,2], y_i ∈ [0,1]
    # (1/n)*sum x_i ∈ [-2,2], (1/n)*sum y_i ∈ [0,1]
    # 6 + (...) ∈ [4,9] > 0
    e_x = _const_vec([1] * n)
    e_y = _const_vec([1] * n)
    den = 6 + (1.0 / n) * (sp.Trace(e_x.T * x) + sp.Trace(e_y.T * y))
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    _add_global_linear_couplings(p, [x, y], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    # FR7: 分式在约束里，线性分母，含 binary
    p = ps.QCQPProblem("FR7_fraction_in_constraint_with_linear_den")
    n = 5
    x = _vec(p, "x", n, lb=-1, ub=1)
    y = _vec(p, "y", n, lb=0,  ub=1, vtype="binary")
    num = sp.Trace(x.T * y)
    # x_i ∈ [-1,1]，(1/n)*sum x_i ∈ [-1,1]，3 + (...) ∈ [2,4] > 0
    e_x = _const_vec([1] * n)
    den = 3 + (1.0 / n) * sp.Trace(e_x.T * x)
    p.set_objective(0.5 * sp.Trace(x.T * x) - 0.4 * sp.Trace(y.T * y), "min")
    lb_lin, _ = _lin_bounds_trace(p, x, e_x)
    rhs = _safe_rhs_ratio(_ub_bilinear_noq(p, x, y), 3 + (1.0 / n) * lb_lin)
    p.add_constraint(num / den, "<=", rhs)
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    _add_global_linear_couplings(p, [x, y], max_cons=2, p_add=0.8)

    p.map_all_terms(); gs.append(p)

    return gs

# ===================================================================
# [NR 组] Neutral / regularizer
# ===================================================================
def _neutral_regularizer() -> List[ps.QCQPProblem]:
    """
    中性/防遗忘子集：
      - NR1: 纯双线性 (x^T y)
      - NR2: 不定二次 (x^T Q x)
      - NR3: 分母为常数的分式 (x^T y / const)
    [产出: 3 个问题]
    """
    gs = []

    # NR1: 纯双线性（无分母）
    p = ps.QCQPProblem(f"NR1_plain_bilinear_regularizer_{random.randint(100,999)}")
    n = 4
    x = _vec(p, "x", n, lb=0, ub=1); y = _vec(p, "y", n, lb=0, ub=1)
    p.set_objective(sp.Trace(x.T * y), "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    p.map_all_terms(); gs.append(p)

    # NR2: 不定二次（无分母）
    p = ps.QCQPProblem(f"NR2_plain_indef_quadratic_regularizer_{random.randint(100,999)}")
    n = 5
    x = _vec(p, "x", n)
    Q = sp.Matrix([[ 2, -1, 0,  1, 0],
                   [-1,-2, 1,  0, 2],
                   [ 0,  1,-1, -2, 0],
                   [ 1,  0,-2,  3,-1],
                   [ 0,  2, 0, -1,-2]])
    p.set_objective(sp.Trace(x.T * Q * x), "min")
    _add_safe_leq(p, sp.Trace(x.T * x), _ub_norm2(p, x))
    p.map_all_terms(); gs.append(p)

    # NR3: 分母为常数（不该触发 remove_fraction）
    p = ps.QCQPProblem(f"NR3_bilin_over_constant_{random.randint(100,999)}")
    n = 3
    x = _vec(p, "x", n); y = _vec(p, "y", n)
    p.set_objective(sp.Trace(x.T * y) / 5, "min")  # 常数分母
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y),
                  _ub_sum_norm2(p, [x, y]))
    p.map_all_terms(); gs.append(p)

    # NR4: 真分式 —— 分子双线性，分母为线性函数（全连续，>0） → 应该用 remove_fraction
    p = ps.QCQPProblem(f"NR4_bilin_over_linear")
    n = 4
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    y = _vec(p, "y", n, lb=-2, ub=2, vtype="continuous")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    num = sp.Trace(x.T * y)
    # z_i ∈ [-2,2] → (1/n)∑ z_i ∈ [-2,2]，5 + (...) ∈ [3,7] > 0
    e_z = _const_vec([1] * n)
    den = 5 + (1.0 / n) * sp.Trace(e_z.T * z)
    p.set_objective(num / den, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(y.T * y) + sp.Trace(z.T * z),
                  _ub_sum_norm2(p, [x, y, z]))
    p.map_all_terms(); gs.append(p)
    
    return gs

# ===================================================================
# Driver: 分式微调数据集（兼容你原来的接口）
# ===================================================================
def create_finetune_vector_fraction(num_repeat: int = 50) -> List[List[ps.QCQPProblem]]:
    """
    生成分式微调数据集：
      - 每次调用 _finetune_fraction() 产出 7 个 FR 问题
      - 每次调用 _neutral_regularizer() 产出 3 个 NR 问题
      - 一次循环一共 10 个问题
      - 重复 num_repeat 次后，一共 10 * num_repeat 个问题

    返回结构：[all_problems]，与当前训练入口期望的单组数据形式一致。
    """
    all_problems: List[ps.QCQPProblem] = []

    for _ in range(num_repeat):
        all_problems.extend(_finetune_fraction())
        all_problems.extend(_neutral_regularizer())

    # 打乱一下顺序，避免同一类连在一起
    random.shuffle(all_problems)

    # 外层套一层 list，保持 [[prob, prob, ...]] 的格式
    return [all_problems]

# ===================================================================
# [INT 组] 专门用于微调 relax_integrality 的问题
# ===================================================================
def _finetune_relax_integrality() -> List[ps.QCQPProblem]:
    """
    目标：
      - 所有表达式在连续意义下都是凸的；
      - 唯一的非凸性来自 binary / integer vtype；
      - 也就是说，最合理的第一步动作就是 relax_integrality。
    """
    gs = []

    # IR1: binary 只出现在目标中，目标 = ||x||^2 + ||b||^2
    p = ps.QCQPProblem(f"IR1_binary_in_obj_norm_{random.randint(100,999)}")
    n = random.randint(4, 8)
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    b = _vec(p, "b", n, lb=0,  ub=1, vtype="binary")
    obj = sp.Trace(x.T * x) + sp.Trace(b.T * b)   # 连续视角下严格凸
    p.set_objective(obj, "min")
    _add_safe_leq(p, sp.Trace(x.T * x), _ub_norm2(p, x))
    p.map_all_terms(); gs.append(p)

    # IR2: binary 出现在约束中，约束 convex，非凸只来自 vtype
    p = ps.QCQPProblem(f"IR2_binary_in_constraint_norm_{random.randint(100,999)}")
    n = random.randint(4, 8)
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    b = _vec(p, "b", n, lb=0,  ub=1, vtype="binary")
    obj = sp.Trace(x.T * x)                       # 凸目标
    p.set_objective(obj, "min")
    # 约束：||x||^2 + ||b||^2 <= c，表达式本身是凸的
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(b.T * b),
                  _ub_sum_norm2(p, [x, b]))
    p.map_all_terms(); gs.append(p)

    # IR3: integer 向量 + 线性目标（离散导致非凸）
    p = ps.QCQPProblem(f"IR3_integer_linear_obj_{random.randint(100,999)}")
    n = random.randint(3, 7)
    z = _vec(p, "z", n, lb=-3, ub=3, vtype="integer")
    c = _const_vec([random.randint(-3, 3) for _ in range(n)])
    obj = sp.Trace(c.T * z)                       # 线性但 z 是 integer
    p.set_objective(obj, "min")
    _add_safe_leq(p, sp.Trace(z.T * z), _ub_norm2(p, z))
    p.map_all_terms(); gs.append(p)

    # IR4: mixed 连续 + integer，目标为 PSD 二次型
    p = ps.QCQPProblem(f"IR4_mixed_cont_integer_psd_{random.randint(100,999)}")
    n_c = random.randint(3, 6)
    n_z = random.randint(3, 6)
    x = _vec(p, "x", n_c, lb=-2, ub=2, vtype="continuous")
    z = _vec(p, "z", n_z, lb=-3, ub=3, vtype="integer")
    # 构造 PSD Qx, Qz
    Qx = sp.Matrix(np.eye(n_c))
    Qz = sp.Matrix(np.eye(n_z))
    obj = sp.Trace(x.T * Qx * x) + sp.Trace(z.T * Qz * z)
    p.set_objective(obj, "min")
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(z.T * z),
                  _ub_sum_norm2(p, [x, z]))
    p.map_all_terms(); gs.append(p)

    # IR5: binary + continuous 线性约束组（典型 MILP / MIQP 场景）
    p = ps.QCQPProblem(f"IR5_mixed_binary_linear_constraints_{random.randint(100,999)}")
    n = random.randint(4, 8)
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    b = _vec(p, "b", n, lb=0,  ub=1, vtype="binary")
    c1 = _const_vec([random.randint(-2, 2) for _ in range(n)])
    c2 = _const_vec([random.randint(-2, 2) for _ in range(n)])
    obj = sp.Trace(x.T * x)                       # 凸目标
    p.set_objective(obj, "min")
    # 线性约束（表达式凸），但有 binary
    _add_safe_leq(p,
                  sp.Trace(c1.T * x) + sp.Trace(c2.T * b),
                  _ub_trace_linear(p, x, c1) + _ub_trace_linear(p, b, c2))
    _add_safe_leq(p,
                  sp.Trace(x.T * x) + sp.Trace(b.T * b),
                  _ub_sum_norm2(p, [x, b]))
    p.map_all_terms(); gs.append(p)

    # IR6: binary 只出现在等式约束中
    p = ps.QCQPProblem(f"IR6_binary_in_equality_{random.randint(100,999)}")
    n = random.randint(3, 6)
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    b = _vec(p, "b", n, lb=0,  ub=1, vtype="binary")
    obj = sp.Trace(x.T * x)
    p.set_objective(obj, "min")
    e = _const_vec([random.randint(-1, 1) for _ in range(n)])
    # 线性等式约束
    lb_x, ub_x = _lin_bounds_trace(p, x, e)
    lb_b, ub_b = _lin_bounds_trace(p, b, e)
    rhs = random.uniform(lb_x + lb_b, ub_x + ub_b)
    p.add_constraint(sp.Trace(e.T * x) + sp.Trace(e.T * b), "=", float(rhs))
    p.map_all_terms(); gs.append(p)

    return gs

def create_finetune_vector_integrality(num_repeat: int = 40,
                                       nr_multiplier: int = 3) -> List[List[ps.QCQPProblem]]:
    """
    nr_multiplier 控制 NR 的放大倍数：
      - =1: 原来的比例（IR:NR ≈ 1:0.5）
      - =2: NR 翻倍（IR:NR ≈ 1:1）
      - =3: NR 三倍（IR:NR ≈ 1:1.5）
    """
    all_problems: List[ps.QCQPProblem] = []

    for _ in range(num_repeat):
        # 正样本：含离散变量，应该选 relax_integrality
        all_problems.extend(_finetune_relax_integrality())
        # 负样本：全连续非凸，选 relax_integrality 是错误的
        for _ in range(nr_multiplier):
            all_problems.extend(_neutral_regularizer())

    random.shuffle(all_problems)
    return [all_problems]



# ===================================================================
# 主 CLI 部分
# ===================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['pure', 'hybrid', 'fraction', 'all', 'integrality']:
        print("Error: Please specify dataset type to generate.")
        print("Usage: python vector_qcqp_all_generators.py [pure|hybrid|fraction|all]")
        sys.exit(1)

    mode = sys.argv[1]
    data_root = OUTPUT_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    all_problems: List[ps.QCQPProblem] = []

    # -----------------------------------------------------------------
    # [阶段 1] "pure" 模式: 只生成纯粹问题 (A组 + B组)
    # -----------------------------------------------------------------
    if mode == 'pure':
        TOTAL_PROBLEMS = 1000
        
        # A 组 (Expand): 40%
        # B 组 (McCormick): 30%
        # B 组 (SDP): 30%
        NUM_A_EXPAND = int(TOTAL_PROBLEMS * 0.4)  # 400
        NUM_B_MC     = int(TOTAL_PROBLEMS * 0.3)  # 300
        NUM_B_SDP    = int(TOTAL_PROBLEMS * 0.3)  # 300

        CALLS_EXPAND = int(np.ceil(NUM_A_EXPAND / EXP_PER_CALL))  # 100
        CALLS_DIRECT = int(np.ceil(NUM_B_MC     / MC_PER_CALL))   # 43
        CALLS_SDP    = int(np.ceil(NUM_B_SDP    / SDP_PER_CALL))  # 60
        CALLS_HYBRID = 0

        fname = data_root / f"vector_finetune_PURE_{TOTAL_PROBLEMS}.pkl"
        print(f"Generating [Phase 1: PURE] dataset...")

    # -----------------------------------------------------------------
    # [阶段 2] "hybrid" 模式: 生成高比例混合问题 (A+B+C组)
    # -----------------------------------------------------------------
    elif mode == 'hybrid':
        TOTAL_PROBLEMS = 1200
        
        # 目标: 600 纯粹 (A+B) + 600 混合 (C)
        NUM_PURE_TOTAL   = int(TOTAL_PROBLEMS * 0.5)  # 600
        NUM_HYBRID_TOTAL = int(TOTAL_PROBLEMS * 0.5)  # 600

        # --- 分配 600 个纯粹问题 (A+B) ---
        NUM_A_EXPAND = int(NUM_PURE_TOTAL * 0.5)  # 300
        NUM_B_SIMPLE = int(NUM_PURE_TOTAL * 0.5)  # 300
        
        # 纯 B 组中再拆 MC / SDP
        NUM_B_SDP = int(NUM_B_SIMPLE * (3/8))     # ~112
        NUM_B_MC  = NUM_B_SIMPLE - NUM_B_SDP      # ~188

        CALLS_EXPAND = int(np.ceil(NUM_A_EXPAND / EXP_PER_CALL))
        CALLS_DIRECT = int(np.ceil(NUM_B_MC  / MC_PER_CALL))
        CALLS_SDP    = int(np.ceil(NUM_B_SDP / SDP_PER_CALL))
        CALLS_HYBRID = int(np.ceil(NUM_HYBRID_TOTAL / HYB_PER_CALL))
        
        fname = data_root / f"vector_finetune_HYBRID_MIX_{TOTAL_PROBLEMS}.pkl"
        print(f"Generating [Phase 2: HYBRID] dataset...")
        
    # -----------------------------------------------------------------
    # [专门连续化微调] "integrality" 模式: 只生成离散变量相关数据集
    # -----------------------------------------------------------------
    elif mode == 'integrality':
        print("Generating [Finetune: INTEGRALITY] dataset...")
        finetune_set = create_finetune_vector_integrality()
        fname = data_root / "vector_finetune_integrality.pkl"
        with open(fname, "wb") as f:
            pickle.dump(finetune_set, f)
        total = sum(len(g) for g in finetune_set)
        print(f"\nSaved {total} problems across {len(finetune_set)} group ➜ {fname}")
        for gi, group in enumerate(finetune_set, 1):
            print(f"\n=== Integrality Finetune Group {gi} ({len(group)} problems) ===")
            for p in group:
                print("•", p.name)
        sys.exit(0)

    # -----------------------------------------------------------------
    # [分式微调] "fraction" 模式: 只生成分母相关数据集
    # -----------------------------------------------------------------
    elif mode == 'fraction':
        print("Generating [Finetune: FRACTION] dataset...")
        finetune_set = create_finetune_vector_fraction()
        fname = data_root / "vector_finetune_fraction.pkl"
        with open(fname, "wb") as f:
            pickle.dump(finetune_set, f)
        total = sum(len(g) for g in finetune_set)
        print(f"\nSaved {total} problems across {len(finetune_set)} group ➜ {fname}")
        for gi, group in enumerate(finetune_set, 1):
            print(f"\n=== Finetune Group {gi} ({len(group)} problems) ===")
            for p in group:
                print("•", p.name)
        sys.exit(0)

    # -----------------------------------------------------------------
    # [全集合] "all" 模式: 一个混合全集
    # -----------------------------------------------------------------
    elif mode == 'all':
        TOTAL_PROBLEMS = 1600
        # 大致比例：
        #  - Expand:     15%
        #  - McCormick:  25%
        #  - SDP:        25%
        #  - Fraction:   30%
        #  - Neutral:     5%
        NUM_A_EXPAND  = int(TOTAL_PROBLEMS * 0.15)  # 240
        NUM_B_MC      = int(TOTAL_PROBLEMS * 0.25)  # 400
        NUM_B_SDP     = int(TOTAL_PROBLEMS * 0.25)  # 400
        NUM_FRACTION  = int(TOTAL_PROBLEMS * 0.30)  # 480
        NUM_NEUTRAL   = TOTAL_PROBLEMS - (NUM_A_EXPAND + NUM_B_MC + NUM_B_SDP + NUM_FRACTION)  # 80

        CALLS_EXPAND  = int(np.ceil(NUM_A_EXPAND  / EXP_PER_CALL))
        CALLS_DIRECT  = int(np.ceil(NUM_B_MC      / MC_PER_CALL))
        CALLS_SDP     = int(np.ceil(NUM_B_SDP     / SDP_PER_CALL))
        CALLS_FRAC    = int(np.ceil(NUM_FRACTION  / FR_PER_CALL))
        CALLS_NEUTRAL = int(np.ceil(NUM_NEUTRAL   / NR_PER_CALL))
        CALLS_HYBRID  = 0  # 全集里就不再单独用 Hybrid 了，避免过多重复模式

        fname = data_root / f"vector_all_mix_{TOTAL_PROBLEMS}.pkl"
        print(f"Generating [ALL-MIX] dataset...")

    # ============ 实际生成（pure / hybrid / all 共用） ============
    print("Generating 'Expand' (Pure A) problems...")
    for _ in range(CALLS_EXPAND):
        all_problems.extend(_e_expand_generator())
        
    print("Generating 'Direct McCormick' (Pure B-MC) problems...")
    for _ in range(CALLS_DIRECT):
        all_problems.extend(_e_direct_mccormick_generator())
    
    print("Generating 'SDP Candidate' (Pure B-SDP) problems...")
    for _ in range(CALLS_SDP):
        all_problems.extend(_e_sdp_candidate_generator())

    if mode in ('hybrid',):
        if CALLS_HYBRID > 0:
            print("Generating 'Hybrid' (Mixed C) problems...")
            for _ in range(CALLS_HYBRID):
                all_problems.extend(_e_hybrid_generator())

    if mode == 'all':
        print("Generating 'Fraction' (FR) problems...")
        for _ in range(CALLS_FRAC):
            all_problems.extend(_finetune_fraction())

        print("Generating 'Neutral' (NR) problems...")
        for _ in range(CALLS_NEUTRAL):
            all_problems.extend(_neutral_regularizer())

    # 3. 打乱顺序
    print(f"\nTotal problems generated (before shuffle): {len(all_problems)}")
    print("Shuffling problems...")
    random.shuffle(all_problems)
    
    # 4. 外层一组 (符合你以前的 [[prob,...]] 格式)
    evalset = [all_problems]

    # 5. 保存
    with open(fname, "wb") as f:
        pickle.dump(evalset, f)

    # 6. 打印总结
    total = len(all_problems)
    print(f"\nSaved {total} total problems ➜ {fname}")
    print("\nSample problem names:")
    for p in all_problems[:10]:
        print("•", p.name)
