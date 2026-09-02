# reward_fn.py
# =========================================================
# 依赖：QCQPProblem（需实现 term_class() / get_convexity_classes()）
# 无额外第三方库

import sympy as sp
from typing import Dict, Tuple
import math
# ----------------------------------------------------------------------
# A. 动作编号 → 名称映射（保持与环境一致）
# ----------------------------------------------------------------------
# ACTION_TYPE = {
#     0: "expand",
#     1: "factor_merge",

#     2: "cancel",

#     3: "expand_log",
#     4: "logcombine",
#     5: "remove_log",

#     6: "relax_integrality",
#     7: "remove_fraction",
#     8: "remove_abs",

#     9: "trace_transformation",

#     10: "mccormick_relaxation",
#     11: "sdp_relaxation",
#     12: "first_order_taylor",
    
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

# ----------------------------------------------------------------------
# B. 复杂度 / 误差表（与动作名对应；保持之前设定）
# ----------------------------------------------------------------------
# ACTION_STATS: Dict[str, Dict[str, float]] = {
#     "expand":               {"complexity": 0.00, "error": 0.00},
#     "cancel":               {"complexity": 0.00, "error": 0.00},
#     "expand_log":           {"complexity": 0.05, "error": 0.00},
#     "logcombine":           {"complexity": 0.05, "error": 0.00},
#     "factor_merge":         {"complexity": 0.00, "error": 0.00},

#     "remove_log":           {"complexity": 0.15, "error": 0.20},
#     "trace_transformation": {"complexity": 0.15, "error": 0.05},
#     "remove_fraction":      {"complexity": 0.01, "error": 0.10},

#     "relax_integrality":    {"complexity": 0.00, "error": 0.25},
#     "remove_abs":           {"complexity": 0.10, "error": 0.20},

#     "mccormick_relaxation": {"complexity": 0.20, "error": 0.20},
#     "first_order_taylor":   {"complexity": 0.10, "error": 0.30},

#     "sdp_relaxation":       {"complexity": 0.10, "error": 0.05},
    
#     # 谱投影：相当于把 Q 投到 PSD 锥，效果接近 SDP，但结构还保持 x^T Q⁺ x 形式
#     "spectral_psd_projection":  {"complexity": 0.08, "error": 0.08},
#     # 对角松弛：只保留 diag(Q)，非常便宜，但松弛很粗
#     "diagonal_relaxation":  {"complexity": 0.03, "error": 0.25},
# }

# 复杂度/误差（建议初值：可再调）
ACTION_STATS = {
    "relax_integrality":    {"complexity": 0.02, "error": 0.25},
    "remove_fraction":      {"complexity": 0.03, "error": 0.10},
    "mccormick_relaxation": {"complexity": 0.18, "error": 0.20},
    "sdp_relaxation":       {"complexity": 0.30, "error": 0.05},
    "qcr":              {"complexity": 0.20, "error": 0.08},
    "bound_tightening":     {"complexity": 0.18, "error": 0.00},
    "global_cut_generation":{"complexity": 0.22, "error": 0.02},
    # # 谱投影：相当于把 Q 投到 PSD 锥，效果接近 SDP，但结构还保持 x^T Q⁺ x 形式
    # "spectral_psd_projection":  {"complexity": 0.08, "error": 0.08},
    # # 对角松弛：只保留 diag(Q)，非常便宜，但松弛很粗
    # "diagonal_relaxation":  {"complexity": 0.03, "error": 0.25},
}

# ----------------------------------------------------------------------
# Reward stability knobs (for smoother training curves)
# ----------------------------------------------------------------------
# Dense progress: reward every step that actually reduces non-convex terms.
DENSE_PROGRESS_COEF = 0.35
# Reduce penalty spikes from no-op actions.
NOOP_PENALTY = 0.45
# Soft-clip final reward to suppress extreme outliers while preserving sign/order.
REWARD_SOFT_CLIP_SCALE = 2.5

# ----------------------------------------------------------------------
# C. 辅助工具函数
# ----------------------------------------------------------------------
def _ensure_term_map(prob, update_problem: bool = False):
    """
    reward 侧只需要读 term map，不应该每步触发 expand/Trace 展开并回写 problem。
    """
    if not getattr(prob, "id_to_item", None) or not getattr(prob, "item_to_id", None):
        prob.map_all_terms(update_problem=update_problem)


def _finite(x):
    return x is not None and x != float("inf") and x != float("-inf")

def _iter_var_bounds(prob):
    # scalar vars
    for name, v in getattr(prob, "variables", {}).items():
        yield ("scalar", name, v.lb, v.ub)
    # vector/matrix vars (你这里是 matrix_variables 统一界)
    for name, mv in getattr(prob, "matrix_variables", {}).items():
        yield ("vector", name, mv.lb, mv.ub)

def _bounds_shrink_score(before_prob, after_prob, eps=1e-12):
    """
    0~1 左右的缩界分数：平均相对缩小比例的截断和
    - old_range 有限时用 (old-new)/old
    - old_range 不有限但 new 引入有限界时给一个小奖励
    """
    # build dict
    b = {(t, n): (lb, ub) for (t, n, lb, ub) in _iter_var_bounds(before_prob)}
    a = {(t, n): (lb, ub) for (t, n, lb, ub) in _iter_var_bounds(after_prob)}

    total = 0.0
    cnt = 0

    keys = set(b.keys()) | set(a.keys())
    for k in keys:
        lb0, ub0 = b.get(k, (None, None))
        lb1, ub1 = a.get(k, (None, None))

        # 只在确实收紧时计分
        tightened = False
        if lb0 is None and lb1 is not None: tightened = True
        if ub0 is None and ub1 is not None: tightened = True
        if lb0 is not None and lb1 is not None and lb1 > lb0: tightened = True
        if ub0 is not None and ub1 is not None and ub1 < ub0: tightened = True
        if not tightened:
            continue

        cnt += 1

        # 相对缩小
        if _finite(lb0) and _finite(ub0) and ub0 > lb0 + eps and _finite(lb1) and _finite(ub1) and ub1 > lb1 + eps:
            r0 = ub0 - lb0
            r1 = ub1 - lb1
            total += max(0.0, min(1.0, (r0 - r1) / max(r0, eps)))
        else:
            # 只有“变得更有限”但不好算区间，就给一个小的固定分
            total += 0.10

    if cnt == 0:
        return 0.0
    return total / cnt

def _constraint_added_score(before_prob, after_prob, cap=30):
    """
    新增约束的 0~1 分：log1p(delta)/log1p(cap)
    """
    b = len(getattr(before_prob, "constraints", [])) + len(getattr(before_prob, "psd_constraints", []))
    a = len(getattr(after_prob, "constraints", [])) + len(getattr(after_prob, "psd_constraints", []))
    d = max(0, a - b)
    if d == 0:
        return 0.0
    d = min(d, cap)
    return math.log1p(d) / math.log1p(cap)

# def _psd_or_cons_changed(before, after) -> bool:
#     """约束/PSD约束有变化则认为问题发生了结构性改变。"""
#     # 普通约束数量或内容变化
#     if len(getattr(before, "constraints", [])) != len(getattr(after, "constraints", [])):
#         return True
#     # 粗略比较内容（可选，注释掉也行）
#     try:
#         if [str(c) for c in before.constraints] != [str(c) for c in after.constraints]:
#             return True
#     except Exception:
#         pass
#     # PSD 约束数量或内容变化
#     if len(getattr(before, "psd_constraints", [])) != len(getattr(after, "psd_constraints", [])):
#         return True
#     try:
#         if [str(c) for c in before.psd_constraints] != [str(c) for c in after.psd_constraints]:
#             return True
#     except Exception:
#         pass
#     return False


def _strip_scalar_mul(expr):
    """
    去掉前面的“纯数系数”，比如：
      -x^T Q x         -> x^T Q x
      2 * x^T Q y      -> x^T Q y
      0.5 * Trace(x^T Q x) -> Trace(x^T Q x)

    只在系数是纯数字 (无 free_symbols) 时剥掉；带符号参数的系数保留。
    """
    import sympy as sp
    if expr is None:
        return None
    e = _unwrap_scalar(expr)

    if isinstance(e, sp.Mul):
        coeff, rest = e.as_coeff_Mul()  # e = coeff * rest
        # coeff 没有符号，说明是纯数字
        if getattr(coeff, "free_symbols", None):
            # 有符号（比如 a * x^T Q x），不剥
            return e
        return rest
    return e


def _mat_factors_of(expr):
    """取 Trace(...) 的内核矩阵因子列表；非 Trace 就看其本身/MatMul。"""
    if isinstance(expr, sp.Trace):
        inner = expr.arg
    else:
        inner = expr
    if isinstance(inner, sp.MatMul):
        return [a for a in inner.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
    return [inner] if isinstance(inner, (sp.MatrixExpr, sp.MatrixBase)) else []

def _is_transpose_pair(a, b):
    return (isinstance(a, sp.Transpose) and a.arg == b) or \
           (isinstance(b, sp.Transpose) and b.arg == a)


def _unwrap_scalar(expr):
    """仅在 1×1 稠密矩阵时返回标量；其余（MatrixBase>1×1、MatrixExpr、Basic）保持原样。"""
    import sympy
    return expr[0, 0] if isinstance(expr, sympy.MatrixBase) and expr.shape == (1, 1) else expr

def _expr_depth(expr) -> int:
    """安全计算树深：MatrixBase>1×1 当作原子，MatrixExpr/Basic照常递归。"""
    import sympy
    expr = _unwrap_scalar(expr)
    # 稠密矩阵（>1×1）当作叶子
    if isinstance(expr, sympy.MatrixBase):
        return 1
    if not isinstance(expr, sympy.Basic):
        return 1
    if not expr.args:
        return 1
    return 1 + max(_expr_depth(a) for a in expr.args)

def _class_of(problem, expr) -> int:
    try:
        return problem.term_class(expr)
    except Exception:
        return 3

def _compute_expr_stats(problem) -> Dict[str, float]:
    """分式/非光滑/乘积/平均树深：保持“块级”，不展开向量/矩阵元素。"""
    import sympy
    problem.map_all_terms()
    terms = problem.get_all_items()

    stats = {
        "n_fraction_terms": 0,
        "n_nonsmooth": 0,
        "n_mul_terms": 0,
        "avg_tree_depth": 0.0,
    }
    depths = []

    for t in terms:
        t0 = _unwrap_scalar(t)

        # 分式/负指数/有理数（仅对 Basic/MatrixExpr 做原子分析）
        has_frac = False
        if isinstance(t0, sympy.Basic):
            # 负指数 x**(-1) 或 a**(-k)
            negpow = any(isinstance(p, sympy.Pow) and p.exp.is_number and float(p.exp) < 0
                         for p in t0.atoms(sympy.Pow))
            rat = any(isinstance(r, sympy.Rational) and r.q != 1
                      for r in t0.atoms(sympy.Rational))
            has_frac = negpow or rat

        # 非光滑
        has_nonsmooth = isinstance(t0, sympy.Basic) and \
                        t0.has(sympy.Abs, sympy.Max, sympy.Min, sympy.sign, sympy.Heaviside)

        # 乘积：标量乘积 Mul 或 矩阵乘积 MatMul 都算“乘积结构”
        has_mul = isinstance(t0, (sympy.Mul, sympy.MatMul))

        if has_frac:     stats["n_fraction_terms"] += 1
        if has_nonsmooth:stats["n_nonsmooth"]      += 1
        if has_mul:      stats["n_mul_terms"]      += 1

        depths.append(_expr_depth(t0))

    stats["avg_tree_depth"] = sum(depths) / len(depths) if depths else 0.0
    return stats

def _diff_new_terms(before_problem, after_problem):
    """返回 after 相对 before 新出现的 **块级** 表达式对象列表（不展开元素）。"""
    _ensure_term_map(before_problem, update_problem=False)
    _ensure_term_map(after_problem,  update_problem=False)

    old_keys = set(before_problem.item_to_id.keys())
    new_keys = [k for k in after_problem.item_to_id.keys() if k not in old_keys]

    new_terms = []
    for k in new_keys:
        try:
            _id = after_problem.item_to_id[k]
            term_obj, _loc = after_problem.id_to_item[_id]
            new_terms.append(_unwrap_scalar(term_obj))
        except Exception:
            # 兜底：key 直接含对象
            if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], sp.Basic):
                new_terms.append(_unwrap_scalar(k[0]))
            else:
                continue
    return new_terms

# ====== 复杂度 & 结构感知：实际增量统计 + 上下文判定 ======
def _count_constraints_total(prob) -> int:
    return len(getattr(prob, "constraints", [])) + len(getattr(prob, "psd_constraints", []))

COST_WEIGHTS = {
    "scalar_var":       0.020,
    "matrix_var_entry": 0.012,
    "aux_var":          0.050,
    "term":             0.010,
    "cons":             0.012,
    "psd_entry":        0.006,
    "psd_cubic":        0.0003,
}

AUX_NAME_PREFIX = ("w_", "W_")   # 你的 RelaxationEngine 生成的辅助变量命名前缀


# 触发阈值（超出 → 认为 McCormick 成本很高）
MCC_HIGH_COST_THRESH = {
    "aux_var": 6,        # 新增辅助变量数 ≥ 6
    "term":    8,        # 新增项数 ≥ 8
}

# SDP 情境奖励 / McCormick 高成本惩罚
BONUS_SDP_CTX   = 0.5    # 仅在“明显该用 SDP”的 old 子式情况下加
PEN_MCC_HIGH    = 0.4    # 仅在“明显高成本/不定二次”且仍选 McCormick 时减

def _count_aux_vars(prob) -> int:
    """根据变量名前缀粗略统计辅助变量数量。"""
    n = 0
    try:
        for name in getattr(prob, "variables", {}):
            if any(name.startswith(p) for p in AUX_NAME_PREFIX):
                n += 1
    except Exception:
        pass
    return n

def _count_terms_blockwise(prob) -> int:
    """以当前 map_all_terms 的块级项计数（不展开元素）。"""
    try:
        # 已有映射就别重复 map（否则每一步 reward 都在重建 term map）
        if not getattr(prob, "id_to_item", None):
            prob.map_all_terms()
        return len(prob.get_all_items())
    except Exception:
        return 0


def _count_scalar_vars(prob) -> int:
    return len(getattr(prob, "variables", {}))


def _count_matrix_var_entries(prob) -> int:
    n = 0
    for mv in getattr(prob, "matrix_variables", {}).values():
        try:
            n += int(mv.rows) * int(mv.cols)
            continue
        except Exception:
            pass
        try:
            shape = getattr(getattr(mv, "symbol", None), "shape", None)
            if shape is not None and len(shape) == 2:
                n += int(shape[0]) * int(shape[1])
        except Exception:
            pass
    return n


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


def _psd_dim_stats(prob) -> Tuple[int, int, int]:
    dims = []
    for psd in getattr(prob, "psd_constraints", []):
        d = _matrix_dim(getattr(psd, "matrix_expr", None))
        if d > 0:
            dims.append(d)
    return len(dims), sum(d * d for d in dims), sum(d * d * d for d in dims)


def problem_size_cost(prob) -> float:
    """
    Lightweight proxy for solver-side size. PSD cones get explicit quadratic and
    cubic terms so an SDP step must earn its lower-bound gain.
    """
    _, psd_entry_sum, psd_cubic_sum = _psd_dim_stats(prob)
    return (
        COST_WEIGHTS["scalar_var"] * _count_scalar_vars(prob)
        + COST_WEIGHTS["matrix_var_entry"] * _count_matrix_var_entries(prob)
        + COST_WEIGHTS["term"] * _count_terms_blockwise(prob)
        + COST_WEIGHTS["cons"] * _count_constraints_total(prob)
        + COST_WEIGHTS["psd_entry"] * psd_entry_sum
        + COST_WEIGHTS["psd_cubic"] * psd_cubic_sum
    )


def _indef_sign_heuristic(M: sp.Matrix) -> bool:
    """
    不做特征值，给个“很安全的”不定迹象启发：
      - 对角线同时出现正/负；或
      - 存在较多正负混合的非零对角 + 明显的非对称/大交叉项
    """
    try:
        if not isinstance(M, sp.MatrixBase):
            return False
        diag = [M[i, i] for i in range(M.shape[0])]
        # 可数值化：只取可以 evalf 的情况
        def _num(v):
            try:
                return float(sp.N(v))
            except Exception:
                return None
        diag_num = [_num(d) for d in diag]
        pos = any((v is not None) and (v >  1e-9) for v in diag_num)
        neg = any((v is not None) and (v < -1e-9) for v in diag_num)
        if pos and neg:
            return True
        # 次优：看非对角的显著交叉（|a_ij| 大且分布广）
        nz_off = 0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i == j: continue
                v = _num(M[i, j])
                if v is not None and abs(v) > 0.5:
                    nz_off += 1
        return nz_off >= max(3, M.shape[0])  # 粗阈值
    except Exception:
        return False

# ----------------------------------------------------------------------
# D. R₁ – 凸松弛进展（严格：old 是 3 级 & new ∈ {0,1,2}）
# ----------------------------------------------------------------------
def convexity_progress(before, after, last_rewrite) -> float:
    """严格晋级：old 整体为 3，且 new 整体进入 {0,1,2}；保持块级，不拆元素。"""
    if not last_rewrite:
        return -0.1
    old_expr = _unwrap_scalar(last_rewrite["old"])
    new_expr = _unwrap_scalar(last_rewrite["new"])
    # print("old_expr:", old_expr)
    # print("new_expr:", new_expr)

    c_before = _class_of(before, old_expr)
    c_after  = _class_of(after,  new_expr)
    # print("class before:", c_before)
    # print("class after :", c_after)

    return 1.0 if (c_before == 3 and c_after in (0, 1, 2)) else -0.1

# ----------------------------------------------------------------------
# E. R₂ – 结构解锁（仅在“没有晋级子式”时才给启发式奖励）
# ----------------------------------------------------------------------
_HEURISTIC_W = {
    "frac_term_reduction": 0.20,
    "tree_depth_reduction": 0.15,
    "nonsmooth_removed":   0.25,
    "term_decomposition":  0.20,
}

def structural_unlock_score(before_problem, after_problem) -> float:
    """若所有“新增块级项”都晋级则 1.0，否则给启发式差分。"""
    before_problem.map_all_terms()
    after_problem.map_all_terms()

    new_terms = _diff_new_terms(before_problem, after_problem)
    if not new_terms:
        return 0.0

    classes = [after_problem.term_class(t) for t in new_terms]
    if classes and all(c in (0, 1, 2) for c in classes):
        return 1.0

    b, a = _compute_expr_stats(before_problem), _compute_expr_stats(after_problem)
    _HEURISTIC_W = {
        "frac_term_reduction": 0.20,
        "tree_depth_reduction": 0.15,
        "nonsmooth_removed":   0.25,
        "term_decomposition":  0.20,
    }
    score = 0.0
    score += _HEURISTIC_W["frac_term_reduction"]  * max(0,   b["n_fraction_terms"] - a["n_fraction_terms"])
    score += _HEURISTIC_W["tree_depth_reduction"] * max(0.0, b["avg_tree_depth"]   - a["avg_tree_depth"])
    score += _HEURISTIC_W["nonsmooth_removed"]    * max(0,   b["n_nonsmooth"]      - a["n_nonsmooth"])
    score += _HEURISTIC_W["term_decomposition"]   * max(0,   b["n_mul_terms"]      - a["n_mul_terms"])
    return score


# ---- 检测是否有“离散 -> 连续”的放松发生 ----
DISCRETE_SET = {"binary", "integer"}

def _collect_all_vtypes(prob) -> dict:
    """
    返回 {name: vtype}，包含标量 variables 和 向量/矩阵 matrix_variables
    """
    d = {}
    try:
        for name, v in getattr(prob, "variables", {}).items():
            d[name] = getattr(v, "vtype", None)
    except Exception:
        pass
    try:
        for name, v in getattr(prob, "matrix_variables", {}).items():
            d[name] = getattr(v, "vtype", None)
    except Exception:
        pass
    return d

def _integrality_relaxed(before, after) -> bool:
    """
    只要同名变量存在 且 (before 是离散, after 不是离散) 就认为发生了放松。
    """
    vb = _collect_all_vtypes(before)
    va = _collect_all_vtypes(after)
    for name in vb.keys() & va.keys():
        if (vb[name] in DISCRETE_SET) and (va[name] not in DISCRETE_SET):
            return True
    return False

# === 新增：检测“这个 old_expr 里是否含离散变量” ===
def _term_has_discrete(expr, prob) -> bool:
    """
    看这个表达式涉及到的符号中，是否有离散变量（integer/binary）。
    假设符号名和 prob 里的 name 对齐即可。
    """
    if expr is None:
        return False
    expr = _unwrap_scalar(expr)
    try:
        vtypes = _collect_all_vtypes(prob)
        sym_names = {s.name for s in getattr(expr, "free_symbols", set())}
        for n in sym_names:
            if n in vtypes and vtypes[n] in DISCRETE_SET:
                return True
    except Exception:
        return False
    return False


def _action_effective(action_id, before, after, last_rewrite) -> bool:
    a_name = ACTION_TYPE.get(int(action_id), "unknown")
    # 1) 表达式有变化
    if last_rewrite and str(last_rewrite.get("old")) != str(last_rewrite.get("new")):
        return True
    # 2) 问题对象有变化（变量域/约束/PSD等）；你实现了 __eq__，直接用
    if before != after:
        return True
    # 3) 特殊动作：即使表达式不变，也算有效
    if a_name in {"relax_integrality"}:
        if _integrality_relaxed(before, after):
            return True
    return False

def _expr_has_symbolic_denominator(expr) -> bool:
    """
    判断一个表达式自身是否存在“分母依赖变量”的分式结构。
    不展开整个问题，只看这一个 old_expr。
    """
    import sympy as sp
    if expr is None:
        return False
    # 把 1×1 稠密矩阵拆掉
    expr = _unwrap_scalar(expr)
    if not isinstance(expr, sp.Basic):
        return False
    try:
        num, den = sp.fraction(sp.simplify(expr))
    except Exception:
        return False
    return hasattr(den, "free_symbols") and len(den.free_symbols) > 0


def _has_symbolic_denominator(prob) -> bool:
    """
    粗略判断：问题中是否存在“分母依赖变量”的分式。
    """
    import sympy as sp
    _ensure_term_map(prob, update_problem=False)

    for t in prob.get_all_items():
        t0 = _unwrap_scalar(t)
        if not isinstance(t0, sp.Basic):
            continue
        try:
            num, den = sp.fraction(t0)
        except Exception:
            continue
        # 分母含有符号变量 → 算作“有分式”
        if hasattr(den, "free_symbols") and len(den.free_symbols) > 0:
            return True
    return False


def _nonconvex_count(prob) -> int:
    """
    Count class-3 (non-convex) terms robustly; fallback to flags if needed.
    """
    try:
        classes = prob.get_convexity_classes()
        return int(sum(1 for c in classes if int(c) == 3))
    except Exception:
        pass
    try:
        flags = prob.get_convexity_flags()
        if hasattr(flags, "tolist"):
            flags = flags.tolist()
        return int(sum(1 for f in flags if not bool(f)))
    except Exception:
        return 0


def _dense_progress_score(before_problem, after_problem) -> float:
    """
    [-1, 1] approximately: positive if non-convex term count decreases.
    """
    nb = _nonconvex_count(before_problem)
    na = _nonconvex_count(after_problem)
    if nb <= 0 and na <= 0:
        return 0.0
    return (nb - na) / max(1, nb)


# ----------------------------------------------------------------------
# F. 主入口：以 action_id 为参数的奖励函数
# ----------------------------------------------------------------------
def get_reward(before_problem,
               after_problem,
               action_id: int,
               last_rewrite=None,
               alpha1: float = 2.0,
               alpha2: float = 1.5,
               beta1: float  = 0.5,
               beta2: float  = 0.5,
               cost_coef: float = 0.2) -> float:
    """
    需要 QCQPProblem 实现：
      - map_all_terms()
      - term_class(term) -> {0,1,2,3}
      - get_convexity_classes() -> list[int]
    """
    if action_id not in ACTION_TYPE:
        raise KeyError(f"[reward_fn] Unknown action_id: {action_id}")

    action_name = ACTION_TYPE[action_id]
    
    # 当前这步操作所在的 old 子式（块级），用于判断是否是“分式 term / 含离散 term”
    old_expr = None
    if last_rewrite and "old" in last_rewrite:
        old_expr = _unwrap_scalar(last_rewrite["old"])

    is_global = bool(action_name in ["bound_tightening", "global_cut_generation"])

    if is_global:
        r1, r2 = 0.0, 0.0
    else:
        r1 = convexity_progress(before_problem, after_problem, last_rewrite)
        if r1 < 0:
            r2 = structural_unlock_score(before_problem, after_problem)
        else:
            r2 = 0.0

    # --- 对全局动作：不让 r1/r2 产生负惩罚（因为 last_rewrite 是哑元，不代表“子式晋级”）---
    if last_rewrite and last_rewrite.get("location") == "GLOBAL":
        r1 = max(0.0, r1)
        r2 = max(0.0, r2)

    # ③ / ④ 惩罚
    c = ACTION_STATS[action_name]["complexity"]
    e = ACTION_STATS[action_name]["error"]
    
    
    # === 实际复杂度增量：新增辅助变量与块级项 ===
    vars_b = _count_scalar_vars(before_problem)
    vars_a = _count_scalar_vars(after_problem)
    mat_entries_b = _count_matrix_var_entries(before_problem)
    mat_entries_a = _count_matrix_var_entries(after_problem)
    aux_b = _count_aux_vars(before_problem)
    aux_a = _count_aux_vars(after_problem)
    terms_b = _count_terms_blockwise(before_problem)
    terms_a = _count_terms_blockwise(after_problem)

    d_aux   = max(0, aux_a   - aux_b)
    d_terms = max(0, terms_a - terms_b)
    d_vars = max(0, vars_a - vars_b)
    d_mat_entries = max(0, mat_entries_a - mat_entries_b)

    cons_b = _count_constraints_total(before_problem)
    cons_a = _count_constraints_total(after_problem)
    d_cons = max(0, cons_a - cons_b)

    _, psd_entries_b, psd_cubic_b = _psd_dim_stats(before_problem)
    _, psd_entries_a, psd_cubic_a = _psd_dim_stats(after_problem)
    d_psd_entries = max(0, psd_entries_a - psd_entries_b)
    d_psd_cubic = max(0, psd_cubic_a - psd_cubic_b)
    
    # 动态复杂度成本（累加到静态 cost 上）
    dyn_cost = (
        COST_WEIGHTS["scalar_var"] * d_vars
        + COST_WEIGHTS["matrix_var_entry"] * d_mat_entries
        + COST_WEIGHTS["aux_var"] * d_aux
        + COST_WEIGHTS["term"] * d_terms
        + COST_WEIGHTS["cons"] * d_cons
        + COST_WEIGHTS["psd_entry"] * d_psd_entries
        + COST_WEIGHTS["psd_cubic"] * d_psd_cubic
    )


    # print("last_rewrite:", last_rewrite)
    # print(f"[reward_fn] Action: {action_name}, r1: {r1:.3f}, r2: {r2:.3f}, c: {c:.3f}, e: {e:.3f}")
    
    # # --- 基础：静态 + r1/r2 ---
    # if action_name in {"sdp_relaxation"} and r1 > 0:
    #     alpha1 = alpha1 * 1.5  # 你原有的放大保留

    dense_progress = _dense_progress_score(before_problem, after_problem)
    reward = (
        alpha1 * r1
        + alpha2 * r2
        + DENSE_PROGRESS_COEF * dense_progress
        - cost_coef * beta1 * (c + dyn_cost)
        - beta2 * e
    )
    
    
    # === 新增：全局 tightening 类动作的效果奖励 ===
    global_eff = None
    if action_name in {"bound_tightening", "global_cut_generation"}:
        s_bounds = _bounds_shrink_score(before_problem, after_problem)
        s_cons   = _constraint_added_score(before_problem, after_problem)

        EPS = 1e-12
        global_eff = (s_bounds > EPS)   # 第一阶段：只认“bounds 真收紧”才算有效

        if action_name == "bound_tightening":
            # 关键：不要把“新增约束”当正收益；tightening 应该主要靠 bounds
            gain = s_bounds
            reward += 0.90 * gain
            if not global_eff:
                reward -= 0.30
            # 如果它居然还加了很多约束，反而应该扣（可选，但很符合你想法）
            if s_cons > 0:
                reward -= 0.10 * s_cons

        else:  # global_cut_generation
            # 仍然保留你原来的 gating：没收紧 bounds，就不认可“加约束”是有效
            s_cons_eff = s_cons if global_eff else 0.0
            gain = 0.85 * s_bounds + 0.15 * s_cons_eff
            reward += 0.55 * gain
            if not global_eff:
                reward -= 0.40

        # 关键：符号兜底 —— 确保“有效为正、无效为负”
        GLOBAL_FLOOR_POS = 0.20
        GLOBAL_FLOOR_NEG = 0.20
        if global_eff:
            reward = max(reward, GLOBAL_FLOOR_POS)
        else:
            reward = min(reward, -GLOBAL_FLOOR_NEG)



    # --- 情境偏置：在“明显该用 PSD 类松弛”的 old 子式时给 bonus/penalty ---
    if last_rewrite and "old" in last_rewrite:
        old_expr = _unwrap_scalar(last_rewrite["old"])

    # --- McCormick 的“高成本”特例惩罚：本步新增太多 ---
    if action_name == "mccormick_relaxation":
        if (d_aux   >= MCC_HIGH_COST_THRESH["aux_var"]) or \
           (d_terms >= MCC_HIGH_COST_THRESH["term"]):
            reward -= PEN_MCC_HIGH

    # === McCormick 保底奖励 ===
    MCC_FLOOR = 0.2   # 建议范围 [0.1, 0.3]
    if ACTION_TYPE.get(action_id) == "mccormick_relaxation":
        # 若动作有效且 reward 过低，则保底
        if _action_effective(action_id, before_problem, after_problem, last_rewrite):
            reward = max(reward, MCC_FLOOR)

    # === 分式存在与否（分母带变量） ===
    has_frac_before = False
    has_frac_after  = False
    if action_name == "remove_fraction":
        has_frac_before = _has_symbolic_denominator(before_problem)
        has_frac_after  = _has_symbolic_denominator(after_problem)

    FRACTION_RESOLVE_BONUS = 0.2   # 这步把“变量分母的分式”全部干掉

    if has_frac_before and (not has_frac_after) and action_name == "remove_fraction":
        # 只有在“原来有分式 + 这一步之后分式没了 + 动作是 remove_fraction”时加一大块奖励
        reward += FRACTION_RESOLVE_BONUS
    

    # === A) 离散→连续 放松兜底 ===
    INTEGRAL_RELAX_BONUS = 0.5   # 0.2~1.0
    if _integrality_relaxed(before_problem, after_problem):
        reward = max(reward, INTEGRAL_RELAX_BONUS)

    # # === B) 解锁型 expand 兜底（关键：让 expand 后至少不为负）===
    # EXPAND_UNLOCK_BONUS = 0.5   # 0.3~0.6
    # if ACTION_TYPE.get(action_id) == "expand" and _expand_unlocked(before_problem, after_problem, last_rewrite):
    #     reward = max(reward, EXPAND_UNLOCK_BONUS)

    # # === C) factor_merge 的解锁兜底 ===
    # FACTOR_MERGE_BONUS = 0.3
    # if ACTION_TYPE.get(action_id) == "factor_merge" and r1 < 0 and _factor_merge_unlocked(before_problem, after_problem, last_rewrite):
    #     reward = max(reward, FACTOR_MERGE_BONUS)

    # 统一的“无效动作”惩罚（集中在这里）
    
    
    # === 针对 remove_fraction / relax_integrality 的局部 shaping ===

    # 这个 term 本身是不是分式（分母带变量）
    has_frac_in_old = _expr_has_symbolic_denominator(old_expr) if old_expr is not None else False
    
    # 这个 term 是否涉及离散变量
    has_discrete_in_old = _term_has_discrete(old_expr, before_problem) if old_expr is not None else False

    # # 4.1 remove_fraction：在“分式 term 上用”加奖励，在非分式 term 上用额外惩罚
    # if action_name == "remove_fraction":
    #     # 正确用在分式 term 上：不要求一次性把全局分式清空，也给一点局部奖励
    #     if has_frac_in_old:
    #         LOCAL_FRAC_TERM_BONUS = 0.3   # 可以在 [0.2, 0.5] 之间调
    #         reward += LOCAL_FRAC_TERM_BONUS
    #     else:
    #         # 在一个根本不是分式的项上乱用 remove_fraction：给更重的惩罚
    #         WRONG_REMOVE_FRAC_PENALTY = 1.2  # 比 NOOP_PENALTY(0.7) 更大一些
    #         reward -= WRONG_REMOVE_FRAC_PENALTY

            
    # if has_frac_in_old and action_name in {
    #     "sdp_relaxation",
    #     "spectral_psd_projection",
    #     "diagonal_relaxation",
    #     "mccormick_relaxation",
    #     # "first_order_taylor",
    # }:
    #     SKIP_FRACTION_PENALTY = 0.3
    #     reward -= SKIP_FRACTION_PENALTY
        
        
    # # === 4.2：跨向量双线性(x^T Q y / x^T y)偏向 McCormick ===
    # if is_cross_bilinear and (not has_discrete_in_old):
    #     if action_name == "mccormick_relaxation":
    #         BILINEAR_MCC_BONUS = 1.2   # 给一块比较明显的奖励
    #         reward += BILINEAR_MCC_BONUS
    #     elif action_name in {
    #         "sdp_relaxation",
    #         "spectral_psd_projection",
    #         "diagonal_relaxation",
    #         # "first_order_taylor",
    #     }:
    #         WRONG_ON_BILINEAR_PENALTY = 0.4
    #         reward -= WRONG_ON_BILINEAR_PENALTY
            
    # # === 规则 4.3：同向量二次型 x^T Q x 尽量不用 McCormick，偏向 SDP/PSD ===
    # if is_samevec_quad and (not has_discrete_in_old):
    #     # 基础偏置：无论 Q 是否明显不定，都偏好 SDP / 谱投影 / 对角
    #     if action_name == "sdp_relaxation":
    #         SDP_ON_SAMEVEC_BONUS = 0.8     # 比较大
    #         reward += SDP_ON_SAMEVEC_BONUS
    #     elif action_name == "spectral_psd_projection":
    #         PSD_PROJ_ON_SAMEVEC_BONUS = 0.4
    #         reward += PSD_PROJ_ON_SAMEVEC_BONUS
    #     elif action_name == "diagonal_relaxation":
    #         DIAG_ON_SAMEVEC_BONUS = 0.2    # 稍微加一点
    #         reward += DIAG_ON_SAMEVEC_BONUS

    #     if action_name == "mccormick_relaxation":
    #         WRONG_MCC_ON_SAMEVEC_PENALTY = 0.6
    #         reward -= WRONG_MCC_ON_SAMEVEC_PENALTY

    # === 4.4：若当前 term 含离散变量，把 relax_integrality 明显抬高，
    #           同时在“尚未放松前”不鼓励用其它连续松弛 ===
    if has_discrete_in_old:
        # 1) 正确：在含离散的 term 上做 relax_integrality，且这一小步确实放松了 vtype
        if action_name == "relax_integrality":
            LOCAL_DISCRETE_TERM_BONUS = 0.4
            reward += LOCAL_DISCRETE_TERM_BONUS
        # 2) 错误：这个 term 还有离散变量，却直接上连续松弛
        elif action_name in {
            "sdp_relaxation",
            "mccormick_relaxation",
            # "spectral_psd_projection",
            # "diagonal_relaxation",
            # "first_order_taylor",
            "remove_fraction",
            "qcr",
        }:
            WRONG_BEFORE_RELAX_INT_PENALTY = 0.5
            reward -= WRONG_BEFORE_RELAX_INT_PENALTY
            
            
    if action_name not in {"bound_tightening", "global_cut_generation"}:
        if not _action_effective(action_id, before_problem, after_problem, last_rewrite):
            reward -= NOOP_PENALTY

    if REWARD_SOFT_CLIP_SCALE > 0:
        reward = REWARD_SOFT_CLIP_SCALE * math.tanh(reward / REWARD_SOFT_CLIP_SCALE)

    # print(f"[reward_fn] Action: {action_name}, reward: {reward:.3f}")
    # print(f"Last rewrite: {last_rewrite}")
    return reward
