# qcqp_to_graph.py
import sympy as sp
import torch
import numpy as np
from torch_geometric.data import HeteroData

from autoconvexrelax.core.problem import QCQPProblem, Variable
from sympy.matrices.expressions.matexpr import MatrixElement

from .visualize import visualize_hetero_graph

TERM_TYPE_SET = [
    'linear', 'quad_diag', 'quad_cross',
    'log', 'log_two_term_sum',
    'abs', 'abs_two_term_sum',
    'sqrt', 'sqrt_two_term_sum',
    'exp', 'exp_two_term_sum',
    'square', 'square_two_term_sum',
    'sin', 'sin_two_term_sum',
    'cos', 'cos_two_term_sum',
    'tan', 'tan_two_term_sum',
    'div', 'div_two_term_sum_num',
    'div_two_term_sum_den', 'div_two_term_sum_both',
    'const', 'other',
    'matrix', 'transpose', 'matmul', 'trace'
]
TERM_TYPE_DICT = {name: i for i, name in enumerate(TERM_TYPE_SET)}
FEATURE_ABS_CLIP = 1e6


def _finite_clip_scalar(x, default=0.0, abs_clip=FEATURE_ABS_CLIP):
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    if v > abs_clip:
        v = abs_clip
    elif v < -abs_clip:
        v = -abs_clip
    return float(v)

def _is_matlike(a):
    return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))

def _base_matrix_symbols(expr):
    """
    收集表达式中的 MatrixSymbol（含 Transpose/MatrixElement 的 parent）
    用于 variable→term 的 uses 边抽取变量名。
    """
    mats = set(expr.atoms(sp.MatrixSymbol))
    for tr in expr.atoms(sp.Transpose):
        if isinstance(tr.arg, sp.MatrixSymbol):
            mats.add(tr.arg)
    for me in expr.atoms(MatrixElement):
        parent = getattr(me, "parent", None) or getattr(me, "base", None) or me.args[0]
        if isinstance(parent, sp.MatrixSymbol):
            mats.add(parent)
    return mats

def _expand_trace_linearity(expr):
    """
    仅对 Trace 做线性展开，并把 Trace(α*M) 的 α 提到外层（变成 α*Trace(M)）。
    不做其它 simplify，避免 SymPy 合并多条 Trace。
    """
    import sympy as sp

    def is_matlike(a):
        return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))

    def rec(e):
        if isinstance(e, sp.Trace):
            inner = rec(e.arg)

            # Trace(A + B + ...) → Trace(A)+Trace(B)+...
            if isinstance(inner, (sp.Add, sp.MatAdd)):
                return sp.Add(*[sp.Trace(rec(a)) for a in inner.args])

            # Trace(α * M(...))：把标量 α 提出来，保留为 α*Trace(core)
            if isinstance(inner, sp.Mul):
                mats = [a for a in inner.args if is_matlike(a)]
                scas = [a for a in inner.args if not is_matlike(a)]
                if scas and mats:
                    alpha = sp.Mul(*scas)
                    core  = mats[0] if len(mats)==1 else sp.MatMul(*mats)
                    return alpha * sp.Trace(rec(core))

            return sp.Trace(inner)

        # 结构递归，避免触发全局化简
        if isinstance(e, (sp.Add, sp.MatAdd, sp.Mul, sp.MatMul, sp.Function, sp.Pow, sp.Transpose)):
            return e.func(*(rec(a) for a in e.args))
        return e

    return rec(expr)
    
def _peel_scalar_from_matmul(expr):
    """
    对 MatrixExpr 的 MatMul: 把数值/标量因子并到 coeff，
    仅保留矩阵/向量因子作为 core（保持原有顺序）。
    例如: MatMul(0.5, x.T, Q, x) → (0.5, MatMul(x.T, Q, x))
    """
    if not isinstance(expr, sp.MatMul):
        return 1.0, expr
    mats, scas = [], []
    for a in expr.args:
        if _is_matlike(a):
            mats.append(a)
        else:
            scas.append(a)
    if not scas:
        return 1.0, expr
    try:
        coeff = float(sp.Mul(*scas))
    except Exception:
        coeff = 1.0
    if len(mats) == 1:
        core = mats[0]
    else:
        core = sp.MatMul(*mats)
    return coeff, core

# === NEW: helpers for vector-aware fraction parsing ===
def _is_scalar_matrixexpr(e: sp.Expr) -> bool:
    """是否为“求值为标量”的矩阵表达式：Trace(...) 或 1x1 MatrixExpr"""
    try:
        if isinstance(e, sp.Trace):
            return True
        if isinstance(e, sp.MatrixExpr) and getattr(e, "shape", None) == (1, 1):
            return True
    except Exception:
        pass
    return False

def _vars_in_expr_including_matrix(expr) -> set:
    names = {s.name for s in expr.free_symbols}
    for ms in _base_matrix_symbols(expr):
        names.add(str(ms.name))          # 关键：Str('x')->"x"
    return {sp.Symbol(n) for n in names} # n 已是 str


# === 在 qcqp_to_graph.py 顶部工具区新增 ===
def _add_reverse_edges(data: HeteroData) -> HeteroData:
    """
    把已有的 (src, rel, dst) 一次性补成 (dst, rel+'_rev', src)。
    - 复用正向边的 edge_attr（若有）
    - 防重复：标记 _has_rev 或检测 *_rev 关系
    """
    if getattr(data, "_has_rev", False):
        return data

    orig_edge_types = [et for et in data.edge_types if not et[1].endswith("_rev")]

    # 暂存所有 edge_attr，方便复用
    edge_attr_store = {
        et: data[et].edge_attr for et in data.edge_types
        if "edge_attr" in data[et]
    }

    for (src, rel, dst) in orig_edge_types:
        rev_key = (dst, f"{rel}_rev", src)
        if rev_key in data.edge_types:
            continue

        # 如果正向边没有任何边，则跳过，避免空张量出错
        if data[(src, rel, dst)].edge_index.numel() == 0:
            continue

        # 反向索引
        data[rev_key].edge_index = data[(src, rel, dst)].edge_index.flip(0)

        # 反向 edge_attr：直接复用（如需物理独立，可 .clone()）
        if (src, rel, dst) in edge_attr_store:
            data[rev_key].edge_attr = edge_attr_store[(src, rel, dst)]

    data._has_rev = True
    return data


# === REPLACE: old decompose_num_den_vars ===
def decompose_num_den_vars(expr: sp.Expr):
    """
    返回 (num_vars, den_vars, has_frac)，向量/矩阵表达式友好。
    会把“标量值的矩阵子表达式”（如 x.T*y、Trace(...)）先替换为 Dummy 标量，再做分式拆分，
    最后把 Dummy 映回其真实变量集合。
    """
    expr = sp.sympify(expr)

    # 1) 找出所有“标量值”的矩阵子表达式，替换成 Dummy
    scalar_subexprs = [e for e in expr.atoms(sp.Basic) if _is_scalar_matrixexpr(e)]
    # 从大到小替换，避免嵌套覆盖
    scalar_subexprs.sort(key=lambda e: len(list(sp.preorder_traversal(e))), reverse=True)

    si_vars, repl = {}, {}
    for idx, e in enumerate(scalar_subexprs):
        si = sp.Dummy(f"_s{idx}", real=True)
        si_vars[si] = _vars_in_expr_including_matrix(e)
        repl[e] = si

    expr_comm = expr.xreplace(repl)

    # 2) 在“可交换世界”里做 together+fraction
    try:
        num, den = sp.fraction(sp.together(expr_comm))
    except Exception:
        return set(), set(), False

    # 3) 把 Dummy 的自由符号映回真实变量集合
    def _expand(symset):
        out = set()
        for s in symset:
            if s in si_vars:
                out |= si_vars[s]
            elif isinstance(s, sp.Symbol):
                out.add(s)
        return out

    num_vars = _expand(set(num.free_symbols))
    den_vars = _expand(set(den.free_symbols))

    has_frac = not (den.is_Number and den == 1) and not (hasattr(den, "is_one") and den.is_one)
    return num_vars, den_vars, has_frac


# ---------- ❶ 递归切项 (最终修正版) ----------
def split_terms(expr, parent_id=None, depth=0, terms=None, max_depth=20,
                parent_is_mul=False, parent_core=None):
    if terms is None:
        terms = {}
    if depth > max_depth:
        return terms

    # --- 新增补丁：在检查 .is_Number 之前，先处理矩阵类型 ---
    if isinstance(expr, sp.MatrixBase):
        # 如果是不含变量的常量矩阵，我们约定忽略它 (与 problem_structure.py 保持一致)
        if not expr.free_symbols:
            return terms
        # 否则，它是一个带变量的矩阵表达式，后面会为它创建'matrix'类型的节点。
        # 这里不 return，让它自然地走到下面的“建当前项”逻辑中去。
    
    # === 常数处理 ===
    # 经过上面的判断，这里的 expr 不可能是 MatrixBase，可以安全调用 .is_Number
    if hasattr(expr, 'is_Number') and expr.is_Number:
        # 我们的约定是忽略所有常数项，所以直接返回
        return terms

    # === 加法：每个加号分出来的是“外层项” ===
    if isinstance(expr, (sp.Add, sp.MatAdd)):
        for arg in expr.args:
            split_terms(arg, parent_id, depth, terms, max_depth,
                        parent_is_mul=parent_is_mul, parent_core=expr)
        return terms

    # ... 函数的其余部分保持不变 ...
    # ... (从 “特判 1/x” 那部分开始) ...

    # === 特判 1/x：如果它是乘法因子里的倒数，则不单独成项，直接下探 base ===
    if isinstance(expr, sp.Pow) and getattr(expr, "exp", None) == -1 and parent_is_mul:
        return split_terms(expr.base, parent_id, depth, terms, max_depth,
                           parent_is_mul=False, parent_core=expr)

    # === 建当前项：剥离外层数值系数（含 MatMul 内部的标量系数） ===
    tid = len(terms)
    coeff, core = 1.0, expr

    # Mul(α, MatrixExpr) 形式：把标量并到 coeff
    if isinstance(expr, sp.Mul) and any(_is_matlike(a) for a in expr.args):
        mats = [a for a in expr.args if _is_matlike(a)]
        scas = [a for a in expr.args if not _is_matlike(a)]
        try:
            coeff = float(sp.Mul(*scas)) if scas else 1.0
        except Exception:
            coeff = 1.0
        core = mats[0] if len(mats) == 1 else expr

    # MatMul 里继续剥内部标量
    if isinstance(core, sp.MatMul):
        c2, core2 = _peel_scalar_from_matmul(core)
        coeff *= c2
        core = core2

    terms[tid] = {"sym": core, "coeff": coeff, "depth": depth, "parent": parent_id}

    # === 递归子结构（务必把 parent_core=core 传下去） ===
    def _recurse(sub, as_mul=False):
        # 同样，在递归前也检查一下，避免对纯数字递归
        is_const_like = (hasattr(sub, 'is_Number') and sub.is_Number) or \
                        (isinstance(sub, sp.MatrixBase) and not sub.free_symbols)
        if isinstance(sub, (sp.Integer, sp.Float)) or is_const_like:
            return
        split_terms(sub, tid, depth + 1, terms, max_depth,
                    parent_is_mul=as_mul, parent_core=core)

    if isinstance(core, sp.Pow):
        base, expo = core.args
        _recurse(base, as_mul=False)
        if not (expo == 2 or expo == -1):
            _recurse(expo, as_mul=False)

    elif isinstance(core, sp.Mul):
        for sub in core.args:
            _recurse(sub, as_mul=True)

    elif isinstance(core, sp.Function):
        for sub in core.args:
            _recurse(sub, as_mul=False)

    elif isinstance(core, sp.Trace):
        _recurse(core.arg, as_mul=False)

    elif isinstance(core, sp.Transpose):
        _recurse(core.arg, as_mul=False)

    elif isinstance(core, sp.MatMul):
        for sub in core.args:
            _recurse(sub, as_mul=True)

    return terms

def is_two_term_sum(expr):
    return isinstance(expr, sp.Add) and len(expr.args) == 2

def get_term_type(expr: sp.Expr) -> str:
    expr = sp.sympify(expr)

    # —— 向量/矩阵优先 —— #
    if isinstance(expr, sp.Trace):      return 'trace'
    if isinstance(expr, sp.Transpose):  return 'transpose'
    if isinstance(expr, sp.MatMul):     return 'matmul'
    if isinstance(expr, sp.MatrixSymbol) or isinstance(expr, sp.MatrixBase):
        return 'matrix'

    # —— 下面是你原来的标量分支 —— #
    if expr.is_number:
        return 'const'
    if expr.is_Symbol:
        return 'linear'

    num, den = sp.fraction(sp.together(expr))
    if not (den.is_Number and den == 1) and not (hasattr(den, "is_one") and den.is_one):
        is_num_two = is_two_term_sum(num)
        is_den_two = is_two_term_sum(den)
        if is_num_two and is_den_two:   return 'div_two_term_sum_both'
        if is_num_two:                  return 'div_two_term_sum_num'
        if is_den_two:                  return 'div_two_term_sum_den'
        return 'div'

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == 2:
            if isinstance(base, sp.Symbol):
                return 'quad_diag'
            elif isinstance(base, sp.Add) and len(base.args) == 2:
                return 'square_two_term_sum'
            else:
                return 'square'
        else:
            return 'other'

    if isinstance(expr, sp.Mul):
        symbols = list(expr.free_symbols)
        if len(symbols) == 1: return 'linear'
        if len(symbols) == 2: return 'quad_cross'
        return 'other'

    if isinstance(expr, sp.Function):
        func = expr.func
        arg = expr.args[0]
        name = func.__name__.lower()
        if name in ['log', 'sqrt', 'abs', 'exp', 'sin', 'cos', 'tan']:
            return f'{name}_two_term_sum' if is_two_term_sum(arg) else name

    return 'other'

# ---------- ① 规范化 & 工具 ----------
# def _sortify(e):
#     if isinstance(e, (sp.Add, sp.Mul, sp.MatAdd, sp.MatMul)):
#         args = tuple(sorted((_sortify(a) for a in e.args), key=sp.default_sort_key))
#         return e.func(*args)
#     if isinstance(e, (sp.Trace, sp.Transpose)):
#         return e.func(_sortify(e.arg))
#     if isinstance(e, sp.Pow):
#         return sp.Pow(_sortify(e.base), _sortify(e.exp))
#     if isinstance(e, sp.Function):
#         return e.func(*(_sortify(a) for a in e.args))
#     return e

# ---------- ① 规范化 & 工具 (修正版) ----------
def _sortify(e):
    """
    递归地规范化表达式，用于创建唯一的键。
    对 Add, Mul, MatAdd (交换操作) 的参数进行排序。
    对 MatMul, Pow, Function (非交换操作) 的参数保持顺序。
    """
    # 1. 递归基例 (叶子节点，如 Symbol, Number)
    if not hasattr(e, 'args') or not e.args:
        return e

    # 2. 交换操作 (Commutative): Add, Mul, MatAdd
    # 递归并排序
    if isinstance(e, (sp.Add, sp.Mul, sp.MatAdd)):
        try:
            # 只对可以排序的参数进行排序
            args = tuple(sorted((_sortify(a) for a in e.args), key=sp.default_sort_key))
            return e.func(*args)
        except Exception: 
            # 如果排序失败（例如混合了无法比较的类型），回退到不排序
            args_unsorted = tuple(_sortify(a) for a in e.args)
            return e.func(*args_unsorted)
    
    # 3. 非交换操作 (Non-commutative): MatMul, Function, Pow, Trace, Transpose 等
    # 只递归，保持参数顺序
    # (Pow, Trace, Transpose 都会落入这个分支)
    if hasattr(e, 'func') and hasattr(e, 'args'):
        args = tuple(_sortify(a) for a in e.args)
        return e.func(*args)
    
    # 4. 兜底 (不应该到这里)
    return e

def canonical_key(expr: sp.Expr) -> str:
    expr = sp.sympify(expr)

    # --- 特判常量矩阵，直接用字符串化结果统一 ---
    if isinstance(expr, sp.MatrixBase) and expr.free_symbols == set():
        return f"MATRIXCONST:{sp.srepr(expr.tolist())}"
    
    try:
        _, core = expr.as_coeff_Mul()
        if _ == 0:
            core = expr
    except Exception:
        core = expr
    core = _sortify(core)
    return sp.srepr(core)


def outer_core_and_sign(expr: sp.Expr):
    """
    给任意表达式 expr，统一抽取：
    - core：去掉纯标量系数后的“核”
    - sign：+1 / -1（系数的符号，无法判断时默认 +1）

    这会用于：
    - 为 outer term 生成唯一 key
    - 计算 top_key 的 sign
    """
    expr = sp.sympify(expr)

    # MatrixExpr 用你之前的 peel 逻辑
    if isinstance(expr, sp.MatMul):
        coeff, core = _peel_scalar_from_matmul(expr)
        try:
            s = 1 if float(coeff) >= 0 else -1
        except Exception:
            s = 1
        return core, s

    # 一般标量：as_coeff_Mul
    try:
        coeff, core = expr.as_coeff_Mul()
        # coeff 可能是 sympy 类型
        try:
            s = 1 if float(coeff) >= 0 else -1
        except Exception:
            s = 1
        # 如果 coeff 为 0，把整个 expr 当成 core
        if coeff == 0:
            core = expr
        return core, s
    except Exception:
        # 兜底：不动 expr，当成正的
        return expr, 1


def find_top_ancestor_tid(terms: dict, tid: int) -> int:
    """沿 parent 往上找到顶层项（parent is None）的 tid。"""
    cur = tid
    while terms[cur]["parent"] is not None:
        cur = terms[cur]["parent"]
    return cur

# # ---------- ② 先收集全局唯一外层项（用于 one-hot 位置） ----------
# def collect_global_top_keys(prob: QCQPProblem):
#     top_keys, seen = [], set()
#     vector_names = set(getattr(prob, "matrix_variables", {}).keys())

#     def _is_decision_vector_matrix(e):
#         return isinstance(e, sp.MatrixSymbol) and e.name in vector_names

#     def _scan(expr):
#         expr = _expand_trace_linearity(expr)
#         tdict = split_terms(expr)
#         for tid, info in tdict.items():
#             if info["parent"] is None:
#                 core = info["sym"]
#                 if _is_decision_vector_matrix(core):
#                     continue  # 忽略 x / y 本体
#                 sign = 1 if info["coeff"] >= 0 else -1
#                 key = (canonical_key(core), sign)
#                 if key not in seen:
#                     seen.add(key)
#                     top_keys.append(key)

#     _scan(prob.obj_expr)
#     for c in prob.constraints:
#         _scan(c.expr)
#     return top_keys

def collect_global_top_keys(prob: QCQPProblem):
    """
    基于 prob.id_to_item 收集“外层项”的全局唯一 key 列表。
    每个 key = (canonical_key(core), sign)，
    其中 core / sign 由 outer_core_and_sign 提取。
    """
    top_keys, seen = [], set()

    for term_id in sorted(prob.id_to_item.keys()):
        term_expr, loc = prob.id_to_item[term_id]

        core, sign = outer_core_and_sign(term_expr)
        key = (canonical_key(core), sign)

        if key not in seen:
            seen.add(key)
            top_keys.append(key)

    return top_keys


# # ---------- ③ 构建节点 & 边（全局去重） ----------
# def qcqp_to_heterodata(prob: QCQPProblem):
#     data = HeteroData()

#     # variable 节点
#     var2id = {}
#     v_feats = []
#     outer_lookup = {}  # (srepr(term), sign) -> gid

#     vector_names = set(getattr(prob, "matrix_variables", {}).keys())

#     def is_decision_vector_matrix(expr):
#         return isinstance(expr, sp.MatrixSymbol) and expr.name in vector_names

    
#     # ===== 在 qcqp_to_heterodata(prob) 内，var2id/v_feats 建好之后，添加 =====
#     var_matrix_symbols = set()
#     if hasattr(prob, "matrix_variables"):
#         for mv in prob.matrix_variables.values():
#             # 只有向量变量（n×1）当作“变量”，不作为 matrix term
#             if getattr(mv, "cols", 1) == 1:
#                 var_matrix_symbols.add(mv.symbol)

#     # 标量变量
#     for i, (name, v) in enumerate(prob.variables.items()):
#         var2id[name] = len(var2id)
#         lb = -1e9 if v.lb is None else v.lb
#         ub =  1e9 if v.ub is None else v.ub
#         v_feats.append([
#             lb, ub,
#             1 if v.vtype=='continuous' else 0,
#             1 if v.vtype=='integer' else 0,
#             1 if v.vtype=='binary'  else 0,
#         ])

#     # 向量/矩阵变量（整体作为一个“变量”参与 uses）
#     if hasattr(prob, "matrix_variables"):
#         for name, mv in prob.matrix_variables.items():
#             if name in var2id:  # 名字冲突则跳过或改名，这里直接跳过
#                 continue
#             var2id[name] = len(var2id)
#             lb = -1e9 if getattr(mv, "lb", None) is None else mv.lb
#             ub =  1e9 if getattr(mv, "ub", None) is None else mv.ub
#             vtype = getattr(mv, "vtype", "continuous")
#             v_feats.append([
#                 lb, ub,
#                 1 if vtype=='continuous' else 0,
#                 1 if vtype=='integer' else 0,
#                 1 if vtype=='binary'  else 0,
#             ])

#     data["variable"].x = torch.tensor(v_feats, dtype=torch.float) if v_feats else torch.zeros((0,5), dtype=torch.float)

#     # 约束/目标 节点
#     c_feats = []
#     sens_map = {'<=':[1,0,0], '=':[0,1,0], '>=':[0,0,1]}
#     # c-1（目标）
#     c_feats.append([0,0,0] + [0.0] + [1])  # sense, rhs, is_objective
#     # c1..cm（约束）
#     for constr in prob.constraints:
#         # print("constr is ", constr)
#         # print("constr.expr is ", constr.expr)
#         # print("constr.sense is ", constr.sense)
#         # print("constr.rhs is ", constr.rhs)
#         if constr.rhs is None:
#             rhs = 0.0
#         elif hasattr(constr.rhs, 'evalf'):  # 首先检查它是否有 .evalf 方法（说明是SymPy对象）
#             print("Warning: constr.rhs is a SymPy expression, evalf() applied.")
#             print("  constr =", constr)
#             rhs = float(constr.rhs.evalf())
#         else:  # 如果没有，就认为它是一个普通的Python数字
#             rhs = float(constr.rhs)
#         sense = sens_map.get(constr.sense, [0,0,0])
#         c_feats.append(sense + [rhs] + [0])
#     data["constraint"].x = torch.tensor(c_feats, dtype=torch.float)

#     # --- 全局外层项索引（one-hot 的长度 & 位置） ---
#     top_keys = collect_global_top_keys(prob)
#     top_key2idx = {k:i for i,k in enumerate(top_keys)}
#     num_top_terms = len(top_keys)

#     # --- term 节点的全局去重表 ---
#     term_key2id = {}            # key → gid
#     term_feats  = []            # [type_onehot..., sign_onehot, depth(min)]
#     outer_gid_by_id = []        # 按 “环境 term_id=1..n” 顺序存放的 gid

#     def get_or_create_term(core, coeff, depth, is_outer: bool):
#         if is_decision_vector_matrix(core):
#             return None
#         sign = 1 if coeff >= 0 else -1
#         basekey = canonical_key(core)
#         key = ('OUTER', basekey, sign) if is_outer else ('NESTED', basekey)

#         gid = term_key2id.get(key)
#         if gid is None:
#             gid = len(term_key2id)
#             term_key2id[key] = gid
#             term_type = get_term_type(core)
#             t_id = TERM_TYPE_DICT.get(term_type, TERM_TYPE_DICT['other'])
#             type_hot = [0]*len(TERM_TYPE_SET); type_hot[t_id] = 1
#             sign_hot = [1] if sign > 0 else [0]
#             term_feats.append(type_hot + sign_hot + [depth])
#         else:
#             term_feats[gid][-1] = min(term_feats[gid][-1], depth)
#         return gid


#     # --- 边容器（集合去重） ---
#     edges_v_t = []         # 允许多重边（不同 one-hot）
#     edges_v_t_attr = []
#     edges_t_t_set = set()  # 去重
#     edges_t_c_set = set()

#     def process_expr(expr, cid_store):
#         """
#         cid_store: -1 表示目标（与 c_feats 对齐：c-1 在 index 0）
#                    >=1 表示第 cid_store 个约束（与 c_feats 对齐：c1..cm 在 1..m）
#         """
#         expr = _expand_trace_linearity(expr)
#         terms = split_terms(expr)
#         local_tid2gid = {}

#         # 先给所有项分配全局 gid（不带 loc_label）
#         for tid, info in terms.items():
#             is_outer = (info["parent"] is None)
#             gid = get_or_create_term(info["sym"], info["coeff"], info["depth"], is_outer=is_outer)
#             local_tid2gid[tid] = gid

#         # --- 再建边 ---
#         for tid, info in terms.items():
#             gid = local_tid2gid.get(tid)
#             if gid is None:
#                 # 被跳过的向量变量 leaf（x 或 y）→ 不建 term、也没有 nested/in 边
#                 continue

#             core = info["sym"]

#             # -------- term -> constraint (只给外层项连 in 边) --------
#             if info["parent"] is None:
#                 c_index = 0 if cid_store == -1 else cid_store
#                 edges_t_c_set.add((gid, c_index))
#                 outer_gid_by_id.append(gid)   # ← 关键：不要去重，按出现顺序逐个追加


#             # -------- term -> term (nested) ：子 -> 父 --------
#             if info["parent"] is not None:
#                 pid_gid = local_tid2gid.get(info["parent"])
#                 if pid_gid is not None:
#                     edges_t_t_set.add((gid, pid_gid))

#             # =========================================================
#             #  变量 -> 项 (uses) ：所有项（外层+嵌套）都要连！
#             #  - 分子/分母：从“顶层祖先项”的表达式做一次 fraction 判定
#             #  - 顶层项 one-hot：固定 MAX_TOP_TERMS=10
#             # =========================================================
#             top_tid  = find_top_ancestor_tid(terms, tid)
#             top_info = terms[top_tid]
#             top_core = top_info["sym"]

#             # 顶层项 one-hot（固定长度10）
#             MAX_TOP_TERMS = 10
#             top_key = (canonical_key(top_core), 1 if top_info["coeff"] >= 0 else -1)
#             if top_key not in top_key2idx:
#                 top_key2idx[top_key] = len(top_key2idx)
#             top_idx = top_key2idx[top_key]
#             term_one_hot = [0] * MAX_TOP_TERMS
#             if top_idx < MAX_TOP_TERMS:
#                 term_one_hot[top_idx] = 1

#             # 参与该 term 的变量名（标量符号 + MatrixSymbol）
#             names = {s.name for s in core.free_symbols}
#             for ms in _base_matrix_symbols(core):
#                 names.add(ms.name)

#             # 用“顶层祖先项”做分子/分母角色判定（这样嵌套项也能带上对的角色）
#             # 注意：如果顶层项是矩阵表达式（少见），就不打分子/分母标签
#             if _is_matlike(top_core):
#                 has_frac = False
#                 num_vars = den_vars = set()
#             else:
#                 num_vars, den_vars, has_frac = decompose_num_den_vars(top_core)

#             for vname in names:
#                 if vname not in var2id:
#                     continue  # 非决策变量（比如常量矩阵 Q）
#                 var_id = var2id[vname]

#                 is_num = 1 if (has_frac and sp.Symbol(vname) in num_vars) else 0
#                 is_den = 1 if (has_frac and sp.Symbol(vname) in den_vars) else 0
#                 role_two_hot = [is_num, is_den]

#                 edge_attr = role_two_hot + term_one_hot
#                 edges_v_t.append([var_id, gid])
#                 edges_v_t_attr.append(edge_attr)

#     # 目标
#     process_expr(prob.obj_expr, cid_store=-1)
#     # 约束
#     for cid, constr in enumerate(prob.constraints, start=1):
#         process_expr(constr.expr, cid_store=cid)
        
#     # --- >>> 新增代码：显式注入变量类型特征 <<< ---
#     # 为所有 term 节点计算并附加“显式变量类型”特征
#     # 1. 初始化一个全零张量，形状为 [总term数, 3] (对应 C, I, B)
#     num_total_terms = len(term_feats)
#     explicit_vtype_features = torch.zeros((num_total_terms, 3), dtype=torch.float)
    
#     # 2. 创建 GID -> SymPy 表达式核心 的反向映射
#     gid_to_core_expr = {}
#     for key, gid in term_key2id.items():
#         # key[0] is 'OUTER' or 'NESTED', key[1] is the srepr string
#         gid_to_core_expr[gid] = sp.sympify(key[1])

#     # 3. 遍历所有“外层项”的 GID (outer_gid_by_id 是按顺序记录的，正好对应环境的 term_id 顺序)
#     for gid in outer_gid_by_id:
#         core_expr = gid_to_core_expr.get(gid)
#         if core_expr is None:
#             continue
        
#         # 4. 找到该项使用的所有变量名
#         names = {s.name for s in core_expr.free_symbols}
#         for ms in _base_matrix_symbols(core_expr):
#             names.add(ms.name)
        
#         # 5. 查找这些变量的类型
#         has_continuous = 0
#         has_integer = 0
#         has_binary = 0
#         for vname in names:
#             v_info = prob.variables.get(vname) or prob.matrix_variables.get(vname)
#             if v_info is None:
#                 continue
            
#             vtype = getattr(v_info, "vtype", "continuous")
#             if vtype == 'continuous':
#                 has_continuous = 1
#             elif vtype == 'integer':
#                 has_integer = 1
#             elif vtype == 'binary':
#                 has_binary = 1
        
#         # 6. 在对应 GID 的位置填入特征
#         explicit_vtype_features[gid, 0] = has_continuous
#         explicit_vtype_features[gid, 1] = has_integer
#         explicit_vtype_features[gid, 2] = has_binary
#     # --- >>> 新增代码结束 <<< ---

#     # 写入 HeteroData
#     data["term"].x = torch.tensor(term_feats, dtype=torch.float)
#     data["term"].outer_index = torch.tensor(outer_gid_by_id, dtype=torch.long)
#     data["term"].explicit_vtype = explicit_vtype_features # <-- 将新特征存入图中


#     def to_edge(idx):
#         return torch.tensor(idx, dtype=torch.long).t().contiguous()

#     data["variable", "uses", "term"].edge_index = to_edge(edges_v_t)
#     data["variable", "uses", "term"].edge_attr  = torch.tensor(edges_v_t_attr, dtype=torch.float)
#     data["term", "nested", "term"].edge_index   = to_edge(list(edges_t_t_set))
#     data["term", "in", "constraint"].edge_index = to_edge(list(edges_t_c_set))
    
#     _add_reverse_edges(data)
    
    
#     # ======================= DEBUG CODE START =======================
#     # 在函数返回前，直接在此处进行根本性对齐检查，信息更全
#     import sympy

#     # 1. 从环境侧 (QCQPProblem.map_all_terms) 获取项列表
#     env_terms_info = []
#     # prob.id_to_item 的键是从 1 开始的 term_id
#     for term_id in sorted(prob.id_to_item.keys()):
#         term_expr, location = prob.id_to_item[term_id]
#         env_terms_info.append({'id': term_id, 'expr': term_expr, 'loc': location})

#     # 2. 从图构建侧 (split_terms) 获取外层项列表
#     # 需要利用 term_key2id 将 outer_gid_by_id 里的 gid 映射回表达式
#     gid_to_key = {gid: key for key, gid in term_key2id.items()}
#     graph_outer_terms_info = []
#     for gid in outer_gid_by_id:
#         # key 的格式是 ('OUTER', basekey, sign)
#         key = gid_to_key.get(gid)
#         if key and key[0] == 'OUTER':
#             srepr_str = key[1]
#             sign = key[2]
#             # srepr 字符串可以通过 sympify 转回 sympy 表达式
#             core_expr = sympy.sympify(srepr_str)
#             graph_outer_terms_info.append({'gid': gid, 'expr': core_expr, 'sign': sign})

#     # 3. 对比并打印详细信息
#     if len(env_terms_info) != len(graph_outer_terms_info):
#         print("\n" + "="*90)
#         print("!!! DEBUGGING: Mismatch found in `qcqp_to_heterodata` !!!")
#         print(f"  - Environment (`map_all_terms`) found {len(env_terms_info)} terms.")
#         print(f"  - Graph builder (`split_terms`) found {len(graph_outer_terms_info)} outer terms.")
#         print("  - This is the root cause of the error in StateRepresentation.")
#         print("="*90)

#         # 并排打印，直到较短的列表结束
#         n_min = min(len(env_terms_info), len(graph_outer_terms_info))
#         print(f"{'#':<4}| {'ENVIRONMENT (`problem.id_to_item`)':<70} | {'GRAPH (`outer_gid_by_id`)'}")
#         print(f"{'-'*4}|{'-'*72}|{'-'*72}")

#         for i in range(n_min):
#             env_t = env_terms_info[i]
#             gra_t = graph_outer_terms_info[i]
#             env_str = f"ID={env_t['id']:<3} | Loc: {env_t['loc']:<20} | Expr: {env_t['expr']}"
#             gra_str = f"GID={gra_t['gid']:<3} | Expr: {gra_t['sign']} * ({gra_t['expr']})"
#             print(f"{i:<4}| {env_str:<70} | {gra_str}")

#         # 打印较长列表的剩余部分
#         if len(env_terms_info) > n_min:
#             print("\n--- Remaining Environment Terms (unmatched): ---")
#             for i in range(n_min, len(env_terms_info)):
#                 env_t = env_terms_info[i]
#                 print(f"{i:<4}| ID={env_t['id']:<3} | Loc: {env_t['loc']:<20} | Expr: {env_t['expr']}")

#         if len(graph_outer_terms_info) > n_min:
#             print("\n--- Remaining Graph Terms (unmatched): ---")
#             for i in range(n_min, len(graph_outer_terms_info)):
#                 gra_t = graph_outer_terms_info[i]
#                 print(f"{i:<4}| GID={gra_t['gid']:<3} | Expr: {gra_t['sign']} * ({gra_t['expr']})")

#         print("="*90 + "\n")

#         # 直接在此处抛出错误，中断执行
#         raise RuntimeError(
#             f"[qcqp_to_heterodata] Alignment failed: "
#             f"env_terms(n={len(env_terms_info)}) != graph_outer_terms(n={len(graph_outer_terms_info)}). "
#             f"Check parsing logic between `QCQPProblem._split_or_add` and `split_terms`."
#         )
#     # ======================== DEBUG CODE END ========================

#     return data

def qcqp_to_heterodata(prob: QCQPProblem):
    """
    用 QCQPProblem.id_to_item 驱动建图：
    - 外层 term = prob.id_to_item 中的每一项 (expr, loc)
    - 再对每个 expr 本地跑一次 split_terms(expr)，补 nested / uses 边
    """
    data = HeteroData()

    # =========================================================
    # 1. variable 节点（原逻辑基本不动）
    # =========================================================
    var2id = {}
    v_feats = []

    # 标量变量
    for i, (name, v) in enumerate(prob.variables.items()):
        var2id[name] = len(var2id)
        lb = -1e9 if v.lb is None else v.lb
        ub =  1e9 if v.ub is None else v.ub
        lb = _finite_clip_scalar(lb, default=-FEATURE_ABS_CLIP)
        ub = _finite_clip_scalar(ub, default=FEATURE_ABS_CLIP)
        v_feats.append([
            lb, ub,
            1 if v.vtype=='continuous' else 0,
            1 if v.vtype=='integer'    else 0,
            1 if v.vtype=='binary'     else 0,
        ])

    # 向量/矩阵变量整体也当成“变量节点”，方便 uses 边
    if hasattr(prob, "matrix_variables"):
        def _scalarize_bounds(b, default):
            """Convert vector/matrix bounds to a scalar (min/max) for graph features."""
            if b is None:
                return default
            # numeric scalar
            if isinstance(b, (int, float, np.number)):
                return float(b)
            # list/tuple/np array
            try:
                arr = np.array(b, dtype=float).reshape(-1)
            except Exception:
                return default
            if arr.size == 0:
                return default
            # ignore NaN/inf
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return default
            return float(np.min(arr)), float(np.max(arr))

        for name, mv in prob.matrix_variables.items():
            if name in var2id:
                continue
            var2id[name] = len(var2id)
            lb_raw = getattr(mv, "lb", None)
            ub_raw = getattr(mv, "ub", None)
            lb = -1e9 if lb_raw is None else lb_raw
            ub =  1e9 if ub_raw is None else ub_raw
            # If lb/ub are vectors/matrices, compress to scalar bounds.
            lb_val = _scalarize_bounds(lb, -1e9)
            ub_val = _scalarize_bounds(ub, 1e9)
            if isinstance(lb_val, tuple):
                lb_val = lb_val[0]
            if isinstance(ub_val, tuple):
                ub_val = ub_val[1]
            lb_val = _finite_clip_scalar(lb_val, default=-FEATURE_ABS_CLIP)
            ub_val = _finite_clip_scalar(ub_val, default=FEATURE_ABS_CLIP)
            vtype = getattr(mv, "vtype", "continuous")
            v_feats.append([
                lb_val, ub_val,
                1 if vtype=='continuous' else 0,
                1 if vtype=='integer'    else 0,
                1 if vtype=='binary'     else 0,
            ])

    data["variable"].x = torch.tensor(v_feats, dtype=torch.float) if v_feats else torch.zeros((0,5), dtype=torch.float)

    # =========================================================
    # 2. constraint / objective 节点（沿用原逻辑）
    #    index=0: objective, index=1..m: constraints
    # =========================================================
    c_feats = []
    sens_map = {'<=':[1,0,0], '=':[0,1,0], '>=':[0,0,1]}

    # c-1（目标）
    c_feats.append([0,0,0] + [0.0] + [1])  # sense, rhs, is_objective

    # c1..cm（约束）
    for constr in prob.constraints:
        if constr.rhs is None:
            rhs = 0.0
        elif hasattr(constr.rhs, 'evalf'):
            # Sympy 表达式
            print("Warning: constr.rhs is a SymPy expression, evalf() applied.")
            print("  constr =", constr)
            rhs = float(constr.rhs.evalf())
        else:
            rhs = float(constr.rhs)
        rhs = _finite_clip_scalar(rhs, default=0.0)
        sense = sens_map.get(constr.sense, [0,0,0])
        c_feats.append(sense + [rhs] + [0])

    data["constraint"].x = torch.tensor(c_feats, dtype=torch.float)

    # =========================================================
    # 3. term 节点：全局去重 + 显式变量类型特征
    # =========================================================
    # 3.1 先准备 top-term one-hot 的 key 空间（基于 id_to_item）
    top_keys = collect_global_top_keys(prob)
    top_key2idx = {k: i for i, k in enumerate(top_keys)}
    MAX_TOP_TERMS = 10

    term_key2id = {}    # ('OUTER', core_key, sign) / ('NESTED', core_key) -> gid
    term_feats  = []    # 每个 term 节点的特征向量
    outer_gid_by_id = []  # env term_id 顺序 → term gid
    gid_to_core_expr = {}  # gid -> 原始 core 表达式对象（不要 sympify）


    def get_or_create_term(core, sign, depth, is_outer: bool):
        """
        core: 抽掉标量系数后的表达式核
        sign: +1 / -1
        depth: 在 split_terms 中的最小深度
        is_outer: 外层 term 用 ('OUTER', key, sign)，嵌套项用 ('NESTED', key)
        """
        basekey = canonical_key(core)
        key = ('OUTER', basekey, sign) if is_outer else ('NESTED', basekey)

        gid = term_key2id.get(key)
        if gid is None:
            gid = len(term_key2id)
            term_key2id[key] = gid

            term_type = get_term_type(core)
            t_id = TERM_TYPE_DICT.get(term_type, TERM_TYPE_DICT['other'])
            type_hot = [0]*len(TERM_TYPE_SET)
            type_hot[t_id] = 1

            sign_hot = [1] if sign > 0 else [0]
            term_feats.append(type_hot + sign_hot + [depth])
            
            gid_to_core_expr[gid] = core

        else:
            # depth 取最小
            term_feats[gid][-1] = min(term_feats[gid][-1], depth)

        return gid

    # =========================================================
    # 4. 构造 term 节点 + 边
    #    —— 关键：外层项用 prob.id_to_item 驱动
    # =========================================================
    edges_v_t = []
    edges_v_t_attr = []
    edges_t_t_set = set()
    edges_t_c_set = set()

    def loc_to_constraint_index(loc: str) -> int:
        """
        根据 prob.id_to_item 保存的 loc 字符串：
        - 'Objective'            -> 0
        - 'Constraint_i_LHS/RHS' -> i
        """
        if loc == "Objective":
            return 0
        if loc.startswith("Constraint_"):
            parts = loc.split("_")
            # 'Constraint', 'i', 'LHS/RHS'
            try:
                idx = int(parts[1])
                return idx
            except Exception:
                pass
        # 兜底：当成目标
        return 0

    for term_id in sorted(prob.id_to_item.keys()):
        term_expr, loc = prob.id_to_item[term_id]
        expr = sp.sympify(term_expr)

        # --- 对这个 outer term 跑一次 split_terms，拿到局部树结构 ---
        terms = split_terms(expr)
        local_tid2gid = {}

        # split_terms 可能因过滤规则返回空；为保证 outer 项与环境 term_id 对齐，这里补一个 outer 节点。
        if not terms:
            core, sign = outer_core_and_sign(expr)
            gid = get_or_create_term(core, sign, depth=0, is_outer=True)
            if gid is not None:
                outer_gid_by_id.append(gid)
                c_index = loc_to_constraint_index(loc)
                edges_t_c_set.add((gid, c_index))
            continue

        # 1) 先给每个局部 tid 分配全局 term gid
        for tid, info in terms.items():
            core = info["sym"]
            coeff = info["coeff"]
            depth = info["depth"]
            is_outer_local = (info["parent"] is None)

            # 对外层 root 来说，coeff 就是我们要的 sign；
            # 对 nested，sign 其实影响不大（反正 key 用 'NESTED'）
            sign = 1 if coeff >= 0 else -1

            gid = get_or_create_term(core, sign, depth, is_outer=is_outer_local)
            local_tid2gid[tid] = gid

            if is_outer_local and gid is not None:
                # 记录：环境 term_id 对应哪个 gid
                outer_gid_by_id.append(gid)

                # term -> constraint 边
                c_index = loc_to_constraint_index(loc)
                edges_t_c_set.add((gid, c_index))

        # 2) 再建 nested / uses 边
        for tid, info in terms.items():
            gid = local_tid2gid.get(tid)
            if gid is None:
                continue

            core = info["sym"]

            # -------- term -> term (nested)：子 → 父 --------
            if info["parent"] is not None:
                pid_gid = local_tid2gid.get(info["parent"])
                if pid_gid is not None:
                    edges_t_t_set.add((gid, pid_gid))

            # -------- variable -> term (uses) --------
            # 顶层祖先项（在 *这个 outer term* 的局部树里）
            top_tid  = find_top_ancestor_tid(terms, tid)
            top_info = terms[top_tid]
            top_core = top_info["sym"]
            top_sign = 1 if top_info["coeff"] >= 0 else -1

            # 4.1 top-term one-hot（长度固定 MAX_TOP_TERMS）
            top_key = (canonical_key(top_core), top_sign)
            # 若 top_key 没出现在 collect_global_top_keys 里，则补充一个 index
            if top_key not in top_key2idx:
                top_key2idx[top_key] = len(top_key2idx)
            top_idx = top_key2idx[top_key]

            term_one_hot = [0] * MAX_TOP_TERMS
            if top_idx < MAX_TOP_TERMS:
                term_one_hot[top_idx] = 1

            # 4.2 分子/分母角色（看 top_core 是否是分式）
            if _is_matlike(top_core):
                has_frac = False
                num_vars = den_vars = set()
            else:
                num_vars, den_vars, has_frac = decompose_num_den_vars(top_core)

            # 4.3 这个 core 中出现的变量名（标量 + MatrixSymbol）
            names = {str(s.name) for s in core.free_symbols}
            for ms in _base_matrix_symbols(core):
                names.add(str(ms.name))


            for vname in names:
                if vname not in var2id:
                    continue  # 常量矩阵 Q 等

                var_id = var2id[vname]
                is_num = 1 if (has_frac and sp.Symbol(vname) in num_vars) else 0
                is_den = 1 if (has_frac and sp.Symbol(vname) in den_vars) else 0
                role_two_hot = [is_num, is_den]

                edge_attr = role_two_hot + term_one_hot
                edges_v_t.append([var_id, gid])
                edges_v_t_attr.append(edge_attr)

    # =========================================================
    # 5. 写入 term 节点特征 + 显式 vtype 特征
    # =========================================================
    data["term"].x = torch.tensor(term_feats, dtype=torch.float) if term_feats else torch.zeros((0, len(TERM_TYPE_SET)+1+1), dtype=torch.float)
    data["term"].outer_index = torch.tensor(outer_gid_by_id, dtype=torch.long)

    # 显式变量类型特征（原逻辑沿用）
    num_total_terms = len(term_feats)
    explicit_vtype_features = torch.zeros((num_total_terms, 3), dtype=torch.float)

    for gid, core_expr in gid_to_core_expr.items():
        # 变量名统一转 str，避免出现非 str 的 name
        names = {str(s.name) for s in core_expr.free_symbols}
        for ms in _base_matrix_symbols(core_expr):
            names.add(str(ms.name))

        has_cont = has_int = has_bin = 0
        for vname in names:
            v_info = prob.variables.get(vname) or prob.matrix_variables.get(vname)
            if v_info is None:
                continue
            vtype = getattr(v_info, "vtype", "continuous")
            if vtype == "continuous":
                has_cont = 1
            elif vtype == "integer":
                has_int = 1
            elif vtype == "binary":
                has_bin = 1

        explicit_vtype_features[gid, 0] = has_cont
        explicit_vtype_features[gid, 1] = has_int
        explicit_vtype_features[gid, 2] = has_bin

    data["term"].explicit_vtype = explicit_vtype_features

    # =========================================================
    # 6. 边写入 HeteroData + 反向边
    # =========================================================
    def to_edge(idx_list):
        if not idx_list:
            return torch.zeros((2,0), dtype=torch.long)
        return torch.tensor(idx_list, dtype=torch.long).t().contiguous()

    data["variable", "uses", "term"].edge_index = to_edge(edges_v_t)
    data["variable", "uses", "term"].edge_attr  = torch.tensor(edges_v_t_attr, dtype=torch.float) if edges_v_t_attr else torch.zeros((0, 2+MAX_TOP_TERMS), dtype=torch.float)
    data["term", "nested", "term"].edge_index   = to_edge(list(edges_t_t_set))
    data["term", "in", "constraint"].edge_index = to_edge(list(edges_t_c_set))

    _add_reverse_edges(data)

    return data

if __name__ == "__main__":
    # 构造 mini 示例
    # prob = QCQPProblem("demo", "min")
    # x = prob.add_variable('x')
    # y = prob.add_variable('y', vtype='binary')
    # prob.obj_expr = x**2 + sp.log(x*y) + 3*x*y
    # prob.add_constraint(expr = x**2 + y**2, sense = "<=", rhs = 1)
    # prob.map_all_terms()

    # g = qcqp_to_heterodata(prob)
    # print(g)
    
    # x, y, z = symbols('x y z')

    # exprs = [
    #     x, x**2, x*y,
    #     log(x), log(x + y),
    #     sqrt(x), sqrt(x + y),
    #     Abs(x), Abs(x + y),
    #     exp(x), exp(x + y),
    #     sin(x), sin(x + y),
    #     cos(x), cos(x + y),
    #     tan(x), tan(x + y),
    #     (x + y) / z,
    #     x / (y + z),
    #     (x + y) / (z + x),
    #     5, log(x**2)
    # ]

    # for e in exprs:
    #     print(f"{str(e):30} → {get_term_type(e)}")
    
    
    # p = QCQPProblem("test_frac", "min")
    # x = p.add_variable('x', vtype='continuous')
    # y = p.add_variable('y', vtype='binary')
    # p.obj_expr = x**2 / y + x*y
    # p.add_constraint(expr=x + y, sense='<=', rhs=1)
    # p.map_all_terms()
    # g = qcqp_to_heterodata(p)
    # print(g)
    
    from sympy import MatrixSymbol, Trace, diag
    prob = QCQPProblem("vec_demo", "min")
    # 注册两个“向量变量”（你的 QCQPProblem 已经有 add_vector_variable 可用的话，用它更好）
    x = prob.add_vector_variable('x', 3, vtype='continuous')
    y = prob.add_vector_variable('y', 3, vtype='binary')
    Q = diag(1,2,3)  # 常量矩阵

    # 目标：- x^T x + x^T Q x + y^T x
    prob.obj_expr = - x.T * x + x.T * y
    # prob.obj_expr = (sp.Trace(x.T * x) + 1) / sp.Trace(y.T * y)
    # 约束：Tr(x^T x) <= 10
    prob.add_constraint(expr = x.T * x, sense = "<=", rhs = 10)
    prob.map_all_terms()

    g = qcqp_to_heterodata(prob)
    print(g)

    # 调用可视化函数
    visualize_hetero_graph(g)
