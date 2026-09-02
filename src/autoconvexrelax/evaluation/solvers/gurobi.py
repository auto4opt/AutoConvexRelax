# solver_interface.py
# -*- coding: utf-8 -*-
"""
把 QCQPProblem 映射为 Gurobi 模型，并提供：
    - solve_nonconvex_qcqp: 原始非凸 QCQP（允许非凸二次，MIP）
    - solve_convex_relax:   凸松弛（整数放宽为连续 + 不允许非凸二次）

注意：
  * 只支持到二次项；
  * 不支持分式（/），所以目前用于 PURE / HYBRID 中“不含分式”的问题。
"""

from __future__ import annotations

import sympy as sp
from sympy.matrices.expressions.matexpr import MatrixElement
import gurobipy as gp
from gurobipy import GRB

# --- MatrixExpr/Trace -> scalar Sympy (只覆盖常见 QCQP 形式) -----------------
from sympy import Add, Trace, MatMul, MatAdd, Transpose, MatrixExpr, MatrixBase, MatrixSymbol
from autoconvexrelax.evaluation.expressions import _trace_to_scalar_expr, normalize_expr

from autoconvexrelax.core.problem import (
    QCQPProblem,
    normalize_scalar,
    Variable,
    VectorVariableSymbol,
    MatrixVariableSymbol,
)

# --- Sympy Expr -> Gurobi Expr（只到二次） ------------------------------
def _sympy_to_gurobi(expr, sym2var):
    """
    把 **已经是标量** 的 sympy 表达式翻译到 Gurobi。

    sym2var 的键：
        - 直接用 sympy 的 atom 当键：Symbol / MatrixElement
    """
    expr = normalize_scalar(expr)
    
    # --- Trace handling (MatrixExpr / MatrixSymbol / custom matrix-like) ---
    # Convert Trace(...) to an explicit scalar expression whenever possible.
    if isinstance(expr, Trace):
        scalar_expr = _trace_to_scalar_expr(expr)
        if scalar_expr is None:
            raise NotImplementedError(f"Unsupported Trace form in QCQP -> Gurobi: {repr(expr)}")
        return _sympy_to_gurobi(scalar_expr, sym2var)

    expr = normalize_expr(expr)

    # 规范化后如果仍出现 Trace / MatrixExpr，说明 expression_utils 不支持该形态
    if isinstance(expr, sp.Trace):
        raise NotImplementedError(f"Trace still present after normalize_expr: {repr(expr)}")
    if isinstance(expr, sp.MatrixExpr):
        # 只允许 1x1（理论上 normalize_expr 已经转成标量了）
        if getattr(expr, "shape", None) == (1, 1):
            return _sympy_to_gurobi(expr[0, 0], sym2var)
        raise NotImplementedError(f"Non-scalar MatrixExpr after normalize_expr: {repr(expr)}")


    # 常数
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, sp.Number):
        return float(expr)

    # 标量变量 Symbol
    if isinstance(expr, sp.Symbol):
        if expr not in sym2var:
            raise KeyError(f"Scalar symbol {repr(expr)} not found in sym2var")
        return sym2var[expr]

    # 向量/矩阵的元素 MatrixElement
    if isinstance(expr, MatrixElement):
        if expr not in sym2var:
            raise KeyError(f"Matrix element {repr(expr)} not found in sym2var")
        return sym2var[expr]

    # 加法
    if isinstance(expr, sp.Add):
        terms = [_sympy_to_gurobi(arg, sym2var) for arg in expr.args]
        res = terms[0]
        for t in terms[1:]:
            res = res + t
        return res

    # 乘法：可包含常数、线性、二次
    if isinstance(expr, sp.Mul):
        coeff = 1.0
        grb_factors = []
        for arg in expr.args:
            if isinstance(arg, (int, float, sp.Number)):
                coeff *= float(arg)
            else:
                grb_factors.append(_sympy_to_gurobi(arg, sym2var))

        # 只有常数
        if not grb_factors:
            return coeff

        res = grb_factors[0]
        for t in grb_factors[1:]:
            res = res * t
        if abs(coeff - 1.0) > 1e-12:
            res = coeff * res
        return res

    # 幂：只支持平方
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if int(exp) == 2:
            v = _sympy_to_gurobi(base, sym2var)
            return v * v
        raise NotImplementedError(f"Only power 2 supported in QCQP translator, got {expr}")

    raise NotImplementedError(f"Unsupported sympy node in QCQP -> Gurobi: {repr(expr)}")

def _add_matrix_vars(m, rows, cols, **kwargs):
    rows = int(rows)
    cols = int(cols)
    return m.addVars(range(rows), range(cols), **kwargs)

def _collect_scalar_symbols(expr):
    """收集表达式中出现的标量 Symbol（排除 MatrixSymbol / Function 等）。"""
    if expr is None:
        return set()
    syms = set()
    for s in expr.free_symbols:
        # 只要标量 Symbol；MatrixSymbol 不能直接当标量变量
        if isinstance(s, sp.Symbol) and not isinstance(s, sp.MatrixSymbol):
            syms.add(s)
    return syms

def _ensure_aux_scalar_vars(m, obj_expr, constraints, sym2var):
    """
    对 objective + constraints 中出现但 sym2var 里没有的标量 Symbol，
    自动在 Gurobi 模型里创建连续变量并加入 sym2var。
    """
    need = set()

    # objective
    need |= _collect_scalar_symbols(obj_expr)

    # constraints
    for c in constraints:
        need |= _collect_scalar_symbols(c.expr)
        if c.rhs is not None and hasattr(c.rhs, "free_symbols"):
            need |= _collect_scalar_symbols(c.rhs)

    # add missing scalar vars
    for s in need:
        if s not in sym2var:
            v = m.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=str(s),
            )
            sym2var[s] = v


# --- QCQPProblem -> Gurobi.Model --------------------------------------


def build_gurobi_model_from_qcqp(
    prob: QCQPProblem,
    model_name: str | None = None,
    nonconvex: bool = True,
    relax_integrality: bool = False,
):
    """
    参数:
        nonconvex:
            True  -> m.Params.NonConvex = 2，允许非凸二次（Gurobi 做 B&B + QCQP）
            False -> m.Params.NonConvex = 0，假设已经是凸问题（否则 Gurobi 会报错）
        relax_integrality:
            True  -> integer/binary 改成 continuous，用于 root convex relax
    """
    if model_name is None:
        model_name = prob.name

    m = gp.Model(model_name)
    
    # 非凸参数
    if nonconvex:
        m.Params.NonConvex = 2
    else:
        m.Params.NonConvex = 0

    # 1. 创建变量
    sym2var = {}

    # 1.1 标量变量：prob.variables: {name -> Variable}
    for name, vinfo in getattr(prob, "variables", {}).items():
        assert isinstance(vinfo, Variable)
        lb = vinfo.lb if vinfo.lb is not None else -gp.GRB.INFINITY
        ub = vinfo.ub if vinfo.ub is not None else gp.GRB.INFINITY

        if relax_integrality:
            grb_type = gp.GRB.CONTINUOUS
        else:
            if vinfo.vtype == "continuous":
                grb_type = gp.GRB.CONTINUOUS
            elif vinfo.vtype == "integer":
                grb_type = gp.GRB.INTEGER
            elif vinfo.vtype == "binary":
                grb_type = gp.GRB.BINARY
            else:
                raise ValueError(f"Unknown vtype for scalar {name}: {vinfo.vtype}")

        grb_var = m.addVar(lb=lb, ub=ub, vtype=grb_type, name=name)
        sym2var[sp.Symbol(name, real=True)] = grb_var

    # 1.2 向量 / 矩阵变量：prob.matrix_variables: {name -> VectorVariableSymbol/MatrixVariableSymbol}
    for name, mv in getattr(prob, "matrix_variables", {}).items():
        if isinstance(mv, VectorVariableSymbol):
            rows, cols = mv.dim, 1
        elif isinstance(mv, MatrixVariableSymbol):
            rows, cols = mv.rows, mv.cols
        else:
            raise ValueError(f"Unknown matrix variable type for {name}: {type(mv)}")

        # 这里 lb/ub 只支持统一的标量上下界
        lb = mv.lb if getattr(mv, "lb", None) is not None else -gp.GRB.INFINITY
        ub = mv.ub if getattr(mv, "ub", None) is not None else gp.GRB.INFINITY

        if relax_integrality:
            grb_type = gp.GRB.CONTINUOUS
        else:
            vtype = getattr(mv, "vtype", "continuous")
            if vtype == "continuous":
                grb_type = gp.GRB.CONTINUOUS
            elif vtype == "integer":
                grb_type = gp.GRB.INTEGER
            elif vtype == "binary":
                grb_type = gp.GRB.BINARY
            else:
                raise ValueError(f"Unknown vtype for matrix variable {name}: {vtype}")

        grb_mat = _add_matrix_vars(
            m,
            rows, cols,
            lb=lb, ub=ub, vtype=grb_type, name=name
        )

        # 建立 MatrixElement -> Gurobi Var 映射
        base = mv.symbol  # 这是构造表达式时用到的 MatrixSymbol
        for i in range(rows):
            for j in range(cols):
                me = MatrixElement(base, i, j)
                sym2var[me] = grb_mat[i, j]

    m.update()

    # 2. 目标函数：prob.obj_expr / prob.obj_sense
    obj_expr = normalize_expr(normalize_scalar(prob.obj_expr))

    _ensure_aux_scalar_vars(m, obj_expr, prob.constraints, sym2var)

    grb_obj = _sympy_to_gurobi(obj_expr, sym2var)

    sense = str(prob.obj_sense).lower()
    if sense.startswith("min"):
        m.setObjective(grb_obj, gp.GRB.MINIMIZE)
    else:
        m.setObjective(grb_obj, gp.GRB.MAXIMIZE)

    # 3. 约束：prob.constraints 里的每个 Constraint(expr, sense, rhs)
    for idx, c in enumerate(prob.constraints):
        lhs = normalize_expr(normalize_scalar(c.expr))

        # rhs 可能是常数，也可能是 Sympy 表达式
        rhs_expr = c.rhs
        if rhs_expr is None:
            grb_rhs = 0.0
        elif isinstance(rhs_expr, (int, float, sp.Number)):
            grb_rhs = float(rhs_expr)
        else:
            rhs_expr = normalize_expr(normalize_scalar(rhs_expr))
            # 如果 RHS 仍然是纯常数（无符号），再数值化；否则翻译成 Gurobi 表达式
            if hasattr(rhs_expr, "free_symbols") and len(rhs_expr.free_symbols) == 0:
                grb_rhs = float(rhs_expr.evalf())
            else:
                grb_rhs = _sympy_to_gurobi(rhs_expr, sym2var)


        grb_lhs = _sympy_to_gurobi(lhs, sym2var)
        cname = f"c_{idx}"

        if c.sense == "<=":
            m.addConstr(grb_lhs <= grb_rhs, name=cname)
        elif c.sense == ">=":
            m.addConstr(grb_lhs >= grb_rhs, name=cname)
        elif c.sense in ("=", "=="):
            m.addConstr(grb_lhs == grb_rhs, name=cname)
        else:
            raise ValueError(...)


    # 4) PSD constraints (semidefinite)
    # 说明：
    #   - 如果 psd_constraint.matrix_expr 是显式的 SymPy Matrix（每个元素都是仿射表达式），直接转为 PSD 锥约束；
    #   - 如果是常见的 Z - x*x.T 形式（SDP 松弛里常见的 Schur 补写法），会自动改写为块矩阵 [[1, x.T],[x, Z]] ⪰ 0；
    #   - 其余包含二次项的矩阵不属于线性矩阵不等式（LMI），Gurobi SDP 也无法直接接受，此处会抛错以避免“悄悄变松弛”。
    if hasattr(prob, "psd_constraints") and prob.psd_constraints:
        def _dense_from_matsym(A: sp.MatrixSymbol) -> sp.Matrix:
            r, c = map(int, A.shape)
            return sp.Matrix([[MatrixElement(A, i, j) for j in range(c)] for i in range(r)])

        def _split_scalar_matmul(mm: sp.MatMul):
            scalar = sp.Integer(1)
            mats = []
            for a in mm.args:
                if a.is_Number:
                    scalar *= a
                else:
                    mats.append(a)
            return scalar, mats

        def _try_schur_block(expr):
            # Match: Z - x*x.T  (or Z + (-1)*x*x.T), where Z is (n,n), x is (n,1)
            Z_sym = None
            x_sym = None
            coeff_Z = sp.Integer(0)
            coeff_xxt = sp.Integer(0)

            terms = expr.args if isinstance(expr, sp.MatAdd) else (expr,)
            for t in terms:
                if isinstance(t, sp.MatrixSymbol):
                    Z_sym = t
                    coeff_Z += 1
                elif isinstance(t, sp.MatMul):
                    s, mats = _split_scalar_matmul(t)
                    # scalar * Z
                    if len(mats) == 1 and isinstance(mats[0], sp.MatrixSymbol):
                        Z_sym = mats[0]
                        coeff_Z += s
                    # scalar * x * x.T
                    elif len(mats) == 2 and isinstance(mats[0], sp.MatrixSymbol) and isinstance(mats[1], sp.Transpose) and mats[1].args[0] == mats[0]:
                        x_sym = mats[0]
                        coeff_xxt += s

            if Z_sym is None or x_sym is None:
                return None
            if coeff_Z != 1 or coeff_xxt != -1:
                return None

            n1, n2 = map(int, Z_sym.shape)
            nx, mx = map(int, x_sym.shape)
            if n1 != n2 or mx != 1 or nx != n1:
                return None

            Z_dense = _dense_from_matsym(Z_sym)  # n x n
            x_col = sp.Matrix([MatrixElement(x_sym, i, 0) for i in range(nx)])  # n x 1

            top = sp.Matrix.hstack(sp.Matrix([[sp.Integer(1)]]), x_col.T)      # 1 x (n+1)
            bottom = sp.Matrix.hstack(x_col, Z_dense)                          # n x (n+1)
            return sp.Matrix.vstack(top, bottom)                               # (n+1) x (n+1)

        def _to_affine_psd_matrix(mat_expr):
            # 1) Explicit Matrix
            if isinstance(mat_expr, sp.MatrixBase):
                return sp.Matrix(mat_expr)
            # 2) Matrix symbol: treat as itself
            if isinstance(mat_expr, sp.MatrixSymbol):
                return _dense_from_matsym(mat_expr)
            # 3) Try Schur block for Z - x*x.T
            blk = _try_schur_block(mat_expr)
            if blk is not None:
                return blk
            # 4) Last resort: try to coerce to Matrix (may fail for non-affine matrix expressions)
            try:
                return sp.Matrix(mat_expr)
            except Exception:
                return None

        for k, psd_c in enumerate(prob.psd_constraints):
            mat = _to_affine_psd_matrix(psd_c.matrix_expr)
            if mat is None:
                raise ValueError(f"Unsupported PSD constraint #{k}: cannot coerce to affine SymPy Matrix.")
            if mat.rows != mat.cols:
                raise ValueError(f"Unsupported PSD constraint #{k}: matrix is not square ({mat.rows}x{mat.cols}).")

            n = int(mat.rows)
            Y = m.addMVar((n, n), lb=-GRB.INFINITY, name=f"PSD_aux_{k}")

            # Tie entries (must be affine)
            for i in range(n):
                for j in range(n):
                    gij = _sympy_to_gurobi(mat[i, j], sym2var)
                    # Reject quadratic / higher-order entries: PSD cone requires LMI (affine)
                    if isinstance(gij, gp.QuadExpr):
                        raise ValueError(
                            f"PSD constraint #{k} has non-affine entry ({i},{j}) = {mat[i,j]}; "
                            "please rewrite PSD constraint as an LMI (e.g., Schur block [[1, x.T],[x, Z]])."
                        )
                    m.addConstr(Y[i, j] == gij, name=f"PSD_aux_{k}_{i}_{j}")

            m.addConstr(Y >> 0, name=f"PSD_cone_{k}")
    m.update()
    return m


# --- 封装两种求解模式 -----------------------------------------------
def solve_nonconvex_qcqp(
    prob: QCQPProblem,
    time_limit: float = 60.0,
    root_only: bool = True,
    return_status: bool = False,
):
    m = build_gurobi_model_from_qcqp(prob, nonconvex=True, relax_integrality=False)
    m.update()

    root_bound_box = {"val": None}

    def cb(model, where):
        if where == GRB.Callback.MIP:
            nodcnt = model.cbGet(GRB.Callback.MIP_NODCNT)
            if nodcnt == 0:
                # capture root bound; keep last (strongest) at root
                root_bound_box["val"] = float(model.cbGet(GRB.Callback.MIP_OBJBND))

    m.Params.NonConvex = 2
    m.Params.TimeLimit = time_limit

    if root_only:
        # Root-only, minimal strengthening
        m.Params.Presolve   = 0
        m.Params.Cuts       = 0
        m.Params.CutPasses = 0
        m.Params.Heuristics = 0
        m.Params.OBBT = 0
        m.Params.Aggregate   = 0
        m.Params.PreDepRow   = 0
        m.Params.DualReductions = 0
        m.Params.Symmetry       = 0
        m.Params.NodeLimit  = 1

    m.optimize(cb)

    best_obj = float(m.ObjVal) if m.SolCount > 0 else None
    best_bound = float(m.ObjBound) if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] else None
    root_bound = root_bound_box["val"]

    if root_bound is None:
        root_bound = best_bound

    result = {
        "sol_count": m.SolCount,
        "status": m.Status,
        "best_obj": best_obj,
        "best_bound": best_bound,
        "root_bound": root_bound,
    }

    if return_status:
        return result
    return result["best_obj"], result["best_bound"], result["root_bound"]

