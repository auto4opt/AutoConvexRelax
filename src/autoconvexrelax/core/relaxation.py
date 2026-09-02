import random
import typing as t
import autoconvexrelax.core.problem as ps
import sympy
import hashlib

from sympy import *
from sympy.matrices.expressions.matexpr import MatrixElement
import numpy as np
from sympy.core.relational import Relational


class RelaxationEngine:
    def __init__(self):
        # … 你的其它初始化 …
        self.w_cache = {}   # {(x_name,y_name): w_sym}  避免重复线性化
        self.affine_cache   = {}   # expr (srepr) → t_sym
        self.affine_counter = 0
        self.vec_affine_cache   = {}   # 向量仿射  srepr(vec) → T_sym
        self.vec_affine_counter = 0
        self.last_rewrite = None

        # self.enable_psd_cone = False
        self.DISCRETE_VTYPES = {"integer", "binary"}
        self.frac_cache = {}   # key → lam_name  （避免重复 λ）
        
        self.sdp_Z_cache = {}   # key: (id(problem), x_name) -> Z_symbol
        self.enable_bt_warmup = True
        self.enable_bt_before_global_cut = False
        # Optional switches for a strict/basic McCormick baseline.
        self.enable_mccormick_square_midpoint_tangent = True
        self.enable_mccormick_square_nonneg_cut = True
        self.enable_mccormick_link_w_to_sdp = True
        # debug flags
        self.debug_bt = False
        self.debug_fix = False
        # QCR: number of chord segments for each negative eigen direction (1 = single chord)
        self.qcr_segments = 1
        self.qcr_aux_counter = 0

    def _sym(self, problem, name: str, **assumptions):
        """
        优先复用 problem 里已注册变量的 sym 句柄，避免重复构造 Symbol；
        不再显式使用 real=True。
        """
        v = None
        if hasattr(problem, "variables"):
            v = problem.variables.get(name)
            if v is not None and getattr(v, "sym", None) is not None:
                return v.sym
        return sympy.Symbol(name, **assumptions)

        
    # --------------------------------------------------
    # Bounds helpers (for tighter McCormick / SDP+RLT)
    # --------------------------------------------------
    def _vec_elem(self, x_sym, i: int):
        """Return the i-th scalar element of a vector MatrixSymbol-like object."""
        try:
            return x_sym[i, 0]
        except Exception:
            return x_sym[i]

    def _elem_bounds(self, mv, idx):
        """
        支持 mv.lb/mv.ub 为：
        - 标量（统一盒约束）
        - 1D list/np（向量逐元素）
        - 2D list/np（矩阵逐元素）
        """
        import numbers
        lb = getattr(mv, "lb", None)
        ub = getattr(mv, "ub", None)
        if lb is None or ub is None:
            return None, None

        # 标量统一界：直接返回
        if isinstance(lb, numbers.Number) and isinstance(ub, numbers.Number):
            return float(lb), float(ub)

        if isinstance(idx, tuple):
            i, j = idx
            try:
                return float(lb[i][j]), float(ub[i][j])
            except Exception:
                # 如果给了矩阵 idx，但 lb/ub 实际仍是标量或 1D，就兜底失败
                return None, None
        else:
            try:
                return float(lb[idx]), float(ub[idx])
            except Exception:
                return None, None



    def _add_sdp_rlt_constraints(self, prob, x_var, X_sym):
        """Add McCormick/RLT envelopes linking X_ij to x_i, x_j (SDP+RLT tightening).
        Only adds constraints where both bounds are known.
        """
        n = int(getattr(x_var, "dim", 0) or 0)
        x_sym = getattr(x_var, "sym", None) or x_var  # x_var may itself be a SymPy symbol
        for i in range(n):
            li, ui = self._elem_bounds(x_var, i)
            if li is None or ui is None:
                continue
            xi = self._vec_elem(x_sym, i)

            # Diagonal tightening from (x-l)(u-x) >= 0 and (x-l)^2 >= 0, (x-u)^2 >= 0
            Xii = X_sym[i, i]
            prob.add_constraint(Xii - (li + ui) * xi + li * ui, "<=", 0)   # Xii <= (l+u)xi - lu
            prob.add_constraint(Xii - 2 * li * xi + li * li, ">=", 0)      # Xii >= 2l xi - l^2
            prob.add_constraint(Xii - 2 * ui * xi + ui * ui, ">=", 0)      # Xii >= 2u xi - u^2

        # Off-diagonal McCormick envelopes
        for i in range(n):
            li, ui = self._elem_bounds(x_var, i)
            if li is None or ui is None:
                continue
            xi = self._vec_elem(x_sym, i)
            for j in range(i + 1, n):
                lj, uj = self._elem_bounds(x_var, j)
                if lj is None or uj is None:
                    continue
                xj = self._vec_elem(x_sym, j)
                Xij = X_sym[i, j]
                Xji = X_sym[j, i]
                # Four McCormick inequalities:
                prob.add_constraint(Xij - li * xj - lj * xi + li * lj, ">=", 0)
                prob.add_constraint(Xij - ui * xj - uj * xi + ui * uj, ">=", 0)
                prob.add_constraint(Xij - ui * xj - lj * xi + ui * lj, "<=", 0)
                prob.add_constraint(Xij - li * xj - uj * xi + li * uj, "<=", 0)
                prob.add_constraint(Xij - Xji, "=", 0)


    def _lift_norm_constraints_to_Z(self, problem: ps.QCQPProblem, x_sym, Z):
        """
        把约束中的 Trace(x.T*x) 提升为 Trace(Z)，用于收紧 SDR：
        - 对 <= / = ：添加一条提升后的“冗余但更紧”的线性/仿射约束
        - 对 >= / > ：直接用提升后的约束替换原约束（否则 MOSEK 会认为是非凸二次约束）
        """
        import sympy as sp

        norm_x = sp.Trace(x_sym.T * x_sym)
        norm_Z = sp.Trace(Z)

        # 注意：problem.constraints 里是对象（你在别处用过 cons.expr/cons.sense/cons.rhs）
        for cons in list(getattr(problem, "constraints", [])):
            try:
                expr = cons.expr
            except Exception:
                continue

            lifted = expr.xreplace({norm_x: norm_Z})
            if lifted == expr:
                continue

            # >= 形式的二次范数约束在 MOSEK 里会被判为非凸；提升后变成线性/仿射，可以安全保留
            if cons.sense in (">=", ">"):
                cons.expr = lifted
            else:
                problem.add_constraint(expr=lifted, sense=cons.sense, rhs=cons.rhs)


    def _mark_identity_rewrite(self, location, witness_expr):
        """不改变表达式，但记录一次改写，让奖励函数能看到上下文变化。"""
        self.last_rewrite = {
            "location": location,
            "old": witness_expr,
            "new": witness_expr,
        }
        
    def _den_has_matrix_decision(self, expr, problem):
        """判断表达式 expr 是否含有矩阵决策变量。"""
        import sympy as sp

        for ms in expr.atoms(sp.MatrixSymbol):
            if ms.name in getattr(problem, "matrix_variables", {}):
                return True
        return False
    
    def _linearize_trace(self, expr):
        import sympy as sp
        def is_matlike(a):
            return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))
        def rec(e):
            if isinstance(e, sp.Trace):
                inner = rec(e.arg)

                # Trace(A + B + ...) -> Trace(A)+Trace(B)+...
                if isinstance(inner, (sp.Add, sp.MatAdd)):
                    return sp.Add(*[sp.Trace(rec(a)) for a in inner.args])

                # Trace(alpha * M) -> alpha * Trace(M)
                if isinstance(inner, sp.Mul):
                    mats = [a for a in inner.args if is_matlike(a)]
                    scas = [a for a in inner.args if not is_matlike(a)]
                    if mats:
                        alpha = sp.Mul(*scas) if scas else sp.Integer(1)
                        core  = mats[0] if len(mats)==1 else sp.MatMul(*mats)
                        return alpha * sp.Trace(rec(core))

                return sp.Trace(inner)

            # 结构递归，避免触发过度化简
            if isinstance(e, (sp.Add, sp.MatAdd, sp.Mul, sp.MatMul, sp.Function, sp.Pow, sp.Transpose)):
                return e.func(*(rec(a) for a in e.args))
            return e
        return rec(expr)
        
    def _is_decision_name(self, problem, vname: str) -> bool:
        return (vname in problem.variables) or (vname in getattr(problem, "matrix_variables", {}))

    def _get_var_obj(self, problem, vname: str):
        """返回 (对象, 'scalar'|'matrix'|None)"""
        if vname in problem.variables:
            return problem.variables[vname], "scalar"
        mv = getattr(problem, "matrix_variables", {}).get(vname)
        if mv is not None:
            return mv, "matrix"
        return None, None
    
    def _as_vector_symbol(self, problem, location, vec):
        """
        返回 MatrixSymbol（列向量）。如果 vec 是 MatAdd/MatMul/其它 MatrixExpr，
        则用 _vector_affine_to_var 物化成新的向量变量 T_k，并加好逐元素等式和界。
        """
        import sympy as sp
        if isinstance(vec, sp.MatrixSymbol):
            return vec
        return self._vector_affine_to_var(problem, location, vec)

    # ➊ 新增：结构相等比较
    def _same(self, a, b):
        return sympy.srepr(a) == sympy.srepr(b)

    # ➋ 新增：带 Trace 桥接的替换器（局部表达式 → 新表达式）
    def _replace_with_trace_bridge(self, target_expr, old_expr, new_expr):
        """
        先尝试常规 xreplace；
        若未命中且 old_expr=Trace(core)，再尝试 Trace(target_expr) 与 old_expr 等价时的“整块替换”。
        返回 (new_target_expr, changed: bool)
        """
        try:
            replaced = target_expr.xreplace({old_expr: new_expr})
        except Exception:
            replaced = target_expr

        if not self._same(replaced, target_expr):
            return replaced, True

        # 回退：Trace 桥接（解决 y*(x.T+y.T) vs Trace(y.T*(x+y))）
        if isinstance(old_expr, sympy.Trace):
            # 情况A：target 本身是 1×1，且 target[0,0] == old_expr
            if getattr(target_expr, "shape", None) == (1, 1):
                try:
                    if self._same(target_expr[0, 0], old_expr):
                        return new_expr, True
                except Exception:
                    pass

            # 情况B：target 是方阵，且 Trace(target) == old_expr
            shp = getattr(target_expr, "shape", None)
            if shp and shp[0] == shp[1]:
                try:
                    if self._same(sympy.Trace(target_expr), old_expr):
                        # 直接把“矩阵目标”替换为新标量
                        return new_expr, True
                except Exception:
                    pass

        return target_expr, False
        
    def _base_of_matrixlike(self, expr):
        """
        给定 MatrixElement / Transpose / MatrixSymbol，
        返回其“底层 MatrixSymbol”（比如 Transpose(x) → x；MatrixElement(x,i,j) → x）。
        其余返回 None。
        """
        if isinstance(expr, MatrixSymbol):
            return expr
        if isinstance(expr, MatrixElement):
            base = expr.parent
            return base if isinstance(base, MatrixSymbol) else None
        if isinstance(expr, Transpose):
            arg = expr.arg
            return arg if isinstance(arg, MatrixSymbol) else None
        return None

    def _iter_var_handles(self, expr) -> t.Iterable[t.Tuple[str, object]]:
        """
        遍历 sub_expr 中所有“可能对应决策变量”的句柄，产出 (变量名, 句柄本身)。
        句柄可能是：
        - Symbol（标量）
        - MatrixSymbol（向量/矩阵）
        - MatrixElement（向量/矩阵的元素）
        - Indexed（若你用 IndexedBase 存元素）
        - 自定义的 VectorVariableSymbol / MatrixVariableSymbol
        """
        # 1) 标量 Symbol
        for s in expr.free_symbols:
            yield (s.name, s)

        # 2) MatrixSymbol（包括出现在 Trace / 乘法 / 转置里的）
        mats: t.Set[MatrixSymbol] = set()
        mats |= set(expr.atoms(MatrixSymbol))
        # 2.1) 从转置里还原底层 MatrixSymbol
        for tr in expr.atoms(Transpose):
            base = self._base_of_matrixlike(tr)
            if isinstance(base, MatrixSymbol):
                mats.add(base)
        # 2.2) 从元素里还原底层 MatrixSymbol
        for me in expr.atoms(MatrixElement):
            base = self._base_of_matrixlike(me)
            if isinstance(base, MatrixSymbol):
                mats.add(base)

        for M in mats:
            yield (M.name, M)

        # 3) MatrixElement / Indexed（元素级）
        for me in expr.atoms(MatrixElement):
            base = self._base_of_matrixlike(me)
            if isinstance(base, MatrixSymbol):
                yield (base.name, me)
        for ind in expr.atoms(Indexed):
            base = ind.base
            if hasattr(base, 'label'):  # IndexedBase('x') → label='x'
                yield (str(base.label), ind)

        # 4) 你的自定义符号类（如果有的话）
        try:
            from autoconvexrelax.core.problem import VectorVariableSymbol, MatrixVariableSymbol
            for v in expr.atoms(VectorVariableSymbol):
                yield (v.name, v)
            for v in expr.atoms(MatrixVariableSymbol):
                yield (v.name, v)
        except Exception:
            pass


    def _is_simple_bound_rel(self, rel, sym_like) -> bool:
        """
        识别形如  sym <= c, sym >= c, c <= sym, c >= sym, sym == c 的简单范围约束。
        sym_like 可以是：
        - Symbol
        - MatrixElement（元素级别）
        仅用于“清理旧 bound 约束”。
        """
        if not isinstance(rel, Relational):
            return False

        lhs, rhs = rel.lhs, rel.rhs

        def is_num(x):
            return x.is_number if hasattr(x, 'is_number') else False

        # 允许左右互换（常数在左/右）
        pairs = [(lhs, rhs), (rhs, lhs)]
        for a, b in pairs:
            # a 是目标变量（或元素），b 是数
            # 注意：element-wise bound 我们只清理 MatrixElement 层面的
            if (a == sym_like) and is_num(b):
                # <= >= ==
                if isinstance(rel, (Le, Ge, Eq)):
                    return True
        return False


    def _remove_old_bounds(self, problem, vname: str, v_obj, handles_for_this_var: t.Iterable[object]) -> int:
        """
        从 problem.constraints 中移除与变量 vname 相关的“简单范围约束”。
        - 对标量：sym <= c / >= c / == c
        - 对向量/矩阵：元素级 MatrixElement(x,i,j) <= c / >= c / == c
        """
        new_cons = []
        removed = 0

        # 收集对该变量的“元素句柄”（MatrixElement）以及标量 Symbol
        elems = set()
        syms  = set()
        for h in handles_for_this_var:
            if isinstance(h, MatrixElement):
                elems.add(h)
            elif isinstance(h, Symbol):
                syms.add(h)
            # 其它类型（MatrixSymbol/自定义）不直接作为单个约束左端项存在

        for cons in problem.constraints:
            rel = getattr(cons, 'rel', cons) if hasattr(cons, 'rel') else cons  # 若你有封装，取出底层 Relational
            keep = True

            # 标量 bound
            for s in syms:
                if self._is_simple_bound_rel(rel, s):
                    keep = False
                    break

            # 元素级 bound
            if keep:
                for e in elems:
                    if self._is_simple_bound_rel(rel, e):
                        keep = False
                        break

            if keep:
                new_cons.append(cons)
            else:
                removed += 1

        problem.constraints = new_cons
        return removed


    def _add_new_bounds(self, problem, vname: str, v_obj, any_handle_for_this_var):
        """
        仅对标量变量添加显式范围约束；
        向量/矩阵变量不添加元素级界，完全依赖 v_obj.lb / v_obj.ub 元信息。
        """
        lb, ub = v_obj.lb, v_obj.ub

        # —— 标量：加约束 —— #
        if isinstance(any_handle_for_this_var, Symbol):
            if lb is not None:
                problem.add_constraint_unique(any_handle_for_this_var, '>=', lb)
            if ub is not None:
                problem.add_constraint_unique(any_handle_for_this_var, '<=', ub)
            return
    
    def _is_decision_var(self, sym, problem):
        return isinstance(sym, sympy.Symbol) and self._is_decision_name(problem, sym.name)
    
    def _as_scalar(self, expr):
        """1×1 MatrixExpr → 标量；其它保持原样"""
        return expr[0, 0] if isinstance(expr, MatrixExpr) else expr

    def _as_matrix(self, expr):
        """标量 Expr → 1×1 Matrix；MatrixExpr 原样返回"""
        return expr if isinstance(expr, MatrixExpr) else Matrix([[expr]])

    # --------------------------------------------
    # 通用取界函数（可放类外也可作为 @staticmethod）
    # --------------------------------------------
    def _get_bounds(self, problem, sym):
        """返回 (lb, ub)；若未设置界返回 (None, None)"""
        v = problem.variables.get(sym.name)
        if v is not None:
            return v.lb, v.ub
        mv = getattr(problem, "matrix_variables", {}).get(sym.name)
        if mv is not None:
            return mv.lb, mv.ub

        # element-style name: Base_i or Base_i_j
        try:
            name = sym.name
            parts = name.split("_")
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                base = "_".join(parts[:-2])
                if base in getattr(problem, "matrix_variables", {}):
                    if hasattr(problem, "get_me_bounds"):
                        lb, ub = problem.get_me_bounds(base, int(parts[-2]), int(parts[-1]))
                        if lb is not None or ub is not None:
                            return lb, ub
            elif len(parts) >= 2 and parts[-1].isdigit():
                base = "_".join(parts[:-1])
                mv = getattr(problem, "matrix_variables", {}).get(base)
                if mv is not None:
                    lb, ub = self._elem_bounds(mv, int(parts[-1]))
                    if lb is not None or ub is not None:
                        return lb, ub
        except Exception:
            pass
        return None, None
    

    def _preprocess_fixed_vars(self, problem):
        """
        One-time preprocessing:
        - Fix scalar vars with lb == ub
        - Detect vector sum constraints like sum(y) = k with y in [l,u] and fix all elements when implied
        """
        import sympy as sp

        if getattr(problem, "_fixed_var_preprocessed", False):
            return False

        subs = {}

        # 1) scalar fixed vars
        for name, v in getattr(problem, "variables", {}).items():
            lb = getattr(v, "lb", None)
            ub = getattr(v, "ub", None)
            if lb is None or ub is None:
                continue
            if lb == ub:
                subs[sp.Symbol(name)] = float(lb)

        # 2) vector sum equality -> fix elements when implied
        # build map for constraints of form sum(y_i) <= k and >= k
        def _linear_coeffs(expr):
            # return (coeffs, const) or None if non-linear
            atoms = list(expr.atoms(sp.Symbol, MatrixElement))
            # build dummy symbol map
            repl = {}
            sym_map = {}
            for i, a in enumerate(atoms):
                s = sp.Symbol(f"__bt_dummy_{i}")
                repl[a] = s
                sym_map[s] = a
            expr2 = expr.xreplace(repl)
            try:
                poly = sp.Poly(expr2, *sym_map.keys(), domain="RR")
            except Exception:
                return None
            if poly.total_degree() > 1:
                return None
            d = poly.as_dict()
            const = float(d.get((0,) * len(sym_map), 0.0))
            coeffs = {}
            for exp, coef in d.items():
                if sum(exp) == 0:
                    continue
                if sum(exp) == 1:
                    idx = exp.index(1)
                    coeffs[sym_map[list(sym_map.keys())[idx]]] = float(coef)
                else:
                    return None
            return coeffs, const

        # collect candidate equalities
        sum_eq = []  # list of (vec_name, k)

        # direct sum(y) detection (Trace(ones_row * y) or Trace(y.T * ones_col))
        sum_eq_direct = []
        for c in getattr(problem, "constraints", []):
            expr = c.expr
            rhs = c.rhs
            # normalize to <= form
            sense = c.sense
            if sense in [">", ">="]:
                expr = -expr
                rhs = -rhs
                sense = "<="
            elif sense in ["=", "=="]:
                sense = "="

            try:
                rhs_val = float(rhs)
            except Exception:
                continue

            # match Trace(row * y)
            if isinstance(expr, sp.Trace):
                inner = expr.arg
                if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                    A, B = inner.args
                    # Trace(row * y) with row all-ones
                    if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                        if A.shape[0] == 1 and B.shape[1] == 1 and A.shape[1] == B.shape[0]:
                            try:
                                if all(float(A[0, i]) == 1.0 for i in range(A.shape[1])):
                                    if sense == "=":
                                        sum_eq_direct.append((B, rhs_val, "<="))
                                        sum_eq_direct.append((B, rhs_val, ">="))
                                    else:
                                        sum_eq_direct.append((B, rhs_val, sense))
                            except Exception:
                                pass
                    # Trace(y.T * row) with row all-ones
                    if isinstance(A, sp.Transpose) and isinstance(B, (sp.MatrixBase, sp.ImmutableMatrix)):
                        if isinstance(A.arg, sp.MatrixSymbol) and B.shape[1] == 1 and A.arg.shape[0] == B.shape[0]:
                            try:
                                if all(float(B[i, 0]) == 1.0 for i in range(B.shape[0])):
                                    if sense == "=":
                                        sum_eq_direct.append((A.arg, rhs_val, "<="))
                                        sum_eq_direct.append((A.arg, rhs_val, ">="))
                                    else:
                                        sum_eq_direct.append((A.arg, rhs_val, sense))
                            except Exception:
                                pass

        # merge direct detections into sum_eq
        for parent, k, s in sum_eq_direct:
            sum_eq.append((parent, k, s))
        # normalize all constraints to expr <= rhs
        for c in getattr(problem, "constraints", []):
            expr = c.expr
            rhs = c.rhs
            # expand Trace if possible (lightweight, local)
            try:
                expr = self._linearize_trace(expr)
                def _expand_trace(tr):
                    inner = tr.arg
                    # Trace(1x1) -> scalar
                    if isinstance(inner, sp.MatrixExpr) and getattr(inner, "shape", None) == (1, 1):
                        return inner[0, 0]
                    # Trace(Z) where Z is square MatrixSymbol -> sum_i Z[i,i]
                    if isinstance(inner, sp.MatrixSymbol) and inner.shape[0] == inner.shape[1]:
                        n = int(inner.shape[0])
                        return sp.Add(*[inner[i, i] for i in range(n)])
                    # Trace(row * y) : row is constant 1xn Matrix, y is n x 1 vector
                    if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                        A, B = inner.args
                        if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                            if A.shape[0] == 1 and B.shape[1] == 1 and A.shape[1] == B.shape[0]:
                                n = int(B.shape[0])
                                return sp.Add(*[A[0, i] * B[i, 0] for i in range(n)])
                    return tr
                expr = expr.replace(lambda x: isinstance(x, sp.Trace), _expand_trace)
                # expand Sum(...) with integer limits into explicit Add
                def _expand_sum(s):
                    try:
                        if not isinstance(s, sp.Sum):
                            return s
                        (sym, start, end) = s.limits[0]
                        start = int(start)
                        end = int(end)
                        terms = [s.function.subs(sym, i) for i in range(start, end + 1)]
                        return sp.Add(*terms)
                    except Exception:
                        return s
                expr = expr.replace(lambda x: isinstance(x, sp.Sum), _expand_sum)
            except Exception:
                pass
            # convert to standard form
            if c.sense in [">", ">="]:
                expr = -expr
                rhs = -rhs
            elif c.sense in ["=", "=="]:
                # treat as two inequalities later
                pass

            # only handle numeric rhs
            try:
                rhs_val = float(rhs)
            except Exception:
                continue

            # equality as two inequalities
            senses = ["<="] if c.sense not in ["=", "=="] else ["<=", ">="]
            for s in senses:
                e = expr if s == "<=" else -expr
                r = rhs_val if s == "<=" else -rhs_val

                lin = _linear_coeffs(sp.expand(e))
                if lin is None:
                    continue
                coeffs, const = lin
                # only MatrixElement from a single vector
                if not coeffs:
                    continue
                parents = []
                for a in coeffs.keys():
                    if not isinstance(a, MatrixElement):
                        parents = []
                        break
                    parent = getattr(a, "parent", None) or getattr(a, "base", None)
                    if not isinstance(parent, sp.MatrixSymbol):
                        parents = []
                        break
                    parents.append(parent)
                if not parents:
                    continue
                # same parent
                parent = parents[0]
                if any(p != parent for p in parents):
                    continue
                # all coeffs equal
                coeff_vals = list(coeffs.values())
                if max(coeff_vals) - min(coeff_vals) > 1e-9:
                    continue
                cval = coeff_vals[0]
                if abs(cval) < 1e-12:
                    continue
                # sum coeff * y_i + const <= r
                k = (r - const) / cval
                # normalize sign so coeff is positive; flip sense accordingly
                if cval < 0:
                    cval = -cval
                    s = ">=" if s == "<=" else "<="
                sum_eq.append((parent, k, s))

        # resolve equalities: need both <= and >= for same k
        eq_map = {}
        for parent, k, s in sum_eq:
            key = (parent, round(float(k), 9))
            eq_map.setdefault(key, set()).add(s)

        for (parent, k), senses in eq_map.items():
            if "<=" not in senses or ">=" not in senses:
                continue
            # now we have sum(y_i) = k
            mv = getattr(problem, "matrix_variables", {}).get(parent.name)
            if mv is None:
                continue
            n = int(parent.shape[0])
            # assume vector shape (n,1)
            if getattr(parent, "shape", None) != (n, 1):
                continue
            # uniform bounds check
            lbs = []
            ubs = []
            for i in range(n):
                li, ui = self._elem_bounds(mv, i)
                if li is None or ui is None:
                    lbs = []
                    break
                lbs.append(li)
                ubs.append(ui)
            if not lbs:
                continue
            if max(lbs) - min(lbs) > 1e-9 or max(ubs) - min(ubs) > 1e-9:
                continue
            l = lbs[0]
            u = ubs[0]
            # if sum = n*u or sum = n*l, then all fixed
            if abs(k - n * u) < 1e-6:
                val = u
            elif abs(k - n * l) < 1e-6:
                val = l
            else:
                continue
            # set substitutions for each element
            for i in range(n):
                subs[MatrixElement(parent, i, 0)] = float(val)

        # promote fully fixed MatrixSymbols to constant matrices (so Trace/MatMul rewrite works)
        fixed_parents = {}
        for sym, val in subs.items():
            if isinstance(sym, MatrixElement):
                parent = getattr(sym, "parent", None) or getattr(sym, "base", None)
                if parent is None:
                    continue
                try:
                    i = int(sym.i)
                    j = int(sym.j)
                except Exception:
                    continue
                fixed_parents.setdefault(parent, {})[(i, j)] = float(val)

        for parent, elems in fixed_parents.items():
            try:
                rows, cols = int(parent.shape[0]), int(parent.shape[1])
            except Exception:
                continue
            if len(elems) != rows * cols:
                continue
            # ensure all indices present
            complete = True
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in elems:
                        complete = False
                        break
                if not complete:
                    break
            if not complete:
                continue
            mat = sp.Matrix(rows, cols, lambda i, j: elems[(i, j)])
            subs[parent] = mat

        if not subs:
            problem._fixed_var_preprocessed = True
            return False

        try:
            fixed_preview = list(subs.items())[:6]
            if self.debug_fix:
                print(f"[FIXED] substitutions={len(subs)} preview={fixed_preview}")
        except Exception:
            pass

        # apply substitutions to objective and constraints
        try:
            problem.obj_expr = sp.simplify(sp.expand(problem.obj_expr.xreplace(subs)))
        except Exception:
            problem.obj_expr = problem.obj_expr.xreplace(subs)

        for c in getattr(problem, "constraints", []):
            try:
                c.expr = sp.simplify(sp.expand(c.expr.xreplace(subs)))
            except Exception:
                c.expr = c.expr.xreplace(subs)

        # update scalar bounds if needed
        for sym, val in subs.items():
            if isinstance(sym, sp.Symbol):
                v = problem.variables.get(sym.name)
                if v is not None:
                    v.lb = val
                    v.ub = val

        problem._fixed_var_preprocessed = True
        return True

    def _infer_affine_bounds(self, expr, problem):
        expr = sympy.expand(expr)

        # 1) scalar decision vars
        bounds = {}
        for s in expr.free_symbols:
            if isinstance(s, sympy.Symbol) and s.name in problem.variables:
                v = problem.variables[s.name]
                bounds[s] = (v.lb, v.ub)

        # 2) matrix/vector element vars (e.g., x[0,0]) -> temporary scalar symbols
        me_subs = {}
        me_idx = 0
        for me in sorted(expr.atoms(MatrixElement), key=sympy.srepr):
            parent = getattr(me, "parent", None) or getattr(me, "base", None)
            if not isinstance(parent, sympy.MatrixSymbol):
                continue

            lb = ub = None
            mv = getattr(problem, "matrix_variables", {}).get(parent.name)
            if mv is not None:
                try:
                    i = int(me.i)
                    j = int(me.j)
                    if hasattr(mv, "dim"):
                        lb, ub = self._elem_bounds(mv, i)
                    else:
                        lb, ub = self._elem_bounds(mv, (i, j))
                except Exception:
                    lb = ub = None

            if (lb is None or ub is None) and hasattr(problem, "get_me_bounds"):
                try:
                    lb_me, ub_me = problem.get_me_bounds(parent.name, int(me.i), int(me.j))
                    if lb is None:
                        lb = lb_me
                    if ub is None:
                        ub = ub_me
                except Exception:
                    pass

            aux = sympy.Symbol(f"__me_{me_idx}")
            me_idx += 1
            me_subs[me] = aux
            bounds[aux] = (lb, ub)

        if me_subs:
            expr = sympy.expand(expr.xreplace(me_subs))

        # unresolved matrix decision object in scalar bound inference -> unsafe
        for ms in expr.atoms(sympy.MatrixSymbol):
            if ms.name in getattr(problem, "matrix_variables", {}):
                print("[INFER-FAIL] unresolved matrix decision symbol:", ms)
                return None, None

        # no decision vars: constant only
        if not bounds:
            try:
                c = float(expr)
                return c, c
            except Exception:
                print("[INFER-FAIL] no bounded decision vars in expr:", expr)
                return None, None

        # any missing bounds -> give up
        for s, (lb, ub) in bounds.items():
            if lb is None or ub is None:
                print(f"[INFER-FAIL] var {s} has no bounds ({lb}, {ub})")
                return None, None

        # only affine polynomial
        try:
            if not expr.is_polynomial(*bounds.keys()):
                print("[INFER-FAIL] not polynomial:", expr)
                return None, None
            if sympy.total_degree(expr, *bounds.keys()) > 1:
                print("[INFER-FAIL] degree >1:", expr)
                return None, None
        except sympy.polys.polyerrors.PolynomialError:
            print("[INFER-FAIL] poly error")
            return None, None

        # Parse affine coefficients with Poly (more robust than as_independent for pure symbols).
        try:
            gens = tuple(sorted(bounds.keys(), key=sympy.srepr))
            poly = sympy.Poly(expr, *gens, domain="RR")
        except Exception:
            print("[INFER-FAIL] poly parse error:", expr)
            return None, None

        if poly.total_degree() > 1:
            print("[INFER-FAIL] degree >1:", expr)
            return None, None

        pmap = poly.as_dict()
        zero = (0,) * len(gens)
        try:
            c0 = float(pmap.get(zero, 0.0))
        except Exception:
            print("[INFER-FAIL] const term not real:", pmap.get(zero, 0.0))
            return None, None

        lb = ub = c0
        for exp, coef in pmap.items():
            deg = sum(exp)
            if deg == 0:
                continue
            if deg != 1:
                print("[INFER-FAIL] non-affine monomial:", (exp, coef))
                return None, None
            idx = exp.index(1)
            var = gens[idx]
            try:
                coeff = float(coef)
            except Exception:
                print("[INFER-FAIL] coeff not real:", coef)
                return None, None
            v_lb, v_ub = bounds[var]
            lb += coeff * (v_lb if coeff >= 0 else v_ub)
            ub += coeff * (v_ub if coeff >= 0 else v_lb)

        print(f"[INFER] {expr} -> ({lb}, {ub})")
        return lb, ub

    def _match_vec_dot(self, expr):
        base = expr
        if isinstance(expr, sympy.matrices.expressions.matexpr.MatrixElement):
            if getattr(expr, "i", None) == 0 and getattr(expr, "j", None) == 0:
                base = getattr(expr, "parent", None) or getattr(expr, "base", None)
            else:
                return False, None, None, None

        # ① α * (A.T * B)
        if isinstance(base, sympy.MatMul):
            alpha, core = self._pull_scalar_from_matmul(base)
            if isinstance(core, sympy.MatMul) and len(core.args) == 2:
                L, R = core.args
                if isinstance(L, sympy.Transpose) and isinstance(L.arg, sympy.MatrixExpr) and isinstance(R, sympy.MatrixExpr):
                    A, B = L.arg, R
                    if A.shape[1]==1 and B.shape[1]==1 and A.shape[0]==B.shape[0]:
                        return True, alpha, A, B

        # ② α * Trace(A.T * B)
        if isinstance(base, sympy.Trace) and isinstance(base.arg, sympy.MatMul):
            alpha2, core = self._pull_scalar_from_matmul(base.arg)
            if isinstance(core, sympy.MatMul) and len(core.args) == 2:
                L, R = core.args
                if isinstance(L, sympy.Transpose) and isinstance(L.arg, sympy.MatrixExpr) and isinstance(R, sympy.MatrixExpr):
                    A, B = L.arg, R
                    if A.shape[1]==1 and B.shape[1]==1 and A.shape[0]==B.shape[0]:
                        return True, alpha2, A, B

        return False, None, None, None

    def _linearize_pair(self, problem, x_sym, y_sym):
        # 1) 只处理标量决策 Symbol
        for s in (x_sym, y_sym):
            if not (isinstance(s, sympy.Symbol) and self._is_decision_name(problem, s.name)):
                return x_sym * y_sym

        # avoid re-linearizing existing McCormick proxy variables
        if self._is_mccormick_proxy_symbol(problem, x_sym) or self._is_mccormick_proxy_symbol(problem, y_sym):
            return x_sym * y_sym

        # 2) canonical
        a_name, b_name = sorted([x_sym.name, y_sym.name])
        key = (id(problem), a_name, b_name)
        w_name = f"w_{a_name}_{b_name}"
        
        def _maybe_link_w_to_Z(w_sym, x_sym, y_sym):
            """If an SDP lift Z_<vec> already exists, tie McCormick auxiliary w to Z[i,j]."""
            if not self.enable_mccormick_link_w_to_sdp:
                return
            import re as _re
            def _parse_vec_elem(vname: str):
                # x_0 -> ("x", 0) ; T_hash_3 -> ("T_hash", 3)
                m = _re.match(r"^(.*)_([0-9]+)$", vname)
                if not m:
                    return None
                return m.group(1), int(m.group(2))

            px = _parse_vec_elem(getattr(x_sym, "name", ""))
            py = _parse_vec_elem(getattr(y_sym, "name", ""))
            if (px is None) or (py is None):
                return
            base_x, ix = px
            base_y, iy = py
            if base_x != base_y:
                return

            Z_name = f"Z_{base_x}"
            if not hasattr(problem, "matrix_variables") or Z_name not in problem.matrix_variables:
                return

            try:
                Zsym = problem.get_matrix_symbol(Z_name)
                i, j = (ix, iy) if ix <= iy else (iy, ix)
                problem.add_constraint(expr=w_sym - sympy.MatrixElement(Zsym, i, j), sense='=', rhs=0)
            except Exception:
                return

        # 3) 先取界（缺界就 Big-M 回填）
        lbx, ubx = self._get_bounds(problem, x_sym)
        lby, uby = self._get_bounds(problem, y_sym)

        # 尝试用 interval propagation 补界（只对标量）
        if lbx is None or ubx is None:
            lx, ux = self._interval_bounds(x_sym, problem)
            lbx, ubx = (lx, ux) if (lx is not None and ux is not None) else (lbx, ubx)

        if lby is None or uby is None:
            ly, uy = self._interval_bounds(y_sym, problem)
            lby, uby = (ly, uy) if (ly is not None and uy is not None) else (lby, uby)

        # 仍缺界：不要 Big-M；直接不线性化
        if lbx is None or ubx is None or lby is None or uby is None:
            return x_sym * y_sym


        # ======== 4) special-case: square (x*x) ========
        if x_sym.name == y_sym.name:
            l, u = lbx, ubx

            # 正确的平方 bounds：min 在 0（若区间跨 0），否则 min(l^2,u^2)
            w_lb = 0.0 if (l <= 0 <= u) else min(l*l, u*u)
            w_ub = max(l*l, u*u)

            # 创建/收紧 w
            if w_name not in problem.variables:
                problem.add_variable(w_name, lb=w_lb, ub=w_ub, vtype="continuous")
            else:
                v = problem.variables[w_name]
                v.lb = w_lb if v.lb is None else max(v.lb, w_lb)
                v.ub = w_ub if v.ub is None else min(v.ub, w_ub)

            w_sym = self._sym(problem, w_name)
            _maybe_link_w_to_Z(w_sym, x_sym, y_sym)

            # 线性 envelope（LP 版 convex hull 近似：两条端点切线 + 一条弦线）
            cons = [
                (w_sym - (2*l * x_sym - l*l), '>=', 0),                 # tangent at l
                (w_sym - (2*u * x_sym - u*u), '>=', 0),                 # tangent at u
                (w_sym - ((l+u) * x_sym - l*u), '<=', 0),               # secant
            ]
            
            if self.enable_mccormick_square_midpoint_tangent:
                t = 0.5 * (l + u)
                cons.append((w_sym - (2 * t * x_sym - t * t), '>=', 0))

            # 跨 0 时加 w>=0（能显著防止 w 被推负）
            if self.enable_mccormick_square_nonneg_cut and l <= 0 <= u:
                cons.append((w_sym, '>=', 0))

            for expr, sense, rhs in cons:
                problem.add_constraint_unique(expr, sense, rhs)

            self.w_cache[key] = w_sym
            if not hasattr(problem, "_bilinear_map"):
                problem._bilinear_map = {}
            problem._bilinear_map[w_name] = (x_sym, y_sym)

            return w_sym
        # ======== square special-case end ========

        # ======== 5) general bilinear (x*y, x!=y) ========
        w_lb = min(lbx*lby, lbx*uby, ubx*lby, ubx*uby)
        w_ub = max(lbx*lby, lbx*uby, ubx*lby, ubx*uby)

        if w_name not in problem.variables:
            problem.add_variable(w_name, lb=w_lb, ub=w_ub, vtype="continuous")
        else:
            v = problem.variables[w_name]
            v.lb = w_lb if v.lb is None else max(v.lb, w_lb)
            v.ub = w_ub if v.ub is None else min(v.ub, w_ub)

        w_sym = self._sym(problem, w_name)
        _maybe_link_w_to_Z(w_sym, x_sym, y_sym)

        cons = [
            (w_sym - (lbx * y_sym + lby * x_sym - lbx * lby), '>=', 0),
            (w_sym - (ubx * y_sym + uby * x_sym - ubx * uby), '>=', 0),
            (w_sym - (ubx * y_sym + lby * x_sym - ubx * lby), '<=', 0),
            (w_sym - (lbx * y_sym + uby * x_sym - lbx * uby), '<=', 0),
        ]
        for expr, sense, rhs in cons:
            problem.add_constraint_unique(expr, sense, rhs)

        self.w_cache[key] = w_sym
        if not hasattr(problem, "_bilinear_map"):
            problem._bilinear_map = {}
        problem._bilinear_map[w_name] = (x_sym, y_sym)

        return w_sym



    # ---------- 标量辅助 ----------
    def _scalarize_trace_expr(self, expr):
        """Convert Trace/1x1 matrix scalar expressions into scalar SymPy exprs."""
        import sympy as sp

        if isinstance(expr, sp.Trace):
            return self._scalarize_trace(expr.arg)

        if isinstance(expr, sp.MatrixExpr) and getattr(expr, "shape", None) == (1, 1):
            return self._scalarize_trace(expr)

        if isinstance(expr, sp.Add):
            return sp.Add(*(self._scalarize_trace_expr(a) for a in expr.args))

        if isinstance(expr, sp.Mul):
            return sp.Mul(*(self._scalarize_trace_expr(a) for a in expr.args))

        if isinstance(expr, sp.Pow):
            return sp.Pow(self._scalarize_trace_expr(expr.base), expr.exp)

        return expr

    def _is_affine_scalar(self, expr, problem):
        """是否纯仿射标量表达式（常数 + 线性变量和）"""
        expr = self._scalarize_trace_expr(expr)
        if isinstance(expr, sympy.matrices.expressions.matexpr.MatrixElement):
            return True
        if expr.is_Number or isinstance(expr, sympy.Symbol):
            return True
        if expr.is_Add:
            return all(self._is_affine_scalar(a, problem) for a in expr.args)
        if expr.is_Mul:
            const, sym = expr.as_coeff_Mul()
            return sym == 1 or (isinstance(sym, sympy.Symbol) and const.is_Number)
        return False

    def _affine_scalar_to_var(self, problem, location, expr):
        """
        把仿射标量 expr 封装为新变量 t，并自动求界：
            lb_t = Σ min(cᵢ·vᵢ) ,  ub_t = Σ max(cᵢ·vᵢ)
        若已存在同名变量，则同步更新其 lb / ub。
        """
        if isinstance(expr, sympy.Symbol):
            return expr                        # 已是符号，无需再包

        expr = self._scalarize_trace_expr(expr)
        expr = self._elements_to_scalars(problem, expr, location)

        # -------- 1. 推断上下界 ----------------------------------
        lb, ub = self._infer_affine_bounds(expr, problem)   # ← 新界 (可为 None)

        # -------- 2. 创建 / 复用变量 ------------------------------
        raw_key = sympy.srepr(sympy.expand(expr))   # 用 srepr 做“结构化字符串”键
        key = (id(problem), raw_key)
        if key in self.affine_cache:            # 已经生成过
            return self.affine_cache[key]
        # t_name = f"t_{self.affine_counter}"

        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:12]
        t_name = f"t_{digest}"

        if t_name not in problem.variables:
            problem.add_variable(t_name, lb=lb, ub=ub)
        else:
            v = problem.variables[t_name]
            if lb is not None:
                v.lb = lb if v.lb is None else max(v.lb, lb)
            if ub is not None:
                v.ub = ub if v.ub is None else min(v.ub, ub)

        t_sym = self._sym(problem, t_name)

        # -------- 3. 加等式约束 ----------------------------------
        problem.add_constraint_unique(t_sym - expr, '=', 0)
        self.affine_cache[key] = t_sym
        return t_sym


    def _vector_affine_to_var(self, problem, location, vec):
        """
        将列向量仿射表达式 vec (n×1) 封装为 MatrixSymbol T，
        并为每个元素 T_i 自动推断 (lb, ub)。
        若 vec 已是 MatrixSymbol，直接返回原符号。
        """
        if isinstance(vec, sympy.MatrixSymbol):
            return vec

        n, _ = vec.shape
        # T_name = f"T_{self.vec_affine_counter}"
        raw_key = sympy.srepr(sympy.expand(vec))
        key = (id(problem), raw_key)
        if key in self.vec_affine_cache:
            return self.vec_affine_cache[key]

        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:12]
        T_name = f"T_{digest}"
        # self.vec_affine_counter += 1   # 可保留也可删掉（不再用于命名）


        # NEW: 如果 problem 里已经有这个向量变量，直接复用
        if T_name in getattr(problem, "matrix_variables", {}):
            mv = problem.matrix_variables[T_name]  # VectorVariableSymbol

            # 统一优先用 mv.symbol；兼容你曾经写入过的 mv.sym
            T = getattr(mv, "symbol", None)
            if T is None:
                T = getattr(mv, "sym", None)

            # 极端兜底：两者都没有才补一个，并写回 mv.symbol（必要时也同步 mv.sym）
            if T is None:
                T = sympy.MatrixSymbol(T_name, n, 1)
                mv.symbol = T
                # 可选：如果你担心旧代码仍访问 mv.sym，可以同步一份
                mv.sym = T
        else:
            # add_vector_variable 本来就返回 vec.symbol（MatrixSymbol），无需再挂 mv.sym
            T = problem.add_vector_variable(T_name, n)

        # self.vec_affine_counter += 1

        lb_vec =  float("inf")
        ub_vec = -float("inf")
        # Per-element bounds (for tighter SDP+RLT later)
        lb_list = [None] * n
        ub_list = [None] * n

        for i in range(n):
            expr_i = vec[i, 0]

            # ---------- 1. MatrixElement → 标量 Symbol ----------
            repl = {}
            for me in expr_i.atoms(sympy.matrices.expressions.matexpr.MatrixElement):
                parent = me.parent if hasattr(me, "parent") else me.base
                # parent 是 MatrixSymbol，取出对应的向量变量对象 mv
                mv = problem.matrix_variables.get(parent.name, None)

                # 只在确实是“注册过的向量变量”时继承逐元素界
                lb_i = ub_i = None
                if mv is not None and getattr(parent, "shape", None) is not None and parent.shape[1] == 1:
                    idx_i = int(me.i)
                    lb_i, ub_i = self._elem_bounds(mv, idx_i)  # 逐元素标量界

                scalar_sym = self._ensure_elem_var(problem, parent, int(me.i), lb_i, ub_i)

                repl[me] = scalar_sym
            expr_scalar = expr_i.xreplace(repl)

            # ---------- 2. 推界 ----------
            lb_i = ub_i = None

            # 2.1 如果 expr_scalar 是一个已注册的标量决策变量：直接继承它的 bounds
            if isinstance(expr_scalar, sympy.Symbol) and expr_scalar.name in problem.variables:
                vsrc = problem.variables[expr_scalar.name]
                lb_i, ub_i = vsrc.lb, vsrc.ub
            else:
                # 2.2 否则才走通用仿射推界
                lb_i, ub_i = self._infer_affine_bounds(expr_scalar, problem)
            
            if lb_i is not None:
                lb_vec = min(lb_vec, lb_i)
                lb_list[i] = float(lb_i)
            if ub_i is not None:
                ub_vec = max(ub_vec, ub_i)
                ub_list[i] = float(ub_i)
            # ---------- 3. 创建 / 更新 T_i ----------
            ti = self._ensure_elem_var(problem, T, i,
                                    lb_i if lb_i is not None else -1e4,
                                    ub_i if ub_i is not None else  1e4)

            vinfo = problem.variables[ti.name]
            vinfo.lb = lb_i if lb_i is not None else vinfo.lb
            vinfo.ub = ub_i if ub_i is not None else vinfo.ub

            # ---------- 4. 等式约束 ----------
            problem.add_constraint_unique(ti - expr_scalar, '=', 0)

        mv_info = problem.matrix_variables[T_name]        # VectorVariableSymbol
        # Prefer per-element bounds when available; fall back to global scalar bounds.
        if any(v is not None for v in lb_list) or any(v is not None for v in ub_list):
            mv_info.lb = lb_list
            mv_info.ub = ub_list
        else:
            mv_info.lb = None if lb_vec == float("inf")  else lb_vec
            mv_info.ub = None if ub_vec == -float("inf") else ub_vec
        self.vec_affine_cache[key] = T
        return T

    # ======== 区间传播主入口 ========
    def _interval_bounds(self, expr, problem):
        import sympy as sp
        # 0) 常数
        try:
            if not hasattr(expr, "free_symbols") or len(expr.free_symbols) == 0:
                c = float(expr)
                return c, c
        except Exception:
            pass

        # 1) Matrix -> 标量桥接
        # 1a) 1x1 MatrixExpr → 取 (0,0)
        if isinstance(expr, sp.MatrixExpr) and getattr(expr, "shape", None) == (1, 1):
            return self._interval_bounds(expr[0, 0], problem)

        # 1b) Trace：先把 Trace(α*M) 规整为 α*Trace(M)
        if isinstance(expr, sp.Trace):
            alpha, tr = self._pull_scalar_times_trace(expr)
            if tr is None:
                # 理论上 Trace 就会命中；否则退回
                return None, None
            inner = tr.arg
            # Trace 的线性性：Trace(A+B)=Trace(A)+Trace(B)
            if isinstance(inner, (sp.Add, sp.MatAdd)):
                lb, ub = 0.0, 0.0
                for a in inner.args:
                    la, ua = self._interval_bounds(sp.Trace(a), problem)
                    if la is None: return None, None
                    lb += la; ub += ua
                return float(alpha)*lb, float(alpha)*ub
            # Trace(M) 的下界：把 M 视作标量表达式（若是 1x1 或进一步可化）
            # 常见情形：Trace(x.T*Q*x) / Trace(A.T*B) 等——交给下方二次/乘积规则
            l0, u0 = self._interval_bounds(self._scalarize_trace(inner), problem)
            if l0 is None: return None, None
            return float(alpha)*l0, float(alpha)*u0

        # 2) 仿射（线性 + 常数）：用已有函数
        try:
            if expr.is_polynomial() and sp.total_degree(expr) <= 1:
                lb, ub = self._infer_affine_bounds(expr, problem)
                if lb is not None and ub is not None:
                    return float(lb), float(ub)
        except Exception:
            pass

        # 3) 加减
        if isinstance(expr, sp.Add):
            lb, ub = 0.0, 0.0
            for a in expr.args:
                la, ua = self._interval_bounds(a, problem)
                if la is None: return None, None
                lb += la; ub += ua
            return lb, ub

        # 4) 乘法
        if isinstance(expr, sp.Mul):
            # 分离标量常数系数
            c, rest = sp.sympify(expr).as_coeff_Mul()
            if rest == 1:
                return float(c), float(c)
            lr, ur = self._interval_bounds(rest, problem)
            if lr is None: return None, None
            return self._bounds_prod((float(c), float(c)), (lr, ur))

        # 5) 幂
        if isinstance(expr, sp.Pow):
            base, exp = expr.base, expr.exp
            lb, ub = self._interval_bounds(base, problem)
            if lb is None: return None, None
            # 只处理数字指数或 ±1/2 常见情形
            if exp.is_integer:
                k = int(exp)
                if k == 1:  return lb, ub
                if k == 2:  # 偶次幂
                    cands = [lb**2, ub**2]
                    if lb <= 0 <= ub: cands.append(0.0)
                    return min(cands), max(cands)
                if k % 2 == 1:  # 奇次幂：单调
                    return (lb**k, ub**k) if lb <= ub else (ub**k, lb**k)
                # 其它整数：保守处理，可扩展
                return None, None
            # 处理 1/2 -> sqrt
            if exp == sp.Rational(1,2):
                if lb < 0: return None, None
                return (lb**0.5, ub**0.5)
            return None, None

        # 6) 单调与常见函数
        if expr.func.__name__ == 'Abs':
            (lb, ub) = self._interval_bounds(expr.args[0], problem)
            if lb is None: return None, None
            if lb <= 0 <= ub:
                return 0.0, max(abs(lb), abs(ub))
            return min(abs(lb), abs(ub)), max(abs(lb), abs(ub))

        if expr.func.__name__ == 'exp':
            lb, ub = self._interval_bounds(expr.args[0], problem)
            if lb is None: return None, None
            import math
            return math.exp(lb), math.exp(ub)

        if expr.func.__name__ == 'log':
            lb, ub = self._interval_bounds(expr.args[0], problem)
            if lb is None or lb <= 0: return None, None
            import math
            return math.log(lb), math.log(ub)

        if expr.func.__name__ == 'sqrt':
            lb, ub = self._interval_bounds(expr.args[0], problem)
            if lb is None or lb < 0: return None, None
            import math
            return math.sqrt(lb), math.sqrt(ub)

        if expr.func.__name__ in ('sin', 'cos'):
            # 简易安全版本：跨度≥2π → [-1,1]；否则端点+极值点
            t = expr.args[0]
            lb, ub = self._interval_bounds(t, problem)
            if lb is None: return None, None
            import math
            if ub - lb >= 2*math.pi - 1e-9:
                return -1.0, 1.0
            # 枚举端点与 kπ/2
            critical = [lb, ub]
            kmin = int(math.floor((lb - math.pi/2)/ (math.pi/2)))
            kmax = int(math.ceil ((ub - math.pi/2)/ (math.pi/2)))
            for k in range(kmin-1, kmax+2):
                x = k * (math.pi/2)
                if lb <= x <= ub: critical.append(x)
            vals = [float(sp.sin(x)) if expr.func.__name__=='sin' else float(sp.cos(x)) for x in critical]
            return min(vals), max(vals)

        # 7) 特殊识别：双线性 / 二次型
        # 7a) 标量双线性（两符号相乘）
        if isinstance(expr, sp.Mul) and len(expr.args) == 2:
            a, b = expr.args
            la, ua = self._interval_bounds(a, problem)
            lb, ub = self._interval_bounds(b, problem)
            if la is not None and lb is not None:
                return self._bounds_prod((la, ua), (lb, ub))

        # 7b) x^T Q x 或 迹形式（弱保守：逐项 McCormick）
        ok, alpha, x_sym, Q_expr = self.is_scaled_xTQx(expr)
        if ok and hasattr(Q_expr, "__getitem__"):
            n, _ = x_sym.shape
            # 元素级变量边界（从父向量继承）
            xi_bounds = []
            for i in range(n):
                xi = x_sym[i, 0]
                xi_s = self._elements_to_scalars(problem, xi, "Bounds")
                lb_i, ub_i = self._get_bounds(problem, xi_s)
                if lb_i is None or ub_i is None: return None, None
                xi_bounds.append((float(lb_i), float(ub_i)))
            # 逐项累加区间
            L, U = 0.0, 0.0
            for i in range(n):
                for j in range(n):
                    qij = Q_expr[i, j]
                    if qij == 0: continue
                    lij, uij = self._bilin_box_bounds(xi_bounds[i], xi_bounds[j])
                    if qij >= 0:
                        L += float(qij) * lij
                        U += float(qij) * uij
                    else:
                        L += float(qij) * uij
                        U += float(qij) * lij
            return float(alpha)*L, float(alpha)*U

        # 8) 兜底：失败
        return None, None


    # --- 把 Trace(inner) 尽量转成标量可界的表达式 ---
    def _scalarize_trace(self, inner):
        import sympy as sp
        try:
            shape = getattr(inner, "shape", None)
            if shape == (1, 1):
                return sp.expand(inner[0, 0].doit())
            if shape is not None and shape[0] == shape[1]:
                n = int(shape[0])
                return sp.expand(sum(inner[i, i].doit() for i in range(n)))
        except Exception:
            pass
        return sp.Trace(inner)

    # --- 区间乘积 ---
    def _bounds_prod(self, I, J):
        (a, b), (c, d) = I, J
        cand = [a*c, a*d, b*c, b*d]
        return min(cand), max(cand)

    # --- 给标量盒 [l1,u1]×[l2,u2] 的双线性项 x*y 的 McCormick 区间 ---
    def _bilin_box_bounds(self, I1, I2):
        l1, u1 = I1; l2, u2 = I2
        cand = [l1*l2, l1*u2, u1*l2, u1*u2]
        return min(cand), max(cand)


    
    def _ensure_elem_var(self, problem, mat_sym, idx, lb=None, ub=None):
        """
        支持：
        - 向量元素 idx=int
        - 矩阵元素 idx=(i,j)
        统一生成干净的标量变量名：X_0 或 Z_0_1

        修正点（最小且“止血”）：
        1) 对派生对象（w_, t_, T_, tau_, Z*）不做初始化/同步收紧（lb/ub 强制 None）
        2) 若元素变量已存在且已有有效界，则不再用传入 lb/ub 去二次“同步收紧”
        （避免跨-problem 污染、重复初始化把界撞死）
        """
        import sympy
        import math

        # ---- name ----
        if isinstance(idx, tuple):
            i, j = idx
            name = f"{mat_sym.name}_{int(i)}_{int(j)}"
        else:
            name = f"{mat_sym.name}_{int(idx)}"

        # ---- 派生对象：一律不初始化界 ----
        SKIP_PREFIX = ("w_", "t_", "T_", "tau_")
        if str(mat_sym.name).startswith(SKIP_PREFIX):
            lb, ub = None, None

        # ---- create ----
        if name not in problem.variables:
            problem.add_variable(name, lb=lb, ub=ub, vtype='continuous')
            return sympy.Symbol(name)

        # ---- already exists ----
        v = problem.variables[name]

        # 关键保护：若已有界（尤其是 x/y 元素），不要再用传入的 lb/ub 去“同步收紧”
        # 这样可以彻底避免：某处错误/污染的 lb/ub 把已有正确界（如 [-2,2]）撞成 [2,2]
        has_lb = (v.lb is not None) and (not (isinstance(v.lb, float) and (math.isnan(v.lb) or math.isinf(v.lb))))
        has_ub = (v.ub is not None) and (not (isinstance(v.ub, float) and (math.isnan(v.ub) or math.isinf(v.ub))))
        if has_lb and has_ub:
            return sympy.Symbol(name)

        # ---- 原来的“同步/收紧已有变量界”逻辑：只在已有界不完整时才允许补全 ----
        cur_lb = v.lb
        cur_ub = v.ub
        new_lb = cur_lb
        new_ub = cur_ub

        if lb is not None:
            new_lb = lb if cur_lb is None else max(cur_lb, lb)
        if ub is not None:
            new_ub = ub if cur_ub is None else min(cur_ub, ub)

        # 安全：避免把变量界改成空区间 -> 人为 infeasible
        if (new_lb is not None) and (new_ub is not None) and (new_lb > new_ub):
            # 建议至少留日志定位
            # print(f"[WARN] Skip tightening {name}: [{cur_lb},{cur_ub}] ∩ [{lb},{ub}] is empty")
            pass
        else:
            v.lb = new_lb
            v.ub = new_ub

        return sympy.Symbol(name)




    def _init_Z_element_bounds(self, problem, Z_sym, x_var, tol: float = 1e-9, create_scalar: bool = False):
        """
        Initialize bounds for Z elements using x bounds:
        Z_ij ? x_i * x_j (conservative box bounds).
        Writes to problem._me_bounds and optionally creates scalar Z_i_j vars.
        """
        if x_var is None:
            return
        if not hasattr(problem, "tighten_me_bounds"):
            return

        try:
            n = int(getattr(Z_sym, "shape", (0, 0))[0])
        except Exception:
            return

        for i in range(n):
            li, ui = self._elem_bounds(x_var, i)
            if li is None or ui is None:
                continue

            # diag bounds
            min_sq = 0.0 if (li <= 0.0 <= ui) else min(li * li, ui * ui)
            max_sq = max(li * li, ui * ui)
            problem.tighten_me_bounds(Z_sym.name, i, i, min_sq, max_sq, tol=tol)
            for j in range(i + 1, n):
                lj, uj = self._elem_bounds(x_var, j)
                if lj is None or uj is None:
                    continue
                cand = [li * lj, li * uj, ui * lj, ui * uj]
                Lij, Uij = min(cand), max(cand)
                problem.tighten_me_bounds(Z_sym.name, i, j, Lij, Uij, tol=tol)
                problem.tighten_me_bounds(Z_sym.name, j, i, Lij, Uij, tol=tol)


    def _parent_bounds(self, problem, mat_sym):
        """
        参数
        -------
        problem : QCQPProblem
        mat_sym : MatrixSymbol  或 其转置 .T 的 MatrixExpr

        返回
        -------
        (lb, ub) : Tuple[float | None, float | None]
            若未设置界，返回 (None, None)
        """
        import sympy as sp
        base_sym = mat_sym.arg if isinstance(mat_sym, sp.Transpose) else mat_sym
        if not isinstance(base_sym, sp.MatrixSymbol):
            return None, None
        mv = problem.matrix_variables.get(base_sym.name, None)
        if mv is None:
            return None, None
        return mv.lb, mv.ub
    
    def _elements_to_scalars(self, problem, expr, location):
        """
        将表达式 expr 中出现的 MatrixElement 统一替换为标量 Symbol：
        - 如果元素来自已注册的 MatrixSymbol（出现在 problem.matrix_variables），
        则用 _ensure_elem_var 创建/复用对应标量变量：
            * 向量元素 x_i 继承元素级 (lb, ub)
            * 矩阵元素 A_{i,j} 不继承 bounds（尤其 Z），避免错误 tightening
        - 否则（例如 (Q.T*y)[i,0] 这类仿射向量的元素），视为仿射标量，
        用 _affine_scalar_to_var 封装为 t_k，并自动推断界与加等式约束。
        返回替换后的表达式（标量 Expr）。
        """
        import sympy
        from sympy.matrices.expressions.matexpr import MatrixElement as _ME

        if not hasattr(expr, "atoms"):
            return expr

        repl = {}
        for me in expr.atoms(_ME):
            parent = getattr(me, "parent", None) or getattr(me, "base", None)

            # A) 注册过的 MatrixSymbol：变量元素
            if isinstance(parent, sympy.MatrixSymbol) and parent.name in getattr(problem, "matrix_variables", {}):
                is_vector = (getattr(parent, "shape", None) is not None and parent.shape[1] == 1)

                if is_vector:
                    mv = problem.matrix_variables.get(parent.name, None)
                    idx = int(me.i)

                    lb_i = ub_i = None
                    if mv is not None:
                        lb_i, ub_i = self._elem_bounds(mv, idx)   # 元素级 bounds（你期望的）

                    scalar_sym = self._ensure_elem_var(problem, parent, idx, lb_i, ub_i)

                else:
                    ii = int(me.i) if hasattr(me, "i") else 0
                    jj = int(me.j) if hasattr(me, "j") else 0

                    lb_ij = ub_ij = None
                    mv = problem.matrix_variables.get(parent.name, None)
                    if mv is not None and str(parent.name).startswith("Z_"):
                        # 如果你在 apply_sdr 里写了 Z_mv.lb/ub 的二维数组，这里就同步到标量元素
                        lb_ij, ub_ij = self._elem_bounds(mv, (ii, jj))

                    scalar_sym = self._ensure_elem_var(problem, parent, (ii, jj), lb=lb_ij, ub=ub_ij)

                repl[me] = scalar_sym

            # B) 非注册 MatrixSymbol 的 MatrixElement：仿射标量 t_k
            else:
                repl[me] = self._affine_scalar_to_var(problem, location, me)

        if not repl:
            return expr

        out = expr.xreplace(repl)
        return out[0, 0] if isinstance(out, sympy.MatrixExpr) and out.shape == (1, 1) else out

    
    # -------- NEW: 线性系数剥离（最小补丁） -------------------
    def _strip_linear_coeff(self, expr):
        """
        将线性项 c*sym 规范化为 (c, sym)；
        非上述形式返回 (1, expr)。
        仅用于 _linearize_pair 的安全检查（不改变上层乘法的系数语义）。
        """
        if isinstance(expr, sympy.Symbol):
            return sympy.Integer(1), expr
        if isinstance(expr, sympy.Mul):
            c, rest = sympy.sympify(expr).as_coeff_Mul()
            if isinstance(rest, sympy.Symbol) and (c.is_Number or c.is_NumberSymbol):
                return c, rest
        if isinstance(expr, MatrixElement):
            return sympy.Integer(1), expr
        return sympy.Integer(1), expr

    def _safe_var_name(self, handle):
        """从 Symbol / MatrixElement / Indexed 安全取变量名；失败返回 None。"""
        if isinstance(handle, sympy.Symbol):
            return handle.name
        if isinstance(handle, MatrixElement):
            base = handle.parent if hasattr(handle, "parent") else handle.base
            return getattr(base, "name", None)
        if isinstance(handle, sympy.Indexed):
            base = handle.base
            return getattr(base, "name", None) or str(getattr(base, "label", None))
        return getattr(handle, "name", None)
    
    
    # 放在 RelaxationEngine 里（或工具模块均可）
    def _is_mccormick_proxy_symbol(self, problem, sym):
        if not isinstance(sym, sympy.Symbol):
            return False
        bil_map = getattr(problem, "_bilinear_map", None)
        return isinstance(bil_map, dict) and sym.name in bil_map

    def _left_right_factor(self, t1, t2):
        """
        t1, t2 均为 MatMul / MatrixExpr
        若存在公共 *左* 因子  L ： t1 = L*B , t2 = L*C  -> 返回 (L,  B, C)
        若不存在公共左因子，则再尝试公共 *右* 因子  R ： t1 = B*R , t2 = C*R
        找不到时返回 (None, None, None)
        """
        # ---- 1. 统一转成 MatMul ----
        t1_args = t1.args if isinstance(t1, sympy.MatMul) else (t1,)
        t2_args = t2.args if isinstance(t2, sympy.MatMul) else (t2,)

        # ---- 2. 尝试公共 *左* 因子 ----
        common_left = []
        for a, b in zip(t1_args, t2_args):
            if a == b:
                common_left.append(a)
            else:
                break
        if common_left:
            L = sympy.MatMul(*common_left)
            B = sympy.MatMul(*t1_args[len(common_left):]) if len(t1_args) > len(common_left) else sympy.eye(1)
            C = sympy.MatMul(*t2_args[len(common_left):]) if len(t2_args) > len(common_left) else sympy.eye(1)
            return L, B, C

        # ---- 3. 尝试公共 *右* 因子 ----
        common_right = []
        for a, b in zip(t1_args[::-1], t2_args[::-1]):
            if a == b:
                common_right.insert(0, a)
            else:
                break
        if common_right:
            R = sympy.MatMul(*common_right)
            B = sympy.MatMul(*t1_args[:-len(common_right)]) if len(common_right) else sympy.eye(1)
            C = sympy.MatMul(*t2_args[:-len(common_right)]) if len(common_right) else sympy.eye(1)
            # 返回形式与左因子一致：  L, B, C
            return R, B, C

        return None, None, None

    def _unwrap_1x1_for_key(self, expr):
        if isinstance(expr, sympy.MatrixBase) and getattr(expr, "shape", None) == (1, 1):
            return expr[0, 0]
        return expr

    def _same_term(self, a, b):
        ak = sympy.srepr(self._unwrap_1x1_for_key(a))
        bk = sympy.srepr(self._unwrap_1x1_for_key(b))
        return ak == bk


    def apply_action(self, problem: ps.QCQPProblem, location, sub_expr, action_type, extra_args=None):
        """
        统一的入口：根据 action_type 调用相应的重写函数。
        :param problem: 目标QCQPProblem
        :param location: 字符串, 表示在"objective"或"constraint_i_lhs"等位置
        :param sub_expr: Sympy子表达式(想改写的对象)
        :param action_type: 字符串, 比如 "expand", "simplify", "cancel", "remove_log", ...
        :param extra_args: 可能需要的额外信息(比如 divide_by_var = 'x1')
        """
        self.last_rewrite = None

        # print(f"[ACTION] {action_type} @ {location} sub={sympy.srepr(sub_expr)}")
        
        
        # fixed-variable preprocessing moved to QCQPProblem.map_all_terms()

        if action_type == "expand":
            self.apply_expand(problem, location, sub_expr)
        elif action_type == "simplify":
            self.apply_simplify(problem, location, sub_expr)
        elif action_type == "cancel":
            self.apply_cancel(problem, location, sub_expr)
        elif action_type == "expand_log":
            self.apply_expand_log(problem, location, sub_expr)
        elif action_type == "logcombine":
            self.apply_logcombine(problem, location, sub_expr)
        elif action_type == "remove_log":
            self.apply_remove_log(problem, location, sub_expr)
        elif action_type == "relax_integrality":
            self.apply_relax_integrality(problem, location, sub_expr)
        elif action_type == "add_binary_obj":
            self.apply_add_binary_obj(problem, location, sub_expr, l_value=0.5)

        elif action_type == "remove_fraction":
            lam_name = extra_args.get("name") if extra_args else None
            self.apply_remove_fraction(problem, location, sub_expr, lamda=lam_name, allow_bigM_fallback=False)
        
        elif action_type == "mccormick_relaxation":
            self.apply_mccormick_relaxation(problem, location, sub_expr)
        elif action_type == "trace_transformation":
            self.apply_trace_transformation(problem, location, sub_expr)
        elif action_type == "first_order_taylor":
            var_name = extra_args.get("var_name") if extra_args else None
            x0_sym = extra_args.get("x0") if extra_args else None
            self.apply_first_order_taylor(problem, location, sub_expr, var_name, x0_sym)
        elif action_type == "remove_abs":
            self.apply_remove_abs(problem, location, sub_expr)
        elif action_type == "sdp_relaxation":
            self.apply_sdr(problem, location, sub_expr)
        elif action_type == "factor_merge":
            self.apply_partial_factor_merge(problem, location, sub_expr)
        elif action_type == "spectral_psd_projection":
            self.apply_spectral_psd_projection(problem, location, sub_expr)
        elif action_type == "diagonal_relaxation":
            self.apply_diagonal_relaxation(problem, location, sub_expr)
        elif action_type == "bound_tightening":
            changed = self.apply_bound_tightening(problem, max_rounds=8, tol=1e-9)
            self.last_rewrite = {"location": "GLOBAL", "old": 0, "new": 1} if changed \
                                else {"location": "GLOBAL", "old": 0, "new": 0}

        elif action_type == "global_cut_generation":
            changed = self.apply_global_cut_generation(problem, rlt_budget=200, oa_budget=20, tol=1e-9)
            self.last_rewrite = {"location": "GLOBAL", "old": 0, "new": 1} if changed \
                                else {"location": "GLOBAL", "old": 0, "new": 0}

        elif action_type == "qcr":
            # 对非凸二次型加 qcr 的对角扰动/凸化（通常需要 bounds）
            # 例：self.apply_qcr(problem, location=location, sub_expr=sub_expr, ...)
            # 如果你做成全局扫描也可以不传 location
            self.apply_qcr(problem, location=location, sub_expr=sub_expr)

        elif action_type == "perspective_relaxation":
            # 0/1 开关的透视/开关松弛（perspective reformulation/relaxation）
            # 通常需要识别 y∈{0,1} 与 x 的 on/off 结构或 x <= Uy 之类的结构
            self.apply_perspective_relaxation(problem, location=location, sub_expr=sub_expr)
        else:
            print(f"Unknown action: {action_type}")
        problem.map_all_terms()
        # print(f"[ACTION] done last_rewrite={self.last_rewrite is not None}")
        if self.last_rewrite is None:
            # 兜底记账：用 sub_expr 做 witness
            self._mark_identity_rewrite(location, sub_expr)
        return self.last_rewrite

    def apply_expand(self, problem, location, sub_expr):
        """
        用 sympy.expand 对 sub_expr 进行括号展开(包括合并同类项)。
        """
        expanded_expr = sympy.expand(sub_expr)
        expanded_expr = self._linearize_trace(expanded_expr)
        # 替换
        self._replace_in_problem(problem, location, sub_expr, expanded_expr)

    def apply_simplify(self, problem, location, sub_expr):
        """
        用 sympy.simplify 对 sub_expr 进行综合化简(含合并项, 三角恒等式等).
        """
        simplified_expr = sympy.simplify(sub_expr)
        self._replace_in_problem(problem, location, sub_expr, simplified_expr)

    def apply_cancel(self, problem, location, sub_expr):
        """
        用 sympy.cancel 约去分子分母中共同因子.
        """
        canceled_expr = sympy.cancel(sub_expr)
        self._replace_in_problem(problem, location, sub_expr, canceled_expr)

    def apply_expand_log(self, problem, location, sub_expr):
        """
        log(a*b) -> log(a) + log(b), log(a^2) -> 2*log(a), etc.
        """
        new_expr = sympy.expand_log(sub_expr)
        self._replace_in_problem(problem, location, sub_expr, new_expr)

    def apply_logcombine(self, problem, location, sub_expr):
        """
        log(a)+log(b)->log(a*b),  2*log(a)->log(a^2)等
        """
        new_expr = sympy.logcombine(sub_expr, force=True)
        self._replace_in_problem(problem, location, sub_expr, new_expr)
        
        
    # ------------------------------------------------------------
    #是否是 “单变量取值范围” 约束
    # ------------------------------------------------------------
    def _is_bound_constraint(self, cons: ps.Constraint, var_sym):
        """
        若 cons 仅含 var_sym 且形如
            var ≥ c   或   var ≤ c   或   var == c
        返回 True；否则 False
        """
        # 1) 左式若是 var 本身
        if cons.expr != var_sym:
            return False
        # 2) 右端必须是数字常量或变量类型
        if not isinstance(cons.rhs, (int, float, sympy.Number, sympy.Symbol, str)):
            return False
        # 3) sense 限定
        return cons.sense in ('>=', '<=', '=', 'is')


    def apply_relax_integrality(self, problem, _, sub_expr):
        """
        查找 sub_expr 中所有决策变量（标量/向量/矩阵），
        若 vtype ∈ {binary, integer} 则放松为 continuous，并重建 bound 约束。
        """
        any_change = False
        # 1) 收集：支持两张表
        var_to_handles: dict[str, set] = {}
        for vname, handle in self._iter_var_handles(sub_expr):
            if self._is_decision_name(problem, vname):
                var_to_handles.setdefault(vname, set()).add(handle)

        if not var_to_handles:
            print("[Integrality] No decision variables inside sub_expr.")
            return

        # 2) 逐变量放松
        for vname, handles in var_to_handles.items():
            v_obj, kind = self._get_var_obj(problem, vname)
            if v_obj is None:
                continue

            if getattr(v_obj, "vtype", "continuous") not in self.DISCRETE_VTYPES:
                continue  # 已经是连续

            # 2.1 移除旧的元素级/标量 bound 约束
            removed = self._remove_old_bounds(problem, vname, v_obj, handles)
            if removed:
                print(f"[Integrality] removed {removed} bound-constraint(s) of {vname}")

            # 2.2 修改变量属性
            # preserve fixed bounds (e.g., from preprocess) when relaxing integrality
            fixed = (v_obj.lb is not None and v_obj.ub is not None and abs(float(v_obj.lb) - float(v_obj.ub)) <= 1e-12)
            if v_obj.vtype == "binary" and not fixed:
                v_obj.lb, v_obj.ub = 0.0, 1.0
            # integer：保留原 lb/ub
            v_obj.vtype = "continuous"

            # 2.3 重建新 bound（标量/向量/矩阵都覆盖）
            any_handle = next(iter(handles))
            self._add_new_bounds(problem, vname, v_obj, any_handle)
            
            # 只要进入这里就是一次真实修改（vtype 已变）
            any_change = True
            print(f"[Integrality] {vname}: binary/integer ➞ continuous (lb={v_obj.lb}, ub={v_obj.ub})")
        if any_change:
            self.last_rewrite = {
                "location": "GLOBAL",
                "old": 0,
                "new": 1
            }

    def _term_has_discrete(self, problem: ps.QCQPProblem, expr) -> bool:
        DISCRETE = {"integer", "binary"}

        # 标量 Symbol
        for s in expr.free_symbols:
            v = problem.variables.get(s.name)
            if v and v.vtype in DISCRETE:
                return True

        # MatrixSymbol（含出现在 .T / MatrixElement 里的底层）
        mats = set(expr.atoms(sympy.MatrixSymbol))
        for tr in expr.atoms(sympy.Transpose):
            if isinstance(tr.arg, sympy.MatrixSymbol):
                mats.add(tr.arg)
        for me in expr.atoms(MatrixElement):
            base = getattr(me, "parent", None) or getattr(me, "base", None)
            if isinstance(base, sympy.MatrixSymbol):
                mats.add(base)

        for ms in mats:
            mv = problem.matrix_variables.get(ms.name)
            if mv and getattr(mv, "vtype", "continuous") in DISCRETE:
                return True
        return False
    
    def apply_remove_abs(self, problem, location, sub_expr):
        """
        将单个的 Abs(...) 表达式替换为一个新的变量 y >= 0,
        并添加 y >= expr, y >= -expr 约束，从而达到 |expr| 的线性化目的。

        注意: 这里只能处理最简单的形如 Abs(x) 或 Abs(线性项) 这种情况。
        如果 sub_expr = Abs(非线性)，需要更复杂的做法。
        """
        if isinstance(sub_expr, sympy.Abs):
            # 取被绝对值包着的内部 expr
            inside = sub_expr.args[0]

            # 生成合理的变量名，避免运算符
            # 提取所有变量名并排序，确保一致性
            var_names = sorted([sym.name for sym in inside.free_symbols])
            if len(var_names) == 1:
                base_name = f"abs_{var_names[0]}"
            else:
                base_name = f"abs_{'_'.join(var_names)}"
            
            # 添加后缀避免重复
            new_var_name = f"{base_name}_1"
            
            # 加到问题的变量集里
            problem.add_variable(new_var_name, lb=0.0, ub=None, vtype='continuous')

            # 在Sympy中用Symbol来代表它
            y_sym = self._sym(problem, new_var_name, nonnegative=True)

            # 用 y_sym 替换掉 原来的 Abs(expr)
            self._replace_everywhere(problem, location, sub_expr, y_sym)

            # 添加约束: y >= expr 和 y >= -expr
            # 根据规则: Abs(expr) → y, 并加 y ≥ expr, y ≥ -expr
            
            # 约束1: y >= expr (即 y - expr >= 0)
            lhs1 = y_sym - inside
            problem.add_constraint(lhs1, '>=', 0)
            
            # 约束2: y >= -expr (即 y + expr >= 0)
            lhs2 = y_sym + inside
            problem.add_constraint(lhs2, '>=', 0)

            print(f"Replaced |{inside}| with new variable {new_var_name} and added constraints.")
        else:
            print("apply_remove_abs: not a simple Abs(...) form, skip.")

    def apply_remove_log(self, problem, location, sub_expr):
        """
        如果 sub_expr = log(x_variable),
        那么引入新变量 u, 并加 x = exp(u) 约束, 并替换 log(x) => u.
        """
        if sub_expr.func == sympy.log and len(sub_expr.args) == 1:
            # sub_expr = log( something )
            inside = sub_expr.args[0]  # 例如 x
            # 创建新变量
            new_var_name = f"log_{str(inside)}_{id(sub_expr)}"
            problem.variables[new_var_name] = ps.Variable(new_var_name, lb=None, ub=None)

            # 替换 log(x) =>  Symbol(new_var_name)
            u_sym = self._sym(problem, new_var_name)
            self._replace_in_problem(problem, location, sub_expr, u_sym)

            # 添加  x = exp(u)
            # => x - exp(u) = 0
            # 这里 inside 是 x_sym, 可能是 x1, x2, etc.
            # 需要将 x_s = x_s.xreplace(...) 处理吗？这里示例假设 inside 就是Symbol
            if isinstance(inside, sympy.Symbol):
                # 约束: inside - exp(u_sym) == 0
                #  也可以写 inside == exp(u_sym)
                eq_expr = inside - sympy.exp(u_sym)
                problem.constraints.append(
                    ps.Constraint(eq_expr, '=', 0)
                )
        else:
            # 其他情况(比如 log(x+2)), 就更复杂: 需要更多解析
            print("apply_remove_log: not a simple log(x) form, skip.")


    def apply_add_binary_obj(self, problem, location, sub_expr, l_value):
        """
        如果变量 var_name 的范围是 [0,1],
        就在目标函数里“加一项”  a^l - (a^l)^2  (并保持最小化的含义)，
        并且移除所有 constraints。

        :param problem: QCQPProblem
        :param var_name: 字符串, 比如 "a"
        :param l_value: 指数 l
        """
        # 1) 检查var是否存在
        if isinstance(sub_expr, sympy.Symbol):
            var_name = sub_expr.name
        else:
            print("[apply_add_binary_obj] sub_expr is not a Symbol, skip.")
            return
        if var_name not in problem.variables:
            print(f"[apply_add_binary_obj] Variable {var_name} not found in problem.")
            return

        var_info = problem.variables[var_name]

        # 2) 检查 [0,1]
        if var_info.lb == 0 and var_info.ub == 1:
            # 构造表达式 a^l - (a^l)^2
            a_sym = self._sym(problem, var_name)
            l = sympy.Symbol("l")  # 这里本来就没 real=True，可不动

            # a^l
            a_l_expr = a_sym ** l
            # a^l - (a^l)^2
            new_term = a_l_expr - (a_l_expr ** 2)

            # 3) 加到目标函数里
            if problem.obj_expr is None:
                # 若之前没有目标, 就直接用这个表达式
                problem.obj_expr = new_term
                problem.obj_sense = "min"  # 默认设成最小化
            else:
                # 如果之前就有目标, 就把它“加”进去
                # => min [ old_obj + (a^l - (a^l)^2) ]
                problem.obj_expr = problem.obj_expr + new_term
                # problem.obj_sense 保持原有 "min" 或 "max"
                # 如果你需要强制它是 "min", 就写:
                # problem.obj_sense = "min"
        else:
            print(f"[apply_add_binary_obj] {var_name} not in {0,1}, skip.")

    def apply_remove_fraction(self, problem, location, sub_expr, lamda=None,
                            allow_bigM_fallback=False, BIG_M=1e4):
        """
        规范去分式：
        A) 分母不含决策变量 → 直接乘倒数（不引 λ）
        B) 分母含决策变量 → 引入 λ，并自动推界；若分母跨 0 或推不出界 → 默认跳过（可选 Big-M 兜底）
        仅依赖本类已有接口：_infer_affine_bounds / _replace_in_problem / _mark_identity_rewrite
        """

        import sympy as sp

         # —— 1) 指纹函数（局部闭包）：对 num/den 做规范化并生成稳定 key —— #
        def canon_key(num, den):
            # 轻量规范化：展开 + 约去公因子；把符号统一到分子
            num_s, den_s = sp.together(sp.expand(num/den)).as_numer_denom()
            # 让分母的首符号为正（避免 ± 全局号导致重复 λ）
            if den_s.could_extract_minus_sign():
                num_s, den_s = -num_s, -den_s
            # 用 srepr 作为结构指纹（比 str 更稳定）
            return (sp.srepr(sp.expand(num_s)), sp.srepr(sp.expand(den_s)))

        # ---- 局部小工具（只在本函数里用）----
        def den_has_decision(d):
            # 标量决策
            for s in d.free_symbols:
                if isinstance(s, sp.Symbol) and (s.name in problem.variables or s.name in getattr(problem, "matrix_variables", {})):
                    return True
            # 矩阵决策（理论上分母应当是标量；这里兜底探测）
            for ms in d.atoms(sp.MatrixSymbol):
                if ms.name in getattr(problem, "matrix_variables", {}):
                    return True
            return False

        def infer_scalar_bounds_general(expr):
            """
            更通用的标量推界：
            - 正确识别 MatrixSymbol / Trace 中的决策变量（free_symbols 不包含 MatrixSymbol）
            - 支持：
                (1) Trace(x.T*y)  向量点积（双线性，用盒约束做区间上界/下界）
                (2) Trace(row*z)  row 为常量 1×n 矩阵，z 为 n×1 决策向量（仿射）
                (3) Add/Mul(标量) 的简单递归组合
            - 兜底：仍然尝试你已有的 _infer_affine_bounds
            """
            import sympy as sp
            from sympy.matrices.expressions.matexpr import MatrixElement

            def has_decision(e):
                # 标量决策变量
                for s in getattr(e, "free_symbols", set()):
                    if isinstance(s, sp.Symbol) and (s.name in problem.variables):
                        return True
                # 向量/矩阵决策变量（MatrixSymbol 不在 free_symbols 里）
                for ms in e.atoms(sp.MatrixSymbol):
                    if ms.name in getattr(problem, "matrix_variables", {}):
                        return True
                # 元素级兜底
                for me in e.atoms(MatrixElement):
                    base = getattr(me, "parent", None)
                    if isinstance(base, sp.MatrixSymbol) and base.name in getattr(problem, "matrix_variables", {}):
                        return True
                return False

            def vec_elem_bounds(vec_name: str, i: int):
                mv = getattr(problem, "matrix_variables", {}).get(vec_name)
                if mv is None:
                    return None, None
                lb, ub = self._elem_bounds(mv, i)   # 你已有的元素界提取函数
                return lb, ub

            def bilin_box(aL, aU, bL, bU):
                # min/max of a*b over a∈[aL,aU], b∈[bL,bU]
                cand = [aL*bL, aL*bU, aU*bL, aU*bU]
                return min(cand), max(cand)

            def scale_interval(k, L, U):
                if k is None:
                    return None, None
                if k >= 0:
                    return k * L, k * U
                else:
                    return k * U, k * L
                
            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            # ---------- 0) 常数 ----------
            if not has_decision(expr):
                try:
                    c = float(expr)
                    return c, c
                except Exception:
                    return None, None
                
                
            # ---------- 0.5) 先特判：Trace(x.T*Q*x) / (alpha * Trace(...)) ----------
            # 利用已有识别器：is_scaled_xTQx(expr) -> (ok, alpha, x_sym, Q_expr)
            try:
                ok, alpha, x_sym, Q_expr = self.is_scaled_xTQx(expr)
                if ok and isinstance(x_sym, MatrixSymbol) and x_sym.shape[1] == 1 and hasattr(Q_expr, "__getitem__"):
                    n = int(x_sym.shape[0])

                    # x_i bounds
                    x_bounds = []
                    for i in range(n):
                        xL, xU = vec_elem_bounds(x_sym.name, i)
                        if xL is None or xU is None:
                            return None, None
                        x_bounds.append((xL, xU))

                    L, U = 0.0, 0.0
                    for i in range(n):
                        for j in range(n):
                            qij = Q_expr[i, j]
                            if qij == 0:
                                continue
                            q = to_float(qij)
                            if q is None:
                                return None, None
                            lij, uij = bilin_box(x_bounds[i][0], x_bounds[i][1],
                                                x_bounds[j][0], x_bounds[j][1])
                            if q >= 0:
                                L += q * lij
                                U += q * uij
                            else:
                                L += q * uij
                                U += q * lij

                    a = to_float(alpha)
                    if a is None:
                        return None, None
                    return a * L, a * U
            except Exception:
                pass

            # ---------- 1) 加法：区间可加 ----------
            if isinstance(expr, sp.Add):
                lbs, ubs = [], []
                for a in expr.args:
                    l, u = infer_scalar_bounds_general(a)
                    if l is None or u is None:
                        return None, None
                    lbs.append(l); ubs.append(u)
                return float(sum(lbs)), float(sum(ubs))

            # ---------- 2) 乘法：只处理“纯标量系数 × 其余” ----------
            if isinstance(expr, sp.Mul):
                scalar = 1.0
                rest = []
                for a in expr.args:
                    if getattr(a, "is_number", False):
                        try:
                            scalar *= float(a)
                        except Exception:
                            rest.append(a)
                    else:
                        rest.append(a)
                if len(rest) == 0:
                    return scalar, scalar
                sub = sp.Mul(*rest)
                l, u = infer_scalar_bounds_general(sub)
                if l is None or u is None:
                    return None, None
                return scale_interval(scalar, l, u)

            # ---------- 3) Trace 特判 ----------
            if isinstance(expr, sp.Trace):
                inner = expr.arg

                # 3.1 识别 Trace(x.T * y)：向量点积（双线性）
                #     只覆盖你日志里这一类：x,y 都是 (n×1) MatrixSymbol
                try:
                    if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                        A, B = inner.args
                        # x.T * y
                        if isinstance(A, sp.Transpose) and isinstance(A.arg, sp.MatrixSymbol) and isinstance(B, sp.MatrixSymbol):
                            x = A.arg
                            y = B
                            if getattr(x, "shape", None) == getattr(y, "shape", None) and x.shape[1] == 1:
                                n = int(x.shape[0])
                                L, U = 0.0, 0.0
                                for i in range(n):
                                    xL, xU = vec_elem_bounds(x.name, i)
                                    yL, yU = vec_elem_bounds(y.name, i)
                                    if xL is None or xU is None or yL is None or yU is None:
                                        return None, None
                                    li, ui = bilin_box(xL, xU, yL, yU)
                                    L += li; U += ui
                                return float(L), float(U)

                        # 3.2 识别 Trace(row * z)：row 是显式常量 1×n，z 是 n×1 决策向量（仿射）
                        if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                            row = A
                            z = B
                            if row.shape[0] == 1 and z.shape[1] == 1 and row.shape[1] == z.shape[0]:
                                n = int(z.shape[0])
                                L, U = 0.0, 0.0
                                for i in range(n):
                                    coef = float(row[0, i])
                                    zL, zU = vec_elem_bounds(z.name, i)
                                    if zL is None or zU is None:
                                        return None, None
                                    # coef * z_i
                                    if coef >= 0:
                                        L += coef * zL; U += coef * zU
                                    else:
                                        L += coef * zU; U += coef * zL
                                return float(L), float(U)
                except Exception:
                    pass

                # 3.3 兜底：Trace(1×1 矩阵) → 取 [0,0] 再推界
                try:
                    if hasattr(inner, "shape") and inner.shape == (1, 1):
                        return infer_scalar_bounds_general(inner[0, 0])
                except Exception:
                    pass

                return None, None

            # ---------- 4) 兜底：你原来的仿射推界器 ----------
            try:
                lb, ub = self._infer_affine_bounds(expr, problem)
                if lb is not None and ub is not None:
                    return float(lb), float(ub)
            except Exception:
                pass

            return None, None


        def bounds_fixed_sign(lb, ub):
            if lb is None or ub is None:
                return False
            return (lb > 0 and ub > 0) or (lb < 0 and ub < 0)

        def interval_div(n_lb, n_ub, d_lb, d_ub):
            # 要求分母不跨 0
            if d_lb <= 0 <= d_ub:
                raise ZeroDivisionError("denominator interval crosses 0")
            cand = []
            for a in (n_lb, n_ub):
                for b in (d_lb, d_ub):
                    cand.append(a / b)
            return min(cand), max(cand)

        def register_bounded_symbol(name, lb, ub):
            # 在 problem.variables 里登记 / 更新一个有界标量变量，返回 Symbol
            if name not in problem.variables:
                problem.add_variable(name, lb=lb, ub=ub, vtype="continuous")
            else:
                v = problem.variables[name]
                if lb is not None: v.lb = lb
                if ub is not None: v.ub = ub
            return self._sym(problem, name)

        # ---- 正式逻辑 ----
        num, den = sp.fraction(sub_expr)
        if den == 1:
            self._mark_identity_rewrite(location, sub_expr)
            print("[remove_fraction] 不是分式，跳过")
            return
        
        # —— 2) 命中缓存直接复用 —— #
        raw_key = canon_key(num, den)
        key = (id(problem), raw_key)
        if key in self.frac_cache:
            lam_name = self.frac_cache[key]
            lam_sym  = sp.Symbol(lam_name)
            # 直接替换，不再重复加约束；你的 add_constraint_unique 也能防重复
            self._replace_in_problem(problem, location, sub_expr, lam_sym)
            print(f"[remove_fraction] 复用已有 λ={lam_name}；{sub_expr} → {lam_sym}")
            return

        # A) 分母不含决策变量：直接乘倒数
        if not den_has_decision(den):
            try:
                inv_den = sp.Pow(den, -1, evaluate=True)
                new_expr = sp.simplify(num * inv_den)
            except Exception:
                new_expr = num * sp.Pow(den, -1)
            self._replace_in_problem(problem, location, sub_expr, new_expr)
            print(f"[remove_fraction] 分母不含决策变量：{sub_expr} → {new_expr}（不引 λ）")
            return

        # B) 分母含决策变量：尝试 λ 代换 + 自动推界
        n_lb, n_ub = infer_scalar_bounds_general(num)
        d_lb, d_ub = infer_scalar_bounds_general(den)

        lam_bounds_ok = False
        if (n_lb is not None and n_ub is not None) and bounds_fixed_sign(d_lb, d_ub):
            try:
                lam_lb, lam_ub = interval_div(n_lb, n_ub, d_lb, d_ub)
                lam_bounds_ok = True
            except Exception as e:
                print(f"[remove_fraction] 计算 λ 界失败：{e}")

        if lam_bounds_ok:
            # 生成/选择 λ 名称
            if isinstance(lamda, str):
                lam_name = lamda
            else:
                lam_name = f"lam_frac_{abs(hash(str(sub_expr)))%10**6}"
            lam_sym = register_bounded_symbol(lam_name, lam_lb, lam_ub)

            # 等式约束：lam*den = num。该约束本身可能含有 lam 与仿射分母的乘积，
            # 需要立即松弛，否则后续 solver 仍会看到非 PSD 二次项。
            frac_eq_lhs = lam_sym * den - num
            n_cons_before = len(getattr(problem, "constraints", []))
            problem.add_constraint_unique(frac_eq_lhs, '=', 0)
            if len(getattr(problem, "constraints", [])) > n_cons_before:
                frac_loc = f"Constraint_{len(problem.constraints)}_LHS"
                self.apply_mccormick_relaxation(problem, frac_loc, problem.constraints[-1].expr)

            # 用 λ 替换原分式
            self._replace_in_problem(problem, location, sub_expr, lam_sym)

            # 写入缓存（复用关键）
            self.frac_cache[key] = lam_name

            print(f"[remove_fraction] 引入 λ={lam_name}，bounds=[{lam_lb:.6g},{lam_ub:.6g}]；已缓存该分式")
            return
    
        # 推不出界 / 分母跨0：默认跳过或 Big-M 兜底
        reasons = []
        if n_lb is None or n_ub is None: reasons.append("分子无界")
        if d_lb is None or d_ub is None: reasons.append("分母无界")
        if d_lb is not None and d_ub is not None and not bounds_fixed_sign(d_lb, d_ub):
            reasons.append("分母跨 0")
        msg = "；".join(reasons) if reasons else "条件不足"

        if not allow_bigM_fallback:
            print(f"[remove_fraction] 跳过：{msg}。为避免不安全乘积，未改写。")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # Big-M 兜底（谨慎使用）
        t_name = f"t_frac_{abs(hash(str(sub_expr)))%10**6}"
        t_sym  = register_bounded_symbol(t_name, -BIG_M, BIG_M)
        problem.add_constraint_unique(t_sym * den - num, '=', 0)
        self._replace_in_problem(problem, location, sub_expr, t_sym)
        print(f"[remove_fraction] [BigM兜底] {msg}。设 {t_name}∈[-{BIG_M},{BIG_M}]，加 t*den=num；"
            f"{sub_expr} → {t_sym}")

    # ------------------------------------------------------------------
    # 收集器：返回 [(coef, xi, yj)]，若无法分解则空列表
    # ------------------------------------------------------------------
    def _collect_bilinear_terms(self, expr):
        """
        能识别并分解：
        - α * x.T * Q * y      （含 Trace 形式）
        - α * x.T * Q * x      （y=x 的特例）
        返回 [(coef, xi, yj)] 的列表；无法分解则返回 []。
        """
        terms = []

        # ① 先试 xᵀQy / Trace(xᵀQy)
        ok, alpha, x_sym, Q_expr, y_sym = self.is_scaled_xTQy(expr)
        if ok:
            n = x_sym.shape[0]
            m = y_sym.shape[0]

            # 尝试只对“可索引”的 Q 展开。如果 Q 是 MatrixSymbol/MatrixBase 均可直接索引；
            # 若是更复杂的 MatrixExpr（例如某些乘积），建议在上游先规范化，或此处降级为返回 [] 交给回退逻辑。
            if hasattr(Q_expr, "__getitem__"):
                for i in range(n):
                    for j in range(m):
                        q_ij = Q_expr[i, j]
                        if q_ij != 0:
                            # 注意：xi,yj 还是 MatrixElement，需要稍后转为标量变量
                            xi = x_sym[i, 0]
                            yj = y_sym[j, 0]
                            terms.append((alpha * q_ij, xi, yj))
                if terms:
                    return terms
            # Q 不是可直接索引的形态 → 交给后续回退逻辑
            # （也可以在这里对 Q 做 simplify/factor 以期获得可索引形态）

        # ② 老的 xᵀQx 专用解析（若上面未命中且你想保留）
        ok2, alpha2, x2_sym, Q2_expr = self.is_scaled_xTQx(expr)
        if ok2 and hasattr(Q2_expr, "__getitem__"):
            n, _ = x2_sym.shape
            for i in range(n):
                for j in range(n):
                    q_ij = Q2_expr[i, j]
                    if q_ij != 0:
                        terms.append((alpha2 * q_ij, x2_sym[i, 0], x2_sym[j, 0]))
            if terms:
                return terms

        # ③ 其它回退逻辑：递归加法分解等
        if expr.is_Add:
            for arg in expr.args:
                terms.extend(self._collect_bilinear_terms(arg))
            return terms

        return terms

    
    def _match_vec_dot_with_coeff(self, expr):
        """
        匹配 α * x.T * y 或 α * Trace(x.T * y) / Trace(y.T * Q * x)
        返回 (ok, alpha, X, Y)，其中 X,Y 形状均为 (n,1) 的 MatrixExpr（可不是纯 MatrixSymbol）。
        """
        import sympy as sp
        from sympy.matrices.expressions.matexpr import MatrixElement

        # 若是 [0,0] 取元素，尽量还原到矩阵表达式本体
        base = expr
        if isinstance(expr, MatrixElement) and getattr(expr, "i", None) == 0 and getattr(expr, "j", None) == 0:
            base = getattr(expr, "parent", None) or getattr(expr, "base", None)

        # ---- Case A: α * Trace(core) ----
        alpha_out, tr = self._pull_scalar_times_trace(base)
        if tr is not None:
            inner = tr.arg
            if isinstance(inner, sp.MatMul):
                alpha_in, core = self._pull_scalar_from_matmul(inner)
            else:
                alpha_in, core = sp.Integer(1), inner
            alpha = alpha_out * alpha_in

            # 2 因子：Trace( x.T * y )
            if isinstance(core, sp.MatMul) and len(core.args) == 2:
                L, R = core.args
                if isinstance(L, sp.Transpose) and isinstance(R, sp.MatrixExpr):
                    X, Y = L.arg, R
                    if X.shape[1] == 1 and Y.shape[1] == 1 and X.shape[0] == Y.shape[0]:
                        return True, alpha, X, Y

            # 3 因子：Trace( y.T * Q * x )  →  等价于 (Q.T*y).T * x
            if isinstance(core, sp.MatMul) and len(core.args) == 3:
                L, M, R = core.args
                if isinstance(L, sp.Transpose) and isinstance(R, sp.MatrixExpr):
                    Y, X = L.arg, R
                    if Y.shape[1] == 1 and X.shape[1] == 1:
                        A = M.T * Y  # 列向量
                        if A.shape[1] == 1 and A.shape[0] == X.shape[0]:
                            return True, alpha, A, X

        # ---- Case B: α * (x.T * y)（非 Trace）----
        if isinstance(base, sp.MatMul):
            alpha, core = self._pull_scalar_from_matmul(base)
            # 需要 core = x.T * y 两个因子
            if isinstance(core, sp.MatMul) and len(core.args) == 2:
                L, R = core.args
                if isinstance(L, sp.Transpose) and isinstance(R, sp.MatrixExpr):
                    X, Y = L.arg, R
                    if X.shape[1] == 1 and Y.shape[1] == 1 and X.shape[0] == Y.shape[0]:
                        return True, alpha, X, Y

        return False, None, None, None


    def apply_mccormick_relaxation(self, problem, location, sub_expr):
        """
        自动识别并线性化：
        (1) α·x·y        α 可为数字或外部参数
        (2) x·y / x²     纯双线性或同变量平方
        (3) xᵀy          列向量点积
        (4) (ax+b)(cy+d) 仿射×仿射   先展开
        递归处理表达式中的所有双线性子项。
        """
        if self._term_has_discrete(problem, sub_expr):
            self._mark_identity_rewrite(location, sub_expr)  # 让 reward 看到“空改写”
            # 可选：print("[McC] term contains discrete vars; skip.")
            return
        
        import sympy as sp
        # do not linearize bilinear terms involving existing McCormick proxies
        def _is_w_proxy(sym):
            return self._is_mccormick_proxy_symbol(problem, sym)

        num, den = sp.fraction(sub_expr)
        if den != 1:
            if any((s.name in problem.variables) for s in den.free_symbols) \
            or self._den_has_matrix_decision(den, problem):
                self._mark_identity_rewrite(location, sub_expr)
                return
        
        # ---- one-time warm-up: do a cheap FBBT pass before building envelopes ----
        # This can only tighten bounds; it should never make the relaxation invalid.
        if self.enable_bt_warmup and not getattr(problem, "_bt_warm_done", False):
            try:
                if self.debug_bt:
                    print("[BT][warmup] mccormick pre-BT (max_rounds=1)")
                changed = self.apply_bound_tightening(problem, max_rounds=1)
                if self.debug_bt:
                    print(f"[BT][warmup] mccormick done changed={changed}")
            except Exception:
                pass
            problem._bt_warm_done = True
            
        # === 早处理 1：alpha * Trace(Add(...)) 直接拆 ===
        alpha, tr = self._pull_scalar_times_trace(sub_expr)
        if tr is not None and isinstance(tr.arg, (sp.Add, sp.MatAdd)):
            parts = [alpha * sp.Trace(arg) for arg in tr.arg.args]
            new_sum = sp.Add(*parts)
            # 先把“一个 Trace(加法)”替换成“和若干项的和”
            self._replace_everywhere(problem, location, sub_expr, new_sum)
            # 再对每个子项递归做 mccormick（此时每一项都是单独的 Trace(...)）
            for p in parts:
                self.apply_mccormick_relaxation(problem, location, p)
            return

        # === 早处理 2：Trace(Add(...))（无显式系数） ===
        if isinstance(sub_expr, sp.Trace) and isinstance(sub_expr.arg, (sp.Add, sp.MatAdd)):
            parts = [sp.Trace(arg) for arg in sub_expr.arg.args]
            new_sum = sp.Add(*parts)
            self._replace_everywhere(problem, location, sub_expr, new_sum)
            for p in parts:
                self.apply_mccormick_relaxation(problem, location, p)
            return
    
        bilins = self._collect_bilinear_terms(sub_expr)
        if bilins:
            new_sum = 0
            for c, xi, yj in bilins:
                # 关键：统一把元素/仿射元素落地为标量
                xi_s = self._elements_to_scalars(problem, xi, location)
                yj_s = self._elements_to_scalars(problem, yj, location)
                if _is_w_proxy(xi_s) or _is_w_proxy(yj_s):
                    new_sum += c * xi_s * yj_s
                    continue
                w = self._linearize_pair(problem, xi_s, yj_s)
                new_sum += c * w
            # 收尾再扫一遍，防止系数组合里残留元素
            new_sum = self._elements_to_scalars(problem, new_sum, location)
            self._replace_everywhere(problem, location, sub_expr, new_sum)
            return
        
        # try:
        #     self.apply_bound_tightening(problem, max_rounds=1)
        # except Exception:
        #     pass

        # 没有直接收集到双线性项，尝试各种模式匹配
        # ---------- 1. 若是系数 * expr，先拆系数 ----------
        coeff, core = sub_expr.as_coeff_Mul()

        # core=Scalar双线性 (x*y)
        # ---------- 0-bis.  处理 x**2 →  x*x ----------
        if isinstance(sub_expr, sympy.Pow) and sub_expr.exp == 2:
            x_sym = sub_expr.base
            if isinstance(x_sym, sympy.Symbol):
                # 生成 / 复用 w = x*x
                w_sym = self._linearize_pair(problem, x_sym, x_sym)
                # 替换表达式
                self._replace_everywhere(problem, location, sub_expr, w_sym)
                return

        if (isinstance(core, sympy.Mul) and len(core.args) == 2
                and all(self._is_decision_var(t, problem) for t in core.args)):
            x_sym, y_sym = core.args
            if _is_w_proxy(x_sym) or _is_w_proxy(y_sym):
                return
            # 若系数含这两个符号 => 三元乘积，跳过
            if coeff.free_symbols & {x_sym, y_sym}:
                print(f"[McC] 三元乘积 {sub_expr} 暂不处理")
                return

            w_sym = self._linearize_pair(problem, x_sym, y_sym)
            new_expr = coeff * w_sym
            self._replace_everywhere(problem, location, sub_expr, new_expr)
            return

        # ---------- 2. 向量点积  xᵀy（带任意标量系数） ----------
        if isinstance(sub_expr, sympy.MatMul):
            alpha_mm, core_mm = self._pull_scalar_from_matmul(sub_expr)  # 把 -1、3、参数等都吸出来
            if isinstance(core_mm, sympy.MatMul) and len(core_mm.args) == 2 \
            and isinstance(core_mm.args[0], sympy.Transpose) and isinstance(core_mm.args[1], sympy.MatrixExpr):

                x_mat = core_mm.args[0].args[0]
                y_mat = core_mm.args[1]

                # 统一物化为向量符号，避免 MatAdd/MatMul 直接过界
                x_mat = self._as_vector_symbol(problem, location, x_mat)
                y_mat = self._as_vector_symbol(problem, location, y_mat)

                if x_mat.shape != y_mat.shape or x_mat.shape[1] != 1:
                    print("[McC] 仅支持列向量同维点积")
                    return
                n = x_mat.shape[0]

                sum_w = 0
                for i in range(n):
                    xi_elem = x_mat[i, 0]
                    yi_elem = y_mat[i, 0]
                    xi_s = self._elements_to_scalars(problem, xi_elem, location)
                    yi_s = self._elements_to_scalars(problem, yi_elem, location)
                    if _is_w_proxy(xi_s) or _is_w_proxy(yi_s):
                        sum_w += xi_s * yi_s
                        continue
                    sum_w += self._linearize_pair(problem, xi_s, yi_s)

                sum_w = self._elements_to_scalars(problem, sum_w, location)
                self._replace_everywhere(problem, location, sub_expr, alpha_mm * sum_w)
                return


        # ---------- 3. 标量仿射×变量 或 向量聚合 ----------
        # ------- (a) 标量情形 -------
        if isinstance(sub_expr, sympy.Mul):
            factors = list(sympy.Mul.make_args(sub_expr))
            for i, factor in enumerate(factors):
                if not (isinstance(factor, sympy.Symbol) and self._is_decision_name(problem, factor.name)):
                    continue
                if _is_w_proxy(factor):
                    continue

                rest = factors[:i] + factors[i + 1:]
                if not rest:
                    continue
                affine_part = sympy.Mul(*rest)
                if self._is_affine_scalar(affine_part, problem):
                    affine_sym = self._affine_scalar_to_var(problem, location, affine_part)
                    if _is_w_proxy(affine_sym):
                        continue
                    w = self._linearize_pair(problem, factor, affine_sym)
                    if w != factor * affine_sym:
                        self._replace_everywhere(problem, location, sub_expr, w)
                        return

        if isinstance(sub_expr, sympy.Mul) and len(sub_expr.args) == 2:
            a, b = sub_expr.args
            # print("[DEBUG] sub_expr args:", a, "|", b)
            if self._is_affine_scalar(a, problem) and isinstance(b, sympy.Symbol):
                # print("[DEBUG] hit scalar-B  (affine * symbol)")
                a_sym = self._affine_scalar_to_var(problem, location, a)
                w = self._linearize_pair(problem, a_sym, b)
                self._replace_everywhere(problem, location, sub_expr, w);  return
            if self._is_affine_scalar(b, problem) and isinstance(a, sympy.Symbol):
                # print("[DEBUG] hit scalar-B  (affine * symbol)")
                b_sym = self._affine_scalar_to_var(problem, location, b)
                w = self._linearize_pair(problem, a, b_sym)
                self._replace_everywhere(problem, location, sub_expr, w);  return

        # ------- (b) 向量聚合情形 -------
        ok, alpha_v, x_mat, y_mat = self._match_vec_dot(sub_expr)
        if ok:
            # 仍可能返回 MatAdd/MatMul 等合成向量 —— 统一物化
            x_mat = self._as_vector_symbol(problem, location, x_mat)
            y_mat = self._as_vector_symbol(problem, location, y_mat)

            n = x_mat.shape[0]
            sum_w = 0
            for i in range(n):
                xi_elem = x_mat[i, 0]
                yi_elem = y_mat[i, 0]
                xi_s = self._elements_to_scalars(problem, xi_elem, location)
                yi_s = self._elements_to_scalars(problem, yi_elem, location)
                if _is_w_proxy(xi_s) or _is_w_proxy(yi_s):
                    sum_w += xi_s * yi_s
                    continue
                sum_w += self._linearize_pair(problem, xi_s, yi_s)

            sum_w = self._elements_to_scalars(problem, sum_w, location)
            self._replace_everywhere(problem, location, sub_expr, alpha_v * sum_w)
            return



        # ---------- 4. 递归扫描子项 ---------------
        if sub_expr.is_Add:
            for term in sympy.Add.make_args(sub_expr):
                self.apply_mccormick_relaxation(problem, location, term)
        elif sub_expr.is_Mul:
            for factor in sub_expr.args:
                self.apply_mccormick_relaxation(problem, location, factor)
        # 其它类型暂不处理
    
    
    def _pull_scalar_times_trace(self, expr):
        """
        支持 α * Trace(core)。返回 (alpha, Trace(core)) 或 (None, None) 表示不匹配。
        仅当除了标量系数外只含一个 Trace 因子时有效。
        """
        import sympy as sp
        if isinstance(expr, sp.Trace):
            return sp.Integer(1), expr

        if isinstance(expr, sp.Mul):
            alpha = sp.Integer(1)
            trace_term = None
            for a in expr.args:
                if isinstance(a, sp.Trace):
                    if trace_term is not None:
                        return None, None  # 多个 Trace，先不支持
                    trace_term = a
                elif not isinstance(a, sp.MatrixExpr):
                    alpha *= a            # 吸收纯标量/外部参数
                else:
                    # 出现了额外的 MatrixExpr（不是 Trace），那就不是“标量×Trace(...)”
                    return None, None
            if trace_term is not None:
                return alpha, trace_term
        return None, None


    def _Fn2(self, M):
        """
        ||M||_F^2 = Trace(M.T * M)
        若 M = (标量)*MatExpr，把标量提到 Trace 外并平方，避免 Trace(4*...) 的非法形态。
        """
        import sympy as sp
        from sympy.matrices.expressions.matexpr import MatrixExpr

        def _is_mat(a):
            return isinstance(a, MatrixExpr) or isinstance(a, sp.MatrixBase)

        if isinstance(M, sp.MatMul):
            mat_args = [a for a in M.args if _is_mat(a)]
            scalars  = [a for a in M.args if not _is_mat(a)]
            if scalars:
                c = sp.Mul(*scalars)
                core = sp.MatMul(*mat_args) if mat_args else sp.Integer(1)
                # ||c * core||_F^2 = c^2 * Trace(core.T * core)
                return c**2 * sp.Trace(sp.Transpose(core) * core)

        # 默认：没有标量或不是 MatMul
        return sp.Trace(sp.Transpose(M) * M)


    def apply_trace_transformation(self, problem, location, sub_expr):
        """
        处理 α * Trace(A*B*...*Z)：
        • 先将内部乘积重组为 Left = A*...*Y、Right = Z
        • 只要 (Left.T).shape == Right.shape（等价于 AB 是方阵）就应用：
                Trace(Left*Right) = 1/2( ||Left^T + Right||_F^2
                                        - ||Left^T||_F^2 - ||Right||_F^2 )
        • 外层标量系数 α 保留
        """
        import sympy as sp

        alpha, tr = self._pull_scalar_times_trace(sub_expr)
        if alpha is None:
            print("apply_trace_transformation: sub_expr is not scalar * Trace(...), skip.")
            return

        inner = tr.arg
        # 至少两个因子
        if isinstance(inner, sp.MatMul):
            args = inner.args
        else:
            print("apply_trace_transformation: Trace core is not a product with >=2 factors, skip.")
            return
        if len(args) < 2:
            print("apply_trace_transformation: need at least 2 factors inside Trace, skip.")
            return

        # 重组为 Left * Right（把前面所有因子并到 Left，最后一个因子做 Right）
        Left  = sp.MatMul(*args[:-1])
        Right = args[-1]

        # 条件：AB 为方阵  <=>  (Left.T).shape == Right.shape
        if not (isinstance(Left, sp.MatrixExpr) and isinstance(Right, sp.MatrixExpr)):
            print("apply_trace_transformation: non-matrix factor(s), skip.")
            return
        if Left.T.shape != Right.shape:
            print("apply_trace_transformation: shape mismatch (Left.T vs Right), skip.")
            return

        # 1/2 ( ||Left^T + Right||_F^2 - ||Left^T||_F^2 - ||Right||_F^2 )
        new_core = sp.Rational(1, 2) * ( self._Fn2(Left.T + Right) - self._Fn2(Left.T) - self._Fn2(Right) )
        new_expr = alpha * new_core

        self._replace_in_problem(problem, location, sub_expr, new_expr)
        
        
    def matrix_gradient(self, expr, M):
        """返回 ∂expr / ∂M  —— 与 M 同形的 sympy.Matrix"""
        m, n = M.shape

        # 1) 若 expr 是 1×1 MatrixExpr → 转成 Expr
        if isinstance(expr, sympy.MatrixExpr):
            expr_scalar = expr[0, 0]
        else:
            expr_scalar = expr            # 已经是标量

        # 2) 按元素求导
        grad_entries = [
            sympy.diff(expr_scalar, M[i, j])
            for i in range(m) for j in range(n)
        ]
        return sympy.Matrix(m, n, grad_entries)
    
    # --- 统一把两端转成标量 Expr ---
    def scalarize(self, mat_or_expr):
        if isinstance(mat_or_expr, sympy.MatrixExpr):
            return mat_or_expr[0, 0]          # 取 (0,0) 元素
        return mat_or_expr                    # 已是 Expr


    def apply_first_order_taylor(self, problem, location, sub_expr, var_name=None, x0_sym=None):
        """
        对 sub_expr 在主变量处自动做一阶泰勒展开。
        如果未提供 var_name 或 x0_sym，会自动从 sub_expr 中推断主变量并构造符号点。
        最终结果写回 problem 的指定 location
        """

        # 判断是标量变量还是矩阵变量
        matrix_vars = [s for s in sub_expr.atoms(sympy.MatrixSymbol)]
        scalar_vars = list(sub_expr.free_symbols - set(matrix_vars))

        if matrix_vars:
            # 2-1. 选主变量
            if var_name is not None:
                M = next((m for m in matrix_vars if m.name == var_name), None)
                if M is None:
                    print(f"[Taylor] 指定矩阵 {var_name} 不在表达式里，跳过。")
                    return
            else:
                # 默认选第一个
                M = matrix_vars[0]

            m, n = M.shape

            # 2-2. 选 / 创建展开点 M0
            if x0_sym is not None:
                M0 = x0_sym
            else:
                M0 = problem.add_matrix_variable(f"{M.name}0", m, n)

            # 2-3. f(M0)
            f_M0 = sub_expr.xreplace({M: M0})

            # 2-4. ∇_M f(M)   (同形矩阵表达式)
            # grad = sympy.diff(sub_expr, M)
            grad = self.matrix_gradient(sub_expr, M)
            grad_M0 = grad.xreplace({M: M0})

            # 2-5. 〈grad , ΔM〉 = Tr(gradᵀ (M-M0))
            delta_M = M - M0
            inner_prod = sympy.Trace(sympy.Transpose(grad_M0) * delta_M)

            # 2-6. 若 f_M0 是 MatrixExpr(1×1)，保持矩阵形式
            # if isinstance(f_M0, sympy.MatrixExpr):
            #     approx_expr = f_M0 + inner_prod
            # else:
            #     # 标量 → 打包成 1×1，保持与 Trace 同类型
            #     approx_expr = sympy.Matrix([[f_M0 + inner_prod[0, 0]]])
            # 2-6. 取代旧的 if-else 块
            f_scalar  = self.scalarize(f_M0)
            ip_scalar = self.scalarize(inner_prod)
            approx_expr = sympy.Matrix([[f_scalar + ip_scalar]])

        else:
            # ---- 标量变量情形 ----
            x_syms = {}  # x -> x0_sym
            f_x0 = sub_expr
            linear_terms = []

            for x in scalar_vars:
                x0_sym = sympy.Symbol(f"{x.name}_0")
                x_syms[x] = x0_sym

                # f(x0): 把所有变量换成它们的 x0 版本
                f_x0 = f_x0.subs(x, x0_sym)

            for x in scalar_vars:
                x0_sym = x_syms[x]
                dfdx = sympy.diff(sub_expr, x)          # ∂f/∂x
                dfdx_x0 = dfdx.subs(x_syms)             # ∂f/∂x evaluated at x0
                linear_terms.append(dfdx_x0 * (x - x0_sym))

            approx_expr = f_x0 + sum(linear_terms)
            
        # 替换表达式
        self._replace_everywhere(problem, location, sub_expr, approx_expr)


    def _nontrivial_factor_taken(self, original_sum, merged):
        # 结构不变 ⇒ 没有真正合并
        if sympy.srepr(merged) == sympy.srepr(original_sum):
            return False
        # 形如  c * (...)：c 是乘积的“头部因子”
        if isinstance(merged, sympy.Mul):
            head = merged.args[0]
            # 标量 1 / -1 不是有效公共因子
            if head.is_Number and head in (1, -1):
                return False
            # 单位矩阵也视作“平凡因子”
            if getattr(head, "is_Identity", False):
                return False
            return True
        return False

    def apply_partial_factor_merge(self, problem, location, sub_expr):
        """
        改进版：标量 Add 仍用 sympy.factor；一旦捕获
        “noncommutative scalars …” 异常，就尝试对 MatExpr 手动合并。
        """
        # ------- 1. 取得完整表达式 -------
        if location == "Objective":
            expr = problem.obj_expr
        elif location.startswith("Constraint_"):
            p = location.split("_")
            idx = int(p[1]) - 1
            side = p[2].upper()
            cons = problem.constraints[idx]
            expr = cons.expr if side == "LHS" else cons.rhs
        else:
            print(f"[Error] Unknown location: {location}");  return

        expr = sympy.expand(expr)
        terms = list(sympy.Add.make_args(expr))

        # ------- 2. 找到与 sub_expr 等价的项 -------
        match_term = None
        for t in terms:
            if self._same_term(t, sub_expr):
                match_term = t;  break
        if match_term is None:
            print("[Skip] 子项不在表达式里");  return

        # ------- 3. 找另一个共享变量的候选 -------
        sub_vars = match_term.free_symbols
        cand = [t for t in terms if t != match_term and (t.free_symbols & sub_vars)]
        if not cand:
            print("[Skip] 没有公共符号的候选项");  return
        chosen = random.choice(cand)

        # ------- 4. 先试原 factor() -------
        try:
            merged = sympy.factor(match_term + chosen)
        except Exception as e:
            # ---------- 捕获矩阵因子化失败 ----------
            if "noncommutative" in str(e):
                L, B, C = self._left_right_factor(match_term, chosen)
                if L is not None:         # L * (B + C)
                    merged = L * (B + C)
                else:
                    print("[Skip] 无公共前/后缀，无法合并");  return
            else:
                print(f"[Error] factor 失败: {e}");  return
                
        original_sum = match_term + chosen
        
        # NEW: 只有提取了非平凡因子才允许合并
        if not self._nontrivial_factor_taken(original_sum, merged):
            print("[Skip] 仅存在平凡公共因子（1 或单位矩阵），不进行合并")
            return

        # ------- 5. 重组新表达式并写回 -------
        new_terms = [t for t in terms if t not in (match_term, chosen)] + [merged]
        new_expr = sympy.Add(*new_terms)
        self._replace_in_problem(problem, location, expr, new_expr)
        print(f"[Merge] {match_term} + {chosen}  →  {merged}")


    def _pull_scalar_from_matmul(self, mm):
        # mm 必须是 MatMul
        alpha = sympy.Integer(1)
        mat_args = []
        for a in mm.args:
            # 关键：把 Dense Matrix 也当作矩阵因子
            if isinstance(a, sympy.MatrixExpr) or isinstance(a, sympy.MatrixBase):
                mat_args.append(a)
            else:
                alpha *= a
        core = sympy.MatMul(*mat_args) if mat_args else sympy.Integer(1)
        return alpha, core

    def is_scaled_xTQy(self, expr):
        """
        识别以下形式（含系数提取与 1×1 包装兼容）：

          (i)  α * x.T * Q * y
          (ii) α * Trace(x.T * Q * y)
          (iii)α * y.T * Q * x  或  α * Trace(y.T * Q * x)   （会规范化为 x.T * Q.T * y）
          (iv) α * x.T * y      （视为 Q = I）

        返回 (ok, alpha, x, Q, y)，其中 x,y 为列向量 (n×1)/(m×1)，Q 为 (n×m)。
        """
        import sympy as sp
        try:
            from sympy.matrices.expressions.matexpr import MatrixElement as _MatrixElement  # type: ignore
        except Exception:
            _MatrixElement = None

        # --- unwrap 1x1 MatrixBase to scalar; keep MatrixExpr (e.g., MatMul) intact ---
        if isinstance(expr, sp.MatrixBase) and getattr(expr, "shape", None) == (1, 1):
            try:
                expr = expr[0, 0]
            except Exception:
                pass
        elif _MatrixElement is not None and isinstance(expr, _MatrixElement):
            parent = getattr(expr, "parent", None) or getattr(expr, "base", None)
            if getattr(parent, "shape", None) == (1, 1) and getattr(expr, "i", None) == 0 and getattr(expr, "j", None) == 0:
                expr = parent

        def _is_matlike(a):
            return isinstance(a, (sp.MatrixExpr, sp.MatrixBase))

        def _match_core(core):
            """
            Match core that should represent x.T*Q*y (possibly with an extra scalar inside a MatMul).
            Returns (ok, alpha_in, x, Q, y).
            """
            alpha_in = sp.Integer(1)

            if isinstance(core, sp.MatMul):
                a_in, core2 = self._pull_scalar_from_matmul(core)
                alpha_in *= a_in
                core = core2

            if not isinstance(core, sp.MatMul):
                return False, None, None, None, None

            args = core.args

            # ---- x.T * Q * y (or swapped y.T * Q * x) ----
            if len(args) == 3:
                A, B, C = args
                if isinstance(A, sp.Transpose) and _is_matlike(B) and _is_matlike(C):
                    left = A.arg
                    right = C
                    if getattr(left, "shape", None) is None or getattr(right, "shape", None) is None:
                        return False, None, None, None, None
                    if left.shape[1] != 1 or right.shape[1] != 1:
                        return False, None, None, None, None

                    # canonical: left=x, right=y, B.shape==(n,m)
                    if B.shape == (left.shape[0], right.shape[0]):
                        return True, alpha_in, left, B, right

                    # swapped: left=y, right=x, B.shape==(m,n) -> canonicalize to x.T * B.T * y
                    if B.shape == (right.shape[0], left.shape[0]):
                        return True, alpha_in, right, B.T, left

            # ---- x.T * y (treat as Q = I) ----
            if len(args) == 2:
                A, C = args
                if isinstance(A, sp.Transpose) and _is_matlike(C):
                    x = A.arg
                    y = C
                    if getattr(x, "shape", None) is not None and getattr(y, "shape", None) is not None:
                        if x.shape == y.shape and x.shape[1] == 1:
                            Q = sp.eye(x.shape[0])
                            return True, alpha_in, x, Q, y

            return False, None, None, None, None

        # --- Case 1: alpha * Trace(core) (robustly pulls alpha outside Trace) ---
        alpha_out, tr = self._pull_scalar_times_trace(expr)
        if tr is not None:
            ok, alpha_in, x, Q, y = _match_core(tr.arg)
            if ok:
                return True, sp.expand(alpha_out * alpha_in), x, Q, y
            return False, None, None, None, None

        # --- Case 2: alpha * (x.T*Q*y) without Trace ---
        if isinstance(expr, sp.MatMul):
            alpha_out, core = self._pull_scalar_from_matmul(expr)
            ok, alpha_in, x, Q, y = _match_core(core)
            if ok:
                return True, sp.expand(alpha_out * alpha_in), x, Q, y
            return False, None, None, None, None

        # --- Case 3: generic scalar * rest ---
        a, rest = sp.sympify(expr).as_coeff_Mul()
        if rest != 1:
            if isinstance(rest, sp.Trace):
                ok, alpha_in, x, Q, y = _match_core(rest.arg)
                if ok:
                    return True, sp.expand(a * alpha_in), x, Q, y
            else:
                ok, alpha_in, x, Q, y = _match_core(rest)
                if ok:
                    return True, sp.expand(a * alpha_in), x, Q, y

        return False, None, None, None, None


    def _lift_xTQy_to_zT_A_z(self, problem: ps.QCQPProblem, location: str, x_sym, Q_sym, y_sym):
        """
        将双线性项 x.T * Q * y（或其等价写法）重写为二次型：

            x.T Q y  =  z.T A z,
            z = [x; y],
            A = [[0,  Q/2],
                 [Q.T/2, 0]]

        说明：
        - z 会被“物化”为一个新的向量决策变量（MatrixSymbol），并通过逐元素等式与 (x,y) 绑定，
          以便后续 SDP/αBB 能正确读取 bounds 与生成 RLT/线性项。
        - 若 (x,y) 已是 MatrixSymbol，则物化仅发生在 z 层（并可被 vec_affine_cache 复用）。

        返回 (z_sym, A_sym)，其中 A_sym 是对称的 BlockMatrix。
        """
        import sympy as sp

        x_vec = self._as_vector_symbol(problem, location, x_sym)
        y_vec = self._as_vector_symbol(problem, location, y_sym)

        n = int(x_vec.shape[0])
        m = int(y_vec.shape[0])

        z_expr = sp.BlockMatrix([[x_vec],
                                 [y_vec]])   # (n+m)×1

        z_vec = self._as_vector_symbol(problem, location, z_expr)

        half = sp.Rational(1, 2)
        A_sym = sp.BlockMatrix([
            [sp.ZeroMatrix(n, n),       half * Q_sym],
            [half * Q_sym.T,            sp.ZeroMatrix(m, m)]
        ])

        A = A_sym.as_explicit()   # 关键：把块矩阵展开成 (n+m)x(n+m) 的标量矩阵
        return z_vec, A

    def is_xTQx(self, expr):
        if not isinstance(expr, sympy.MatMul) or len(expr.args) != 3:
            return False, None, None
        a, b, c = expr.args

        def is_transpose_pair(L, R):
            # 用 ==（结构等价）而不是 is
            return (isinstance(L, sympy.Transpose) and L.arg == R) or \
                (isinstance(R, sympy.Transpose) and R.arg == L)

        if is_transpose_pair(a, c) and isinstance(b, sympy.MatrixExpr):
            x = a.arg if isinstance(a, sympy.Transpose) else c.arg
            return True, x, b
        return False, None, None
    
    def _match_trace_xQx(self, expr):

        if not isinstance(expr, Trace):
            return False, None, None

        mat = expr.arg
        # 统一成因子列表
        args = list(mat.args) if isinstance(mat, MatMul) else [mat]

        # ---- Case A: Trace(x.T * x) 或 Trace(x * x.T) → Q = I ----
        if len(args) == 2:
            a, b = args
            # Trace(x.T * x)
            if isinstance(a, Transpose) and isinstance(a.arg, MatrixSymbol) and isinstance(b, MatrixSymbol) and a.arg == b and b.shape[1] == 1:
                return True, b, eye(b.shape[0])
            # Trace(x * x.T)
            if isinstance(a, MatrixSymbol) and isinstance(b, Transpose) and b.arg == a and a.shape[1] == 1:
                return True, a, eye(a.shape[0])

        # ---- Case B: ..., x.T, Q, x, ...  （原有分支，保留）----
        if isinstance(mat, MatMul):
            for i in range(len(args) - 2):
                a, b, c = args[i], args[i+1], args[i+2]
                if isinstance(a, Transpose) and isinstance(a.arg, MatrixSymbol) and isinstance(c, MatrixSymbol):
                    x1, x2 = a.arg, c
                    if (x1 == x2 or getattr(x1, "name", None) == getattr(x2, "name", None)) and x1.shape[1] == 1:
                        pre  = args[:i]
                        post = args[i+3:]
                        # Trace 是循环不变的：Tr( pre * (x.T * b * x) * post )
                        # 等价于 Tr( (post * pre * b) * x x.T )
                        from sympy import MatMul as _MM
                        Q_expr = _MM(*(pre + post + (b,))) if (pre or post) else b
                        return True, x1, Q_expr

            # ---- Case C: ..., x, x.T, ...  （新增分支）----
            for i in range(len(args) - 1):
                a, b = args[i], args[i+1]
                if isinstance(a, MatrixSymbol) and isinstance(b, Transpose) and b.arg == a and a.shape[1] == 1:
                    pre  = args[:i]
                    post = args[i+2:]
                    # Tr(pre * x*x.T * post) = Tr( (post*pre) * x*x.T )
                    from sympy import MatMul as _MM
                    if pre or post:
                        Q_expr = _MM(*(post + pre))
                    else:
                        Q_expr = eye(a.shape[0])
                    return True, a, Q_expr

        return False, None, None


    def is_scaled_xTQx(self, expr):
        import sympy as sp

        # ---- NEW: 支持 α * Trace(core)（外层是 Mul or Trace 都能吃到）----
        alpha_out, tr = self._pull_scalar_times_trace(expr)
        if tr is not None:
            inner = tr.arg
            if isinstance(inner, sp.MatMul):
                alpha_in, core_in = self._pull_scalar_from_matmul(inner)
            else:
                alpha_in, core_in = sp.Integer(1), inner
            alpha = alpha_out * alpha_in
            core = sp.Trace(core_in)           # 统一成 Trace 形态给匹配器
            ok, x_sym, Q_expr = self._match_trace_xQx(core)
            if ok:
                return True, alpha, x_sym, Q_expr

        # ---- 下面保留你原来的“非 Trace 标量路径” ----
        # 先把 1x1 稠密 MatrixBase 展成标量（可选）
        if isinstance(expr, sp.MatrixBase) and expr.shape == (1, 1):
            expr = expr[0, 0]

        # ① Trace 形式：Trace( MatMul(...) ) → 从内部 MatMul 抽 α
        if isinstance(expr, sp.Trace) and isinstance(expr.arg, sp.MatMul):
            alpha_in, core_in = self._pull_scalar_from_matmul(expr.arg)
            core = sp.Trace(core_in)
            alpha = alpha_in
            ok, x_sym, Q_expr = self._match_trace_xQx(core)
            if ok:
                return True, alpha, x_sym, Q_expr

        # ② 顶层 MatMul：抽 α，再走 xᵀ Q x
        if isinstance(expr, sp.MatMul):
            alpha, core = self._pull_scalar_from_matmul(expr)
            ok, x_sym, Q_expr = self.is_xTQx(core)
            if ok:
                return True, alpha, x_sym, Q_expr
            # C) xᵀ x 视为 Q=I，仅当结果是 1×1
            if len(core.args) == 2:
                A, B = core.args
                if (isinstance(A, sp.Transpose)
                    and isinstance(B, sp.MatrixSymbol)
                    and A.arg == B
                    and B.shape[1] == 1
                    and (A*B).shape == (1, 1)):
                    return True, alpha, B, sp.eye(B.shape[0])

        # ③ 其它：尝试标量×核心的常规分解（仅对纯标量有效）
        if not isinstance(expr, sp.MatrixExpr):
            a, c = sp.sympify(expr).as_coeff_Mul()
            alpha, core = a, c
            ok, x_sym, Q_expr = self.is_xTQx(core)
            if ok:
                return True, alpha, x_sym, Q_expr

        return False, None, None, None


    def apply_sdr(self, problem: ps.QCQPProblem, location: str, sub_expr):
        """
        SDP 松弛：α·xᵀQx  →  α·τ   （其中 τ = Tr(QZ),  Z - xxᵀ ⪰ 0）

        与旧版的区别：
        - 不再使用固定名字 Z_x，而是：
            Z_x, Z_x_1, Z_x_2, ...
        每次调用都生成一个在 problem.matrix_variables 里不重名的新 Z。
        - tau 同理，用 tau_x, tau_x_1, ... 避免标量变量重名。
        """

        import sympy as sp

        def _is_aux_T_name(sym) -> bool:
            name = getattr(sym, "name", "")
            return isinstance(name, str) and name.startswith("T_")

        def _already_lifted_T(sym) -> bool:
            name = getattr(sym, "name", "")
            if not _is_aux_T_name(sym):
                return False
            z_name = f"Z_{name}"
            return z_name in getattr(problem, "matrix_variables", {})

        # 0) 含离散变量：跳过（保持你原来的逻辑）
        if self._term_has_discrete(problem, sub_expr):
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 1) 分式且分母含决策变量：跳过（保持你原来的逻辑）
        num, den = sp.fraction(sub_expr)
        if den != 1:
            if any((s.name in problem.variables) for s in den.free_symbols) \
            or self._den_has_matrix_decision(den, problem):
                self._mark_identity_rewrite(location, sub_expr)
                return

        # ---- one-time warm-up: tighten x bounds before building SDR+RLT ----
        if self.enable_bt_warmup and not getattr(problem, "_sdr_bt_warm_done", False):
            try:
                if self.debug_bt:
                    print("[BT][warmup] sdr pre-BT (max_rounds=1)")
                changed = self.apply_bound_tightening(problem, max_rounds=1)
                if self.debug_bt:
                    print(f"[BT][warmup] sdr done changed={changed}")
            except Exception:
                pass
            problem._sdr_bt_warm_done = True
            
        # 2) 非 (1x1) MatrixExpr：跳过，并打印调试信息
        if isinstance(sub_expr, sp.MatrixExpr) and sub_expr.shape != (1, 1):
            print(f"[SDR] skip non-scalar quadratic (shape={sub_expr.shape})")
            # 原来那段 debug 打印可以保留 / 删除，看需要
            # print("\n=============== DEBUGGER ACTIVATED ================")
            # print(f"  Problem Name : {problem.name}")
            # print(f"  Term Location: {location}")
            # print(f"  Sub-expression type: {type(sub_expr)}")
            # print(f"  Sub-expression value:\n{sub_expr}")
            # print(f"  Sub-expression srepr:\n{sp.srepr(sub_expr)}")
            # print("===================================================\n")
            return

        # 3) 解析 α·xᵀQx 或 α·xᵀQy（对 xᵀQy 先自动提升为 zᵀAz）
        lifted_from_xTQy = False
        ok, alpha, x_sym, Q_sym = self.is_scaled_xTQx(sub_expr)
        if not ok:
            ok2, alpha2, x2_sym, Q2_sym, y2_sym = self.is_scaled_xTQy(sub_expr)
            if not ok2:
                print("[SDR] 当前表达式不属于 α·xᵀQx / α·xᵀQy 形式，跳过 SDP 松弛。")
                return
            if _already_lifted_T(x2_sym) or _already_lifted_T(y2_sym):
                self._mark_identity_rewrite(location, sub_expr)
                return
            # x2.T*Q2*y2  ->  z.T*A*z
            z_sym, A_sym = self._lift_xTQy_to_zT_A_z(problem, location, x2_sym, Q2_sym, y2_sym)
            alpha, x_sym, Q_sym = alpha2, z_sym, A_sym
            lifted_from_xTQy = True

        if _already_lifted_T(x_sym) and not lifted_from_xTQy:
            self._mark_identity_rewrite(location, sub_expr)
            return

        n, _ = x_sym.shape

        # # ================== 核心修改：Z 名字不复用 ==================
        # Z_base = f"Z_{x_sym.name}"
        # Z_name = Z_base
        # k = 0
        # # 保证在当前 problem 下不重名
        # while hasattr(problem, "matrix_variables") and Z_name in problem.matrix_variables:
        #     k += 1
        #     Z_name = f"{Z_base}_{k}"

        # # 新建矩阵变量 Z_name (n x n)
        # Z = problem.add_matrix_variable(Z_name, n, n)
        
        
        key = (id(problem), x_sym.name)
        Z = None
        if hasattr(self, "sdp_Z_cache") and key in self.sdp_Z_cache:
            Z = self.sdp_Z_cache[key]
            # 防御：如果 problem 里没有这个名字（例如你 deep copy 过），就当作失效
            if getattr(Z, "name", None) not in getattr(problem, "matrix_variables", {}):
                Z = None

        if Z is None:
            Z_name = f"Z_{x_sym.name}"
            if hasattr(problem, "matrix_variables") and Z_name in problem.matrix_variables:
                # 已存在同名 Z，直接取用（不再创建 _1,_2...）
                Z = problem.get_matrix_symbol(Z_name)
            else:
                Z = problem.add_matrix_variable(Z_name, n, n)

            self.sdp_Z_cache[key] = Z
        # print(f"[SDR] Create new matrix variable {Z_name} with shape ({n},{n}).")

        # # ================== tau 也用不重名的名字 ==================
        # tau_base = f"tau_{x_sym.name}"
        # tau_name = tau_base
        # t_id = 0
        # while tau_name in problem.variables:
        #     t_id += 1
        #     tau_name = f"{tau_base}_{t_id}"

        # # 注册 tau 作为标量决策变量（lb/ub 不设界）
        # problem.add_variable(tau_name, lb=None, ub=None)
        # # 关键：用 problem 里注册的“同一个 Symbol 句柄”
        # v_tau = problem.variables.get(tau_name, None)
        # tau = getattr(v_tau, "sym", None) or sp.Symbol(tau_name)   # 注意：不要 real=True

        # 4) 添加 SDP 约束
        # PSD 约束：Z - x xᵀ ⪰ 0
        # problem.add_psd_constraint(Z - x_sym * x_sym.T)
        one = sp.Matrix([[1]])
        M = sp.BlockMatrix([[one, x_sym.T],
                            [x_sym, Z]])   # (n+1)x(n+1) 仿射矩阵
        problem.add_psd_constraint(M)
        
        # ---- SDP strengthening (minimal, root-tightening) -----------------
        # 1) Enforce symmetry: Z_ij = Z_ji (do NOT rely on PSD cone to imply it)
        # 2) Add simple box bounds L_ij <= Z_ij <= U_ij implied by x bounds
        # NOTE: MatrixVariableSymbol.lb/ub are just metadata in your data structure;
        #       writing Z_mv.lb/ub will NOT automatically bind element scalars.
        try:
            # fetch x bounds (vector) if available
            try:
                x_var_bt = problem.matrix_variables.get(getattr(x_sym, "name", None))
            except Exception:
                x_var_bt = None

            # symmetry
            for i in range(n):
                for j in range(i + 1, n):
                    problem.add_constraint(
                        expr=sp.MatrixElement(Z, i, j) - sp.MatrixElement(Z, j, i),
                        sense='=',
                        rhs=0
                    )

            # box bounds from x bounds (only if both lb/ub are available per-index)
            x_var_bt = problem.matrix_variables.get(getattr(x_sym, "name", None), None)

            if x_var_bt is not None:
                for i in range(n):
                    li, ui = self._elem_bounds(x_var_bt, i)
                    if li is None or ui is None:
                        continue

                    # diag bounds
                    min_sq = 0.0 if (li <= 0.0 <= ui) else min(li*li, ui*ui)
                    max_sq = max(li*li, ui*ui)
                    problem.add_constraint_unique(sp.MatrixElement(Z, i, i), ">=", min_sq)
                    problem.add_constraint_unique(sp.MatrixElement(Z, i, i), "<=", max_sq)

                    for j in range(i+1, n):
                        lj, uj = self._elem_bounds(x_var_bt, j)
                        if lj is None or uj is None:
                            continue
                        cand = [li*lj, li*uj, ui*lj, ui*uj]
                        Lij, Uij = min(cand), max(cand)
                        problem.add_constraint_unique(sp.MatrixElement(Z, i, j), ">=", Lij)
                        problem.add_constraint_unique(sp.MatrixElement(Z, i, j), "<=", Uij)
            # also initialize element bounds for Z so BT can see them (Z[i,j] and Z_i_j)
            try:
                self._init_Z_element_bounds(problem, Z, x_var_bt, tol=1e-9, create_scalar=False)
            except Exception:
                pass

        except Exception:
            # keep SDP relaxation safe (never crash the pipeline)
            pass

        # SDP+RLT tightening: add McCormick/RLT envelopes linking Z_ij to x bounds (when available)
        try:
            x_var = problem.matrix_variables.get(getattr(x_sym, "name", None))
        except Exception:
            x_var = None
            
        if x_var is not None:
            self._add_sdp_rlt_constraints(problem, x_var, Z)


        # # 等式约束：tau = Tr(Q Z)
        # problem.add_constraint(expr=tau - sp.Trace(Q_sym * Z), sense='=', rhs=0)

        # # 5) 用 1×1 Matrix 表达式构造替换项（保持矩阵类型，防止和 1×1 Matrix 相加出错）
        # new_expr = sp.Matrix([[alpha * tau]])
        

        self._lift_norm_constraints_to_Z(problem, x_sym, Z)

        # 直接用 Tr(QZ) 替代，不再引入 tau 变量（避免符号句柄不一致/无谓自由度）
        tau_expr = sp.Trace(Q_sym * Z)

        # 5) 用 1×1 Matrix 表达式构造替换项（保持矩阵类型，防止 1×1 Matrix 相加出错）
        new_expr = alpha * tau_expr

        # 6) 执行替换
        self._replace_everywhere(problem, location, sub_expr, new_expr)

        # optional: tighten bounds after adding SDP+RLT constraints
        # try:
        #     self.apply_bound_tightening(problem, max_rounds=1)
        # except Exception:
        #     pass



    def apply_spectral_psd_projection(self, problem: ps.QCQPProblem, location: str, sub_expr):
        """
        谱投影：对 α·xᵀQx 做特征值截断，得到 Q⁺⪰0：
            Q = UΛUᵀ → Λ⁺ = max(Λ,0) → Q⁺ = UΛ⁺Uᵀ
        然后用 α·xᵀQ⁺x 替换原项。
        这是一个“非凸二次 → 凸二次”的外凸松弛：xᵀQx ≤ xᵀQ⁺x.
        """
        import sympy as sp

        # 0) 含离散变量：跳过（和 sdp_relaxation 同逻辑）
        if self._term_has_discrete(problem, sub_expr):
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 1) 分式且分母含决策变量：跳过（和 sdp_relaxation 同逻辑）
        num, den = sp.fraction(sub_expr)
        if den != 1:
            if any((s.name in problem.variables) for s in den.free_symbols) \
               or self._den_has_matrix_decision(den, problem):
                self._mark_identity_rewrite(location, sub_expr)
                return

        # 2) 非 (1x1) MatrixExpr：直接跳过
        if isinstance(sub_expr, sp.MatrixExpr) and sub_expr.shape != (1, 1):
            print(f"[PSD-Proj] skip non-scalar quadratic (shape={sub_expr.shape})")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 3) 解析 α·xᵀQx —— 完全复用 is_scaled_xTQx（和 sdp_relaxation 一致）
        ok, alpha, x_sym, Q_sym = self.is_scaled_xTQx(sub_expr)
        if not ok:
            # print("[PSD-Proj] not of form α·xᵀQx, skip.")
            self._mark_identity_rewrite(location, sub_expr)
            return

        n, _ = x_sym.shape

        # 4) 把 Q_sym 转成数值矩阵（只支持 Dense / MatrixExpr；MatrixSymbol 直接转 sp.Matrix）
        try:
            if isinstance(Q_sym, sp.MatrixBase):
                Q_mat = Q_sym
            else:
                Q_mat = sp.Matrix(Q_sym)
        except Exception:
            print("[PSD-Proj] Q is not a concrete matrix, skip.")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 要求 Q 的每个元素不再含符号（纯数值）
        for e in Q_mat:
            if e.free_symbols:
                print("[PSD-Proj] Q has symbolic entries, skip.")
                self._mark_identity_rewrite(location, sub_expr)
                return

        # 5) 数值特征值分解 + 截断
        Q_num = np.array(Q_mat.evalf(), dtype=float)
        vals, vecs = np.linalg.eigh(Q_num)

        # 若已经近似 PSD，就没必要改
        if np.all(vals >= -1e-9):
            self._mark_identity_rewrite(location, sub_expr)
            return

        vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
        Qp_num = vecs @ np.diag(vals_clipped) @ vecs.T
        Qp_num[np.abs(Qp_num) < 1e-10] = 0.0  # 清掉数值噪声
        Qp = sp.Matrix(Qp_num)

        # 6) 构造新的凸二次项 α·xᵀQ⁺x
        new_core = x_sym.T * Qp * x_sym       # 1×1 MatrixExpr
        new_expr = alpha * new_core           # 保留原来的 α，支持 -xᵀx

        # 7) 替换到问题中（用 existing bridge 处理 1×1 / 标量差异）
        self._replace_everywhere(problem, location, sub_expr, new_expr)



    def apply_diagonal_relaxation(self, problem: ps.QCQPProblem, location: str, sub_expr):
        """
        Diagonal Relaxation：
        把 α·xᵀQx 中的 Q 替换为对角矩阵 diag(max(Q_ii, 0))，丢掉所有交叉项，
        得到一个简单的凸二次上界：xᵀQx ≤ xᵀQ_diag x（在最小化情形下为外凸松弛）。
        """
        import sympy as sp

        # 0) 含离散变量：跳过
        if self._term_has_discrete(problem, sub_expr):
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 1) 分式且分母含决策变量：跳过
        num, den = sp.fraction(sub_expr)
        if den != 1:
            if any((s.name in problem.variables) for s in den.free_symbols) \
               or self._den_has_matrix_decision(den, problem):
                self._mark_identity_rewrite(location, sub_expr)
                return

        # 2) 非 (1x1) MatrixExpr：跳过（和 sdp_relaxation 对齐）
        if isinstance(sub_expr, sp.MatrixExpr) and sub_expr.shape != (1, 1):
            print(f"[Diag-Relax] skip non-scalar quadratic (shape={sub_expr.shape})")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 3) 解析 α·xᵀQx —— 完全复用 is_scaled_xTQx
        ok, alpha, x_sym, Q_sym = self.is_scaled_xTQx(sub_expr)
        if not ok:
            # print("[Diag-Relax] not of form α·xᵀQx, skip.")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 4) 把 Q_sym 转成可索引的矩阵
        try:
            if isinstance(Q_sym, sp.MatrixBase):
                Q_mat = Q_sym
            else:
                Q_mat = sp.Matrix(Q_sym)
        except Exception:
            print("[Diag-Relax] Q is not a concrete matrix, skip.")
            self._mark_identity_rewrite(location, sub_expr)
            return

        n, m = Q_mat.shape
        if n != m:
            print("[Diag-Relax] Q is not square, skip.")
            self._mark_identity_rewrite(location, sub_expr)
            return

        # 5) 构造对角矩阵：diag(max(Q_ii, 0))，并保证是 PSD
        Qd = sp.zeros(n, n)

        for i in range(n):
            qii = Q_mat[i, i]

            # 尽量用符号信息判断正负；否则用数值 evalf
            new_entry = None
            if qii.is_number:
                v = float(qii)
                new_entry = qii if v > 0 else sp.Integer(0)
            elif qii.is_positive:
                new_entry = qii
            elif qii.is_negative:
                new_entry = sp.Integer(0)
            else:
                # 符号不清楚，保守起见直接放弃这次松弛（避免破坏凸性）
                print(f"[Diag-Relax] cannot determine sign of Q[{i},{i}], skip.")
                self._mark_identity_rewrite(location, sub_expr)
                return

            Qd[i, i] = new_entry

        # 6) 如果 Q 本身已经等于 Qd（对角非负且无交叉项），就不用改
        try:
            if sp.srepr(Q_mat) == sp.srepr(Qd):
                self._mark_identity_rewrite(location, sub_expr)
                return
        except Exception:
            pass

        # 7) 构造新的凸二次项 α·xᵀQ_diag x
        new_core = x_sym.T * Qd * x_sym       # 1×1 MatrixExpr
        new_expr = alpha * new_core

        # 8) 替换
        self._replace_everywhere(problem, location, sub_expr, new_expr)

    def apply_bound_tightening(self, problem, max_rounds: int = 3, tol: float = 1e-9):
        """
        全局 bound tightening（最小版本）：
        - 仅对“仿射(线性+常数)约束”做 FBBT 推界
        - 对任意状态都安全：缺界/非线性/右端非常数 -> 自动跳过
        - 单调：只收紧 bounds，不删除、不放松
        """
        import sympy as sp
        from sympy.matrices.expressions.matexpr import MatrixElement as _ME
        
        stats = {
            "scanned": 0,
            "skipped_rhs": 0,
            "skipped_poly": 0,
            "skipped_missing_bounds": 0,
            "tighten_cnt": 0,
        }
        
        if not hasattr(problem, "_me_bounds") or problem._me_bounds is None:
            problem._me_bounds = {}

        BT_DEBUG = bool(getattr(self, "debug_bt", False))         # 总开关
        BT_PRINT_MAX = 20        # 每轮最多打印多少条“可处理约束”
        BT_PRINT_TIGHTEN_MAX = 50  # 最多打印多少次 tighten
        stats.update({
            "affine_seen": 0,
            "affine_printed": 0,
            "tighten_printed": 0,
            "tighten_decision": 0,
            "tighten_derived": 0,
            "reject_lbgtub": 0,
            "no_change_candidates": 0,
            "ssq_hits": 0,
            "ssq_tighten": 0,
            "ssq_logged": 0,
        })
        
        SKIP_PREFIX = ("lam_", "s_", "rlt_", "cut_", "aux_", "w_", "t_", "T_", "qcr_s_")
        decision_var_names = {name for name in problem.variables.keys()
                            if not name.startswith(SKIP_PREFIX)}


        
        # def _is_decision_sym(s):
        #     # s: sympy.Symbol
        #     return isinstance(s, sp.Symbol) and (s.name in problem.variables)

        # def _is_constant_wrt_decisions(expr):
        #     # 用 atoms(Symbol, MatrixElement) 判断，比 free_symbols 更稳（能抓住 x[i,0]/Z[i,j]）
        #     expr = _bt_preprocess(expr)
        #     if expr is None:
        #         return False
        #     if not hasattr(expr, "atoms"):
        #         return True
        #     for a in expr.atoms(sp.Symbol, _ME):
        #         if _is_decision_atom(a):
        #             return False
        #     return True


        
        def _is_decision_atom(a):
            # scalar var
            if isinstance(a, sp.Symbol):
                # 只跳过明确不参与 BT 的前缀
                if a.name.startswith(SKIP_PREFIX):
                    return False
                return a.name in decision_var_names

            # MatrixElement: allow vector elements and matrix elements (incl. Z)
            if isinstance(a, _ME):
                parent = getattr(a, "parent", None) or getattr(a, "base", None)
                if not isinstance(parent, sp.MatrixSymbol):
                    return False
                # 必须登记过
                if parent.name not in getattr(problem, "matrix_variables", {}):
                    return False
                # 跳过明确不参与 BT 的前缀
                if str(parent.name).startswith(SKIP_PREFIX):
                    return False
                return True

            return False


        def _sq_bounds(lb, ub):
            # x^2 bounds over [lb,ub]
            import math
            cand = [lb*lb, ub*ub]
            if lb <= 0 <= ub:
                return 0.0, max(cand)
            return min(cand), max(cand)

        def _bilin_bounds(l1, u1, l2, u2):
            cand = [l1*l2, l1*u2, u1*l2, u1*u2]
            return min(cand), max(cand)

        def _ensure_Z_elem_bounds(z_name: str, i: int, j: int):
            """
            为 Z_*_{i,j} 生成保守 bounds：
            - 假设 Z_x 对应 x（通过名字 Z_x 推断），用 x 的元素界推导乘积界
            - 这是“加强松弛”的合法约束（不会排除原 lifted 可行集），同时能让 BT/Poly 工作
            """
            # 1) 推断对应的向量名：Z_x -> x
            base = None
            if z_name.startswith("Z_") and len(z_name) > 2:
                base = z_name[2:]          # "Z_x" -> "x"
            if base is None:
                return None, None

            # 2) base 必须是向量 matrix_variable，并且元素变量存在/可创建
            mv = getattr(problem, "matrix_variables", {}).get(base, None)
            if mv is None:
                return None, None

            # 确保 x_i / x_j 标量元素存在，用 mv.lb/mv.ub 补全（原始向量通常有盒约束）
            # --- robust dimension for base vector ---
            n = None
            # 兼容：有些实现把维度存在 dim / n / size / length
            for attr in ("shape", "dim", "n", "size", "length"):
                if hasattr(mv, attr):
                    vv = getattr(mv, attr)
                    # shape 可能是 tuple；其他可能是 int
                    if attr == "shape" and isinstance(vv, (tuple, list)) and len(vv) >= 1:
                        n = int(vv[0])
                    elif isinstance(vv, (int,)):
                        n = int(vv)
                    break
            # 兜底：至少覆盖当前用到的 i/j
            if n is None:
                n = int(max(int(i), int(j)) + 1)

            base_vec_sym = sp.MatrixSymbol(base, n, 1)

            # 确保 x_i / x_j 标量元素存在，用 mv.lb/mv.ub 补全（原始向量通常有盒约束）
            li, ui = self._elem_bounds(mv, int(i))
            lj, uj = self._elem_bounds(mv, int(j))
            xi = self._ensure_elem_var(problem, base_vec_sym, int(i), lb=li, ub=ui)
            xj = self._ensure_elem_var(problem, base_vec_sym, int(j), lb=lj, ub=uj)


            vi = problem.variables.get(xi.name)
            vj = problem.variables.get(xj.name)
            if vi is None or vj is None or vi.lb is None or vi.ub is None or vj.lb is None or vj.ub is None:
                return None, None

            l1, u1 = float(vi.lb), float(vi.ub)
            l2, u2 = float(vj.lb), float(vj.ub)

            if int(i) == int(j):
                return _sq_bounds(l1, u1)
            return _bilin_bounds(l1, u1, l2, u2)


        def _atom_bounds(a):
            # scalar Symbol
            if isinstance(a, sp.Symbol) and a.name in problem.variables:
                v = problem.variables[a.name]
                return v.lb, v.ub, ("scalar", a.name)

            if isinstance(a, _ME):
                parent = getattr(a, "parent", None) or getattr(a, "base", None)
                if isinstance(parent, sp.MatrixSymbol) and parent.name in getattr(problem, "matrix_variables", {}):
                    # 跳过 SKIP_PREFIX
                    if str(parent.name).startswith(SKIP_PREFIX):
                        return None

                    shp = getattr(parent, "shape", None)
                    if shp is None or len(shp) != 2:
                        return None

                    # ---- 向量元素：x[i,0] -> x_i
                    if shp[1] == 1:
                        idx = int(a.i)
                        elem_name = f"{parent.name}_{idx}"
                        if elem_name in problem.variables:
                            v = problem.variables[elem_name]
                            return v.lb, v.ub, ("elem", elem_name)

                        mv = problem.matrix_variables[parent.name]
                        li, ui = self._elem_bounds(mv, idx)
                        elem_sym = self._ensure_elem_var(problem, parent, idx, lb=li, ub=ui)
                        v = problem.variables[elem_sym.name]
                        return v.lb, v.ub, ("elem", elem_sym.name)

                    # ---- 矩阵元素：Z[i,j]（方案1：Z 元素不创建标量变量；bounds 存在 problem._me_bounds）
                    i, j = int(a.i), int(a.j)

                    lb, ub = (None, None)
                    if hasattr(problem, "get_me_bounds"):
                        lb, ub = problem.get_me_bounds(parent.name, i, j)

                    # 若缺界且是 Z_*：用 x bounds 初始化一个保守乘积界，写入 _me_bounds（不创建 Z_i_j 标量变量）
                    if (lb is None or ub is None) and str(parent.name).startswith("Z_"):
                        lb0, ub0 = _ensure_Z_elem_bounds(str(parent.name), i, j)
                        if lb is None:
                            lb = lb0
                        if ub is None:
                            ub = ub0
                        if (lb is not None) or (ub is not None):
                            if hasattr(problem, "tighten_me_bounds"):
                                problem.tighten_me_bounds(parent.name, i, j, lb, ub, tol=tol)
                                # 对称项也顺带初始化（你本身已有 Z_ij=Z_ji 约束）
                                problem.tighten_me_bounds(parent.name, j, i, lb, ub, tol=tol)

                    # 给 Poly 用的“虚拟符号名”，避免与真实标量变量冲突
                    sym_name = f"ME__{parent.name}_{i}_{j}"
                    return lb, ub, ("me", sym_name)


            return None

        def _tighten_atom(a, new_lb, new_ub):
            """
            安全收紧 bounds：
            - 只允许单调收紧
            - 严禁制造 lb > ub（否则会把问题剪成 infeasible）
            """
            import math
            changed = False

            def _safe_update(v, cand_lb, cand_ub):
                nonlocal changed
                updated = False
                import math

                # ---- normalize candidate ----
                def _norm(x):
                    if x is None:
                        return None
                    x = float(x)
                    if math.isnan(x) or math.isinf(x):
                        return None
                    return x

                cand_lb = _norm(cand_lb)
                cand_ub = _norm(cand_ub)

                cur_lb = None if v.lb is None else float(v.lb)
                cur_ub = None if v.ub is None else float(v.ub)

                # ---- compute proposed (monotone) WITHOUT writing ----
                prop_lb = cur_lb
                prop_ub = cur_ub

                if cand_lb is not None:
                    if prop_lb is None or cand_lb > prop_lb + tol:
                        prop_lb = cand_lb
                if cand_ub is not None:
                    if prop_ub is None or cand_ub < prop_ub - tol:
                        prop_ub = cand_ub

                # ---- feasibility guard BEFORE commit ----
                if (prop_lb is not None) and (prop_ub is not None) and (prop_lb > prop_ub + tol):
                    stats["reject_lbgtub"] += 1
                    if BT_DEBUG and stats["tighten_printed"] < BT_PRINT_TIGHTEN_MAX:
                        print(f"[BT][REJECT] {v.name}: candidate makes lb>ub: cur=({cur_lb},{cur_ub}) cand=({cand_lb},{cand_ub}) prop=({prop_lb},{prop_ub})")
                    return False

                # ---- commit once ----
                old_lb, old_ub = cur_lb, cur_ub

                updated_any = False
                if prop_lb != cur_lb:
                    v.lb = prop_lb
                    updated_any = True
                if prop_ub != cur_ub:
                    v.ub = prop_ub
                    updated_any = True

                if updated_any:
                    changed = True
                    stats["tighten_cnt"] += 1
                    # decision vs derived 统计（C类的一部分）
                    if v.name in decision_var_names:
                        stats["tighten_decision"] += 1
                    else:
                        stats["tighten_derived"] += 1

                    if BT_DEBUG and stats["tighten_printed"] < BT_PRINT_TIGHTEN_MAX:
                        print(f"[BT][TIGHTEN] {v.name}: ({old_lb}, {old_ub}) -> ({v.lb}, {v.ub})")
                        stats["tighten_printed"] += 1
                else:
                    stats["no_change_candidates"] += 1
                    if BT_DEBUG and stats["tighten_printed"] < BT_PRINT_TIGHTEN_MAX:
                        # 这里打印“为什么没变”：候选是什么、prop 结果是什么
                        print(f"[BT][NO-CHANGE] {v.name}: cur=({cur_lb},{cur_ub}) cand=({cand_lb},{cand_ub}) prop=({prop_lb},{prop_ub})")
                        stats["tighten_printed"] += 1

                return updated_any



            # scalar Symbol
            if isinstance(a, sp.Symbol) and a.name in problem.variables:
                # 禁止 BT 直接改派生变量的 bounds（可选：也可禁止派生变量参与推界，见下）
                if a.name.startswith(SKIP_PREFIX):
                    return False
                v = problem.variables[a.name]
                return _safe_update(v, new_lb, new_ub)

            # MatrixElement -> tighten element bounds
            if isinstance(a, _ME):
                parent = getattr(a, "parent", None) or getattr(a, "base", None)
                if isinstance(parent, sp.MatrixSymbol) and parent.name in getattr(problem, "matrix_variables", {}):
                    # 统一禁止 BT 更新派生矩阵变量（lam_/s_/rlt_/cut_/aux_）
                    if str(parent.name).startswith(SKIP_PREFIX):
                        return False

                    shp = getattr(parent, "shape", None)
                    if shp is None or len(shp) != 2:
                        return False

                    # ---- 向量元素：x[i,0] -> x_i（更新标量变量界）
                    if shp[1] == 1:
                        idx = int(a.i)
                        elem_name = f"{parent.name}_{idx}"
                        if elem_name in problem.variables:
                            v = problem.variables[elem_name]
                        else:
                            elem_sym = self._ensure_elem_var(problem, parent, idx, lb=None, ub=None)
                            v = problem.variables[elem_sym.name]
                        return _safe_update(v, new_lb, new_ub)

                    # ---- 矩阵元素：Z[i,j] 等（更新 problem._me_bounds，不创建 Z_i_j 标量变量）
                    i, j = int(a.i), int(a.j)
                    if not hasattr(problem, "tighten_me_bounds"):
                        return False

                    ok = problem.tighten_me_bounds(parent.name, i, j, new_lb, new_ub, tol=tol)
                    if ok:
                        # 对称项也同步收紧
                        problem.tighten_me_bounds(parent.name, j, i, new_lb, new_ub, tol=tol)

                        stats["tighten_cnt"] += 1
                        stats["tighten_derived"] += 1

                        if BT_DEBUG and stats["tighten_printed"] < BT_PRINT_TIGHTEN_MAX:
                            lb2, ub2 = problem.get_me_bounds(parent.name, i, j)
                            print(f"[BT][TIGHTEN-ME] {parent.name}[{i},{j}]: -> ({lb2}, {ub2})")
                            stats["tighten_printed"] += 1
                    else:
                        stats["no_change_candidates"] += 1

                    return ok

            return False



        def _as_scalar(e):
            if isinstance(e, sp.MatrixExpr) and getattr(e, "shape", None) == (1, 1):
                return e[0, 0]
            return e

        def _bt_preprocess(e):
            """
            BT 专用预处理：把 Trace / 1x1 矩阵壳 / 常见 MatMul 壳尽量展开成显式标量和式，
            使 Poly(domain="RR") 能识别“仿射表达式”。
            """
            if e is None:
                return None

            e = _as_scalar(e)

            # 先把 Trace 的线性性拆开：Trace(A+B)->Trace(A)+Trace(B)，Trace(alpha*M)->alpha*Trace(M)
            e = self._linearize_trace(e)

            def _expand_trace(tr):
                # tr is sp.Trace
                inner = tr.arg

                # Trace(1x1) -> (0,0)
                if isinstance(inner, sp.MatrixExpr) and getattr(inner, "shape", None) == (1, 1):
                    return _bt_preprocess(inner[0, 0])

                # Trace(Z) where Z is square MatrixSymbol -> sum_i Z[i,i]
                if isinstance(inner, sp.MatrixSymbol) and inner.shape[0] == inner.shape[1]:
                    n = int(inner.shape[0])
                    return sp.Add(*[inner[i, i] for i in range(n)])

                # Trace(row * x) : row is constant 1×n Matrix, x is n×1 MatrixSymbol
                if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                    A, B = inner.args
                    if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                        if A.shape[0] == 1 and B.shape[1] == 1 and A.shape[1] == B.shape[0]:
                            n = int(B.shape[0])
                            return sp.Add(*[A[0, i] * B[i, 0] for i in range(n)])

                # Trace(x.T * y) : x,y are n×1 MatrixSymbol
                if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                    A, B = inner.args
                    if isinstance(A, sp.Transpose) and isinstance(A.arg, sp.MatrixSymbol) and isinstance(B, sp.MatrixSymbol):
                        if A.arg.shape[1] == 1 and B.shape[1] == 1 and A.arg.shape[0] == B.shape[0]:
                            n = int(B.shape[0])
                            return sp.Add(*[A.arg[i, 0] * B[i, 0] for i in range(n)])

                # Trace(Q*Z) or Trace(Z*Q) : Q is constant matrix, Z is matrix decision MatrixSymbol (square)
                if isinstance(inner, sp.MatMul) and len(inner.args) == 2:
                    A, B = inner.args

                    # Trace(Q*Z) = sum_{i,j} Q[i,j] * Z[j,i]
                    if isinstance(A, (sp.MatrixBase, sp.ImmutableMatrix)) and isinstance(B, sp.MatrixSymbol):
                        if B.shape[0] == B.shape[1] and A.shape == B.shape:
                            n = int(B.shape[0])
                            return sp.Add(*[A[i, j] * B[j, i] for i in range(n) for j in range(n)])

                    # Trace(Z*Q) = sum_{i,j} Z[i,j] * Q[j,i]
                    if isinstance(A, sp.MatrixSymbol) and isinstance(B, (sp.MatrixBase, sp.ImmutableMatrix)):
                        if A.shape[0] == A.shape[1] and A.shape == B.shape:
                            n = int(A.shape[0])
                            return sp.Add(*[A[i, j] * B[j, i] for i in range(n) for j in range(n)])

                # 兜底：不展开
                return tr

            # 把所有 Trace(...) 尽量替换成显式求和
            try:
                e = e.replace(lambda x: isinstance(x, sp.Trace), _expand_trace)
            except Exception:
                pass

            # expand x.T*Q*x / x.T*Q*y / x.T*y (with or without Trace)
            try:
                def _expand_xTQx(expr):
                    ok, alpha, x_sym, Q_expr = self.is_scaled_xTQx(expr)
                    if not ok:
                        return expr
                    # only expand if Q is numeric matrix
                    if not isinstance(Q_expr, (sp.MatrixBase, sp.ImmutableMatrix)):
                        return expr
                    n = int(Q_expr.shape[0])
                    # if diagonal numeric, prefer sum of squares
                    diag_only = True
                    for i in range(n):
                        for j in range(n):
                            if i != j and Q_expr[i, j] != 0:
                                diag_only = False
                                break
                        if not diag_only:
                            break
                    if diag_only:
                        return sp.Add(*[alpha * Q_expr[i, i] * x_sym[i, 0] * x_sym[i, 0] for i in range(n)])
                    # full expansion (may include cross terms)
                    return sp.Add(*[alpha * Q_expr[i, j] * x_sym[i, 0] * x_sym[j, 0]
                                    for i in range(n) for j in range(n)])

                def _expand_xTQy(expr):
                    ok, alpha, x_sym, Q_expr, y_sym = self.is_scaled_xTQy(expr)
                    if not ok:
                        return expr
                    if not isinstance(Q_expr, (sp.MatrixBase, sp.ImmutableMatrix)):
                        return expr
                    n = int(x_sym.shape[0])
                    m = int(y_sym.shape[0])
                    # assume Q is n x m
                    return sp.Add(*[alpha * Q_expr[i, j] * x_sym[i, 0] * y_sym[j, 0]
                                    for i in range(n) for j in range(m)])

                # apply on the full expression if it matches
                e = _expand_xTQx(e)
                e = _expand_xTQy(e)
            except Exception:
                pass

            # 最后做一次展开（避免残留 (x_0 + x_1) 这类结构影响 Poly）
            try:
                e = sp.expand(e)
            except Exception:
                pass

            return e

        def _min_sq_on_interval(l, u):
            # min of x^2 on [l,u]
            l, u = float(l), float(u)
            if l <= 0 <= u:
                return 0.0
            return min(l*l, u*u)

        def _tighten_from_sum_squares_ineq(poly, atoms, bnds):
            """
            Handle: sum_i a_i * x_i^2 + c <= 0
            - a_i > 0
            - no cross terms
            - no linear terms
            """
            # poly is sp.Poly in atoms
            stats["ssq_hits"] += 1
            d = poly.as_dict()
            n = len(atoms)

            # constant
            c = float(d.get((0,) * n, 0.0))

            # collect quadratic coefficients, reject if any non-(0 or 2) exponents appear
            a2 = [0.0] * n

            for exp, coef in d.items():
                deg = sum(exp)
                if deg == 0:
                    continue
                # reject linear terms
                if deg == 1:
                    return False
                if deg == 2:
                    # must be pure square: only one variable has exponent 2
                    if exp.count(2) != 1 or exp.count(1) != 0:
                        return False
                    i = exp.index(2)
                    a2[i] += float(coef)
                    continue
                # higher degree
                return False

            # require all used quadratic coeffs >= 0, and at least one positive
            any_pos = False
            for ai in a2:
                if ai < -1e-12:
                    return False
                if ai > 1e-12:
                    any_pos = True
            if not any_pos:
                return False

            # inequality: sum ai*x_i^2 <= -c
            R = -c
            if R < 0:
                # already infeasible under current form; do not "tighten" here
                # (let solver detect infeasibility)
                return False

            # light debug: log a few ssq cases to see why no tightening
            if BT_DEBUG and stats.get("ssq_logged", 0) < 3:
                try:
                    preview = [(str(atoms[i]), bnds[atoms[i]]) for i in range(min(5, len(atoms)))]
                    print(f"[BT][ssq] R={R:.6g}, a2={a2}, preview_bounds={preview}")
                except Exception:
                    pass
                stats["ssq_logged"] += 1

            changed_any = False

            # precompute min contributions from others
            min_contrib = [0.0] * n
            for j in range(n):
                aj = a2[j]
                if aj <= 1e-12:
                    min_contrib[j] = 0.0
                    continue
                lj, uj = bnds[atoms[j]]
                min_contrib[j] = aj * _min_sq_on_interval(lj, uj)

            total_min_other = sum(min_contrib)

            for i in range(n):
                ai = a2[i]
                if ai <= 1e-12:
                    continue

                # remaining budget for x_i^2
                rem = R - (total_min_other - min_contrib[i])
                if rem < 0:
                    # cannot tighten safely
                    continue

                rad = (rem / ai) ** 0.5
                new_lb = -rad
                new_ub = +rad
                changed_any |= _tighten_atom(atoms[i], new_lb, new_ub)

            if changed_any:
                stats["ssq_tighten"] += 1
            return changed_any

        def _tighten_from_linear_ineq(lhs_expr, rhs_const):
            """
            tighten from: lhs_expr <= rhs_const
            其中 lhs_expr 是 affine in decision atoms
            """
            if BT_DEBUG:
                stats["affine_seen"] += 1
                if stats["affine_printed"] < BT_PRINT_MAX:
                    try:
                        # 打印约束表达式（替换 MatrixElement 前后都看一下）
                        expr_preview = sp.srepr(sp.expand(lhs_expr - rhs_const))
                    except Exception:
                        expr_preview = str(lhs_expr - rhs_const)

                    # 打印 atoms（决策相关的）
                    atoms_all = list((lhs_expr - rhs_const).atoms(sp.Symbol, _ME))
                    decision_atoms = [a for a in atoms_all if _is_decision_atom(a)]
                    print("[BT][AFFINE?] try ineq: lhs<=rhs with rhs_const=", rhs_const)
                    print("  expr:", expr_preview)
                    print("  atoms_all:", [str(a) for a in atoms_all][:30], ("..." if len(atoms_all) > 30 else ""))
                    print("  decision_atoms:", [str(a) for a in decision_atoms][:30], ("..." if len(decision_atoms) > 30 else ""))
                    stats["affine_printed"] += 1

            # 1) 收集候选 atom，并把 MatrixElement -> elem_sym（Symbol）
            atom_syms = []
            me_to_sym = {}   # MatrixElement -> Symbol
            sym_to_atom = {}  # Symbol -> original atom (MatrixElement)

            for a in lhs_expr.atoms(sp.Symbol, _ME):
                if not _is_decision_atom(a):
                    continue
                if isinstance(a, sp.Symbol) and a.name.startswith(SKIP_PREFIX):
                    continue
                if isinstance(a, _ME):
                    parent = getattr(a, "parent", None) or getattr(a, "base", None)
                    if parent is not None and str(parent.name).startswith(SKIP_PREFIX):
                        continue

                    ret = _atom_bounds(a)
                    if ret is None:
                        return False
                    lb, ub, (_, sym_name) = ret  # ret 里你返回了 ("elem", elem_sym.name)
                    if lb is None or ub is None:
                        stats["skipped_missing_bounds"] += 1
                        return False

                    sym = sp.Symbol(sym_name)    # 使用元素标量名作为 Poly 变量
                    me_to_sym[a] = sym
                    sym_to_atom[sym] = a
                    atom_syms.append(sym)
                else:
                    # scalar Symbol
                    ret = _atom_bounds(a)
                    if ret is None:
                        return False
                    lb, ub, _ = ret
                    if lb is None or ub is None:
                        stats["skipped_missing_bounds"] += 1
                        return False

                    atom_syms.append(a)

            # 去重保持顺序（Poly 变量顺序要稳定）
            seen = set()
            atom_syms = [s for s in atom_syms if (s not in seen and not seen.add(s))]
            if not atom_syms:
                return False

            # 2) 把 lhs_expr 中的 MatrixElement 全部替换成 elem_sym（标量 Symbol）
            lhs_lin = lhs_expr
            if me_to_sym:
                lhs_lin = lhs_lin.xreplace(me_to_sym)
                
            # NEW: 数值化常数（避免 pi/sqrt(2) 等导致 domain="RR" 失败）
            lhs_lin = sp.N(lhs_lin)

            # 3) 建立 bounds dict（键必须是 atom_syms 里的 Symbol）
            bnds = {}
            missing_bounds = False
            # scalar variables bounds
            for s in atom_syms:
                # s 可能来自 MatrixElement 的“虚拟符号”，也可能是原始 scalar Symbol
                if s in sym_to_atom:
                    ret2 = _atom_bounds(sym_to_atom[s])
                    if ret2 is None:
                        missing_bounds = True
                        bnds[s] = (0.0, 0.0)
                        continue
                    lb2, ub2, _ = ret2
                    if lb2 is None or ub2 is None:
                        missing_bounds = True
                        bnds[s] = (0.0, 0.0)
                    else:
                        bnds[s] = (float(lb2), float(ub2))
                    continue

                # scalar variables bounds
                if s.name in problem.variables:
                    v = problem.variables[s.name]
                else:
                    return False
                if v.lb is None or v.ub is None:
                    missing_bounds = True
                else:
                    bnds[s] = (float(v.lb), float(v.ub))


            residual = (lhs_lin - rhs_const).atoms(_ME)
            if residual:
                stats["skipped_poly"] += 1
                if BT_DEBUG and stats["affine_printed"] < BT_PRINT_MAX:
                    print("[BT][SKIP_POLY] residual MatrixElement not replaced:", [str(r) for r in list(residual)[:10]])
                return False

            # 4) 用纯 Symbol 做 Poly
            try:
                poly = sp.Poly(lhs_lin - rhs_const, *atom_syms, domain="RR")
                deg = poly.total_degree()

                if deg > 1:
                    # try separable quadratic (sum of squares) only when all bounds known
                    if (deg == 2) and (not missing_bounds):
                        ok = _tighten_from_sum_squares_ineq(poly, atom_syms, bnds)
                        if ok:
                            return True

                    stats["skipped_poly"] += 1
                    if BT_DEBUG and stats["affine_printed"] < BT_PRINT_MAX:
                        print("[BT][SKIP_POLY] degree>1:", deg, "expr=", sp.srepr(lhs_lin - rhs_const))
                    return False
            except Exception as e:
                stats["skipped_poly"] += 1
                if BT_DEBUG and stats["affine_printed"] < BT_PRINT_MAX:
                    print("[BT][SKIP_POLY] Poly failed:", repr(e))
                    print("  expr=", sp.srepr(lhs_lin - rhs_const))
                    print("  atom_syms=", [str(s) for s in atom_syms])
                return False



            # 5) 后续求系数时，把 atoms 从 atoms 改成 atom_syms，并且 bnds 用 Symbol 键
            if missing_bounds:
                stats["skipped_missing_bounds"] += 1
                return False

            atoms = atom_syms
            
            def _orig_atom(x):
                return sym_to_atom.get(x, x)


            # lhs <= rhs  <=> (lhs - rhs) <= 0
            # 取系数：sum ai*ai_atom + c <= 0  => sum ai*atom <= -c
            d = poly.as_dict()
            c = float(d.get((0,) * len(atoms), 0.0))
            acoef = []
            for i in range(len(atoms)):
                mon = [0] * len(atoms)
                mon[i] = 1
                acoef.append(float(d.get(tuple(mon), 0.0)))
            b = -c  # sum ai*atom <= b

            changed_any = False
            for i, ai in enumerate(acoef):
                if abs(ai) < 1e-12:
                    continue

                # compute S_min / S_max for others
                S_min = 0.0
                S_max = 0.0
                for j, aj in enumerate(acoef):
                    if j == i or abs(aj) < 1e-12:
                        continue
                    lj, uj = bnds[atoms[j]]
                    # min of aj*xj
                    S_min += aj * (lj if aj >= 0 else uj)
                    # max of aj*xj
                    S_max += aj * (uj if aj >= 0 else lj)

                if ai > 0:
                    # ai*x_i <= b - (min others)  => x_i <= (b - S_min)/ai
                    new_ub = (b - S_min) / ai
                    changed_any |= _tighten_atom(_orig_atom(atoms[i]), None, new_ub)
                else:
                    # ai<0: ai*x_i <= b - (others)
                    # 推导得到 x_i >= (b - others)/ai  (ai<0)
                    # 全局安全下界：使用 others 的最小值 S_min（不能用 S_max）
                    new_lb = (b - S_min) / ai  # ai negative
                    changed_any |= _tighten_atom(_orig_atom(atoms[i]), new_lb, None)


            return changed_any

        # ---------------- main loop ----------------
        changed = False
        for _ in range(max_rounds):
            round_changed = False
            for c in getattr(problem, "constraints", []):
                stats["scanned"] += 1
                lhs = _bt_preprocess(c.expr)
                rhs = _bt_preprocess(c.rhs)

                # normalize sense
                sense = c.sense
                g = _bt_preprocess(lhs - rhs)
                
                if sense in ["<", "<="]:
                    round_changed |= _tighten_from_linear_ineq(g, 0.0)
                elif sense in [">", ">="]:
                    round_changed |= _tighten_from_linear_ineq(-g, 0.0)
                elif sense in ["=", "=="]:
                    round_changed |= _tighten_from_linear_ineq(g, 0.0)
                    round_changed |= _tighten_from_linear_ineq(-g, 0.0)

            changed |= round_changed
            if not round_changed:
                break

        if BT_DEBUG:
            print(
                f"[BT] summary: scanned={stats['scanned']}, skipped_rhs={stats['skipped_rhs']}, "
                f"skipped_poly={stats['skipped_poly']}, skipped_missing_bounds={stats['skipped_missing_bounds']}, "
                f"tighten_cnt={stats['tighten_cnt']} (decision={stats['tighten_decision']}, derived={stats['tighten_derived']}), "
                f"reject_lbgtub={stats['reject_lbgtub']}, no_change={stats['no_change_candidates']}, changed={changed}, "
                f"ssq_hits={stats['ssq_hits']}, ssq_tighten={stats['ssq_tighten']}"
            )

        
        return changed
    
    
    def apply_global_cut_generation(
        self,
        problem,
        rlt_budget: int = 200,
        oa_budget: int = 20,
        tol: float = 1e-9,
        enable_affine_rlt: bool = True,     # make A2 available by default (budgeted)
    ):
        """
        全局 cut generation（预算化）：
        A) RLT(McCormick) refresh：对已存在的 w 变量，按最新 bounds 再加一批 envelopes
        B) Outer-approx：对凸二次约束 α x^T Q x <= b 生成一条切平面（用 bounds 中点作 x0）
        A2) affine-RLT（可选，默认关闭）：从仿射约束生成 level-1 RLT cuts（非常容易爆炸）
        """
        import sympy as sp
        import numpy as np

        added = 0

        if self.enable_bt_before_global_cut and not getattr(problem, "_bt_before_global_cut_done", False):
            try:
                if self.debug_bt:
                    print("[BT][warmup] global_cut pre-BT (max_rounds=1)")
                changed = self.apply_bound_tightening(problem, max_rounds=1)
                if self.debug_bt:
                    print(f"[BT][warmup] global_cut done changed={changed}")
            except Exception:
                pass
            problem._bt_before_global_cut_done = True

        # ---------------- A) RLT refresh on existing bilinear proxies ----------------
        bil_map = getattr(problem, "_bilinear_map", None)
        if isinstance(bil_map, dict):
            for w_name, pair in bil_map.items():
                if added >= rlt_budget:
                    break
                if not isinstance(pair, tuple) or len(pair) != 2:
                    continue
                x_sym, y_sym = pair

                # ensure scalar decisions
                if not (isinstance(x_sym, sp.Symbol) and isinstance(y_sym, sp.Symbol)):
                    continue
                if not (self._is_decision_name(problem, x_sym.name) and self._is_decision_name(problem, y_sym.name)):
                    continue

                # need finite bounds (otherwise skip; do NOT Big-M here)
                lbx, ubx = self._get_bounds(problem, x_sym)
                lby, uby = self._get_bounds(problem, y_sym)
                if lbx is None or ubx is None or lby is None or uby is None:
                    continue
                
                # NEW: bounds sanity (avoid lb > ub)
                try:
                    lbx_f, ubx_f = float(lbx), float(ubx)
                    lby_f, uby_f = float(lby), float(uby)
                except Exception:
                    continue
                if lbx_f > ubx_f + tol or lby_f > uby_f + tol:
                    continue


                # reuse existing w symbol
                if w_name not in getattr(problem, "variables", {}):
                    continue
                w_sym = self._sym(problem, w_name)
                # tighten w bounds from current x/y bounds
                try:
                    w_lb = min(lbx * lby, lbx * uby, ubx * lby, ubx * uby)
                    w_ub = max(lbx * lby, lbx * uby, ubx * lby, ubx * uby)
                    w_var = problem.variables.get(w_name)
                    if w_var is not None:
                        w_var.lb = w_lb if w_var.lb is None else max(w_var.lb, w_lb)
                        w_var.ub = w_ub if w_var.ub is None else min(w_var.ub, w_ub)
                except Exception:
                    pass

                # McCormick envelopes
                # McCormick envelopes (ALL in <= form to match canonicalization/solver expectations)
                cons = [
                    # w >= lbx*y + lby*x - lbx*lby  <=>  (lbx*y + lby*x - lbx*lby - w) <= 0
                    ((lbx * y_sym + lby * x_sym - lbx * lby - w_sym), '<=', 0),
                    # w >= ubx*y + uby*x - ubx*uby  <=>  (ubx*y + uby*x - ubx*uby - w) <= 0
                    ((ubx * y_sym + uby * x_sym - ubx * uby - w_sym), '<=', 0),
                    # w <= ubx*y + lby*x - ubx*lby  <=>  (w - (...)) <= 0
                    ((w_sym - (ubx * y_sym + lby * x_sym - ubx * lby)), '<=', 0),
                    # w <= lbx*y + uby*x - lbx*uby  <=>  (w - (...)) <= 0
                    ((w_sym - (lbx * y_sym + uby * x_sym - lbx * uby)), '<=', 0),
                ]

                before = len(problem.constraints)
                for expr, sense, rhs in cons:
                    problem.add_constraint_unique(expr, sense, rhs)
                if len(problem.constraints) > before:
                    added += 1

        # ---------------- A2) Fuller RLT from affine constraints (optional) -----------
        if enable_affine_rlt and (added < rlt_budget):
            added += self._add_rlt_from_affine_constraints(
                problem,
                budget=rlt_budget - added,
                tol=tol,
                # 下面这几个默认参数会极大降低爆炸风险
                max_xk_per_ineq=1,
                max_lhs_vars=6,
                allow_new_pairs=True,   # create a few new pairs to make cuts effective
            )

        # ---------------- B) Outer-approx cuts for convex quadratic -------------------
        # 这段保持你原来的实现（我不改你 OA 的逻辑）
        def _is_numeric_psd(Q):
            try:
                Qn = np.array(Q.tolist(), dtype=float)
                Qn = 0.5 * (Qn + Qn.T)
                eig = np.linalg.eigvalsh(Qn)
                return bool(np.min(eig) >= -1e-10), Qn
            except Exception:
                return False, None

        def _get_quad_terms(expr):
            # 只处理你现有的 alpha*Trace(x.T*Q*x)
            try:
                ok, a, x_sym, Q_expr = self.is_scaled_xTQx(expr)
                return ok, float(a), x_sym, Q_expr
            except Exception:
                return False, None, None, None

        # iterate constraints for OA
        for c in getattr(problem, "constraints", []):
            if oa_budget <= 0:
                break
            if getattr(c, "sense", None) not in ["<=", "<"]:
                continue
            if not hasattr(c, "expr"):
                continue
            lhs = c.expr
            rhs = c.rhs
            ok, a, x_sym, Q_expr = _get_quad_terms(lhs)
            if not ok:
                continue
            try:
                rhs_f = float(sp.N(rhs))
            except Exception:
                continue

            psd, Qn = _is_numeric_psd(Q_expr)
            if not psd:
                continue

            mv = getattr(problem, "matrix_variables", {}).get(getattr(x_sym, "name", None), None)
            if mv is None or mv.lb is None or mv.ub is None:
                continue
            # Per-element midpoint if bounds are vector-shaped; fall back to scalar bounds.
            n_dim = int(Qn.shape[0])
            elem_bounds = [self._elem_bounds(mv, i) for i in range(n_dim)]
            if any((lb is None or ub is None) for lb, ub in elem_bounds):
                continue
            x0 = np.array([0.5 * (lb + ub) for lb, ub in elem_bounds], dtype=float)

            grad0 = a * (Qn + Qn.T).dot(x0)
            f0 = a * float(x0.T.dot(Qn).dot(x0))

            # cut: f(x0) + grad(x0)^T (x - x0) <= rhs
            # => grad^T x <= rhs - f0 + grad^T x0
            # build scalar dot: sum_i grad0[i] * x_i
            try:
                x_name = x_sym.name
                dot = 0
                for i in range(int(x_sym.shape[0])):
                    xi = self._ensure_elem_var(problem, x_sym, i, None, None)
                    dot += float(grad0[i]) * xi
                b_cut = rhs_f - f0 + float(np.dot(grad0, x0))
            except Exception:
                continue

            before = len(problem.constraints)
            problem.add_constraint_unique(sp.expand(dot), "<=", b_cut)
            if len(problem.constraints) > before:
                oa_budget -= 1

        # optional: propagate tighter bounds after adding cuts
        # try:
        #     self.apply_bound_tightening(problem, max_rounds=1, tol=tol)
        # except Exception:
        #     pass

        return added

    
    def _add_rlt_from_affine_constraints(
        self,
        problem,
        budget: int = 200,
        tol: float = 1e-9,
        max_xk_per_ineq: int = 2,       # NEW: 每条约束最多选 K 个 xk
        max_lhs_vars: int = 8,          # NEW: lhs 变量太多直接跳过
        allow_new_pairs: bool = False,  # NEW: 默认不创建新的 w（关键防爆炸）
    ):
        """
        Level-1 RLT cuts from affine inequalities (safe + budgeted).

        只在以下条件下生成：
        - 约束是 affine 且 rhs 是常数
        - lhs 中涉及的每个变量都有有限 bounds（否则跳过，避免 Big-M）
        - xk 也必须有 bounds（来自 bounded list）
        - 默认只复用已有 bilinear 代理（allow_new_pairs=False）
        """
        import sympy as sp

        if budget <= 0:
            return 0

        # ---------- helpers ----------
        def _rhs_is_const(r):
            try:
                _ = float(sp.N(r))
                return True
            except Exception:
                return False

        def _pair_key(a: str, b: str):
            x, y = sorted([a, b])
            return (id(problem), x, y)

        # ---- collect affine inequalities (normalized to <=) ----
        affine_ineqs = []  # list of (coeff_dict{name->float}, const_float, rhs_float)
        for c in getattr(problem, "constraints", []):
            lhs, sense, rhs = c.expr, c.sense, c.rhs

            # normalize to lhs <= rhs
            if sense in [">=", ">"]:
                lhs, rhs = -lhs, -rhs
                sense = "<="

            if sense not in ["<=", "<"]:
                continue
            if not _rhs_is_const(rhs):
                continue
            if not self._is_affine_scalar(lhs, problem):
                continue

            try:
                rhs_f = float(sp.N(rhs))
            except Exception:
                continue

            lhs_exp = sp.expand(lhs)

            # decision symbols in lhs
            syms = [s for s in lhs_exp.free_symbols
                    if isinstance(s, sp.Symbol) and self._is_decision_name(problem, s.name)]
            if any(s.name.startswith("w_") for s in syms):
                continue
            if len(syms) == 0:
                continue
            if len(syms) > max_lhs_vars:
                continue

            # extract numeric coefficients
            coeff = {}
            ok = True
            for s in syms:
                ci = sp.expand(lhs_exp).coeff(s)
                if not ci.is_Number:
                    ok = False
                    break
                coeff[s.name] = float(ci)
            if not ok:
                continue

            # constant term
            const_expr = lhs_exp
            for name, ai in coeff.items():
                const_expr = const_expr - ai * self._sym(problem, name)
            const_expr = sp.simplify(const_expr)
            if hasattr(const_expr, "free_symbols") and len(const_expr.free_symbols) > 0:
                continue
            try:
                const_f = float(sp.N(const_expr))
            except Exception:
                continue

            affine_ineqs.append((coeff, const_f, rhs_f))

        if not affine_ineqs:
            return 0

        # ---- bounded scalar variables ----
        bounded = []  # list of (sym, lb, ub)
        for name, v in getattr(problem, "variables", {}).items():
            if getattr(v, "vtype", None) == "matrix":
                continue
            if name.startswith("w_"):
                continue
            if v.lb is None or v.ub is None:
                continue
            bounded.append((self._sym(problem, name), float(v.lb), float(v.ub)))

        if not bounded:
            return 0

        added = 0

        for coeff, const_f, rhs_f in affine_ineqs:
            if added >= budget:
                break

            g0 = rhs_f - const_f  # g(x) = g0 - Σ a_i x_i
            sym_names = list(coeff.keys())

            # safety: lhs vars must all have finite bounds (avoid Big-M & incorrectness)
            ok_bounds = True
            for name in sym_names:
                s_sym = self._sym(problem, name)
                lb_s, ub_s = self._get_bounds(problem, s_sym)
                if lb_s is None or ub_s is None:
                    ok_bounds = False
                    break
            if not ok_bounds:
                continue

            # choose candidate xk:
            # - must be bounded
            # - must NOT appear in lhs (to avoid square/RLT self-product complications)
            # - if allow_new_pairs=False, require existing w-cache for all pairs (xi,xk)
            candidates = []
            for xk, lk, uk in bounded:
                if xk.name in coeff:
                    continue

                if xk.name.startswith("w_"):
                    continue

                if not allow_new_pairs:
                    # require: for every xi in lhs, the pair already exists in w_cache
                    all_exist = True
                    for name in sym_names:
                        if _pair_key(name, xk.name) not in self.w_cache:
                            all_exist = False
                            break
                    if not all_exist:
                        continue

                # heuristic: prefer tighter xk (smaller interval width)
                width = float(uk - lk)
                candidates.append((width, xk, lk, uk))

            if not candidates:
                continue

            candidates.sort(key=lambda t: t[0])  # tighter first

            picked = 0
            for _, xk, lk, uk in candidates:
                if added >= budget:
                    break
                if picked >= max_xk_per_ineq:
                    break

                # build w_{i,k} map (REUSE cache; only call _linearize_pair if allowed and missing)
                wmap = {}
                bad = False
                for name in sym_names:
                    pk = _pair_key(name, xk.name)
                    if pk in self.w_cache:
                        wmap[name] = self.w_cache[pk]
                    else:
                        if not allow_new_pairs:
                            bad = True
                            break
                        # allow_new_pairs=True 才会走到这里：确保 bounds 完整，否则跳过
                        s_sym = self._sym(problem, name)
                        lb_s, ub_s = self._get_bounds(problem, s_sym)
                        lbk, ubk = self._get_bounds(problem, xk)
                        if lb_s is None or ub_s is None or lbk is None or ubk is None:
                            bad = True
                            break
                        wmap[name] = self._linearize_pair(problem, s_sym, xk)  # may add envelopes
                        # after call, it should be cached
                        if pk in self.w_cache:
                            wmap[name] = self.w_cache[pk]

                if bad:
                    continue

                # Cut1: (g0 - Σ a_i x_i) * (xk - lk) >= 0
                expr1 = g0 * xk - g0 * lk
                for name in sym_names:
                    ai = coeff[name]
                    s_sym = self._sym(problem, name)
                    expr1 += ai * lk * s_sym - ai * wmap[name]

                before = len(problem.constraints)
                problem.add_constraint_unique(sp.expand(expr1), ">=", 0)
                if len(problem.constraints) > before:
                    added += 1
                    if added >= budget:
                        picked += 1
                        break

                # Cut2: (g0 - Σ a_i x_i) * (uk - xk) >= 0
                expr2 = g0 * uk - g0 * xk
                for name in sym_names:
                    ai = coeff[name]
                    s_sym = self._sym(problem, name)
                    expr2 += -ai * uk * s_sym + ai * wmap[name]

                before = len(problem.constraints)
                problem.add_constraint_unique(sp.expand(expr2), ">=", 0)
                if len(problem.constraints) > before:
                    added += 1

                picked += 1

        return added



    def apply_perspective_relaxation(self, problem, location, sub_expr, tol: float = 1e-9):
        """
        Perspective relaxation (minimal, safe version):
        - Detect scalar term: coeff * b * x^2  where b is binary (or integer relaxed to [0,1]) and x is continuous scalar.
        - Introduce z >= x^2 (convex quadratic epigraph), then linearize w = b*z with McCormick on (b,z).
        - Replace coeff*b*x^2 by coeff*w.
        """
        import sympy as sp

        expr = sp.expand(sub_expr)
        coeff = sp.Integer(1)
        core = expr
        if core.is_Mul:
            coeff, core = core.as_coeff_Mul()

        b_sym = None
        x_sym = None

        def _match_square(e):
            if isinstance(e, sp.Pow) and e.exp == 2 and isinstance(e.base, sp.Symbol):
                return e.base
            if e.is_Mul and len(e.args) == 2 and e.args[0] == e.args[1] and isinstance(e.args[0], sp.Symbol):
                return e.args[0]
            return None

        if core.is_Mul:
            args = list(core.args)
            for i, a in enumerate(args):
                if isinstance(a, sp.Symbol) and self._is_decision_name(problem, a.name):
                    cand_b = a
                    rest = sp.Mul(*[args[j] for j in range(len(args)) if j != i])
                    cand_x = _match_square(rest)
                    if cand_x is not None:
                        b_sym, x_sym = cand_b, cand_x
                        break

        if b_sym is None or x_sym is None:
            self._mark_identity_rewrite(location, sub_expr)
            return

        vb = problem.variables.get(b_sym.name, None)
        vx = problem.variables.get(x_sym.name, None)
        if vb is None or vx is None:
            self._mark_identity_rewrite(location, sub_expr);  return

        # require b ∈ [0,1] (binary or relaxed)
        if vb.lb is None or vb.ub is None:
            self._mark_identity_rewrite(location, sub_expr);  return
        if vb.lb < -tol or vb.ub > 1 + tol:
            self._mark_identity_rewrite(location, sub_expr);  return

        # x bounds needed for z bounds
        lbx, ubx = self._get_bounds(problem, x_sym)
        if lbx is None or ubx is None:
            self._mark_identity_rewrite(location, sub_expr);  return

        z_name = f"z_sq_{x_sym.name}"
        z_lb = 0.0 if (lbx <= 0 <= ubx) else min(lbx*lbx, ubx*ubx)
        z_ub = max(lbx*lbx, ubx*ubx)
        if z_name not in problem.variables:
            problem.add_variable(z_name, lb=z_lb, ub=z_ub, vtype="continuous")
        else:
            vz = problem.variables[z_name]
            vz.lb = z_lb if vz.lb is None else max(vz.lb, z_lb)
            vz.ub = z_ub if vz.ub is None else min(vz.ub, z_ub)

        z_sym = self._sym(problem, z_name)

        # convex epigraph: z >= x^2
        problem.add_constraint_unique(z_sym - x_sym**2, ">=", 0)

        # McCormick on (b,z): w ≈ b*z
        w_sym = self._linearize_pair(problem, b_sym, z_sym)

        new_expr = sp.expand(coeff * w_sym)
        self._replace_everywhere(problem, location, sub_expr, new_expr)
        
        
    def apply_qcr(self, problem, location, sub_expr, tol: float = 1e-9):
        """
        αBB-style convex under-estimator for an indefinite quadratic form over a box.

        Pattern: alpha * Trace(x.T * Q * x)   (handled by existing is_scaled_xTQx)
        If Q is indefinite, build a convex under-estimator over x in [l,u].

        Safety gates:
        - Objective only if obj_sense == 'min'
        - Constraint only if on LHS and sense is <= (or <) and rhs constant
        """
        import sympy as sp
        import numpy as np

        if self.enable_bt_warmup and not getattr(problem, "_ab_bt_warm_done", False):
            try:
                if self.debug_bt:
                    print("[BT][warmup] qcr pre-BT (max_rounds=1)")
                changed = self.apply_bound_tightening(problem, max_rounds=1)
                if self.debug_bt:
                    print(f"[BT][warmup] qcr done changed={changed}")
            except Exception:
                pass
            problem._ab_bt_warm_done = True

        # --- location safety gate ---
        if location == "Objective":
            if getattr(problem, "obj_sense", "min") != "min":
                self._mark_identity_rewrite(location, sub_expr);  return
        elif location.startswith("Constraint_"):
            p = location.split("_")
            if len(p) < 3 or p[2].upper() != "LHS":
                self._mark_identity_rewrite(location, sub_expr);  return
            idx = int(p[1]) - 1
            cons = problem.constraints[idx]
            if cons.sense not in ["<=", "<"]:
                self._mark_identity_rewrite(location, sub_expr);  return
            if isinstance(cons.rhs, sp.Expr) and len(cons.rhs.free_symbols) > 0:
                self._mark_identity_rewrite(location, sub_expr);  return
        else:
            self._mark_identity_rewrite(location, sub_expr);  return

        # --- match quadratic form ---
        ok, alpha, x_sym, Q_expr = self.is_scaled_xTQx(sub_expr)
        if not ok:
            ok2, alpha2, x2_sym, Q2_expr, y2_sym = self.is_scaled_xTQy(sub_expr)
            if not ok2:
                self._mark_identity_rewrite(location, sub_expr);  return
            # x2.T*Q2*y2  ->  z.T*A*z
            z_sym, A_expr = self._lift_xTQy_to_zT_A_z(problem, location, x2_sym, Q2_expr, y2_sym)
            alpha, x_sym, Q_expr = alpha2, z_sym, A_expr

        # --- parse numeric matrix ---
        try:
            alpha_f = float(sp.N(alpha))
            Qm = sp.Matrix(Q_expr)
            Qn = np.array(Qm.evalf(), dtype=float)
        except Exception:
            self._mark_identity_rewrite(location, sub_expr);  return

        Qn = 0.5 * (Qn + Qn.T)
        Qeff = alpha_f * Qn

        # --- dimension ---
        try:
            n = int(getattr(x_sym, "shape", (0, 0))[0])
        except Exception:
            n = 0
        if n <= 0:
            self._mark_identity_rewrite(location, sub_expr);  return

        # --- fetch bounds from vector variable x ---
        mv = getattr(problem, "matrix_variables", {}).get(getattr(x_sym, "name", None), None)
        if mv is None:
            self._mark_identity_rewrite(location, sub_expr);  return

        def _as_list(v, n_):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return [float(v)] * n_
            if isinstance(v, (list, tuple)):
                if len(v) != n_:
                    return None
                return [float(t) for t in v]
            return None

        lbs = _as_list(getattr(mv, "lb", None), n)
        ubs = _as_list(getattr(mv, "ub", None), n)
        if lbs is None or ubs is None:
            # 没有逐元素 box bounds，αBB 不安全/不收紧，直接跳过
            self._mark_identity_rewrite(location, sub_expr);  return

        # --- QCR via eigen-decomposition ---
        try:
            eigvals, eigvecs = np.linalg.eigh(Qeff)
        except Exception:
            self._mark_identity_rewrite(location, sub_expr);  return

        if float(np.min(eigvals)) >= -tol:
            self._mark_identity_rewrite(location, sub_expr);  return

        # PSD part: keep positive eigenvalues
        pos_eigs = np.where(eigvals > 0.0, eigvals, 0.0)
        Qpsd = eigvecs @ np.diag(pos_eigs) @ eigvecs.T
        Qpsd = 0.5 * (Qpsd + Qpsd.T)
        Qpsd_expr = sp.Matrix(Qpsd)

        # convex quadratic part
        quad_convex = sp.Trace(x_sym.T * Qpsd_expr * x_sym)

        # linear under-estimator for each negative eigen-direction
        lin_part = 0
        neg_idx = np.where(eigvals < 0.0)[0]
        for k in neg_idx:
            lam = float(eigvals[k])  # negative
            v = eigvecs[:, k]

            t_lb = 0.0
            t_ub = 0.0
            for j in range(n):
                vj = float(v[j])
                lj = float(lbs[j])
                uj = float(ubs[j])
                if vj >= 0:
                    t_lb += vj * lj
                    t_ub += vj * uj
                else:
                    t_lb += vj * uj
                    t_ub += vj * lj
            if t_lb > t_ub:
                t_lb, t_ub = t_ub, t_lb

            t_expr = 0
            for j in range(n):
                vj = float(v[j])
                if vj != 0.0:
                    t_expr += vj * x_sym[j, 0]

            segs = int(getattr(self, "qcr_segments", 1) or 1)
            if segs <= 1 or abs(t_ub - t_lb) <= tol:
                # single chord of concave lambda * t^2 over [t_lb, t_ub]
                lin_part += lam * (t_ub + t_lb) * t_expr - lam * (t_ub * t_lb)
                continue

            # multi-segment chords: use s >= chord_i(t), and replace with s
            points = np.linspace(t_lb, t_ub, segs + 1, dtype=float)
            chords = []
            for a, b in zip(points[:-1], points[1:]):
                chords.append(lam * (a + b) * t_expr - lam * (a * b))

            # auxiliary scalar for max of chords
            self.qcr_aux_counter += 1
            s_name = f"qcr_s_{self.qcr_aux_counter}"
            if s_name not in problem.variables:
                # Do not set tight bounds for qcr_s to avoid accidental infeasibility
                problem.add_variable(s_name, lb=None, ub=None, vtype="continuous")
            s_sym = self._sym(problem, s_name)
            for chord in chords:
                problem.add_constraint_unique(s_sym - chord, ">=", 0)
            lin_part += s_sym

        new_expr = sp.expand(quad_convex + lin_part)
        self._replace_everywhere(problem, location, sub_expr, new_expr)


    def _replace_expr_allowing_scalar_wrap(self, expr, old_expr, new_expr):
        import sympy as sp

        # 统一 new_expr 的类型（标量/1×1 Matrix）以匹配 old_expr
        new_core = self._as_matrix(new_expr) if isinstance(old_expr, sp.MatrixExpr) \
                else self._as_scalar(new_expr)

        # === 0) 直接替换 old_expr → new_core
        out = expr.xreplace({old_expr: new_core})

        # === 1) 处理“标量 × old_expr”（α * old_expr）→ α * new_core
        def _is_scalar_times_old(e):
            if isinstance(e, sp.Mul):
                mats = [a for a in e.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                if not mats:
                    return False
                core = mats[0] if len(mats) == 1 else sp.MatMul(*mats)
                return core == old_expr
            if isinstance(e, sp.MatMul):
                alpha, core = self._pull_scalar_from_matmul(e)
                return core == old_expr
            return False

        def _rewrite_scalar_times_old(e):
            if isinstance(e, sp.Mul):
                mats = [a for a in e.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                scas = [a for a in e.args if not isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                alpha = sp.Mul(*scas) if scas else 1
                return alpha * new_core
            if isinstance(e, sp.MatMul):
                alpha, _ = self._pull_scalar_from_matmul(e)
                return alpha * new_core
            return new_core

        out = out.replace(_is_scalar_times_old, _rewrite_scalar_times_old)

        # === 2) 处理“Trace(α * CORE_old)” → α * new_core
        # old_expr 若是 Trace(core_old)，我们还要把内部带系数的也换掉
        if isinstance(old_expr, sp.Trace):
            core_old = old_expr.arg

            def _is_trace_scaled_old(e):
                if not isinstance(e, sp.Trace):
                    return False
                inner = e.arg
                # 拆出矩阵因子与标量因子
                if isinstance(inner, sp.Mul):
                    mats = [a for a in inner.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                    scas = [a for a in inner.args if not isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                    if not mats:
                        return False
                    core = mats[0] if len(mats) == 1 else sp.MatMul(*mats)
                    # 只要核心等于 old 的核心，就认为是 Trace(α*CORE_old)
                    return sp.srepr(core) == sp.srepr(core_old)
                return False

            def _rewrite_trace_scaled_old(e):
                inner = e.arg
                mats = [a for a in inner.args if isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                scas = [a for a in inner.args if not isinstance(a, (sp.MatrixExpr, sp.MatrixBase))]
                alpha = sp.Mul(*scas) if scas else 1
                return alpha * new_core  # 关键：把 α 提出来乘 new_core

            out = out.replace(_is_trace_scaled_old, _rewrite_trace_scaled_old)

        return out

    def _replace_in_problem(self, problem, location, old_expr, new_expr):
        if isinstance(old_expr, MatrixExpr):
            new_expr = self._as_matrix(new_expr)
        else:
            new_expr = self._as_scalar(new_expr)

        def _rec(loc):
            self.last_rewrite = {"location": loc, "old": old_expr, "new": new_expr}

        if location == "Objective":
            new_obj = self._replace_expr_allowing_scalar_wrap(problem.obj_expr, old_expr, new_expr)
            if sympy.srepr(new_obj) != sympy.srepr(problem.obj_expr):
                problem.obj_expr = new_obj
                _rec("Objective")
            return

        if location.startswith("Constraint_"):
            parts = location.split("_")
            idx  = int(parts[1]) - 1
            side = parts[2].upper()
            cons = problem.constraints[idx]

            if side == "LHS":
                new_lhs = self._replace_expr_allowing_scalar_wrap(cons.expr, old_expr, new_expr)
                if sympy.srepr(new_lhs) != sympy.srepr(cons.expr):
                    cons.expr = new_lhs
                    _rec(f"Constraint_{idx+1}_LHS")
            elif side == "RHS" and isinstance(cons.rhs, sympy.Expr):
                new_rhs = self._replace_expr_allowing_scalar_wrap(cons.rhs, old_expr, new_expr)
                if sympy.srepr(new_rhs) != sympy.srepr(cons.rhs):
                    cons.rhs = new_rhs
                    _rec(f"Constraint_{idx+1}_RHS")
            return

        print(f"[Warning] Unknown location '{location}' — no replacement performed.")

    def _replace_everywhere(self, problem, location, old_expr, new_expr):
        # print(f"Replacing {old_expr} with {new_expr}")
        if isinstance(old_expr, MatrixExpr):
            new_expr = self._as_matrix(new_expr)
        else:
            new_expr = self._as_scalar(new_expr)

        # 目标
        if problem.obj_expr is not None:
            problem.obj_expr = self._replace_expr_allowing_scalar_wrap(problem.obj_expr, old_expr, new_expr)

        # 约束
        for cons in problem.constraints:
            cons.expr = self._replace_expr_allowing_scalar_wrap(cons.expr, old_expr, new_expr)
            if isinstance(cons.rhs, sympy.Expr):
                cons.rhs = self._replace_expr_allowing_scalar_wrap(cons.rhs, old_expr, new_expr)

        self.last_rewrite = {
            "location": location,
            "old": old_expr,
            "new": new_expr,
        }

