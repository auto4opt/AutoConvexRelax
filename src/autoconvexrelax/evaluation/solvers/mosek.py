
# -*- coding: utf-8 -*-
"""
MOSEK backend for solving convex (conic + SDP) relaxations of QCQPProblem.

This backend complements the Gurobi adapter in the same package.
It reuses:
  - QCQPProblem / Variable / VectorVariableSymbol / MatrixVariableSymbol
  - trace_to_scalar_expr
  - _to_affine_psd_matrix

The shared expression utilities keep solver behavior consistent.
"""

from __future__ import annotations

import sympy as sp

try:
    from sympy.matrices.expressions.matexpr import MatrixElement as _MatrixElement
except Exception:
    _MatrixElement = None
    
import numpy as np


from autoconvexrelax.core.problem import QCQPProblem, VectorVariableSymbol, MatrixVariableSymbol
from autoconvexrelax.evaluation.expressions import _trace_to_scalar_expr, _to_affine_psd_matrix, normalize_expr
from mosek.fusion import Matrix, Expr

import sys, os, time, traceback
# Optional: MOSEK (for SDP / conic relaxations)

import mosek.fusion as mf
import mosek



class _NotAffineError(Exception):
    pass

def _to_scalar_expr(e):
    """递归把表达式里的 Trace(...) 全部展开为标量表达式（包括嵌套在 Add/Mul 内的 Trace）。"""
    if e is None:
        return sp.Integer(0)

    # 1x1 matrix -> scalar
    if isinstance(e, sp.MatrixBase) and getattr(e, "shape", None) == (1, 1):
        return _to_scalar_expr(e[0, 0])
    if isinstance(e, sp.MatrixExpr) and getattr(e, "shape", None) == (1, 1):
        try:
            return _to_scalar_expr(e[0, 0])
        except Exception:
            return e

    # Trace -> explicit scalar (e.g., Trace(y.T*y) -> sum_i y_i^2)
    if isinstance(e, sp.Trace) or getattr(e, "func", None) is sp.Trace:
        s = _trace_to_scalar_expr(e)
        return _to_scalar_expr(s) if s is not None else e

    # recurse so Trace inside Add/Mul also gets eliminated
    if isinstance(e, sp.Add):
        return sp.Add(*[_to_scalar_expr(a) for a in e.args])
    if isinstance(e, sp.Mul):
        return sp.Mul(*[_to_scalar_expr(a) for a in e.args])
    if isinstance(e, sp.Pow):
        base, exp = e.args
        return sp.Pow(_to_scalar_expr(base), exp)

    try:
        return normalize_expr(e)
    except Exception:
        return e

def _debug_print_problem(problem, header="[MOSEK ERROR]"):
    print("\n" + "=" * 80)
    print(header)
    try:
        print(f"Problem name: {getattr(problem, 'name', 'N/A')}")
    except Exception:
        pass

    print("\n[Objective]")
    try:
        print(problem.objective)
    except Exception as e:
        print("Failed to print objective:", e)

    print("\n[Variables]")
    for name, v in problem.variables.items():
        try:
            print(f"  {name}: lb={v.lb}, ub={v.ub}, vtype={v.vtype}")
        except Exception:
            print(f"  {name}: <print failed>")

    print("\n[Constraints]")
    for i, cons in enumerate(problem.constraints):
        try:
            print(f"  C{i}: {cons.expr} {cons.sense} {cons.rhs}")
        except Exception:
            print(f"  C{i}: <print failed>")

    print("\n[Last rewrite]")
    try:
        print(problem.last_rewrite)
    except Exception:
        pass

    print("=" * 80 + "\n")


def _ensure_all_scalar_symbols_in_prob(M, prob, sym2var, matvar2fusion):
    """MOSEK 侧兜底：把问题里出现但未注册到 prob.variables 的标量 Symbol 自动建成决策变量。

    典型场景：McCormick / RLT 等操作引入了 w_* / t_* / tau_* 等辅助变量，
    但这些符号如果没进入 prob.variables，sym2var 就缺映射，后续用 Poly(domain='RR')
    会把它当作“系数”，导致 CoercionFailed / _NotAffineError。
    """
    import sympy as sp
    import mosek.fusion as mf

    exprs = []
    # objective
    try:
        exprs.append(_to_scalar_expr(prob.obj_expr))
    except Exception:
        pass

    # constraints
    for cst in getattr(prob, "constraints", []):
        try:
            exprs.append(_to_scalar_expr(cst.expr))
        except Exception:
            pass
        try:
            if getattr(cst, "rhs", None) is not None:
                exprs.append(_to_scalar_expr(cst.rhs))
        except Exception:
            pass

    # PSD constraints (best-effort)
    for pc in getattr(prob, "psd_constraints", []):
        mat = getattr(pc, "mat", None) or getattr(pc, "expr", None)
        if mat is None:
            continue
        try:
            Msp = sp.Matrix(mat)
            exprs.extend(list(Msp))
        except Exception:
            # ignore non-matrix psd reps
            pass

    def _find_existing_fusion_var_by_name(sym2var, name: str):
        for k, v in sym2var.items():
            if hasattr(k, "name") and k.name == name:
                return v
        return None

    for ex in exprs:
        if ex is None:
            continue
        for s in getattr(ex, "free_symbols", set()):
            if not isinstance(s, sp.Symbol):
                continue

            canon = sp.Symbol(s.name)

            # 1) 直接命中（同一个 Symbol 对象）
            if s in sym2var:
                continue

            # 2) canonical 命中（同名不同对象）
            if canon in sym2var:
                sym2var[s] = sym2var[canon]
                continue

            # 3) 兜底：按名字查已有（防止 sym2var 里 key 不是 canon 但同名）
            v_exist = _find_existing_fusion_var_by_name(sym2var, s.name)
            if v_exist is not None:
                sym2var[canon] = v_exist
                sym2var[s] = v_exist
                continue

            # 4) 真没有，才创建
            v_new = M.variable(s.name, 1, mf.Domain.unbounded())
            sym2var[canon] = v_new
            sym2var[s] = v_new


def _shape(*dims: int):
    return np.array([int(d) for d in dims], dtype=np.int32)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return float(sp.N(x))


def _is_const(expr: sp.Expr) -> bool:
    return (expr.free_symbols is None) or (len(expr.free_symbols) == 0)


def _get_scalar_symbol_name_from_matrix_element(me) -> str:
    """
    Your VectorVariableSymbol elements typically appear as base[i,0] in SymPy.
    We alias base[i,0] -> Symbol(f"{base}_{i}") when such a scalar variable exists.
    """
    base = str(me.parent)
    i = int(me.i)
    j = int(me.j)
    if j == 0:
        return f"{base}_{i}"
    return f"{base}_{i}_{j}"


def _np_from_sympy_mat(M):
    M = sp.Matrix(M)
    return np.array(M.tolist(), dtype=float)

def _fusion_trace_coeff_times_var(coeff_np, X_fusion):
    # Trace(coeff * X) = <coeff^T, X> = sum( (coeff^T) ⊙ X )
    A = Matrix.dense(coeff_np.T)
    return Expr.sum(Expr.mulElm(A, X_fusion))

def _sympy_to_fusion_affine(expr, sym2var, matvar2fusion):
    """Convert a SymPy scalar expression into a MOSEK Fusion affine Expr.

    Supports: constants, Symbol, MatrixElement (vector/matrix elements), Add, Mul by constant.
    Raises _NotAffineError if a non-affine construct is encountered (e.g., var*var, pow(var,2)).
    """
    if mf is None:
        raise ImportError("MOSEK is not available (mosek.fusion import failed).")

    # 1x1 matrices -> scalar
    if isinstance(expr, sp.MatrixBase) and expr.shape == (1, 1):
        expr = expr[0, 0]

    if expr is None:
        return mf.Expr.constTerm(0.0)

    if isinstance(expr, (int, float)):
        return mf.Expr.constTerm(float(expr))

    if isinstance(expr, (sp.Integer, sp.Float)):
        return mf.Expr.constTerm(float(expr))

    if isinstance(expr, sp.Rational):
        return mf.Expr.constTerm(float(expr))

    if isinstance(expr, sp.Symbol):
        canon = sp.Symbol(expr.name)
        if expr in sym2var:
            return sym2var[expr]
        if canon in sym2var:
            return sym2var[canon]
        raise KeyError(f"Unknown symbol in expression: {expr}")


    if _MatrixElement is not None and isinstance(expr, _MatrixElement):
        base = str(expr.parent)
        i = int(expr.i)
        j = int(expr.j)

        # If there is a scalar variable alias (e.g., x[2,0] -> x_2), prefer it.
        alias = sp.Symbol(_get_scalar_symbol_name_from_matrix_element(expr))
        if alias in sym2var:
            return sym2var[alias]

        # Otherwise, map to a Fusion matrix/vector variable element.
        if base not in matvar2fusion:
            raise KeyError(f"Unknown matrix variable in expression: {base}")
        fv = matvar2fusion[base]
        return fv.index([i, j])

    if isinstance(expr, sp.Add):
        terms = [_sympy_to_fusion_affine(a, sym2var, matvar2fusion) for a in expr.args]
        out = terms[0]
        for t in terms[1:]:
            out = mf.Expr.add(out, t)
        return out

    if isinstance(expr, sp.Mul):
        # Pull out numeric constant part
        const = 1.0
        nonconst = []
        for a in expr.args:
            if isinstance(a, (int, float, sp.Integer, sp.Float, sp.Rational)) and not isinstance(a, sp.Symbol):
                const *= _as_float(a)
            elif _is_const(a) and not isinstance(a, (sp.Symbol,)) and not (_MatrixElement is not None and isinstance(a, _MatrixElement)):
                const *= _as_float(a)
            else:
                nonconst.append(a)

        if len(nonconst) == 0:
            return mf.Expr.constTerm(const)

        if len(nonconst) == 1:
            inner = _sympy_to_fusion_affine(nonconst[0], sym2var, matvar2fusion)
            if const == 1.0:
                return inner
            return mf.Expr.mul(const, inner)

        # More than one non-constant factor -> nonlinear
        raise _NotAffineError(f"Non-affine product encountered: {expr}")

    if isinstance(expr, sp.Pow):
        # quadratic terms are handled separately (via cones)
        raise _NotAffineError(f"Non-affine power encountered: {expr}")

    # Try numeric evaluation fallback
    if _is_const(expr):
        return mf.Expr.constTerm(_as_float(expr))
    
    # ---- Trace 支持 ----
    if isinstance(expr, sp.Trace) or expr.func is sp.Trace:
        mat = expr.args[0]

        if isinstance(mat, sp.MatrixBase):
            return mf.Expr.constTerm(float(mat.trace()))

        factors = list(mat.args) if isinstance(mat, sp.MatMul) else [mat]

        def _is_matvar_factor(f):
            # matvar2fusion 的 key 是字符串 base
            return isinstance(f, sp.MatrixSymbol) and (str(f) in matvar2fusion)

        var_pos = [i for i, f in enumerate(factors) if _is_matvar_factor(f)]
        if len(var_pos) != 1:
            raise _NotAffineError(f"Trace with !=1 matrix var not supported: {expr}")

        k = var_pos[0]
        X_key = str(factors[k])          # "Z_y"
        X = matvar2fusion[X_key]         # Fusion matrix variable

        post_pre = factors[k+1:] + factors[:k]
        if len(post_pre) == 0:
            n = X.getShape()[0]
            coeff_np = np.eye(n, dtype=float)
        else:
            coeff = post_pre[0]
            for f in post_pre[1:]:
                coeff = coeff * f

            # 只允许 coeff 是常数矩阵；不能含其它矩阵变量
            if (len(coeff.free_symbols) > 0) or any(str(ms) in matvar2fusion for ms in coeff.atoms(sp.MatrixSymbol)):
                raise _NotAffineError(f"Trace coeff must be constant: {expr}")

            coeff_np = _np_from_sympy_mat(coeff)

        return _fusion_trace_coeff_times_var(coeff_np, X)
    
    raise _NotAffineError(f"Unsupported / non-affine SymPy expression: {type(expr)} : {expr}")


def _collect_sum_squares_terms(expr: sp.Expr):
    """Extract sum of squares terms from a SymPy expression.

    Returns:
      linear_expr (SymPy): expr with square terms removed
      groups: dict[float, list[SymPy Atom]]: coefficient -> list of vars whose squares appear

    Recognizes terms of form:
      v**2
      c * v**2  (c is numeric)
    """
    expr = sp.expand(expr)
    if isinstance(expr, sp.Add):
        terms = list(expr.args)
    else:
        terms = [expr]

    groups = {}
    kept = []

    for t in terms:
        coef = None
        var = None

        # c * v**2
        if isinstance(t, sp.Mul):
            c = sp.Integer(1)
            rest = []
            for a in t.args:
                if isinstance(a, (int, float, sp.Integer, sp.Float, sp.Rational)) and not isinstance(a, sp.Symbol):
                    c = c * a
                else:
                    rest.append(a)
            if len(rest) == 1 and isinstance(rest[0], sp.Pow):
                p = rest[0]
                if p.exp == 2:
                    coef = float(sp.N(c))
                    var = p.base

        # v**2
        if coef is None and isinstance(t, sp.Pow) and t.exp == 2:
            coef = 1.0
            var = t.base

        if coef is not None and var is not None:
            groups.setdefault(coef, []).append(var)
        else:
            kept.append(t)

    linear_expr = sum(kept) if kept else sp.Integer(0)
    return linear_expr, groups



# -----------------------------------------------------------------------------
# Quadratic polynomial extraction (for convex quadratic objectives/constraints)
# -----------------------------------------------------------------------------
def _is_decision_atom(a: sp.Expr, sym2var: Dict[sp.Symbol, mf.Variable], matvar2fusion: Dict[str, mf.Variable]) -> bool:
    """Return True if `a` is a scalar decision-variable atom (Symbol or MatrixElement)."""
    if isinstance(a, sp.Symbol):
        return (a in sym2var) or (sp.Symbol(a.name) in sym2var)

    if isinstance(a, _MatrixElement):
        base = str(a.parent)
        if base in matvar2fusion:
            return True
        # fall back to scalar aliases: x[i,0] -> Symbol('x_i')
        try:
            i = int(a.indices[0])
        except Exception:
            return False
        alias = sp.Symbol(f"{base}_{i}")
        return alias in sym2var
    return False

def _decision_atoms_in_expr(expr: sp.Expr, sym2var: Dict[sp.Symbol, mf.Variable], matvar2fusion: Dict[str, mf.Variable]) -> List[sp.Expr]:
    atoms = []
    for a in expr.atoms(sp.Symbol, _MatrixElement):
        if _is_decision_atom(a, sym2var, matvar2fusion):
            atoms.append(a)
    atoms.sort(key=lambda z: str(z))
    return atoms

def _fusion_of_decision_atom(a: sp.Expr, sym2var: Dict[sp.Symbol, mf.Variable], matvar2fusion: Dict[str, mf.Variable]) -> mf.Expression:
    # use the existing affine converter on a single atom
    return _sympy_to_fusion_affine(a, sym2var, matvar2fusion)

def _extract_quadratic_poly(
    expr: sp.Expr,
    sym2var: Dict[sp.Symbol, mf.Variable],
    matvar2fusion: Dict[str, mf.Variable],
) -> Tuple[float, Dict[sp.Expr, float], np.ndarray, List[sp.Expr]]:
    """Extract polynomial up to degree 2: expr = const + sum_i lin[i]*v_i + v^T Q v.
    Q is symmetric (float). Raises _NotAffineError if degree>2 or not polynomial.
    """
    expr = sp.expand(expr)
    atoms = _decision_atoms_in_expr(expr, sym2var, matvar2fusion)
    if not atoms:
        try:
            return float(expr), {}, np.zeros((0, 0), dtype=float), []
        except Exception as e:
            raise _NotAffineError(f"Non-polynomial/unsupported constant expr: {expr}") from e

    try:
        poly = sp.Poly(expr, *atoms, domain='RR')
    except Exception as e:
        raise _NotAffineError(f"Expression is not a polynomial in decision vars: {expr}") from e

    if poly.total_degree() > 2:
        raise _NotAffineError(f"Degree>2 expression encountered: {expr}")

    n = len(atoms)
    Q = np.zeros((n, n), dtype=float)
    lin: Dict[sp.Expr, float] = {}
    const = 0.0

    for monom, coeff in poly.terms():
        deg = sum(monom)
        c = float(coeff)
        if deg == 0:
            const += c
        elif deg == 1:
            i = monom.index(1)
            lin[atoms[i]] = lin.get(atoms[i], 0.0) + c
        elif deg == 2:
            idxs = [k for k, p in enumerate(monom) if p != 0]
            if len(idxs) == 1 and monom[idxs[0]] == 2:
                i = idxs[0]
                Q[i, i] += c
            elif len(idxs) == 2 and monom[idxs[0]] == 1 and monom[idxs[1]] == 1:
                i, j = idxs
                Q[i, j] += c / 2.0
                Q[j, i] += c / 2.0
            else:
                raise _NotAffineError(f"Unsupported quadratic monomial in: {expr}")
        else:
            raise _NotAffineError(f"Unsupported monomial degree in: {expr}")

    return const, lin, Q, atoms

def _psd_factor(Q: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return L such that Q = L^T L (up to numerical tolerance) for PSD Q."""
    if Q.size == 0:
        return np.zeros((0, 0), dtype=float)
    Qs = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Qs)
    w_min = float(np.min(w))
    if w_min < -1e-8:
        raise _NotAffineError(f"Quadratic form is not PSD (min eigenvalue={w_min:.3e}).")
    w = np.where(w > tol, w, 0.0)
    keep = w > 0.0
    if not np.any(keep):
        return np.zeros((0, Q.shape[0]), dtype=float)
    L = (np.diag(np.sqrt(w[keep])) @ V[:, keep].T)  # shape (r, n)
    return L

def _fusion_affine_from_poly(const: float, lin: Dict[sp.Expr, float], sym2var: Dict[sp.Symbol, mf.Variable], matvar2fusion: Dict[str, mf.Variable]) -> mf.Expression:
    out = mf.Expr.constTerm(const)
    for a, c in lin.items():
        if abs(c) < 1e-18:
            continue
        out = mf.Expr.add(out, mf.Expr.mul(c, _fusion_of_decision_atom(a, sym2var, matvar2fusion)))
    return out

def _add_convex_quadratic_term(
    M: mf.Model,
    name: str,
    Q: np.ndarray,
    atoms: List[sp.Expr],
    sym2var: Dict[sp.Symbol, mf.Variable],
    matvar2fusion: Dict[str, mf.Variable],
) -> mf.Variable:
    """Add rotated cone representing u >= 0.5 * v^T Q v. Return scalar var u."""
    if Q.size == 0:
        raise ValueError("Empty Q")
    # shrink to active vars
    mask = np.any(np.abs(Q) > 1e-15, axis=0)
    idx = np.where(mask)[0]
    Qs = Q[np.ix_(idx, idx)]
    atoms_s = [atoms[i] for i in idx.tolist()]

    L = _psd_factor(Qs)
    u = M.variable(name, 1, mf.Domain.greaterThan(0.0))
    if L.shape[0] == 0:
        return u

    # Build w = L * v as a stacked vector of scalar expressions to avoid shape conflicts.
    r = int(L.shape[0])
    w_exprs = []
    for i in range(r):
        row = np.asarray(L[i, :]).reshape(-1)
        wi = mf.Expr.constTerm(0.0)
        for c, a in zip(row.tolist(), atoms_s):
            if abs(c) < 1e-15:
                continue
            wi = mf.Expr.add(wi, mf.Expr.mul(float(c), _fusion_of_decision_atom(a, sym2var, matvar2fusion)))
        w_exprs.append(wi)

    # Use scalar expressions consistently in the cone stack.
    u0 = u.index(0)
    cone_expr = _stack_vector_exprs([u0, mf.Expr.constTerm(1.0)] + w_exprs)
    try:
        M.constraint(cone_expr, mf.Domain.inRotatedQCone(r + 2))
    except Exception as e:
        try:
            print(
                "[MOSEK-DEBUG] add_convex_quad failed: "
                f"name={name} n={len(atoms_s)} r={r} "
                f"L_shape={L.shape} Q_shape={Q.shape} Qs_shape={Qs.shape} "
                f"atoms={len(atoms_s)} cone_shape={cone_expr.getShape()}"
            )
            # per-atom shapes (best-effort)
            shapes = []
            for a in atoms_s:
                try:
                    fa = _sympy_to_fusion_affine(a, sym2var, matvar2fusion)
                    shapes.append((str(a), fa.getShape()))
                except Exception as se:
                    shapes.append((str(a), f"ERR:{se}"))
            print(f"[MOSEK-DEBUG] atom_shapes={shapes}")
        except Exception:
            print("[MOSEK-DEBUG] add_convex_quad failed (shape introspection failed)")
        raise
    return u

def _stack_vector_affine(vars_list, sym2var, matvar2fusion):
    """Build a Fusion column vector Expr from a list of SymPy atoms."""
    exprs = [_sympy_to_fusion_affine(v, sym2var, matvar2fusion) for v in vars_list]
    if len(exprs) == 0:
        return mf.Expr.constTerm(0.0)  # 不太会发生，但防御一下
    if len(exprs) == 1:
        return exprs[0]

    # Fusion 的 vstack(*args) 在 Python 侧最多吃 3 个参数；
    # 长向量用 fold 方式两两堆叠即可。
    out = mf.Expr.vstack(exprs[0], exprs[1])
    for e in exprs[2:]:
        out = mf.Expr.vstack(out, e)
    return out

def _stack_vector_exprs(exprs):
    """Build a Fusion column vector Expr from a list of Fusion scalar expressions."""
    if len(exprs) == 0:
        return mf.Expr.constTerm(0.0)
    if len(exprs) == 1:
        return exprs[0]
    out = mf.Expr.vstack(exprs[0], exprs[1])
    for e in exprs[2:]:
        out = mf.Expr.vstack(out, e)
    return out



def _expand_block_matrix(mat: sp.MatrixBase) -> sp.MatrixBase:
    """Expand a SymPy block matrix whose entries may themselves be matrices (e.g., [[1, y.T],[y,Z]])."""
    if not isinstance(mat, sp.MatrixBase):
        raise TypeError("Expected SymPy MatrixBase for PSD constraint matrix.")

    # Fast path: already scalar entries only
    scalar_only = True
    for i in range(mat.rows):
        for j in range(mat.cols):
            e = mat[i, j]
            if isinstance(e, sp.MatrixBase) and e.shape != (1, 1):
                scalar_only = False
            if isinstance(e, sp.MatrixExpr) and tuple(e.shape) != (1, 1):
                scalar_only = False
    if scalar_only:
        return sp.Matrix(mat)

    # Determine block sizes
    row_heights = []
    col_widths = []

    for i in range(mat.rows):
        h = None
        for j in range(mat.cols):
            e = mat[i, j]
            if isinstance(e, sp.MatrixBase) and e.shape != (1, 1):
                h = e.shape[0]
                break
            if isinstance(e, sp.MatrixExpr) and tuple(e.shape) != (1, 1):
                h = int(e.shape[0])
                break
        row_heights.append(h or 1)

    for j in range(mat.cols):
        w = None
        for i in range(mat.rows):
            e = mat[i, j]
            if isinstance(e, sp.MatrixBase) and e.shape != (1, 1):
                w = e.shape[1]
                break
            if isinstance(e, sp.MatrixExpr) and tuple(e.shape) != (1, 1):
                w = int(e.shape[1])
                break
        col_widths.append(w or 1)

    block_rows = []
    for i in range(mat.rows):
        row_blocks = []
        for j in range(mat.cols):
            e = mat[i, j]
            if isinstance(e, sp.MatrixBase):
                em = sp.Matrix(e)
            elif isinstance(e, sp.MatrixExpr):
                em = sp.Matrix(e)
            else:
                em = sp.Matrix([[e]])
            if em.shape != (row_heights[i], col_widths[j]):
                raise ValueError(f"Block shape mismatch at ({i},{j}): got {em.shape}, expected {(row_heights[i], col_widths[j])}")
            row_blocks.append(em)
        block_rows.append(sp.Matrix.hstack(*row_blocks))
    return sp.Matrix.vstack(*block_rows)


def build_mosek_model_from_qcqp(prob: QCQPProblem, relax_integrality: bool = True):
    """Build a MOSEK Fusion model for (convex) QCQP relaxations with PSD constraints.

    Current support (sufficient for your SDP-relaxed problems):
      - Affine objective/constraints (after McCormick / lifting), with optional sum(v_i^2) terms handled via rotated quadratic cones.
      - PSD constraints of the form 'M is positive semidefinite' (explicit block matrices supported).

    Not supported yet:
      - General quadratic forms (x^T Q x) unless rewritten into cones.
      - Quadratic constraints with affine terms on the RHS (requires more general conic reformulations).
    """
    if mf is None:
        raise ImportError("MOSEK is not available (mosek.fusion import failed).")

    M = mf.Model(f"mosek_{prob.name}")

    sym2var = {}
    matvar2fusion = {}

    # 1) Scalar variables
    for v in (prob.variables.values() if isinstance(prob.variables, dict) else prob.variables):
        name = v.name
        lb = v.lb
        ub = v.ub
        vtype = getattr(v, "vtype", "continuous")

        if (not relax_integrality) and vtype in ("integer", "binary"):
            if vtype == "binary":
                fv = M.variable(name, 1, mf.Domain.binary())
            else:
                dom = mf.Domain.unbounded()
                if lb is not None and ub is not None:
                    dom = mf.Domain.inRange(float(lb), float(ub))
                elif lb is not None:
                    dom = mf.Domain.greaterThan(float(lb))
                elif ub is not None:
                    dom = mf.Domain.lessThan(float(ub))
                fv = M.variable(name, 1, mf.Domain.integral(dom))
        else:
            if lb is not None and ub is not None:
                fv = M.variable(name, 1, mf.Domain.inRange(float(lb), float(ub)))
            elif lb is not None:
                fv = M.variable(name, 1, mf.Domain.greaterThan(float(lb)))
            elif ub is not None:
                fv = M.variable(name, 1, mf.Domain.lessThan(float(ub)))
            else:
                fv = M.variable(name, 1, mf.Domain.unbounded())


        sym2var[sp.Symbol(name)] = fv

    # 2) Matrix variables
    for mv in (prob.matrix_variables.values() if isinstance(prob.matrix_variables, dict) else prob.matrix_variables):
        if isinstance(mv, VectorVariableSymbol):
            base = mv.name  # 必须在任何 base 的使用之前

            # If scalar aliases exist, skip creating a separate Fusion var
            aliases_exist = all(sp.Symbol(f"{base}_{i}") in sym2var for i in range(mv.dim))
            if aliases_exist:
                continue

            lb = mv.lb
            ub = mv.ub
            if lb is not None and ub is not None:
                fv = M.variable(base, _shape(mv.dim, 1), mf.Domain.inRange(float(lb), float(ub)))
            elif lb is not None:
                fv = M.variable(base, _shape(mv.dim, 1), mf.Domain.greaterThan(float(lb)))
            elif ub is not None:
                fv = M.variable(base, _shape(mv.dim, 1), mf.Domain.lessThan(float(ub)))
            else:
                fv = M.variable(base, _shape(mv.dim, 1), mf.Domain.unbounded())

            matvar2fusion[base] = fv


        elif isinstance(mv, MatrixVariableSymbol):
            base = mv.name
            r = getattr(mv, "rows", None)
            c = getattr(mv, "cols", None)
            shp = getattr(mv, "shape", None)
            if isinstance(shp, tuple) and len(shp) == 2:
                r, c = int(shp[0]), int(shp[1])
            fv = M.variable(base, _shape(r, c), mf.Domain.unbounded())
            matvar2fusion[base] = fv

            if r == c:
                for i in range(r):
                    for j in range(i + 1, c):
                        M.constraint(
                            mf.Expr.sub(fv.index([i, j]), fv.index([j, i])),
                            mf.Domain.equalsTo(0.0),
                        )

    # 2.5) Ensure auxiliary scalar symbols are registered (e.g., w_*, t_*, tau_* introduced by relaxations)
    _ensure_all_scalar_symbols_in_prob(M, prob, sym2var, matvar2fusion)
    
    
    # 2.6) Element-wise bounds for MatrixElement (e.g., Z[i,j]) stored in prob._me_bounds
    if hasattr(prob, "_me_bounds") and isinstance(prob._me_bounds, dict) and prob._me_bounds:
        for (base, i, j), (lb, ub) in prob._me_bounds.items():
            base = str(base)
            i = int(i); j = int(j)

            fv = None
            if base in matvar2fusion:
                fv = matvar2fusion[base].index([i, j])
            else:
                # fallback: if a scalar alias exists
                alias = sp.Symbol(f"{base}_{i}_{j}")
                if alias in sym2var:
                    fv = sym2var[alias]

            if fv is None:
                continue

            if lb is not None:
                M.constraint(fv, mf.Domain.greaterThan(float(lb)))
            if ub is not None:
                M.constraint(fv, mf.Domain.lessThan(float(ub)))


    # 3) Constraints
    
    # ------------------- 3) Constraints -------------------
    for ci, cst in enumerate(prob.constraints):
        # Build expr = lhs - rhs
        lhs = _to_scalar_expr(cst.expr)
        rhs = _to_scalar_expr(cst.rhs) if cst.rhs is not None else 0.0
        expr = sp.expand(lhs - rhs)

        const, lin, Q, atoms = _extract_quadratic_poly(expr, sym2var, matvar2fusion)

        # Pure affine constraint
        if Q.size == 0 or float(np.max(np.abs(Q))) < 1e-15:
            aff = _fusion_affine_from_poly(const, lin, sym2var, matvar2fusion)
            if cst.sense in ("<=", "<"):
                M.constraint(aff, mf.Domain.lessThan(0.0))
            elif cst.sense in (">=", ">"):
                M.constraint(aff, mf.Domain.greaterThan(0.0))
            elif cst.sense in ("=", "=="):
                M.constraint(aff, mf.Domain.equalsTo(0.0))
            else:
                raise ValueError(f"Unknown constraint sense: {cst.sense}")
            continue

        # Quadratic constraints: only support convex form (PSD) with <=
        if cst.sense not in ("<=", "<"):
            raise _NotAffineError(
                f"Nonlinear constraint with sense {cst.sense} is not convex in this interface: {expr}"
            )

        u = _add_convex_quadratic_term(
            M,
            f"quad_{getattr(cst, 'name', 'cst')}_{ci}",
            Q, atoms, sym2var, matvar2fusion
        )
        aff = _fusion_affine_from_poly(const, lin, sym2var, matvar2fusion)

        # v^T Q v + aff <= 0  <=>  2u + aff <= 0  with u >= 0.5 v^T Q v
        M.constraint(mf.Expr.add(mf.Expr.mul(2.0, u), aff), mf.Domain.lessThan(0.0))

# 4) PSD constraints
    for k, psd in enumerate(prob.psd_constraints):
        mat_expr = psd.matrix_expr
        mat_expr = _to_affine_psd_matrix(mat_expr)
        mat_expr = _expand_block_matrix(sp.Matrix(mat_expr))

        n = mat_expr.rows
        if n != mat_expr.cols:
            raise ValueError(f"PSD constraint matrix must be square, got {mat_expr.shape} for {psd}")

        n = int(mat_expr.rows)  # 先确保是 Python int

        # 推荐：用 int32
        n32 = np.int32(n)

        try:
            dom_psd = mf.Domain.inPSDCone(n32)          # n×n 对称 PSD
        except Exception:
            dom_psd = mf.Domain.inPSDCone(n32, n32)     # 某些版本只吃两个参数

        Y = M.variable(f"PSD_aux_{k}", dom_psd)

        for i in range(n):
            for j in range(n):
                gij = mat_expr[i, j]
                try:
                    aff = _sympy_to_fusion_affine(gij, sym2var, matvar2fusion)
                except _NotAffineError as e:
                    raise RuntimeError(f"PSD matrix has non-affine entry at ({i},{j}): {gij}. Detail: {e}")
                M.constraint(mf.Expr.sub(Y.index([i, j]), aff), mf.Domain.equalsTo(0.0))

    
    # 5) Objective
    obj_sym = _to_scalar_expr(prob.obj_expr)
    const, lin, Q, atoms = _extract_quadratic_poly(obj_sym, sym2var, matvar2fusion)

    obj_aff = _fusion_affine_from_poly(const, lin, sym2var, matvar2fusion)
    obj_expr_fusion = obj_aff

    if Q.size != 0 and float(np.max(np.abs(Q))) >= 1e-15:
        # Convex quadratic term
        if prob.obj_sense != "min":
            raise _NotAffineError("Maximizing a convex quadratic term is non-convex; not supported here.")
        u = _add_convex_quadratic_term(M, "obj_quad", Q, atoms, sym2var, matvar2fusion)
        obj_expr_fusion = mf.Expr.add(obj_expr_fusion, mf.Expr.mul(2.0, u))

    sense = mf.ObjectiveSense.Minimize if prob.obj_sense == "min" else mf.ObjectiveSense.Maximize
    M.objective("obj", sense, obj_expr_fusion)


    try:
        M.writeTask("mosek_build_dump.ptf")
        M.writeTask("mosek_build_dump.opf")
        print("[MOSEK] build dump written: mosek_build_dump.ptf/.opf")
    except Exception:
        pass

    return M


def solve_convex_relax_mosek(prob: QCQPProblem, time_limit: float = 30.0, verbose: bool = False):
    """Solve a convex (conic + SDP) relaxation using MOSEK.

    Returns:
      objective value (float) if a solution is found, else None.
    """
    if mf is None:
        raise ImportError("MOSEK is not available (mosek.fusion import failed).")

    M = build_mosek_model_from_qcqp(prob, relax_integrality=True)

    # Time limit: set both continuous + MIP params defensively.
    try:
        M.setSolverParam("optimizerMaxTime", float(time_limit))
    except Exception:
        pass
    try:
        M.setSolverParam("mioMaxTime", float(time_limit))
    except Exception:
        pass

    dump_tag = f"{prob.name}_t{int(time.time())}"

    def _dump_task(suffix: str):
        try:
            os.makedirs("mosek_dumps", exist_ok=True)
            M.writeTask(os.path.join("mosek_dumps", f"{dump_tag}_{suffix}.ptf"))
            M.writeTask(os.path.join("mosek_dumps", f"{dump_tag}_{suffix}.opf"))
            print(f"[MOSEK] Task dumped to mosek_dumps/{dump_tag}_{suffix}.ptf/.opf")
        except Exception as ee:
            print(f"[MOSEK][WARN] dump task failed: {repr(ee)}")

    def _print_prob(header: str):
        print("\n" + "=" * 80)
        print(header)
        try:
            print(prob)  # 你说 prob 自带 __str__/__repr__，直接用
        except Exception as e:
            print(f"[QCQP][WARN] print(prob) failed: {repr(e)}")
        # 可选：最近一次 rewrite（如果你有这个字段）
        try:
            print("[Last rewrite]", getattr(prob, "last_rewrite", None))
        except Exception:
            pass
        print("=" * 80 + "\n")

    # === solve() ===
    try:
        M.solve()
    except Exception as e:
        print(f"[MOSEK][ERROR] solve() failed for {prob.name}: {repr(e)}")
        traceback.print_exc()
        _print_prob(f"[QCQP @ solve() exception] {prob.name}")
        _dump_task("solve_fail")

        try:
            M.dispose()
        except Exception:
            pass
        return None

    # === status ===
    pstat = None
    sstat = None
    try:
        pstat = M.getProblemStatus()
        sstat = M.getPrimalSolutionStatus()
        print(f"[MOSEK] status: ProblemStatus={pstat}, PrimalSolutionStatus={sstat}")
    except Exception as e:
        print(f"[MOSEK][WARN] cannot query status: {repr(e)}")

    # 关键：PrimalInfeasible 时不要继续 primalObjValue()，而是把问题打印出来并 dump
    if pstat is not None and "PrimalInfeasible" in str(pstat):
        print(f"[MOSEK][ERROR] PrimalInfeasible for {prob.name}")
        _print_prob(f"[QCQP @ PrimalInfeasible] {prob.name}")
        _dump_task("primal_infeas")
        try:
            M.dispose()
        except Exception:
            pass
        return None

    # === primalObjValue() ===
    try:
        val = float(M.primalObjValue())
    except Exception as e:
        print(f"[MOSEK][ERROR] primalObjValue() failed for {prob.name}: {repr(e)}")
        traceback.print_exc()
        _print_prob(f"[QCQP @ primalObjValue() exception] {prob.name}")
        _dump_task("noobj")
        val = None

    try:
        M.dispose()
    except Exception:
        pass
    return val
