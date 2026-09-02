# solver_interface_scip.py
# -*- coding: utf-8 -*-
"""
Use SCIP (PySCIPOpt) to solve nonconvex QCQP and extract root bound.

This mirrors solver_interface.py (Gurobi) at a high level:
  - Build a SCIP model from QCQPProblem (supports up to quadratic terms).
  - Optionally root-only solve for a "single relaxation" bound.
"""

from __future__ import annotations

import sympy as sp
from sympy.matrices.expressions.matexpr import MatrixElement

try:
    from pyscipopt import Model
except Exception as e:  # pragma: no cover - import guard
    raise ImportError(
        "PySCIPOpt is required for SCIP backend. Please install pyscipopt and SCIP."
    ) from e

from autoconvexrelax.evaluation.expressions import _trace_to_scalar_expr, normalize_expr
from autoconvexrelax.core.problem import (
    QCQPProblem,
    normalize_scalar,
    Variable,
    VectorVariableSymbol,
    MatrixVariableSymbol,
)


def _sympy_to_scip(expr, sym2var):
    """
    Translate a *scalar* sympy expression to PySCIPOpt expression.
    Supports up to quadratic (degree 2).
    """
    expr = normalize_scalar(expr)

    # Trace handling -> scalar
    if isinstance(expr, sp.Trace):
        scalar_expr = _trace_to_scalar_expr(expr)
        if scalar_expr is None:
            raise NotImplementedError(f"Unsupported Trace form in QCQP -> SCIP: {repr(expr)}")
        return _sympy_to_scip(scalar_expr, sym2var)

    expr = normalize_expr(expr)

    if isinstance(expr, sp.Trace):
        raise NotImplementedError(f"Trace still present after normalize_expr: {repr(expr)}")
    if isinstance(expr, sp.MatrixExpr):
        if getattr(expr, "shape", None) == (1, 1):
            return _sympy_to_scip(expr[0, 0], sym2var)
        raise NotImplementedError(f"Non-scalar MatrixExpr after normalize_expr: {repr(expr)}")

    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, sp.Number):
        return float(expr)

    if isinstance(expr, sp.Symbol):
        if expr not in sym2var:
            raise KeyError(f"Scalar symbol {repr(expr)} not found in sym2var")
        return sym2var[expr]

    if isinstance(expr, MatrixElement):
        if expr not in sym2var:
            raise KeyError(f"Matrix element {repr(expr)} not found in sym2var")
        return sym2var[expr]

    if isinstance(expr, sp.Add):
        terms = [_sympy_to_scip(arg, sym2var) for arg in expr.args]
        res = terms[0]
        for t in terms[1:]:
            res = res + t
        return res

    if isinstance(expr, sp.Mul):
        coeff = 1.0
        scip_factors = []
        for arg in expr.args:
            if isinstance(arg, (int, float, sp.Number)):
                coeff *= float(arg)
            else:
                scip_factors.append(_sympy_to_scip(arg, sym2var))

        if not scip_factors:
            return coeff

        res = scip_factors[0]
        for t in scip_factors[1:]:
            res = res * t
        if abs(coeff - 1.0) > 1e-12:
            res = coeff * res
        return res

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if int(exp) == 2:
            v = _sympy_to_scip(base, sym2var)
            return v * v
        raise NotImplementedError(f"Only power 2 supported in QCQP translator, got {expr}")

    raise NotImplementedError(f"Unsupported sympy node in QCQP -> SCIP: {repr(expr)}")


def _collect_scalar_symbols(expr):
    if expr is None:
        return set()
    syms = set()
    for s in expr.free_symbols:
        if isinstance(s, sp.Symbol) and not isinstance(s, sp.MatrixSymbol):
            syms.add(s)
    return syms


def _get_infinity(m: Model):
    try:
        return m.infinity()
    except Exception:
        try:
            return m.getInfinity()
        except Exception:
            return 1e20


def _ensure_aux_scalar_vars(m: Model, obj_expr, constraints, sym2var):
    need = set()
    need |= _collect_scalar_symbols(obj_expr)
    for c in constraints:
        need |= _collect_scalar_symbols(c.expr)
        if c.rhs is not None and hasattr(c.rhs, "free_symbols"):
            need |= _collect_scalar_symbols(c.rhs)

    for s in need:
        if s not in sym2var:
            inf = _get_infinity(m)
            v = m.addVar(
                name=str(s),
                vtype="C",
                lb=-inf,
                ub=inf,
            )
            sym2var[s] = v


def _add_matrix_vars(m: Model, rows, cols, name, lb, ub, vtype):
    mat = []
    for i in range(rows):
        row = []
        for j in range(cols):
            vname = f"{name}[{i},{j}]"
            row.append(m.addVar(name=vname, vtype=vtype, lb=lb, ub=ub))
        mat.append(row)
    return mat


def build_scip_model_from_qcqp(
    prob: QCQPProblem,
    model_name: str | None = None,
    relax_integrality: bool = False,
):
    if model_name is None:
        model_name = prob.name

    m = Model(model_name)

    sym2var = {}

    # scalar variables
    for name, vinfo in getattr(prob, "variables", {}).items():
        assert isinstance(vinfo, Variable)
        inf = _get_infinity(m)
        lb = vinfo.lb if vinfo.lb is not None else -inf
        ub = vinfo.ub if vinfo.ub is not None else inf

        if relax_integrality:
            vtype = "C"
        else:
            if vinfo.vtype == "continuous":
                vtype = "C"
            elif vinfo.vtype == "integer":
                vtype = "I"
            elif vinfo.vtype == "binary":
                vtype = "B"
            else:
                raise ValueError(f"Unknown vtype for scalar {name}: {vinfo.vtype}")

        var = m.addVar(name=name, vtype=vtype, lb=lb, ub=ub)
        sym2var[sp.Symbol(name, real=True)] = var

    # matrix variables
    for name, mv in getattr(prob, "matrix_variables", {}).items():
        if isinstance(mv, VectorVariableSymbol):
            rows, cols = mv.dim, 1
        elif isinstance(mv, MatrixVariableSymbol):
            rows, cols = mv.rows, mv.cols
        else:
            raise ValueError(f"Unknown matrix variable type for {name}: {type(mv)}")

        inf = _get_infinity(m)
        lb = mv.lb if getattr(mv, "lb", None) is not None else -inf
        ub = mv.ub if getattr(mv, "ub", None) is not None else inf

        if relax_integrality:
            vtype = "C"
        else:
            vtype_raw = getattr(mv, "vtype", "continuous")
            if vtype_raw == "continuous":
                vtype = "C"
            elif vtype_raw == "integer":
                vtype = "I"
            elif vtype_raw == "binary":
                vtype = "B"
            else:
                raise ValueError(f"Unknown vtype for matrix variable {name}: {vtype_raw}")

        mat = _add_matrix_vars(m, rows, cols, name=name, lb=lb, ub=ub, vtype=vtype)
        base = mv.symbol
        for i in range(rows):
            for j in range(cols):
                me = MatrixElement(base, i, j)
                sym2var[me] = mat[i][j]

    # objective
    obj_expr = normalize_expr(normalize_scalar(prob.obj_expr))
    _ensure_aux_scalar_vars(m, obj_expr, prob.constraints, sym2var)
    scip_obj = _sympy_to_scip(obj_expr, sym2var)

    sense = str(prob.obj_sense).lower()
    if sense.startswith("min"):
        obj_sense = "minimize"
    else:
        obj_sense = "maximize"

    try:
        m.setObjective(scip_obj, obj_sense)
    except ValueError as e:
        # PySCIPOpt disallows nonlinear objective via setObjective; use recipe if available.
        msg = str(e).lower()
        if "nonlinear objective" in msg:
            try:
                from pyscipopt.recipe.nonlinear import set_nonlinear_objective
                set_nonlinear_objective(m, scip_obj, obj_sense)
            except Exception:
                # Fallback: epigraph / hypograph transformation to linearize objective
                inf = _get_infinity(m)
                t = m.addVar(name="__obj_epi__", vtype="C", lb=-inf, ub=inf)
                m.setObjective(t, obj_sense)
                if obj_sense == "minimize":
                    m.addCons(scip_obj - t <= 0.0, name="obj_epi")
                else:
                    m.addCons(scip_obj - t >= 0.0, name="obj_epi")
        else:
            raise

    # constraints
    for idx, c in enumerate(prob.constraints):
        lhs = normalize_expr(normalize_scalar(c.expr))
        rhs_expr = c.rhs

        if rhs_expr is None:
            rhs = 0.0
        elif isinstance(rhs_expr, (int, float, sp.Number)):
            rhs = float(rhs_expr)
        else:
            rhs_expr = normalize_expr(normalize_scalar(rhs_expr))
            if hasattr(rhs_expr, "free_symbols") and len(rhs_expr.free_symbols) == 0:
                rhs = float(rhs_expr.evalf())
            else:
                rhs = _sympy_to_scip(rhs_expr, sym2var)

        scip_lhs = _sympy_to_scip(lhs, sym2var)
        # If both sides are numeric, avoid creating a boolean constraint.
        # This can happen after normalization when a constraint becomes constant.
        if isinstance(scip_lhs, (int, float, np.floating)) and isinstance(rhs, (int, float, np.floating)):
            val = float(scip_lhs) - float(rhs)
            tol = 1e-12
            if c.sense == "<=":
                if val <= tol:
                    continue
                raise ValueError(f"Infeasible constant constraint: {val} <= 0 (idx={idx})")
            elif c.sense == ">=":
                if val >= -tol:
                    continue
                raise ValueError(f"Infeasible constant constraint: {val} >= 0 (idx={idx})")
            elif c.sense in ("=", "=="):
                if abs(val) <= tol:
                    continue
                raise ValueError(f"Infeasible constant constraint: {val} == 0 (idx={idx})")
            else:
                raise ValueError(f"Unknown constraint sense: {c.sense}")

        cons_expr = scip_lhs - rhs
        cname = f"c_{idx}"

        if c.sense == "<=":
            m.addCons(cons_expr <= 0.0, name=cname)
        elif c.sense == ">=":
            m.addCons(cons_expr >= 0.0, name=cname)
        elif c.sense in ("=", "=="):
            m.addCons(cons_expr == 0.0, name=cname)
        else:
            raise ValueError(f"Unknown constraint sense: {c.sense}")

    # SCIP does not support PSD constraints in this backend
    if hasattr(prob, "psd_constraints") and prob.psd_constraints:
        raise ValueError("SCIP backend does not support PSD constraints.")

    return m


def _safe_set_param(model: Model, name: str, value):
    try:
        model.setParam(name, value)
    except Exception:
        pass


def solve_nonconvex_qcqp_scip(
    prob: QCQPProblem,
    time_limit: float = 60.0,
    root_only: bool = True,
    return_status: bool = False,
):
    m = build_scip_model_from_qcqp(prob, relax_integrality=False)
    try:
        _safe_set_param(m, "limits/time", float(time_limit))
        _safe_set_param(m, "parallel/maxnthreads", 1)
        _safe_set_param(m, "lp/threads", 1)

        if root_only:
            _safe_set_param(m, "limits/nodes", 1)
            _safe_set_param(m, "presolving/maxrounds", 0)
            _safe_set_param(m, "presolving/maxrestarts", 0)
            _safe_set_param(m, "separating/maxrounds", 0)
            _safe_set_param(m, "separating/maxroundsroot", 0)
            _safe_set_param(m, "propagating/maxrounds", 0)
            # best-effort: turn off heuristic/presolve/separating via higher-level setters if available
            try:
                from pyscipopt import SCIP_PARAMSETTING
                m.setHeuristics(SCIP_PARAMSETTING.OFF)
                m.setPresolve(SCIP_PARAMSETTING.OFF)
                m.setSeparating(SCIP_PARAMSETTING.OFF)
            except Exception:
                pass

        m.optimize()

        sol_count = None
        try:
            sol_count = int(m.getNSols())
        except Exception:
            try:
                sol_count = len(m.getSols())
            except Exception:
                sol_count = None

        best_obj = None
        if sol_count is not None and sol_count > 0:
            try:
                best_obj = float(m.getObjVal())
            except Exception:
                best_obj = None

        try:
            best_bound = float(m.getDualbound())
        except Exception:
            best_bound = None

        root_bound = best_bound

        result = {
            "sol_count": sol_count,
            "status": m.getStatus(),
            "best_obj": best_obj,
            "best_bound": best_bound,
            "root_bound": root_bound,
        }

        if return_status:
            return result
        return result["best_obj"], result["best_bound"], result["root_bound"]
    finally:
        try:
            m.freeTransform()
        except Exception:
            pass
        try:
            m.freeProb()
        except Exception:
            pass
