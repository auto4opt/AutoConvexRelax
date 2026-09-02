# -*- coding: utf-8 -*-
# 最小可跑：正常 import，不改你的 problem_structure.py

from autoconvexrelax.core.problem import QCQPProblem
from autoconvexrelax.core.relaxation import RelaxationEngine
import sympy as sp

def rmfrac(engine):
    # 兼容你方法名可能叫 remove_fraction / apply_remove_fraction
    if hasattr(engine, "remove_fraction"):
        return engine.remove_fraction
    if hasattr(engine, "apply_remove_fraction"):
        return engine.apply_remove_fraction
    raise AttributeError("RelaxationEngine 上没找到 remove_fraction / apply_remove_fraction")

def den_is_one(expr):
    num, den = sp.fraction(sp.together(sp.expand(expr)))
    return sp.simplify(den - 1) == 0

def find_link_eq(prob, num, den):
    """找形如  a*den - num = 0 的等式约束；返回 (a_symbol, constraint) 或 (None, None)"""
    tgt = sp.simplify(den)
    for c in prob.constraints:
        if c.sense not in ("=", "is"):
            continue
        eq = sp.simplify((c.expr - c.rhs) if c.sense == "=" else c.expr)
        a = sp.Wild('a')
        m = sp.simplify(eq).match(a*tgt - sp.simplify(num))
        if m and 'a' in m:
            return m['a'], c
    return None, None

# ----------------- 用例 -----------------

def case_A_const_den():
    P = QCQPProblem("A", sense="min")
    x = P.add_variable("x", lb=0, ub=10)
    P.set_objective((x + 1) / 5, "min")
    eng = RelaxationEngine()
    rmfrac(eng)(P, "Objective", P.obj_expr)

    print(P)

def case_B_pos_den_cached():
    # (3x)/y, y∈[1,2] → 引入 λ, 约束 λ*y - 3x = 0；再次调用应复用 λ
    P = QCQPProblem("B", sense="min")
    x = P.add_variable("x", lb=0, ub=10)
    y = P.add_variable("y", lb=1, ub=2)
    P.set_objective((3*x)/y, "min")
    eng = RelaxationEngine()
    f = rmfrac(eng)
    f(P, "Objective", P.obj_expr)
    a, cons = find_link_eq(P, num=3*x, den=y)
    n_eq = sum(1 for c in P.constraints if c.sense in ("=", "is"))
    f(P, "Objective", (3*x)/y)
    n_eq2 = sum(1 for c in P.constraints if c.sense in ("=", "is"))
    print(P)

def case_C_cross_zero_no_bigM():
    # (x+2)/(z-0.5), z∈[-1,2] 跨0；若未开Big-M应跳过，不改动
    P = QCQPProblem("C", sense="min")
    x = P.add_variable("x", lb=-5, ub=5)
    z = P.add_variable("z", lb=-1, ub=2)
    frac = (x + 2) / (z - 0.5)
    P.set_objective(frac, "min")
    eng = RelaxationEngine()
    f = rmfrac(eng)
    try:
        f(P, "Objective", P.obj_expr, allow_bigM_fallback=False)
    except TypeError:
        # 没这个参数就当默认不启 Big-M
        f(P, "Objective", P.obj_expr)
    print(P)

def case_D_cross_zero_with_bigM():
    # (2x+1)/(z-0.5) + Big-M → 引入 t, 约束 t*(z-0.5) - (2x+1) = 0
    P = QCQPProblem("D", sense="min")
    x = P.add_variable("x", lb=-5, ub=5)
    z = P.add_variable("z", lb=-1, ub=2)
    num, den = (2*x + 1), (z - 0.5)
    P.set_objective(num/den, "min")
    eng = RelaxationEngine()
    f = rmfrac(eng)
    try:
        f(P, "Objective", P.obj_expr, allow_bigM_fallback=True, BIG_M=1e3)
    except TypeError:
        f(P, "Objective", P.obj_expr)  # 你的实现可能默认启用
    a, cons = find_link_eq(P, num=num, den=den)
    print(P)

def main():
    print("=== remove_fraction 最小验证 ===")
    case_A_const_den()
    case_B_pos_den_cached()
    case_C_cross_zero_no_bigM()
    case_D_cross_zero_with_bigM()

if __name__ == "__main__":
    main()
