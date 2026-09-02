# -*- coding: utf-8 -*-
"""
Verify positive reward for global actions (bound_tightening / global_cut_generation).

Usage (preferred, inside your repo where QCQP package exists):
    python verify_global_action_reward.py

Usage (fallback, if you just have these three files in the same folder):
    relaxation_engine.py
    problem_structure.py
    new_reward.py
    python verify_global_action_reward.py

What this script does:
1) Builds a QCQP instance where bound_tightening should *strongly* shrink bounds (and thus should get positive reward).
2) Builds a QCQP instance where global_cut_generation should add RLT cuts / auxiliary variables (and thus should get positive reward).
3) Runs the corresponding action via RelaxationEngine.apply_action and prints:
   - last_rewrite
   - bounds before/after
   - #constraints before/after
   - reward and its main diagnostic scores (bounds shrink, constraints added, r1/r2)
"""

import copy
import contextlib
import io
import sympy as sp

from autoconvexrelax import reward as rw
from autoconvexrelax.core.problem import QCQPProblem
from autoconvexrelax.core.relaxation import RelaxationEngine


def build_problem_for_bound_tightening():
    """
    Single-variable problem with two affine constraints that *should* tighten both lb and ub dramatically:
        x in [-10, 10]
        x >= 0
        x <= 0.1
    Expected: after bound_tightening, x bounds become [0, 0.1] (or tighter).
    """
    p = QCQPProblem("bt_positive_reward_test")
    x = p.add_variable("x", lb=-10, ub=10, vtype="continuous")
    p.add_constraint(x, ">=", 0.0)
    p.add_constraint(x, "<=", 0.1)
    p.set_objective(x, sense="min")
    return p


def build_problem_for_global_cut_generation():
    """
    Affine inequality + another bounded variable NOT appearing in that inequality.
    This matches _add_rlt_from_affine_constraints():
        x,y,z in [0,1]
        x + y <= 1
        z not in constraint => will generate RLT cuts involving x*z, y*z and add McCormick envelopes.
    Expected: after global_cut_generation, constraints count increases.
    """
    p = QCQPProblem("gc_positive_reward_test")
    x = p.add_variable("x", lb=0, ub=1, vtype="continuous")
    y = p.add_variable("y", lb=0, ub=1, vtype="continuous")
    z = p.add_variable("z", lb=0, ub=1, vtype="continuous")
    p.add_constraint(x + y, "<=", 1.0)
    p.set_objective(x + y + z, sense="min")
    return p


def run_and_report(action_id: int, action_name: str, build_problem_fn):
    engine = RelaxationEngine()
    problem = build_problem_fn()

    before = copy.deepcopy(problem)
    engine.apply_action(problem, location="GLOBAL", sub_expr=None, action_type=action_name)
    after = problem

    # Diagnostics
    r1 = rw.convexity_progress(before, after, engine.last_rewrite)
    r2 = rw.structural_unlock_score(before, after) if r1 < 0 else 0.0
    s_bounds = rw._bounds_shrink_score(before, after)
    s_cons = rw._constraint_added_score(before, after)

    # get_reward has internal prints; capture them to keep output clean
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reward = rw.get_reward(before, after, action_id=action_id, last_rewrite=engine.last_rewrite)

    print("=" * 80)
    print(f"Action: {action_name} (action_id={action_id})")
    print(f"last_rewrite: {engine.last_rewrite}")

    # Bounds summary (print only scalar vars for simplicity)
    for name, v in getattr(before, "variables", {}).items():
        v2 = getattr(after, "variables", {}).get(name, None)
        if v2 is None:
            continue
        print(f"  var {name}: [{v.lb}, {v.ub}]  ->  [{v2.lb}, {v2.ub}]")

    print(f"  #constraints: {len(getattr(before,'constraints',[]))} -> {len(getattr(after,'constraints',[]))}")
    print(f"  scores: r1={r1:.3f}, r2={r2:.3f}, bounds_shrink={s_bounds:.3f}, constraints_added={s_cons:.3f}")
    print(f"  reward = {reward:.6f}")
    debug_out = buf.getvalue().strip()
    if debug_out:
        print("  (internal get_reward debug)")
        print("  " + debug_out.replace("\n", "\n  "))

    # Simple assertion-style check
    if reward > 0:
        print("  PASS: reward is positive.")
    else:
        print("  FAIL: reward is NOT positive. (This indicates either a no-op or reward coefficients too small.)")


def main():
    # 1) bound_tightening: expect positive reward due to strong bounds shrink
    run_and_report(action_id=5, action_name="bound_tightening", build_problem_fn=build_problem_for_bound_tightening)

    # 2) global_cut_generation: expect positive reward due to added RLT cuts (new constraints)
    run_and_report(action_id=6, action_name="global_cut_generation", build_problem_fn=build_problem_for_global_cut_generation)


if __name__ == "__main__":
    main()
