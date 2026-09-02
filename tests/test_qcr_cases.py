#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_qcr.py
------------------------------------------------------------
Sanity checks for QCR (implemented in apply_qcr):
  - Trace(x.T*Q*x)
  - x.T*Q*y (and Trace)
  - x.T*y
  - -x.T*x
"""

from __future__ import annotations

# This file is an executable diagnostic with its own result aggregation.
# It is invoked by run/smoke_tests.sh rather than collected by pytest.
__test__ = False

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import sympy as sp


from autoconvexrelax.core.problem import QCQPProblem
from autoconvexrelax.core.relaxation import RelaxationEngine


@dataclass
class TestResult:
    name: str
    ok: bool
    details: str = ""


def _apply_qcr_on_objective(p, term) -> str:
    eng = RelaxationEngine()
    before = p.obj_expr
    eng.apply_action(p, "Objective", term, "qcr")
    after = p.obj_expr
    return sp.srepr(before), sp.srepr(after)


def _basic_check(name: str, before_repr: str, after_repr: str) -> TestResult:
    if before_repr == after_repr:
        return TestResult(name, False, "Objective did not change after QCR.")
    return TestResult(name, True, "OK")


def test_trace_xTQx() -> TestResult:
    try:
        p = QCQPProblem("T_trace_xTQx")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-1, ub=1)
        Q = sp.Matrix([[1, 0], [0, -1]])  # indefinite
        term = sp.Trace(x.T * Q * x)
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("trace_xTQx", b, a)
    except Exception:
        return TestResult("trace_xTQx", False, traceback.format_exc())


def test_xTQx_no_trace() -> TestResult:
    try:
        p = QCQPProblem("T_xTQx")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-2, ub=2)
        Q = sp.Matrix([[2, 1], [1, -3]])  # indefinite
        term = x.T * Q * x
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("xTQx", b, a)
    except Exception:
        return TestResult("xTQx", False, traceback.format_exc())


def test_xTQy() -> TestResult:
    try:
        p = QCQPProblem("T_xTQy")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-1, ub=1)
        y = p.add_vector_variable("y", dim=2, vtype="continuous", lb=-2, ub=2)
        Q = sp.Matrix([[1, 2], [3, 4]])
        term = x.T * Q * y
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("xTQy", b, a)
    except Exception:
        return TestResult("xTQy", False, traceback.format_exc())


def test_trace_xTQy() -> TestResult:
    try:
        p = QCQPProblem("T_trace_xTQy")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-1, ub=1)
        y = p.add_vector_variable("y", dim=2, vtype="continuous", lb=-1, ub=1)
        Q = sp.Matrix([[0, 1], [1, 0]])
        term = sp.Trace(x.T * Q * y)
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("trace_xTQy", b, a)
    except Exception:
        return TestResult("trace_xTQy", False, traceback.format_exc())


def test_xTy() -> TestResult:
    try:
        p = QCQPProblem("T_xTy")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-1, ub=1)
        y = p.add_vector_variable("y", dim=2, vtype="continuous", lb=-1, ub=1)
        term = x.T * y
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("xTy", b, a)
    except Exception:
        return TestResult("xTy", False, traceback.format_exc())


def test_neg_xTx() -> TestResult:
    try:
        p = QCQPProblem("T_neg_xTx")
        x = p.add_vector_variable("x", dim=2, vtype="continuous", lb=-1, ub=1)
        term = -sp.Trace(x.T * x)
        p.set_objective(term, "min")
        b, a = _apply_qcr_on_objective(p, term)
        return _basic_check("neg_xTx", b, a)
    except Exception:
        return TestResult("neg_xTx", False, traceback.format_exc())


TESTS: Dict[str, Callable[[], TestResult]] = {
    "trace_xTQx": test_trace_xTQx,
    "xTQx": test_xTQx_no_trace,
    "xTQy": test_xTQy,
    "trace_xTQy": test_trace_xTQy,
    "xTy": test_xTy,
    "neg_xTx": test_neg_xTx,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="comma-separated test names")
    args = parser.parse_args()

    names = list(TESTS.keys())
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]

    results: List[TestResult] = []
    for n in names:
        if n not in TESTS:
            results.append(TestResult(n, False, "Unknown test name"))
            continue
        res = TESTS[n]()
        results.append(res)

    ok_all = True
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}")
        if r.details and not r.ok:
            print(r.details)
        ok_all = ok_all and r.ok

    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
