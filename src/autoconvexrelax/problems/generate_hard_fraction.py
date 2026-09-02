#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate harder fractional QCQP instances for the paper demo.

These instances keep the denominator simple and safely positive, but make the
numerator more discriminative across McCormick, SDP, heuristic, and learned
relaxations. The output pickle uses the same single-group format consumed by
run/evaluate.py with --no_split.

Server usage:
    python run/prepare_data.py hard-fraction \
      --output vector_fraction_hard_candidates_60_seed42.pkl \
      --num-repeat 12 \
      --seed 42
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import sympy as sp

import autoconvexrelax.core.problem as ps
from autoconvexrelax.problems.finetune_fraction import (
    _add_global_linear_couplings,
    _add_safe_leq,
    _const_vec,
    _rand_indefinite_matrix,
    _safe_rhs_ratio,
    _ub_norm2,
    _ub_sum_norm2,
    _vec,
)


HARD_FRACTION_FAMILIES = (
    "FRH1_dense_indef_quad_over_linear",
    "FRH2_dense_bilinear_over_linear",
    "FRH3_mixed_quad_bilin_over_linear",
    "FRH4_binary_continuous_quad_over_linear",
    "FRH5_dense_bilinear_fraction_constraint",
)


def _name(base: str, repeat_idx: int) -> str:
    return base if repeat_idx == 0 else f"{base}_r{repeat_idx}"


def _fixed_linear_denominator(vec) -> sp.Expr:
    n = int(vec.shape[0])
    e = _const_vec([1] * n)
    return 6.0 + (1.0 / n) * sp.Trace(e.T * vec)


def _fraction_over_fixed_denominator(numerator, denominator) -> sp.Expr:
    """Keep the paper-demo fraction as one fixed-denominator term.

    The current RemoveFraction path is reliable when the variable denominator is
    exposed as a single rational term. Avoid distributing dense numerator terms
    into many monomial/denominator fractions.
    """
    return sp.Mul(numerator, sp.Pow(denominator, -1, evaluate=False), evaluate=False)


def _dense_matrix(rows: int, cols: int, scale: float = 2.0) -> sp.Matrix:
    return sp.Matrix(np.round(np.random.uniform(-scale, scale, size=(rows, cols)), 3))


def _map_and_return(problem: ps.QCQPProblem) -> ps.QCQPProblem:
    problem.map_all_terms()
    return problem


def _frh1_dense_indef_quad(repeat_idx: int) -> ps.QCQPProblem:
    """Dense indefinite quadratic numerator over a positive affine denominator."""
    n = random.randint(5, 7)
    p = ps.QCQPProblem(_name("FRH1_dense_indef_quad_over_linear", repeat_idx))
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    q = sp.Matrix(np.round(_rand_indefinite_matrix(n, 4.0, -4.0), 3))
    den = _fixed_linear_denominator(z)

    p.set_objective(_fraction_over_fixed_denominator(sp.Trace(x.T * q * x), den), "min")
    _add_safe_leq(p, sp.Trace(x.T * x) + sp.Trace(z.T * z), _ub_sum_norm2(p, [x, z]))
    _add_global_linear_couplings(p, [x, z], max_cons=3, p_add=0.9)
    return _map_and_return(p)


def _frh2_dense_bilinear(repeat_idx: int) -> ps.QCQPProblem:
    """Fixed denominator with extra dense nonconvex QCQP terms in the numerator."""
    n = random.randint(5, 7)
    p = ps.QCQPProblem(_name("FRH2_dense_bilinear_over_linear", repeat_idx))
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    y = _vec(p, "y", n, lb=-2, ub=2, vtype="continuous")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    qx = sp.Matrix(np.round(_rand_indefinite_matrix(n, 4.0, -4.0), 3))
    qy = sp.Matrix(np.round(_rand_indefinite_matrix(n, 3.0, -3.0), 3))
    a = _dense_matrix(n, n, scale=1.5)
    den = _fixed_linear_denominator(z)
    numerator = sp.Trace(x.T * qx * x)

    p.set_objective(
        _fraction_over_fixed_denominator(numerator, den)
        + 0.45 * sp.Trace(y.T * qy * y)
        + 0.15 * sp.Trace(x.T * a * y),
        "min",
    )
    _add_safe_leq(p, sp.Trace(x.T * x) + sp.Trace(y.T * y) + sp.Trace(z.T * z), _ub_sum_norm2(p, [x, y, z]))
    _add_global_linear_couplings(p, [x, y, z], max_cons=3, p_add=0.9)
    return _map_and_return(p)


def _frh3_mixed_quad_bilinear(repeat_idx: int) -> ps.QCQPProblem:
    """Fixed denominator with numerator and separate QCQP term of different shape."""
    n = random.randint(5, 7)
    p = ps.QCQPProblem(_name("FRH3_mixed_quad_bilin_over_linear", repeat_idx))
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    y = _vec(p, "y", n, lb=-2, ub=2, vtype="continuous")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    q = sp.Matrix(np.round(_rand_indefinite_matrix(n, 3.5, -3.5), 3))
    r = sp.Matrix(np.round(_rand_indefinite_matrix(n, 2.5, -2.5), 3))
    a = _dense_matrix(n, n, scale=1.5)
    den = _fixed_linear_denominator(z)
    numerator = sp.Trace(x.T * q * x)

    p.set_objective(
        _fraction_over_fixed_denominator(numerator, den)
        + 0.35 * sp.Trace(y.T * r * y)
        + 0.25 * sp.Trace(x.T * a * y),
        "min",
    )
    _add_safe_leq(p, sp.Trace(x.T * x) + sp.Trace(y.T * y) + sp.Trace(z.T * z), _ub_sum_norm2(p, [x, y, z]))
    _add_global_linear_couplings(p, [x, y, z], max_cons=3, p_add=0.9)
    return _map_and_return(p)


def _frh4_binary_continuous_mixture(repeat_idx: int) -> ps.QCQPProblem:
    """Fixed denominator with continuous and binary quadratic numerator terms."""
    n = random.randint(4, 5)
    p = ps.QCQPProblem(_name("FRH4_binary_continuous_quad_over_linear", repeat_idx))
    x = _vec(p, "x", n, lb=-2, ub=2, vtype="continuous")
    b = _vec(p, "b", n, lb=0, ub=1, vtype="binary")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    q = sp.Matrix(np.round(_rand_indefinite_matrix(n, 3.0, -3.0), 3))
    a = _dense_matrix(n, n, scale=1.6)
    den = _fixed_linear_denominator(z)
    numerator = sp.Trace(x.T * q * x)

    p.set_objective(
        _fraction_over_fixed_denominator(numerator, den)
        + 0.2 * sp.Trace(x.T * a * b),
        "min",
    )
    _add_safe_leq(p, sp.Trace(x.T * x) + sp.Trace(b.T * b) + sp.Trace(z.T * z), _ub_sum_norm2(p, [x, b, z]))
    _add_global_linear_couplings(p, [x, z], max_cons=2, p_add=0.7)
    return _map_and_return(p)


def _frh5_fraction_constraint(repeat_idx: int) -> ps.QCQPProblem:
    """Fixed-denominator fraction constraint with a hard QCQP numerator."""
    n = random.randint(5, 7)
    p = ps.QCQPProblem(_name("FRH5_dense_bilinear_fraction_constraint", repeat_idx))
    x = _vec(p, "x", n, lb=-1, ub=1, vtype="continuous")
    y = _vec(p, "y", n, lb=-1, ub=1, vtype="continuous")
    z = _vec(p, "z", n, lb=-2, ub=2, vtype="continuous")
    q = sp.Matrix(np.round(_rand_indefinite_matrix(n, 2.5, -2.5), 3))
    r = sp.Matrix(np.round(_rand_indefinite_matrix(n, 2.0, -2.0), 3))
    a = _dense_matrix(n, n, scale=1.5)
    num = sp.Trace(x.T * q * x)
    den = _fixed_linear_denominator(z)
    rhs = 0.35 * _safe_rhs_ratio(_ub_sum_norm2(p, [x, y]), 3.0)

    p.set_objective(0.6 * sp.Trace(x.T * a * y) + 0.4 * sp.Trace(y.T * r * y), "min")
    p.add_constraint(_fraction_over_fixed_denominator(num, den), "<=", rhs)
    _add_safe_leq(p, sp.Trace(x.T * x), _ub_norm2(p, x))
    _add_safe_leq(p, sp.Trace(y.T * y) + sp.Trace(z.T * z), _ub_sum_norm2(p, [y, z]))
    _add_global_linear_couplings(p, [x, y, z], max_cons=3, p_add=0.9)
    return _map_and_return(p)


FAMILY_BUILDERS = (
    _frh1_dense_indef_quad,
    _frh2_dense_bilinear,
    _frh3_mixed_quad_bilinear,
    _frh4_binary_continuous_mixture,
    _frh5_fraction_constraint,
)


def create_hard_fraction_problems(num_repeat: int = 12, seed: int = 42) -> list[ps.QCQPProblem]:
    """Return a flat list of hard fractional problems."""
    if num_repeat <= 0:
        raise ValueError("num_repeat must be positive")

    random.seed(seed)
    np.random.seed(seed)
    problems: list[ps.QCQPProblem] = []
    for repeat_idx in range(num_repeat):
        for builder in FAMILY_BUILDERS:
            problems.append(builder(repeat_idx))
    return problems


def create_hard_fraction_dataset(num_repeat: int = 12, seed: int = 42) -> list[list[ps.QCQPProblem]]:
    """Return runner-compatible single-group dataset format."""
    return [create_hard_fraction_problems(num_repeat=num_repeat, seed=seed)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("vector_fraction_hard_candidates_60_seed42.pkl"))
    parser.add_argument("--num-repeat", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = create_hard_fraction_dataset(num_repeat=args.num_repeat, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(dataset, f)

    problems = dataset[0]
    print(
        f"[DONE] wrote {args.output} with {len(problems)} hard fractional problems "
        f"({len(HARD_FRACTION_FAMILIES)} families x {args.num_repeat} repeats, seed={args.seed})"
    )
    for family in HARD_FRACTION_FAMILIES:
        count = sum(problem.name == family or problem.name.startswith(f"{family}_r") for problem in problems)
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
