from __future__ import annotations

import numpy as np
import sympy as sp

from autoconvexrelax.core.problem import QCQPProblem


def _trace_scalar(expr):
    return sp.Trace(expr)


def _build_feasible_beamformer(channels, sinr_targets):
    """
    Build a valid common beamformer by solving the stronger linear system
        h_k^T w = sqrt(gamma_k)
    in minimum-norm form.

    This gives a feasible point for the original quadratic constraints
        (h_k^T w)^2 >= gamma_k
    whenever the linear system is solvable.
    """
    H = np.hstack(channels)              # n x K
    rhs = np.sqrt(np.asarray(sinr_targets, dtype=float))  # K
    # Solve H^T w = rhs with minimum norm.
    w_feas, *_ = np.linalg.lstsq(H.T, rhs, rcond=None)
    return w_feas.reshape(-1, 1)


def build_multicast_beamforming_problem(
    name: str = "real_multicast_beamforming",
    num_antennas: int = 4,
    num_users: int = 3,
    sinr_targets: list[float] | None = None,
    seed: int = 7,
    variable_bound: float = 2.0,
) -> QCQPProblem:
    """
    Real-valued max-min multicast beamforming QCQP:

        max_{w,t} t
        s.t.      |h_k^T w|^2 >= t,     k = 1, ..., K
                  ||w||_2^2 <= P

    This keeps the same application structure but is much friendlier to the
    current relaxation engine than power minimization with reverse-convex
    quadratic constraints.
    """
    if sinr_targets is None:
        sinr_targets = [1.0] * num_users
    if len(sinr_targets) != num_users:
        raise ValueError("sinr_targets length must equal num_users.")

    rng = np.random.default_rng(seed)
    channels = [rng.standard_normal((num_antennas, 1)) for _ in range(num_users)]
    feasible_w = _build_feasible_beamformer(channels, sinr_targets)
    feasible_power = float(feasible_w.T @ feasible_w)
    power_upper_bound = 1.05 * feasible_power
    variable_bound = min(float(variable_bound), float(np.sqrt(power_upper_bound)))

    prob = QCQPProblem(name, sense="max")
    w = prob.add_vector_variable("w", num_antennas, lb=-variable_bound, ub=variable_bound)
    t = prob.add_variable("t", lb=0.0, ub=power_upper_bound)

    identity = sp.eye(num_antennas)
    prob.set_objective(t, "max")
    prob.add_constraint(_trace_scalar(w.T * identity * w), "<=", power_upper_bound)

    for user_idx, (h_np, gamma_k) in enumerate(zip(channels, sinr_targets), start=1):
        h_mat = sp.Matrix(h_np)
        h_outer = h_mat * h_mat.T
        lhs = _trace_scalar(w.T * h_outer * w)
        prob.add_constraint(lhs - t, ">=", 0.0)

    prob.real_application_data = {
        "application": "beamforming",
        "num_antennas": int(num_antennas),
        "num_users": int(num_users),
        "sinr_targets": [float(v) for v in sinr_targets],
        "variable_bound": float(variable_bound),
        "power_upper_bound": float(power_upper_bound),
        "feasible_w": feasible_w.tolist(),
        "feasible_margin": float(min((float(h.T @ feasible_w) ** 2) for h in channels)),
        "channels": [h.tolist() for h in channels],
    }
    prob.map_all_terms()
    return prob


problem = build_multicast_beamforming_problem()
