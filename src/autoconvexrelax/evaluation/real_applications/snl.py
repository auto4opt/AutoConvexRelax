import math
import numpy as np
import sympy as sp

from autoconvexrelax.core.problem import QCQPProblem


def _sqdist_to_anchor(x_i, anchor_vec: sp.Matrix) -> sp.Expr:
    diff = x_i - anchor_vec
    return sp.Trace(diff.T * diff)


def _sqdist_between_sensors(x_i, x_j) -> sp.Expr:
    diff = x_i - x_j
    return sp.Trace(diff.T * diff)


def build_snl_least_squares_problem(
    name: str = "real_snl_bounded_noise",
    noise_std: float = 0.03,
    seed: int = 11,
    variable_bound: float = 3.0,
    distance_margin_scale: float = 2.0,
) -> QCQPProblem:
    """
    Anchored sensor network localization as a QCQP with bounded-noise intervals:

        min   sum_i ||x_i||^2
        s.t.  d_ij^2 - delta <= ||x_i - x_j||^2 <= d_ij^2 + delta
              d_ik^2 - delta <= ||x_i - a_k||^2 <= d_ik^2 + delta

    This avoids free residual variables, which makes the resulting SDR/RLT
    substantially tighter for the current relaxation engine.
    """
    rng = np.random.default_rng(seed)

    anchors = [
        np.array([[0.0], [0.0]]),
        np.array([[2.0], [0.0]]),
        np.array([[0.0], [2.0]]),
    ]
    true_sensors = [
        np.array([[0.8], [0.7]]),
        np.array([[1.4], [1.2]]),
        np.array([[0.6], [1.5]]),
    ]

    sensor_sensor_edges = [(0, 1), (1, 2), (0, 2)]
    # Use full sensor-anchor observations to reduce geometric ambiguity and tighten the relaxation.
    sensor_anchor_edges = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]

    prob = QCQPProblem(name, sense="min")
    x = []
    objective_terms = []
    observed_ss = []
    observed_sa = []
    coord_boxes = []

    for i in range(len(true_sensors)):
        lb_i = [-float(variable_bound)] * 2
        ub_i = [float(variable_bound)] * 2
        for sensor_idx, anchor_idx in sensor_anchor_edges:
            if sensor_idx != i:
                continue
            anchor = anchors[anchor_idx].reshape(2)
            true_d2 = float(np.sum((true_sensors[i] - anchors[anchor_idx]) ** 2))
            noisy_d2 = true_d2 + float(rng.normal(0.0, noise_std))
            radius = math.sqrt(max(noisy_d2 + distance_margin_scale * noise_std, 1e-9))
            for d in range(2):
                lb_i[d] = max(lb_i[d], float(anchor[d] - radius))
                ub_i[d] = min(ub_i[d], float(anchor[d] + radius))
        coord_boxes.append((lb_i[:], ub_i[:]))
        xi = prob.add_vector_variable(f"x{i}", 2, lb=lb_i, ub=ub_i)
        x.append(xi)
        objective_terms.append(sp.Trace(xi.T * xi))

    for edge_idx, (i, j) in enumerate(sensor_sensor_edges):
        true_d2 = float(np.sum((true_sensors[i] - true_sensors[j]) ** 2))
        noisy_d2 = true_d2 + float(rng.normal(0.0, noise_std))
        observed_ss.append(noisy_d2)
        delta = distance_margin_scale * noise_std
        expr = _sqdist_between_sensors(x[i], x[j])
        prob.add_constraint(expr, ">=", max(noisy_d2 - delta, 1e-6))
        prob.add_constraint(expr, "<=", noisy_d2 + delta)

    for edge_idx, (i, k) in enumerate(sensor_anchor_edges):
        anchor_vec = sp.Matrix(anchors[k].reshape(2, 1))
        true_d2 = float(np.sum((true_sensors[i] - anchors[k]) ** 2))
        noisy_d2 = true_d2 + float(rng.normal(0.0, noise_std))
        observed_sa.append(noisy_d2)
        delta = distance_margin_scale * noise_std
        expr = _sqdist_to_anchor(x[i], anchor_vec)
        prob.add_constraint(expr, ">=", max(noisy_d2 - delta, 1e-6))
        prob.add_constraint(expr, "<=", noisy_d2 + delta)

    prob.set_objective(sp.Add(*objective_terms), "min")
    prob.real_application_data = {
        "application": "snl",
        "noise_std": float(noise_std),
        "variable_bound": float(variable_bound),
        "distance_margin_scale": float(distance_margin_scale),
        "anchors": [a.tolist() for a in anchors],
        "true_sensors": [s.tolist() for s in true_sensors],
        "sensor_sensor_edges": list(sensor_sensor_edges),
        "sensor_anchor_edges": list(sensor_anchor_edges),
        "observed_sensor_sensor_d2": [float(v) for v in observed_ss],
        "observed_sensor_anchor_d2": [float(v) for v in observed_sa],
        "coordinate_boxes": coord_boxes,
        "num_sensors": len(true_sensors),
        "dim": 2,
    }
    prob.map_all_terms()
    return prob


problem = build_snl_least_squares_problem()
