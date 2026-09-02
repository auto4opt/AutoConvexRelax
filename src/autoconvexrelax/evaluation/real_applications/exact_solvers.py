import argparse
import json
import os
from pathlib import Path

import numpy as np

from autoconvexrelax.evaluation.real_applications.instances import (
    build_real_application_instance,
    iter_real_application_specs,
)
from autoconvexrelax.paths import OUTPUT_ROOT, PROJECT_ROOT


def _extract_data(prob):
    data = getattr(prob, "real_application_data", None)
    if not isinstance(data, dict):
        raise ValueError(f"{prob.name} is missing real_application_data metadata.")
    return data


def solve_beamforming_gurobi(prob, time_limit: float = 60.0):
    import gurobipy as gp
    from gurobipy import GRB

    data = _extract_data(prob)
    n = int(data["num_antennas"])
    K = int(data["num_users"])
    channels = [np.array(h, dtype=float) for h in data["channels"]]
    gamma = [float(v) for v in data["sinr_targets"]]
    bound = float(data["variable_bound"])
    power_upper_bound = float(data["power_upper_bound"])

    m = gp.Model(prob.name + "_gurobi")
    m.Params.NonConvex = 2
    m.Params.TimeLimit = float(time_limit)

    w = [m.addVar(lb=-bound, ub=bound, vtype=GRB.CONTINUOUS, name=f"w[{i}]") for i in range(n)]
    t = m.addVar(lb=0.0, ub=power_upper_bound, vtype=GRB.CONTINUOUS, name="t")
    m.update()

    power = gp.QuadExpr()
    for i in range(n):
        power += w[i] * w[i]
    m.setObjective(t, GRB.MAXIMIZE)
    m.addQConstr(power <= power_upper_bound, name="power_cap")

    for k in range(K):
        qexpr = gp.QuadExpr()
        hk = channels[k].reshape(n, 1)
        Ak = hk @ hk.T
        for i in range(n):
            for j in range(n):
                coef = float(Ak[i, j])
                if abs(coef) > 1e-12:
                    qexpr += coef * w[i] * w[j]
        m.addQConstr(qexpr >= t, name=f"margin_{k}")

    m.optimize()

    return {
        "solver": "gurobi",
        "status": int(m.Status),
        "objective": float(m.ObjVal) if m.SolCount > 0 else None,
        "best_bound": float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) else None,
        "solution": {"w": [float(v.X) for v in w], "t": float(t.X)} if m.SolCount > 0 else None,
        "sol_count": int(m.SolCount),
    }


def solve_snl_gurobi(prob, time_limit: float = 60.0):
    import gurobipy as gp
    from gurobipy import GRB

    data = _extract_data(prob)
    num_sensors = int(data["num_sensors"])
    dim = int(data["dim"])
    anchors = [np.array(a, dtype=float).reshape(dim, 1) for a in data["anchors"]]
    sensor_sensor_edges = [tuple(e) for e in data["sensor_sensor_edges"]]
    sensor_anchor_edges = [tuple(e) for e in data["sensor_anchor_edges"]]
    obs_ss = [float(v) for v in data["observed_sensor_sensor_d2"]]
    obs_sa = [float(v) for v in data["observed_sensor_anchor_d2"]]
    coord_boxes = data["coordinate_boxes"]
    noise_std = float(data["noise_std"])
    margin_scale = float(data["distance_margin_scale"])
    delta = margin_scale * noise_std

    m = gp.Model(prob.name + "_gurobi")
    m.Params.NonConvex = 2
    m.Params.TimeLimit = float(time_limit)

    x = [[m.addVar(lb=float(coord_boxes[i][0][d]), ub=float(coord_boxes[i][1][d]), vtype=GRB.CONTINUOUS, name=f"x[{i},{d}]")
          for d in range(dim)] for i in range(num_sensors)]
    m.update()

    obj = gp.QuadExpr()
    for i in range(num_sensors):
        for d in range(dim):
            obj += x[i][d] * x[i][d]
    m.setObjective(obj, GRB.MINIMIZE)

    for e, (i, j) in enumerate(sensor_sensor_edges):
        qexpr = gp.QuadExpr()
        for d in range(dim):
            qexpr += x[i][d] * x[i][d]
            qexpr += -2.0 * x[i][d] * x[j][d]
            qexpr += x[j][d] * x[j][d]
        m.addQConstr(qexpr >= max(float(obs_ss[e]) - delta, 1e-6), name=f"ss_lb_{e}")
        m.addQConstr(qexpr <= float(obs_ss[e]) + delta, name=f"ss_ub_{e}")

    for e, (i, k) in enumerate(sensor_anchor_edges):
        qexpr = gp.QuadExpr()
        anchor = anchors[k].reshape(dim)
        for d in range(dim):
            qexpr += x[i][d] * x[i][d]
            qexpr += -2.0 * float(anchor[d]) * x[i][d]
            qexpr += float(anchor[d] ** 2)
        m.addQConstr(qexpr >= max(float(obs_sa[e]) - delta, 1e-6), name=f"sa_lb_{e}")
        m.addQConstr(qexpr <= float(obs_sa[e]) + delta, name=f"sa_ub_{e}")

    m.optimize()

    sol = None
    if m.SolCount > 0:
        sol = {
            "x": [[float(x[i][d].X) for d in range(dim)] for i in range(num_sensors)],
        }

    return {
        "solver": "gurobi",
        "status": int(m.Status),
        "objective": float(m.ObjVal) if m.SolCount > 0 else None,
        "best_bound": float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) else None,
        "solution": sol,
        "sol_count": int(m.SolCount),
    }


def solve_beamforming_scip(prob, time_limit: float = 60.0):
    from pyscipopt import Model

    data = _extract_data(prob)
    n = int(data["num_antennas"])
    K = int(data["num_users"])
    channels = [np.array(h, dtype=float) for h in data["channels"]]
    gamma = [float(v) for v in data["sinr_targets"]]
    bound = float(data["variable_bound"])
    power_upper_bound = float(data["power_upper_bound"])

    m = Model(prob.name + "_scip")
    m.setParam("limits/time", float(time_limit))

    w = [m.addVar(name=f"w[{i}]", vtype="C", lb=-bound, ub=bound) for i in range(n)]
    t = m.addVar(name="t", vtype="C", lb=0.0, ub=power_upper_bound)
    m.setObjective(t, "maximize")
    m.addCons(sum(w[i] * w[i] for i in range(n)) <= power_upper_bound, name="power_cap")

    for k in range(K):
        hk = channels[k].reshape(n, 1)
        Ak = hk @ hk.T
        expr = 0.0
        for i in range(n):
            for j in range(n):
                coef = float(Ak[i, j])
                if abs(coef) > 1e-12:
                    expr += coef * w[i] * w[j]
        m.addCons(expr >= t, name=f"margin_{k}")

    m.optimize()

    sol = m.getBestSol()
    return {
        "solver": "scip",
        "status": str(m.getStatus()),
        "objective": float(m.getObjVal()) if sol is not None else None,
        "best_bound": float(m.getDualbound()) if hasattr(m, "getDualbound") else None,
        "solution": {"w": [float(m.getSolVal(sol, wi)) for wi in w], "t": float(m.getSolVal(sol, t))} if sol is not None else None,
        "sol_count": int(m.getNSols()) if hasattr(m, "getNSols") else None,
    }


def solve_snl_scip(prob, time_limit: float = 60.0):
    from pyscipopt import Model

    data = _extract_data(prob)
    num_sensors = int(data["num_sensors"])
    dim = int(data["dim"])
    anchors = [np.array(a, dtype=float).reshape(dim, 1) for a in data["anchors"]]
    sensor_sensor_edges = [tuple(e) for e in data["sensor_sensor_edges"]]
    sensor_anchor_edges = [tuple(e) for e in data["sensor_anchor_edges"]]
    obs_ss = [float(v) for v in data["observed_sensor_sensor_d2"]]
    obs_sa = [float(v) for v in data["observed_sensor_anchor_d2"]]
    coord_boxes = data["coordinate_boxes"]
    noise_std = float(data["noise_std"])
    margin_scale = float(data["distance_margin_scale"])
    delta = margin_scale * noise_std

    m = Model(prob.name + "_scip")
    m.setParam("limits/time", float(time_limit))

    x = [[m.addVar(name=f"x[{i},{d}]", vtype="C", lb=float(coord_boxes[i][0][d]), ub=float(coord_boxes[i][1][d]))
          for d in range(dim)] for i in range(num_sensors)]
    t = m.addVar(name="obj_epi", vtype="C", lb=0.0)
    m.setObjective(t, "minimize")
    m.addCons(sum(x[i][d] * x[i][d] for i in range(num_sensors) for d in range(dim)) - t <= 0.0, name="obj_epi_cons")

    for e, (i, j) in enumerate(sensor_sensor_edges):
        expr = 0.0
        for d in range(dim):
            expr += (x[i][d] - x[j][d]) * (x[i][d] - x[j][d])
        m.addCons(expr >= max(float(obs_ss[e]) - delta, 1e-6), name=f"ss_lb_{e}")
        m.addCons(expr <= float(obs_ss[e]) + delta, name=f"ss_ub_{e}")

    for e, (i, k) in enumerate(sensor_anchor_edges):
        expr = 0.0
        anchor = anchors[k].reshape(dim)
        for d in range(dim):
            expr += (x[i][d] - float(anchor[d])) * (x[i][d] - float(anchor[d]))
        m.addCons(expr >= max(float(obs_sa[e]) - delta, 1e-6), name=f"sa_lb_{e}")
        m.addCons(expr <= float(obs_sa[e]) + delta, name=f"sa_ub_{e}")

    m.optimize()

    sol = m.getBestSol()
    return {
        "solver": "scip",
        "status": str(m.getStatus()),
        "objective": float(m.getObjVal()) if sol is not None else None,
        "best_bound": float(m.getDualbound()) if hasattr(m, "getDualbound") else None,
        "solution": {
            "x": [[float(m.getSolVal(sol, x[i][d])) for d in range(dim)] for i in range(num_sensors)],
        } if sol is not None else None,
        "sol_count": int(m.getNSols()) if hasattr(m, "getNSols") else None,
    }


def solve_problem_exact(prob, solver: str, time_limit: float = 60.0):
    app = _extract_data(prob).get("application")
    if solver == "gurobi":
        if app == "beamforming":
            return solve_beamforming_gurobi(prob, time_limit=time_limit)
        if app == "snl":
            return solve_snl_gurobi(prob, time_limit=time_limit)
    if solver == "scip":
        if app == "beamforming":
            return solve_beamforming_scip(prob, time_limit=time_limit)
        if app == "snl":
            return solve_snl_scip(prob, time_limit=time_limit)
    raise ValueError(f"Unsupported solver/application pair: solver={solver}, application={app}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["beamforming", "snl", "all"], default="all")
    parser.add_argument("--solver", choices=["gurobi", "scip"], default="gurobi")
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_ROOT / "real_applications" / "exact_solution_results.json"),
    )
    args = parser.parse_args()

    results = {}
    for spec in iter_real_application_specs(args.problem):
        prob = build_real_application_instance(spec)
        try:
            results[spec.instance_key] = solve_problem_exact(prob, solver=args.solver, time_limit=args.time_limit)
            results[spec.instance_key]["application"] = spec.app_key
            results[spec.instance_key]["instance_index"] = spec.instance_index
            results[spec.instance_key]["problem_name"] = prob.name
        except Exception as e:
            results[spec.instance_key] = {
                "solver": args.solver,
                "application": spec.app_key,
                "instance_index": spec.instance_index,
                "problem_name": prob.name,
                "error": repr(e),
            }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] wrote exact-solver results to {out_path}")


if __name__ == "__main__":
    main()
