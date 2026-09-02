import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional


from autoconvexrelax.evaluation.baselines import apply_heuristic_relaxation
from autoconvexrelax.evaluation.real_applications.instances import (
    build_real_application_instance,
    iter_real_application_specs,
)
from autoconvexrelax.paths import OUTPUT_ROOT, PROJECT_ROOT


def _rl_summary_keys():
    return {
        "beamforming": "real_multicast_beamforming",
        "snl": "real_snl_bounded_noise",
    }


def _sense_for_app(app_key: str) -> str:
    if app_key == "beamforming":
        return "max"
    if app_key == "snl":
        return "min"
    return "min"


def _stats(prob):
    return {
        "name": prob.name,
        "problem_type": prob.problem_type,
        "num_scalar_vars": len(getattr(prob, "variables", {})),
        "num_matrix_vars": len(getattr(prob, "matrix_variables", {})),
        "num_constraints": len(getattr(prob, "constraints", [])),
        "num_psd_constraints": len(getattr(prob, "psd_constraints", [])),
        "is_convex": bool(prob.is_convex()),
    }


def _load_json(path: Optional[str]):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_reference(app_key: str, instance_key: str, gurobi_data, scip_data):
    sense = _sense_for_app(app_key)
    cand = []
    for solver_name, data in (("gurobi", gurobi_data), ("scip", scip_data)):
        if not isinstance(data, dict):
            continue
        one = data.get(instance_key, None)
        if one is None and instance_key.endswith("_1"):
            one = data.get(app_key, None)
        if not isinstance(one, dict):
            continue
        obj = one.get("objective", None)
        if isinstance(obj, (int, float)):
            cand.append((solver_name, float(obj)))
    if not cand:
        return {"reference_objective": None, "reference_solvers": []}

    if sense == "max":
        best_val = max(v for _, v in cand)
    else:
        best_val = min(v for _, v in cand)
    used = [s for s, v in cand if v == best_val]
    return {
        "reference_objective": float(best_val),
        "reference_solvers": used,
    }


def _gap_to_reference(ref_obj, lower_bound):
    if not isinstance(ref_obj, (int, float)) or not isinstance(lower_bound, (int, float)):
        return None, None
    gap = float(ref_obj) - float(lower_bound)
    den = max(abs(float(ref_obj)), 1e-12)
    return gap, 100.0 * gap / den


def _build_rl_from_summary(app_key: str, prob_orig, rl_summary_data):
    if not isinstance(rl_summary_data, dict):
        return None
    one = rl_summary_data.get(prob_orig.name)
    if not isinstance(one, dict):
        legacy_key = _rl_summary_keys().get(app_key)
        if legacy_key and prob_orig.name == legacy_key:
            one = rl_summary_data.get(legacy_key)
    if not isinstance(one, dict):
        return None
    return {
        "stats": one.get("relaxed") or _stats(prob_orig),
        "lb": one.get("convex_relaxation_value"),
        "solve_error": one.get("solve_error"),
        "relax_time_sec": None,
        "solve_time_sec": None,
        "pipeline_time_sec": None,
        "source": "rl_summary",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--exact_gurobi", type=str, default="")
    parser.add_argument("--exact_scip", type=str, default="")
    parser.add_argument("--rl_summary", type=str, default="")
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_ROOT / "real_applications" / "compare_with_baselines.json"),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch  # noqa: F401
        from autoconvexrelax.evaluation.runner import (
            apply_policy_until_convex,
            build_model,
            canonicalize_problem,
        )
        from autoconvexrelax.evaluation.solvers.mosek import solve_convex_relax_mosek
    except Exception as e:
        summary = {
            "error": repr(e),
            "message": "Compare-with-baselines pipeline dependencies are unavailable.",
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[WARN] wrote dependency error summary to {out_path}")
        return

    exact_gurobi = _load_json(args.exact_gurobi)
    exact_scip = _load_json(args.exact_scip)
    rl_summary_data = _load_json(args.rl_summary)

    specs = list(iter_real_application_specs())
    model = None
    if not isinstance(rl_summary_data, dict):
        sample_prob = build_real_application_instance(specs[0])
        model = build_model(args.ckpt, args.device, sample_prob)

    results = {}
    for spec in specs:
        app_key = spec.app_key
        ref = _collect_reference(app_key, spec.instance_key, exact_gurobi, exact_scip)
        ref_obj = ref["reference_objective"]

        one = {
            "application": app_key,
            "instance_key": spec.instance_key,
            "instance_index": spec.instance_index,
            "sense": _sense_for_app(app_key),
            "reference": ref,
        }

        prob_rl = build_real_application_instance(spec)
        one["original"] = _stats(prob_rl)
        rl_record = _build_rl_from_summary(app_key, prob_rl, rl_summary_data)
        if rl_record is None:
            t0 = time.perf_counter()
            t_relax0 = time.perf_counter()
            prob_rl_relaxed = apply_policy_until_convex(
                prob_rl,
                model,
                max_steps=args.max_steps,
                device=args.device,
            )
            rl_relax_sec = time.perf_counter() - t_relax0

            prob_rl_relaxed = canonicalize_problem(prob_rl_relaxed)
            t_solve0 = time.perf_counter()
            rl_lb, rl_err = None, None
            try:
                rl_lb = solve_convex_relax_mosek(prob_rl_relaxed, time_limit=args.time_limit, verbose=False)
            except Exception as e:
                rl_err = repr(e)
            rl_solve_sec = time.perf_counter() - t_solve0
            rl_total_sec = time.perf_counter() - t0
            rl_record = {
                "stats": _stats(prob_rl_relaxed),
                "lb": rl_lb,
                "solve_error": rl_err,
                "relax_time_sec": rl_relax_sec,
                "solve_time_sec": rl_solve_sec,
                "pipeline_time_sec": rl_total_sec,
                "source": "fresh_compare_run",
            }

        rl_lb = rl_record["lb"]
        rl_gap_abs, rl_gap_pct = _gap_to_reference(ref_obj, rl_lb)
        one["rl"] = dict(rl_record)
        one["rl"]["gap_to_reference_abs"] = rl_gap_abs
        one["rl"]["gap_to_reference_pct"] = rl_gap_pct

        # Baselines
        baselines = {}
        for mode in ("mccormick", "sdp"):
            t0 = time.perf_counter()
            prob_base = build_real_application_instance(spec)
            t_relax0 = time.perf_counter()
            prob_base = apply_heuristic_relaxation(prob_base, mode=mode)
            base_relax_sec = time.perf_counter() - t_relax0
            prob_base = canonicalize_problem(prob_base)
            t_solve0 = time.perf_counter()
            base_lb, base_err = None, None
            try:
                base_lb = solve_convex_relax_mosek(prob_base, time_limit=args.time_limit, verbose=False)
            except Exception as e:
                base_err = repr(e)
            base_solve_sec = time.perf_counter() - t_solve0
            base_total_sec = time.perf_counter() - t0
            base_gap_abs, base_gap_pct = _gap_to_reference(ref_obj, base_lb)
            baselines[mode] = {
                "stats": _stats(prob_base),
                "lb": base_lb,
                "solve_error": base_err,
                "relax_time_sec": base_relax_sec,
                "solve_time_sec": base_solve_sec,
                "pipeline_time_sec": base_total_sec,
                "gap_to_reference_abs": base_gap_abs,
                "gap_to_reference_pct": base_gap_pct,
            }
        one["baselines"] = baselines

        mcc_lb = baselines["mccormick"]["lb"]
        sdp_lb = baselines["sdp"]["lb"]
        one["deltas"] = {
            "rl_minus_mccormick_lb": None if (rl_lb is None or mcc_lb is None) else (float(rl_lb) - float(mcc_lb)),
            "rl_minus_sdp_lb": None if (rl_lb is None or sdp_lb is None) else (float(rl_lb) - float(sdp_lb)),
            "rl_minus_mccormick_time_sec": None
            if (one["rl"]["pipeline_time_sec"] is None or baselines["mccormick"]["pipeline_time_sec"] is None)
            else (float(one["rl"]["pipeline_time_sec"]) - float(baselines["mccormick"]["pipeline_time_sec"])),
            "rl_minus_sdp_time_sec": None
            if (one["rl"]["pipeline_time_sec"] is None or baselines["sdp"]["pipeline_time_sec"] is None)
            else (float(one["rl"]["pipeline_time_sec"]) - float(baselines["sdp"]["pipeline_time_sec"])),
        }

        results[spec.instance_key] = one
        print(
            f"[{spec.instance_key}] ref={ref_obj}, rl_lb={rl_lb}, "
            f"mcc_lb={mcc_lb}, sdp_lb={sdp_lb}, "
            f"rl_gap_pct={rl_gap_pct}"
        )

    summary = {
        "config": {
            "ckpt": args.ckpt,
            "device": args.device,
            "max_steps": args.max_steps,
            "time_limit": args.time_limit,
            "exact_gurobi": args.exact_gurobi,
            "exact_scip": args.exact_scip,
            "rl_summary": args.rl_summary,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote compare-with-baselines summary to {out_path}")


if __name__ == "__main__":
    main()
