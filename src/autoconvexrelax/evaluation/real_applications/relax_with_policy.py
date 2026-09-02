import argparse
import json
import os
import pickle
from pathlib import Path

from autoconvexrelax.evaluation.real_applications.instances import build_real_application_instances
from autoconvexrelax.paths import OUTPUT_ROOT, PROJECT_ROOT


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(OUTPUT_ROOT / "real_applications" / "rl"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "message": "Policy relaxation pipeline dependencies are unavailable.",
        }
        summary_path = out_dir / "rl_relaxation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[WARN] wrote dependency error summary to {summary_path}")
        return

    problems = build_real_application_instances()

    model = build_model(args.ckpt, args.device, problems[0])

    summary = {}
    for prob in problems:
        relaxed = apply_policy_until_convex(
            prob,
            model,
            max_steps=args.max_steps,
            device=args.device,
        )
        relaxed = canonicalize_problem(relaxed)

        lb = None
        solve_error = None
        try:
            lb = solve_convex_relax_mosek(relaxed, time_limit=args.time_limit, verbose=False)
        except Exception as e:
            solve_error = repr(e)

        relaxed_pkl = out_dir / f"{prob.name}_relaxed.pkl"
        with open(relaxed_pkl, "wb") as f:
            pickle.dump(relaxed, f)

        summary[prob.name] = {
            "original": _stats(prob),
            "relaxed": _stats(relaxed),
            "relaxed_problem_pickle": str(relaxed_pkl),
            "convex_relaxation_value": lb,
            "solve_error": solve_error,
        }

    summary_path = out_dir / "rl_relaxation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote RL relaxation summary to {summary_path}")


if __name__ == "__main__":
    main()
