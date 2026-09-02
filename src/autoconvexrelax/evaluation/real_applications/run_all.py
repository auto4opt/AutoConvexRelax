import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from autoconvexrelax.paths import OUTPUT_ROOT, PROJECT_ROOT


def _run(cmd):
    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = source_root if not current else source_root + os.pathsep + current
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--exact_time_limit", type=float, default=60.0)
    parser.add_argument("--relax_time_limit", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(OUTPUT_ROOT / "real_applications"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    exact_gurobi_out = out_dir / "exact_gurobi.json"
    exact_scip_out = out_dir / "exact_scip.json"
    rl_out_dir = out_dir / "rl"
    rl_summary_out = rl_out_dir / "rl_relaxation_summary.json"
    compare_out = out_dir / "compare_with_baselines.json"

    steps = [
        [
            py,
            "-m",
            "autoconvexrelax.evaluation.real_applications.build_dataset",
        ],
        [
            py,
            "-m",
            "autoconvexrelax.evaluation.real_applications.exact_solvers",
            "--solver",
            "gurobi",
            "--problem",
            "all",
            "--time_limit",
            str(args.exact_time_limit),
            "--out",
            str(exact_gurobi_out),
        ],
        [
            py,
            "-m",
            "autoconvexrelax.evaluation.real_applications.exact_solvers",
            "--solver",
            "scip",
            "--problem",
            "all",
            "--time_limit",
            str(args.exact_time_limit),
            "--out",
            str(exact_scip_out),
        ],
        [
            py,
            "-m",
            "autoconvexrelax.evaluation.real_applications.relax_with_policy",
            "--ckpt",
            args.ckpt,
            "--device",
            args.device,
            "--time_limit",
            str(args.relax_time_limit),
            "--out_dir",
            str(rl_out_dir),
        ],
        [
            py,
            "-m",
            "autoconvexrelax.evaluation.real_applications.compare_with_baselines",
            "--ckpt",
            args.ckpt,
            "--device",
            args.device,
            "--time_limit",
            str(args.relax_time_limit),
            "--exact_gurobi",
            str(exact_gurobi_out),
            "--exact_scip",
            str(exact_scip_out),
            "--rl_summary",
            str(rl_summary_out),
            "--out",
            str(compare_out),
        ],
    ]

    summary = {
        "commands": [],
        "returncodes": [],
        "outputs": {
            "dataset": str(OUTPUT_ROOT / "data" / "real_applications.pkl"),
            "exact_gurobi": str(exact_gurobi_out),
            "exact_scip": str(exact_scip_out),
            "rl_summary": str(rl_summary_out),
            "compare_with_baselines": str(compare_out),
        },
    }
    for cmd in steps:
        proc = _run(cmd)
        summary["commands"].append(cmd)
        summary["returncodes"].append(proc.returncode)

    summary_path = out_dir / "run_all_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote orchestration summary to {summary_path}")


if __name__ == "__main__":
    main()
