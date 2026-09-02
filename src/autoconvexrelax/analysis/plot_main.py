#!/usr/bin/env python3
"""Generate main-experiment figures for the paper.

This script uses only the main multi-seed evaluation logs and intentionally
excludes checkpoint-sweep diagnostics. Output filenames use descriptive
title-style stems because the figures themselves do not contain titles.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoconvexrelax.paths import OUTPUT_ROOT

EVAL_ROOT = OUTPUT_ROOT / "logs" / "compare_all_basic_baselines_new"
OUT_DIR = OUTPUT_ROOT / "figures" / "paper_figures"
SEEDS = [42, 52, 62, 72, 82, 102, 112]

COLORS = {
    "orange": "#D55E00",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "grey": "#7A7A7A",
    "light_grey": "#D9D9D9",
    "dark": "#222222",
}

FAMILY_COLORS = {
    "Mixed-structure QCQPs": "#4E79A7",
    "Dense quadratic blocks": "#009E73",
    "Diagonal/separable quadratics": "#E69F00",
    "Bilinear continuous couplings": "#56B4E9",
    "Binary-coupled quadratics": "#CC79A7",
    "Nonconvex quadratic constraints": "#D55E00",
    "Concave quadratic objectives": "#7A7A7A",
    "Other": "#BDBDBD",
}

FAMILY_SHORT = {
    "Mixed-structure QCQPs": "Mixed",
    "Dense quadratic blocks": "Dense quadratic",
    "Diagonal/separable quadratics": "Diagonal/separable",
    "Bilinear continuous couplings": "Bilinear",
    "Binary-coupled quadratics": "Binary-coupled",
    "Nonconvex quadratic constraints": "Nonconvex constraints",
    "Concave quadratic objectives": "Concave objectives",
    "Other": "Other",
}


def structural_family(name: str) -> str:
    if name.startswith("H_MIXED_"):
        return "Mixed-structure QCQPs"
    if name.startswith(("E_DIR_MC_P1", "E_DIR_MC_P2", "E_EXP_P4_PURE")):
        return "Diagonal/separable quadratics"
    if name.startswith(("E_SDP_P1", "E_SDP_P3", "E_EXP_P1")):
        return "Dense quadratic blocks"
    if name.startswith(("E_DIR_MC_P3", "E_DIR_MC_P4", "E_DIR_MC_P6", "E_EXP_P2")):
        return "Bilinear continuous couplings"
    if name.startswith("E_EXP_P3"):
        return "Binary-coupled quadratics"
    if name.startswith(("E_DIR_MC_P5", "E_DIR_MC_P7", "E_SDP_P2", "E_SDP_P5_NEG_CONS")):
        return "Nonconvex quadratic constraints"
    if name.startswith("E_SDP_P4_NEG_OBJ"):
        return "Concave quadratic objectives"
    return "Other"


def safe_float(value) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values) -> np.ndarray:
    return np.array([x for x in values if np.isfinite(x)], dtype=float)


def mean(values) -> float:
    xs = finite(values)
    return float(xs.mean()) if xs.size else math.nan


def sem95(values) -> float:
    xs = finite(values)
    if xs.size <= 1:
        return 0.0
    return float(1.96 * xs.std(ddof=1) / math.sqrt(xs.size))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.045,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def load_rows() -> list[dict]:
    rows = []
    for seed in SEEDS:
        path = EVAL_ROOT / f"seed_{seed}" / "eval_compare_all.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            item = dict(row)
            item["seed"] = seed
            item["family"] = structural_family(item["name"])
            item["instance_key"] = item.get("dataset_key") or item["name"]
            rows.append(item)
    return rows


def aggregate_by_instance(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["instance_key"]].append(row)

    out = []
    for key, rs in groups.items():
        first = rs[0]
        out.append(
            {
                "instance_key": key,
                "name": first["name"],
                "family": first["family"],
                "gurobi_pct": mean([safe_float(r.get("lb_improve_pct")) for r in rs]),
                "scip_pct": mean([safe_float(r.get("lb_improve_scip_pct")) for r in rs]),
                "gurobi_diff": mean([safe_float(r.get("rl_lb")) - safe_float(r.get("gurobi_root_bound")) for r in rs]),
                "scip_diff": mean([safe_float(r.get("rl_lb")) - safe_float(r.get("scip_root_bound")) for r in rs]),
                "rl_added_nnz": mean([safe_float(r.get("rl_added_nnz")) for r in rs]),
                "rl_pipeline_time_sec": mean([safe_float(r.get("rl_pipeline_time_sec")) for r in rs]),
            }
        )
    return out


def plot_sorted_improvement_curve(instances: list[dict]) -> None:
    gurobi = np.sort(finite([r["gurobi_pct"] for r in instances]))
    scip = np.sort(finite([r["scip_pct"] for r in instances]))
    x_g = np.linspace(0, 100, gurobi.size)
    x_s = np.linspace(0, 100, scip.size)

    fig, ax = plt.subplots(figsize=(5.05, 3.2))
    ax.plot(x_g, gurobi, color=COLORS["orange"], linewidth=1.9, label="vs Gurobi root")
    ax.plot(x_s, scip, color=COLORS["blue"], linewidth=1.9, label="vs SCIP root")
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Instance percentile, sorted by improvement")
    ax.set_ylabel("Root-bound improvement (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-90, 130)
    ax.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "ranked_root_bound_improvement")


def plot_outcome_rates(instances: list[dict]) -> None:
    eps = 1e-6
    refs = [
        ("Gurobi root", finite([r["gurobi_diff"] for r in instances])),
        ("SCIP root", finite([r["scip_diff"] for r in instances])),
    ]
    stacks = []
    for _, diffs in refs:
        total = diffs.size
        stacks.append(
            [
                100.0 * np.sum(diffs > eps) / total,
                100.0 * np.sum(np.abs(diffs) <= eps) / total,
                100.0 * np.sum(diffs < -eps) / total,
            ]
        )
    stacks_arr = np.array(stacks)

    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    y = np.arange(len(refs))
    left = np.zeros(len(refs))
    labels = ["Better", "Tie", "Worse"]
    colors = [COLORS["orange"], COLORS["light_grey"], COLORS["blue"]]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        vals = stacks_arr[:, idx]
        ax.barh(y, vals, left=left, height=0.52, color=color, edgecolor="white", linewidth=0.8, label=label)
        for yi, val, li in zip(y, vals, left):
            if val >= 7:
                text_color = "white" if label != "Tie" else COLORS["dark"]
                ax.text(li + val / 2, yi, f"{val:.0f}%", ha="center", va="center", color=text_color, fontsize=8.2)
        left += vals

    ax.set_yticks(y, [name for name, _ in refs])
    ax.set_xlabel("Held-out instances (%)")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), handlelength=1.2)
    fig.tight_layout()
    save_figure(fig, "pairwise_root_bound_outcomes")


def plot_family_effect(rows: list[dict]) -> None:
    seed_family = defaultdict(list)
    family_instances = defaultdict(set)
    for row in rows:
        seed_family[(row["seed"], row["family"])].append(safe_float(row.get("lb_improve_pct")))
        family_instances[row["family"]].add(row["instance_key"])

    families = sorted(
        {fam for _, fam in seed_family},
        key=lambda fam: mean([mean(seed_family[(seed, fam)]) for seed in SEEDS]),
    )
    means = []
    cis = []
    labels = []
    for fam in families:
        seed_means = [mean(seed_family[(seed, fam)]) for seed in SEEDS if (seed, fam) in seed_family]
        means.append(mean(seed_means))
        cis.append(sem95(seed_means))
        labels.append(f"{FAMILY_SHORT[fam]} (n={len(family_instances[fam])})")

    y = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(5.35, 3.65))
    for yi, fam, m, ci in zip(y, families, means, cis):
        ax.errorbar(
            m,
            yi,
            xerr=ci,
            fmt="o",
            markersize=5.3,
            color=FAMILY_COLORS[fam],
            ecolor=FAMILY_COLORS[fam],
            elinewidth=1.2,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Root-bound improvement vs Gurobi root (%)")
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "effect_size_varies_by_problem_structure")


def plot_cost_quality(instances: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.45))
    families = [fam for fam in FAMILY_COLORS if any(r["family"] == fam for r in instances)]
    for fam in families:
        pairs = [
            (r["rl_added_nnz"], r["gurobi_pct"])
            for r in instances
            if r["family"] == fam and np.isfinite(r["rl_added_nnz"]) and np.isfinite(r["gurobi_pct"])
        ]
        if not pairs:
            continue
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        ax.scatter(
            x + 1.0,
            y,
            s=20,
            alpha=0.76,
            color=FAMILY_COLORS[fam],
            edgecolor="white",
            linewidth=0.4,
            label=FAMILY_SHORT[fam],
        )

    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Added nonzeros (log scale)")
    ax.set_ylabel("Root-bound improvement (%)")
    ax.grid(True, which="major", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0, labelspacing=0.62)
    fig.subplots_adjust(left=0.17, right=0.69, top=0.96, bottom=0.18)
    save_figure(fig, "cost_performance_trade_off")


def main() -> None:
    configure_style()
    rows = load_rows()
    instances = aggregate_by_instance(rows)
    plot_sorted_improvement_curve(instances)
    plot_outcome_rates(instances)
    plot_family_effect(rows)
    plot_cost_quality(instances)
    print(f"Wrote main-experiment figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
