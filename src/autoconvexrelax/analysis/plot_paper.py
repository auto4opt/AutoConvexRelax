#!/usr/bin/env python3
"""Generate publication-style figures from evaluation logs.

Outputs:
  outputs/figures/paper_figures/rl_relaxations_tighten_solver_root_bounds.{pdf,svg,png}
  outputs/figures/paper_figures/policy_action_composition_varies_across_qcqp_structures.{pdf,svg,png}
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoconvexrelax.paths import OUTPUT_ROOT

LOG_ROOT = OUTPUT_ROOT / "logs" / "compare_all_basic_baselines_new"
ACTION_LOG = OUTPUT_ROOT / "results" / "action_logs" / "APA-eval-multiseed_70913.out"
OUT_DIR = OUTPUT_ROOT / "figures" / "paper_figures"
SEEDS = [42, 52, 62, 72, 82, 102, 112]

COLORS = {
    "rl": "#D55E00",
    "root": "#4D4D4D",
    "scip": "#7A7A7A",
    "sdp": "#009E73",
    "mcc": "#56B4E9",
    "integrality": "#CC79A7",
    "qcr": "#E69F00",
    "other": "#BDBDBD",
}

ACTION_LABELS = {
    "sdp_relaxation": "SDP",
    "mccormick_relaxation": "McCormick",
    "relax_integrality": "Integrality",
    "qcr": "QCR",
    "bound_tightening": "Bound tightening",
}

ACTION_COLORS = {
    "sdp_relaxation": COLORS["sdp"],
    "mccormick_relaxation": COLORS["mcc"],
    "relax_integrality": COLORS["integrality"],
    "qcr": COLORS["qcr"],
    "bound_tightening": "#999999",
}


def structural_family(name: str) -> str:
    """Group by structural nonconvexity, not by expression syntax."""
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


def finite(values):
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def mean_ci95(values):
    xs = finite(values)
    if xs.size == 0:
        return math.nan, math.nan
    if xs.size == 1:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(xs.size))


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        path = LOG_ROOT / f"seed_{seed}" / "eval_compare_all.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            item = dict(row)
            item["seed"] = seed
            item["family"] = structural_family(item["name"])
            rows.append(item)
    return rows


def load_effective_action_counts() -> dict[str, Counter]:
    """Parse accepted actions only: an action is counted when [Changed] True."""
    counts: dict[str, Counter] = defaultdict(Counter)
    if not ACTION_LOG.exists():
        return counts

    seed = None
    problem = None
    pending_action = None
    with ACTION_LOG.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = re.search(r"Running compare-all runner for seed=(\d+)", line)
            if match:
                seed = int(match.group(1))
                pending_action = None
                continue

            match = re.search(r"^=== Problem \d+/\d+: (.+) ===", line)
            if match:
                problem = match.group(1).strip()
                pending_action = None
                continue

            match = re.search(r"Action: ([a-zA-Z0-9_]+) @", line)
            if match:
                pending_action = match.group(1)
                continue

            if "[Changed] True" in line and pending_action and seed in SEEDS and problem:
                counts[structural_family(problem)][pending_action] += 1
                pending_action = None
            elif "[Changed] False" in line:
                pending_action = None
    return counts


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def plot_root_bound_improvement(rows: list[dict]) -> None:
    labels = ["Gurobi root", "SCIP root"]
    keys = ["lb_improve_pct", "lb_improve_scip_pct"]
    colors = [COLORS["root"], COLORS["scip"]]

    seed_values = []
    for key in keys:
        vals = []
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            vals.append(float(np.nanmean([safe_float(r[key]) for r in seed_rows])))
        seed_values.append(vals)

    means = []
    errors = []
    for vals in seed_values:
        mu, err = mean_ci95(vals)
        means.append(mu)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, width=0.58, color=colors, edgecolor="black", linewidth=0.6, capsize=3)

    # Seed-level points make the uncertainty definition explicit without clutter.
    offsets = np.linspace(-0.16, 0.16, len(SEEDS))
    for i, vals in enumerate(seed_values):
        ax.scatter(
            np.full(len(vals), x[i]) + offsets,
            vals,
            s=11,
            color="white",
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )

    for i, mu in enumerate(means):
        ax.text(x[i], mu + errors[i] + 1.2, f"{mu:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(30, max(np.array(means) + np.array(errors)) + 5))
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        -0.24,
        "Points denote random seeds; bars show mean +/- 95% CI.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.96, bottom=0.30)
    save_figure(fig, "rl_relaxations_tighten_solver_root_bounds")


def plot_policy_action_composition(counts: dict[str, Counter]) -> None:
    family_order = [
        "Mixed-structure QCQPs",
        "Dense quadratic blocks",
        "Diagonal/separable quadratics",
        "Bilinear continuous couplings",
        "Binary-coupled quadratics",
        "Nonconvex quadratic constraints",
        "Concave quadratic objectives",
    ]
    families = [fam for fam in family_order if sum(counts.get(fam, Counter()).values()) > 0]
    actions = ["sdp_relaxation", "mccormick_relaxation", "relax_integrality", "qcr"]

    shares = np.zeros((len(families), len(actions)))
    totals = []
    for i, fam in enumerate(families):
        total = sum(counts[fam][a] for a in actions)
        totals.append(total)
        if total == 0:
            continue
        for j, action in enumerate(actions):
            shares[i, j] = 100.0 * counts[fam][action] / total

    fig_height = max(3.3, 0.36 * len(families) + 1.55)
    fig, ax = plt.subplots(figsize=(6.45, fig_height))
    y = np.arange(len(families))
    left = np.zeros(len(families))
    for j, action in enumerate(actions):
        ax.barh(
            y,
            shares[:, j],
            left=left,
            height=0.62,
            color=ACTION_COLORS[action],
            edgecolor="white",
            linewidth=0.7,
            label=ACTION_LABELS[action],
        )
        left += shares[:, j]

    label_map = {
        "Mixed-structure QCQPs": "Mixed-structure QCQPs",
        "Dense quadratic blocks": "Dense quadratic blocks",
        "Diagonal/separable quadratics": "Diagonal/separable quadratics",
        "Bilinear continuous couplings": "Bilinear couplings",
        "Binary-coupled quadratics": "Binary-coupled quadratics",
        "Nonconvex quadratic constraints": "Nonconvex quadratic constraints",
        "Concave quadratic objectives": "Concave quadratic objectives",
    }
    ax.set_yticks(y, [label_map[f] for f in families])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of effective relaxation actions (%)")
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)

    # for i, total in enumerate(totals):
    #     ax.text(101.0, y[i], f"n={total}", va="center", ha="left", fontsize=7.2, color="#555555")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.36, right=0.88, top=0.88, bottom=0.22)
    save_figure(fig, "policy_action_composition_varies_across_qcqp_structures")


def main() -> None:
    configure_style()
    rows = load_rows()
    counts = load_effective_action_counts()
    plot_root_bound_improvement(rows)
    plot_policy_action_composition(counts)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
