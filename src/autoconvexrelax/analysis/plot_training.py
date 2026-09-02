#!/usr/bin/env python3
"""Plot journal-style training curves across multiple seeds.

Expected directory layout (from sbatch_train_5seeds_array.sh):

    train_runs/
      seed_42/
        stage1_no_lb_1600/
          train_reward_log.csv
          train_loss_log.csv
        stage2_finetune_1200/
          train_reward_log.csv
          train_loss_log.csv

The script aggregates the selected seeds per stage and produces:
1. A 2x2 figure for each stage.
2. A CSV summary (mean/std/sem/ci95 over seeds for each update).
3. A manifest text file showing which runs were found or missing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the server env first, "
        "for example: pip install matplotlib"
    ) from exc


STAGE_METRICS = [
    ("avg_reward", "Average Step Reward"),
    ("episode_return_mean", "Average Episode Return"),
    ("total_loss_mean", "Total Loss"),
]

COMBINED_REWARD_METRICS = [
    ("avg_reward", "Average Step Reward"),
    ("episode_return_mean", "Average Episode Return"),
]

DEFAULT_SEEDS = [42, 52, 62, 72]
DEFAULT_STAGES = ["stage1_no_lb_1600", "stage2_finetune_1200"]
SEED_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#B07AA1"]


@dataclass
class RunLog:
    seed: int
    stage: str
    stage_dir: Path
    reward_csv: Path
    loss_csv: Path
    rows: Dict[int, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("train_runs"),
        help="Root directory that contains seed_xx subdirectories.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to aggregate.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=DEFAULT_STAGES,
        help="Stage directories under each seed_xx directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "training_curves",
        help="Output directory for plots and summaries.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=7,
        help="Centered moving-average window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--spread",
        type=str,
        choices=["std", "sem", "ci95"],
        default="std",
        help="Type of uncertainty band around the mean curve.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pdf", "png", "svg"],
        help="Figure formats to save.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster DPI used for PNG output.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional prefix shown before each stage title.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def as_float(value: str) -> float:
    value = value.strip()
    if value == "":
        return math.nan
    return float(value)


def read_csv_rows(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            if not row:
                continue
            update_raw = row.get("update", "").strip()
            if update_raw == "":
                continue
            update = int(float(update_raw))
            parsed = {}
            for key, value in row.items():
                if key is None:
                    continue
                if key == "update":
                    parsed[key] = float(update)
                    continue
                try:
                    parsed[key] = as_float(value)
                except (TypeError, ValueError):
                    parsed[key] = math.nan
            rows[update] = parsed
    return rows


def merge_logs(reward_rows: Dict[int, Dict[str, float]], loss_rows: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    updates = sorted(set(reward_rows) | set(loss_rows))
    merged: Dict[int, Dict[str, float]] = {}
    for update in updates:
        merged_row: Dict[str, float] = {"update": float(update)}
        if update in reward_rows:
            merged_row.update(reward_rows[update])
        if update in loss_rows:
            merged_row.update(loss_rows[update])
        merged[update] = merged_row
    return merged


def load_run(base_dir: Path, seed: int, stage: str) -> RunLog | None:
    stage_dir = base_dir / f"seed_{seed}" / stage
    reward_csv = stage_dir / "train_reward_log.csv"
    loss_csv = stage_dir / "train_loss_log.csv"
    if not reward_csv.is_file() or not loss_csv.is_file():
        return None
    reward_rows = read_csv_rows(reward_csv)
    loss_rows = read_csv_rows(loss_csv)
    rows = merge_logs(reward_rows, loss_rows)
    return RunLog(
        seed=seed,
        stage=stage,
        stage_dir=stage_dir,
        reward_csv=reward_csv,
        loss_csv=loss_csv,
        rows=rows,
    )


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)
    kernel = np.ones(window, dtype=float)
    denom = np.convolve(valid, kernel, mode="same")
    numer = np.convolve(filled, kernel, mode="same")
    out = np.divide(numer, denom, out=np.full_like(values, np.nan, dtype=float), where=denom > 0)
    return out


def collect_metric_series(run: RunLog, metric: str) -> tuple[np.ndarray, np.ndarray]:
    updates = sorted(run.rows)
    x = np.asarray(updates, dtype=float)
    y = np.asarray([run.rows[u].get(metric, math.nan) for u in updates], dtype=float)
    return x, y


def summarize_runs(runs: List[RunLog], metrics: Iterable[str]) -> Dict[str, np.ndarray]:
    all_updates = sorted({update for run in runs for update in run.rows})
    update_to_idx = {update: idx for idx, update in enumerate(all_updates)}
    summary: Dict[str, np.ndarray] = {"update": np.asarray(all_updates, dtype=float)}

    for metric in metrics:
        mat = np.full((len(runs), len(all_updates)), np.nan, dtype=float)
        for run_idx, run in enumerate(runs):
            for update, row in run.rows.items():
                mat[run_idx, update_to_idx[update]] = row.get(metric, math.nan)

        count = np.sum(np.isfinite(mat), axis=0).astype(float)
        sum_vals = np.nansum(mat, axis=0)
        mean = np.divide(
            sum_vals,
            count,
            out=np.full(len(all_updates), np.nan, dtype=float),
            where=count > 0,
        )
        sq_diff = np.where(np.isfinite(mat), (mat - mean[None, :]) ** 2, np.nan)
        var = np.divide(
            np.nansum(sq_diff, axis=0),
            count,
            out=np.full(len(all_updates), np.nan, dtype=float),
            where=count > 0,
        )
        std = np.sqrt(var)
        sem = np.divide(std, np.sqrt(np.maximum(count, 1.0)))
        ci95 = 1.96 * sem

        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_sem"] = sem
        summary[f"{metric}_ci95"] = ci95
        summary[f"{metric}_count"] = count
    return summary


def stage_display_name(stage: str) -> str:
    mapping = {
        "stage1_no_lb_1600": "Stage 1: Pretraining Without Solver Reward",
        "stage2_finetune_1200": "Stage 2: Finetuning With Solver Reward",
    }
    return mapping.get(stage, stage.replace("_", " "))


def band_label(spread: str) -> str:
    return {
        "std": "Mean +/- SD",
        "sem": "Mean +/- SEM",
        "ci95": "Mean +/- 95% CI",
    }[spread]


def write_summary_csv(path: Path, summary: Dict[str, np.ndarray], metrics: Iterable[str]) -> None:
    fieldnames = ["update"]
    for metric in metrics:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_sem",
                f"{metric}_ci95",
                f"{metric}_count",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(summary["update"])):
            row = {"update": int(summary["update"][idx])}
            for key in fieldnames[1:]:
                val = summary[key][idx]
                if key.endswith("_count"):
                    row[key] = int(val)
                elif np.isfinite(val):
                    row[key] = f"{val:.8g}"
                else:
                    row[key] = ""
            writer.writerow(row)


def write_manifest(path: Path, base_dir: Path, stages: List[str], loaded: Dict[str, List[RunLog]], missing: Dict[str, List[int]]) -> None:
    lines = [f"Base dir: {base_dir.resolve()}", ""]
    for stage in stages:
        lines.append(f"[{stage}]")
        found = loaded.get(stage, [])
        lines.append(f"Found seeds: {[run.seed for run in found]}")
        lines.append(f"Missing seeds: {missing.get(stage, [])}")
        for run in found:
            lines.append(f"  seed {run.seed}: {run.stage_dir}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: plt.Figure, out_base: Path, formats: Iterable[str], dpi: int) -> None:
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        save_kwargs = {}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)


def make_combined_stage_runs(stage_order: List[str], runs_by_stage: Dict[str, List[RunLog]]) -> Dict[int, RunLog]:
    if not stage_order:
        return {}

    seed_sets = []
    for stage in stage_order:
        stage_runs = runs_by_stage.get(stage, [])
        seed_sets.append({run.seed for run in stage_runs})
    common_seeds = set.intersection(*seed_sets) if seed_sets else set()

    combined: Dict[int, RunLog] = {}
    for seed in sorted(common_seeds):
        merged_rows: Dict[int, Dict[str, float]] = {}
        offset = 0
        last_stage_dir = None
        for stage in stage_order:
            run = next(run for run in runs_by_stage[stage] if run.seed == seed)
            last_stage_dir = run.stage_dir
            stage_updates = sorted(run.rows)
            for update in stage_updates:
                new_update = int(offset + update)
                row = dict(run.rows[update])
                row["update"] = float(new_update)
                row["stage_update"] = float(update)
                merged_rows[new_update] = row
            if stage_updates:
                offset += int(max(stage_updates))
        combined[seed] = RunLog(
            seed=seed,
            stage="combined",
            stage_dir=last_stage_dir if last_stage_dir is not None else Path("."),
            reward_csv=Path("."),
            loss_csv=Path("."),
            rows=merged_rows,
        )
    return combined


def compute_stage_boundaries(stage_order: List[str], runs_by_stage: Dict[str, List[RunLog]]) -> List[tuple[float, str]]:
    boundaries: List[tuple[float, str]] = []
    offset = 0.0
    for idx, stage in enumerate(stage_order):
        runs = runs_by_stage.get(stage, [])
        max_updates = [max(run.rows) for run in runs if run.rows]
        if not max_updates:
            continue
        stage_len = float(max(max_updates))
        if idx < len(stage_order) - 1:
            boundaries.append((offset + stage_len, stage))
        offset += stage_len
    return boundaries


def plot_stage(
    stage: str,
    runs: List[RunLog],
    out_dir: Path,
    smooth: int,
    spread: str,
    formats: Iterable[str],
    dpi: int,
    title_prefix: str,
) -> None:
    if not runs:
        return

    metric_keys = [metric for metric, _ in STAGE_METRICS]
    summary = summarize_runs(runs, metric_keys)
    write_summary_csv(out_dir / f"{stage}_summary.csv", summary, metric_keys)

    fig, axes = plt.subplots(1, len(STAGE_METRICS), figsize=(12.2, 3.8), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, (metric, label) in zip(axes, STAGE_METRICS):
        for idx, run in enumerate(runs):
            x_seed, y_seed = collect_metric_series(run, metric)
            if x_seed.size == 0:
                continue
            y_seed_plot = centered_moving_average(y_seed, smooth)
            ax.plot(
                x_seed,
                y_seed_plot,
                color=SEED_COLORS[idx % len(SEED_COLORS)],
                alpha=0.35,
                linewidth=1.1,
            )

        x = summary["update"]
        mean = summary[f"{metric}_mean"]
        delta = summary[f"{metric}_{spread}"]
        mean_plot = centered_moving_average(mean, smooth)
        lower = centered_moving_average(mean - delta, smooth)
        upper = centered_moving_average(mean + delta, smooth)

        ax.fill_between(x, lower, upper, color="#9EA3A8", alpha=0.25, linewidth=0.0)
        ax.plot(x, mean_plot, color="#111111", linewidth=2.4)

        ax.set_title(label)
        ax.set_xlabel("Update")
        ax.set_ylabel(label)
        ax.margins(x=0.02)

        if metric == "total_loss_mean":
            finite = mean[np.isfinite(mean)]
            if finite.size:
                ymin = min(0.0, float(np.nanmin(finite)))
                ymax = float(np.nanmax(finite))
                if ymax > ymin:
                    pad = 0.08 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)

    title = stage_display_name(stage)
    if title_prefix:
        title = f"{title_prefix} {title}".strip()
    fig.suptitle(title, y=0.995, fontsize=13)

    legend_handles = [
        Line2D([0], [0], color=SEED_COLORS[0], lw=1.2, alpha=0.5, label="Individual seed"),
        Line2D([0], [0], color="#111111", lw=2.4, label="Mean"),
        Patch(facecolor="#9EA3A8", edgecolor="none", alpha=0.25, label=band_label(spread)),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    save_figure(fig, out_dir / f"{stage}_training_curves", formats, dpi)
    plt.close(fig)


def plot_combined_rewards(
    stage_order: List[str],
    runs_by_stage: Dict[str, List[RunLog]],
    out_dir: Path,
    smooth: int,
    spread: str,
    formats: Iterable[str],
    dpi: int,
    title_prefix: str,
) -> None:
    combined_by_seed = make_combined_stage_runs(stage_order, runs_by_stage)
    combined_runs = list(combined_by_seed.values())
    if not combined_runs:
        print("[WARN] No seeds have all requested stages; combined reward figure skipped.")
        return

    metric_keys = [metric for metric, _ in COMBINED_REWARD_METRICS]
    summary = summarize_runs(combined_runs, metric_keys)
    write_summary_csv(out_dir / "combined_two_stage_reward_summary.csv", summary, metric_keys)

    fig, axes = plt.subplots(1, len(COMBINED_REWARD_METRICS), figsize=(8.8, 3.6), sharex=True)
    axes = np.atleast_1d(axes)
    boundaries = compute_stage_boundaries(stage_order, runs_by_stage)

    for ax, (metric, label) in zip(axes, COMBINED_REWARD_METRICS):
        for idx, run in enumerate(combined_runs):
            x_seed, y_seed = collect_metric_series(run, metric)
            if x_seed.size == 0:
                continue
            y_seed_plot = centered_moving_average(y_seed, smooth)
            ax.plot(
                x_seed,
                y_seed_plot,
                color=SEED_COLORS[idx % len(SEED_COLORS)],
                alpha=0.35,
                linewidth=1.1,
            )

        x = summary["update"]
        mean = summary[f"{metric}_mean"]
        delta = summary[f"{metric}_{spread}"]
        mean_plot = centered_moving_average(mean, smooth)
        lower = centered_moving_average(mean - delta, smooth)
        upper = centered_moving_average(mean + delta, smooth)

        ax.fill_between(x, lower, upper, color="#9EA3A8", alpha=0.25, linewidth=0.0)
        ax.plot(x, mean_plot, color="#111111", linewidth=2.4)

        for boundary, stage in boundaries:
            ax.axvline(boundary, color="#6B6B6B", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(
                boundary,
                0.98,
                stage_display_name(stage).split(":")[0],
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color="#4A4A4A",
            )

        ax.set_title(label)
        ax.set_xlabel("Cumulative Update")
        ax.set_ylabel(label)
        ax.margins(x=0.02)

    title = "Two-Stage Reward Curves"
    if title_prefix:
        title = f"{title_prefix} {title}".strip()
    fig.suptitle(title, y=0.995, fontsize=13)

    legend_handles = [
        Line2D([0], [0], color=SEED_COLORS[0], lw=1.2, alpha=0.5, label="Individual seed"),
        Line2D([0], [0], color="#111111", lw=2.4, label="Mean"),
        Patch(facecolor="#9EA3A8", edgecolor="none", alpha=0.25, label=band_label(spread)),
        Line2D([0], [0], color="#6B6B6B", lw=1.0, linestyle="--", label="Stage boundary"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    save_figure(fig, out_dir / "combined_two_stage_reward_curves", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, List[RunLog]] = {}
    missing: Dict[str, List[int]] = {}

    for stage in args.stages:
        stage_runs: List[RunLog] = []
        stage_missing: List[int] = []
        for seed in args.seeds:
            run = load_run(args.base_dir, seed, stage)
            if run is None:
                stage_missing.append(seed)
            else:
                stage_runs.append(run)
        loaded[stage] = stage_runs
        missing[stage] = stage_missing

    write_manifest(args.out_dir / "manifest.txt", args.base_dir, args.stages, loaded, missing)

    plotted_any = False
    for stage in args.stages:
        runs = loaded.get(stage, [])
        if not runs:
            print(f"[WARN] No valid runs found for stage: {stage}")
            continue
        plot_stage(
            stage=stage,
            runs=runs,
            out_dir=args.out_dir,
            smooth=max(1, args.smooth),
            spread=args.spread,
            formats=args.formats,
            dpi=args.dpi,
            title_prefix=args.title_prefix,
        )
        plotted_any = True
        print(
            f"[OK] stage={stage} seeds={[run.seed for run in runs]} "
            f"missing={missing.get(stage, [])}"
        )

    plot_combined_rewards(
        stage_order=args.stages,
        runs_by_stage=loaded,
        out_dir=args.out_dir,
        smooth=max(1, args.smooth),
        spread=args.spread,
        formats=args.formats,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )

    if not plotted_any:
        raise SystemExit(
            "No valid training logs were found. Check --base-dir, --seeds, and --stages. "
            f"See {args.out_dir / 'manifest.txt'} for details."
        )

    print(f"[OK] Output directory: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
