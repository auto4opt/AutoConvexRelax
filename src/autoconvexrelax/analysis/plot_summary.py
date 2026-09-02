# plot_summary.py
# Usage:
#   python run/plot.py summary
#   python run/plot.py summary --csv outputs/logs/summary.csv --out outputs/logs/figs --topk 10
#
# This version auto-maps your CSV columns:
#   root_lb      <- gurobi_root_bound
#   improve      <- lb_improve
#   improve_pct  <- lb_improve_pct
#
# Optional columns used if present:
#   gap, tight_gain, gurobi_obj, file

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_mkdir(d: str):
    os.makedirs(d, exist_ok=True)

def infer_family(name: str) -> str:
    # "E_EXP_P4_PURE_n7_956" -> "E_EXP"
    # "H_MIXED_3_n6_754"     -> "H_MIXED"
    parts = str(name).split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return parts[0] if parts else "UNKNOWN"

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip spaces
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # define alias mapping (first match wins)
    alias = {
        "name": ["name", "problem", "instance"],
        "root_lb": ["root_lb", "gurobi_root_bound", "root_bound", "root_relax_lb", "gurobi_root_lb"],
        "rl_lb": ["rl_lb", "rl_root_lb", "policy_lb", "lb_rl"],
        "improve": ["improve", "lb_improve", "improvement", "delta_lb"],
        "improve_pct": ["improve_%", "improve_pct", "lb_improve_pct", "improvement_pct"],
        "gap": ["gap", "mip_gap", "gap_to_obj"],
        "tight_gain": ["tight_gain", "rl_lb_tight_gain", "tightening_gain"],
        "gurobi_obj": ["gurobi_obj", "obj", "gurobi_opt", "gurobi_best_obj"],
        "file": ["file", "pkl", "path"],
    }

    colmap = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for std, candidates in alias.items():
        found = None
        for cand in candidates:
            key = cand.lower()
            if key in lower_cols:
                found = lower_cols[key]
                break
        if found is not None:
            colmap[found] = std

    df = df.rename(columns=colmap)

    # required minimal set
    required = ["name", "root_lb", "rl_lb", "improve", "improve_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after mapping: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # ensure numeric
    for c in ["root_lb", "rl_lb", "improve", "improve_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with NaNs in critical cols
    df = df.dropna(subset=["name", "root_lb", "rl_lb", "improve", "improve_pct"]).reset_index(drop=True)

    # add family
    df["family"] = df["name"].apply(infer_family)
    return df

def write_stats(df: pd.DataFrame, out_dir: str):
    imp = df["improve"].to_numpy()
    imp_pct = df["improve_pct"].to_numpy()

    n = len(df)
    eps = 1e-9
    n_pos = int(np.sum(imp > eps))
    n_zero = int(np.sum(np.abs(imp) <= eps))
    n_neg = int(np.sum(imp < -eps))

    lines = []
    lines.append(f"n = {n}")
    lines.append(f"improve > 0: {n_pos} ({n_pos/n*100:.2f}%)")
    lines.append(f"improve = 0: {n_zero} ({n_zero/n*100:.2f}%)")
    lines.append(f"improve < 0: {n_neg} ({n_neg/n*100:.2f}%)")
    lines.append("")
    lines.append("improve (absolute):")
    lines.append(f"  mean   = {np.mean(imp):.6g}")
    lines.append(f"  median = {np.median(imp):.6g}")
    lines.append(f"  min    = {np.min(imp):.6g}")
    lines.append(f"  max    = {np.max(imp):.6g}")
    lines.append("")
    lines.append("improve_pct (%):")
    lines.append(f"  mean   = {np.mean(imp_pct):.6g}")
    lines.append(f"  median = {np.median(imp_pct):.6g}")
    lines.append(f"  min    = {np.min(imp_pct):.6g}")
    lines.append(f"  max    = {np.max(imp_pct):.6g}")
    lines.append("")

    fam = df.groupby("family").agg(
        n=("name", "count"),
        mean_improve=("improve", "mean"),
        med_improve=("improve", "median"),
        mean_improve_pct=("improve_pct", "mean"),
        med_improve_pct=("improve_pct", "median"),
        neg_cnt=("improve", lambda x: int(np.sum(x < -1e-9))),
    ).sort_values("n", ascending=False)

    lines.append("By family (n, mean/median improve, mean/median improve_pct, #neg):")
    for idx, row in fam.iterrows():
        lines.append(
            f"  {idx:10s}  n={int(row['n']):3d}  "
            f"mean={row['mean_improve']:.6g}  med={row['med_improve']:.6g}  "
            f"mean%={row['mean_improve_pct']:.6g}  med%={row['med_improve_pct']:.6g}  "
            f"neg={int(row['neg_cnt'])}"
        )

    with open(os.path.join(out_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_histograms(df: pd.DataFrame, out_dir: str):
    plt.figure()
    plt.hist(df["improve"].to_numpy(), bins=20)
    plt.title("Distribution of lb_improve (absolute)")
    plt.xlabel("lb_improve = rl_lb - gurobi_root_bound")
    plt.ylabel("count")
    save_fig(os.path.join(out_dir, "hist_lb_improve_abs.png"))

    plt.figure()
    plt.hist(df["improve_pct"].to_numpy(), bins=20)
    plt.title("Distribution of lb_improve_pct (%)")
    plt.xlabel("lb_improve_pct")
    plt.ylabel("count")
    save_fig(os.path.join(out_dir, "hist_lb_improve_pct.png"))

def plot_ecdf(df: pd.DataFrame, out_dir: str):
    x = np.sort(df["improve"].to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure()
    plt.step(x, y, where="post")
    plt.title("ECDF of lb_improve")
    plt.xlabel("lb_improve")
    plt.ylabel("fraction ≤ x")
    save_fig(os.path.join(out_dir, "ecdf_lb_improve.png"))

    x2 = np.sort(df["improve_pct"].to_numpy())
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    plt.figure()
    plt.step(x2, y2, where="post")
    plt.title("ECDF of lb_improve_pct")
    plt.xlabel("lb_improve_pct (%)")
    plt.ylabel("fraction ≤ x")
    save_fig(os.path.join(out_dir, "ecdf_lb_improve_pct.png"))

def plot_top_bottom(df: pd.DataFrame, out_dir: str, topk: int = 10):
    df_sorted = df.sort_values("improve", ascending=False).reset_index(drop=True)

    top = df_sorted.head(topk)
    bot = df_sorted.tail(topk)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(top)), top["improve"].to_numpy())
    plt.xticks(range(len(top)), top["name"].tolist(), rotation=60, ha="right")
    plt.title(f"Top {topk} lb_improve (absolute)")
    plt.ylabel("lb_improve")
    save_fig(os.path.join(out_dir, f"top{topk}_lb_improve.png"))

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(bot)), bot["improve"].to_numpy())
    plt.xticks(range(len(bot)), bot["name"].tolist(), rotation=60, ha="right")
    plt.title(f"Bottom {topk} lb_improve (absolute)")
    plt.ylabel("lb_improve")
    save_fig(os.path.join(out_dir, f"bottom{topk}_lb_improve.png"))

def plot_scatter_root_vs_rl(df: pd.DataFrame, out_dir: str):
    x = df["root_lb"].to_numpy()
    y = df["rl_lb"].to_numpy()
    mn = float(np.min(np.concatenate([x, y])))
    mx = float(np.max(np.concatenate([x, y])))

    plt.figure()
    plt.scatter(x, y, s=20)
    plt.plot([mn, mx], [mn, mx])  # y=x
    plt.title("gurobi_root_bound vs rl_lb (higher is better)")
    plt.xlabel("gurobi_root_bound")
    plt.ylabel("rl_lb")
    save_fig(os.path.join(out_dir, "scatter_root_vs_rl.png"))

def plot_box_by_family(df: pd.DataFrame, out_dir: str):
    fams = sorted(df["family"].unique().tolist())
    if len(fams) <= 1:
        return
    data = [df.loc[df["family"] == f, "improve_pct"].to_numpy() for f in fams]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=fams, showfliers=True)
    plt.title("lb_improve_pct by family")
    plt.ylabel("lb_improve_pct (%)")
    plt.xticks(rotation=45, ha="right")
    save_fig(os.path.join(out_dir, "box_lb_improve_pct_by_family.png"))

def plot_cumulative(df: pd.DataFrame, out_dir: str):
    df_sorted = df.sort_values("improve", ascending=False).reset_index(drop=True)
    cum = np.cumsum(df_sorted["improve"].to_numpy())

    plt.figure()
    plt.plot(np.arange(1, len(cum) + 1), cum)
    plt.title("Cumulative sum of lb_improve (sorted desc)")
    plt.xlabel("k (top-k instances)")
    plt.ylabel("cumulative lb_improve")
    save_fig(os.path.join(out_dir, "cumulative_lb_improve.png"))

def plot_optional_gap(df_raw: pd.DataFrame, out_dir: str):
    # Use original df if it has gap; mapping may or may not include it.
    if "gap" not in df_raw.columns:
        return
    g = pd.to_numeric(df_raw["gap"], errors="coerce").dropna().to_numpy()
    if len(g) == 0:
        return

    plt.figure()
    plt.hist(g, bins=20)
    plt.title("Distribution of gap (if defined in your pipeline)")
    plt.xlabel("gap")
    plt.ylabel("count")
    save_fig(os.path.join(out_dir, "hist_gap.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/logs/summary.csv")
    parser.add_argument("--out", type=str, default="outputs/figures/summary")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    safe_mkdir(args.out)

    df_raw = pd.read_csv(args.csv)
    df = standardize_columns(df_raw)

    write_stats(df, args.out)

    plot_histograms(df, args.out)
    plot_ecdf(df, args.out)
    plot_top_bottom(df, args.out, topk=args.topk)
    plot_scatter_root_vs_rl(df, args.out)
    plot_box_by_family(df, args.out)
    plot_cumulative(df, args.out)

    # optional
    plot_optional_gap(df_raw, args.out)

    print(f"[OK] CSV : {args.csv}")
    print(f"[OK] OUT : {args.out}")
    print(f"[OK] Wrote stats: {os.path.join(args.out, 'stats.txt')}")

if __name__ == "__main__":
    main()
