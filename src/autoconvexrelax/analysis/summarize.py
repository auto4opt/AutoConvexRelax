# Summarize a single evaluation directory.
# -*- coding: utf-8 -*-

import os
import json
import csv
import math
import argparse
from statistics import mean, median

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def pct_improve(rl_lb, root_lb, eps=1e-9):
    """relative improvement (%) against gurobi root bound (robust near zero)."""
    if rl_lb is None or root_lb is None:
        return None
    # use a scale that won't explode when root_lb is 0
    denom = max(abs(root_lb), abs(rl_lb), eps)
    return (rl_lb - root_lb) / denom * 100.0

def fmt(x, width=12, prec=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return " " * (width - 4) + "None"
    s = f"{x:.{prec}g}"
    return s.rjust(width)

def _default_json_path(dir_):
    # runner 默认把 json 写在 SAVE_DIR/eval_vs_gurobi.json
    return os.path.join(dir_, "eval_vs_gurobi.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="outputs/logs",
                    help="runner 输出目录（默认 outputs/logs）")
    ap.add_argument("--json", type=str, default=None,
                    help="结果 JSON 路径（默认用 --dir/eval_vs_gurobi.json）")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="输出 csv 路径（默认写到 dir/summary.csv）")
    ap.add_argument("--sort", type=str, default="pct", choices=["pct", "abs", "name"],
                    help="打印排序：pct=按百分比提升；abs=按绝对提升；name=按名称")
    ap.add_argument("--eps", type=float, default=1e-6,
                help="判定 same 的阈值：|lb_improve| <= eps 视为不变")

    args = ap.parse_args()

    save_dir = args.dir
    json_path = args.json or _default_json_path(save_dir)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Result json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Expect a list in json, got: {type(data)}")

    rows = []
    bad = 0
    has_baseline = False

    for r in data:
        if not isinstance(r, dict):
            bad += 1
            continue

        name = r.get("name", "noname")

        root_lb = safe_float(r.get("gurobi_root_bound"))
        scip_root = safe_float(r.get("scip_root_bound"))
        rl_lb   = safe_float(r.get("rl_lb"))
        lb_imp  = safe_float(r.get("lb_improve"))
        scip_imp = safe_float(r.get("lb_improve_scip"))
        scip_pct = safe_float(r.get("lb_improve_scip_pct"))
        base_lb = safe_float(r.get("baseline_lb"))
        base_mcc_lb = safe_float(r.get("baseline_mccormick_lb"))
        base_sdp_lb = safe_float(r.get("baseline_sdp_lb"))
        base_imp = safe_float(r.get("lb_improve_baseline"))
        base_imp_scip = safe_float(r.get("lb_improve_baseline_scip"))
        rl_minus_base = safe_float(r.get("rl_minus_baseline"))
        rl_cost = safe_float(r.get("rl_cost"))
        base_mcc_cost = safe_float(r.get("baseline_mccormick_cost"))
        base_sdp_cost = safe_float(r.get("baseline_sdp_cost"))
        if rl_cost is None:
            rl_cost = rl_lb
        if base_mcc_cost is None:
            base_mcc_cost = base_mcc_lb
        if base_sdp_cost is None:
            base_sdp_cost = base_sdp_lb

        # 如果你没存 lb_improve，也可以用 rl_lb-root_lb 现场算
        if lb_imp is None and rl_lb is not None and root_lb is not None:
            lb_imp = rl_lb - root_lb

        pct = pct_improve(rl_lb, root_lb)

        if scip_imp is None and rl_lb is not None and scip_root is not None:
            scip_imp = rl_lb - scip_root
        if scip_pct is None:
            scip_pct = pct_improve(rl_lb, scip_root)

        if base_imp is None and base_lb is not None and root_lb is not None:
            base_imp = base_lb - root_lb
        if base_imp_scip is None and base_lb is not None and scip_root is not None:
            base_imp_scip = base_lb - scip_root
        if rl_minus_base is None and rl_lb is not None and base_lb is not None:
            rl_minus_base = rl_lb - base_lb

        base_pct = pct_improve(base_lb, root_lb) if base_lb is not None else None
        base_scip_pct = pct_improve(base_lb, scip_root) if base_lb is not None else None
        rl_minus_base_pct = pct_improve(rl_lb, base_lb) if (rl_lb is not None and base_lb is not None) else None
        if base_lb is not None:
            has_baseline = True
        elif base_mcc_lb is not None or base_sdp_lb is not None:
            has_baseline = True

        rows.append({
            "name": name,
            "gurobi_root_bound": root_lb,
            "scip_root_bound": scip_root,
            "rl_lb": rl_lb,
            "rl_cost": rl_cost,
            "lb_improve": lb_imp,
            "lb_improve_pct": pct,
            "lb_improve_scip": scip_imp,
            "lb_improve_scip_pct": scip_pct,
            "baseline_lb": base_lb,
            "baseline_mccormick_lb": base_mcc_lb,
            "baseline_sdp_lb": base_sdp_lb,
            "baseline_mccormick_cost": base_mcc_cost,
            "baseline_sdp_cost": base_sdp_cost,
            "lb_improve_baseline": base_imp,
            "lb_improve_baseline_pct": base_pct,
            "lb_improve_baseline_scip": base_imp_scip,
            "lb_improve_baseline_scip_pct": base_scip_pct,
            "rl_minus_baseline": rl_minus_base,
            "rl_minus_baseline_pct": rl_minus_base_pct,
            "tight_gain": safe_float(r.get("tight_gain")),
            "gap": safe_float(r.get("gap")),
            "gurobi_obj": safe_float(r.get("gurobi_obj")),
            "gurobi_bound": safe_float(r.get("gurobi_bound")),
            "rl_lb_tight": safe_float(r.get("rl_lb_tight")),
            "rl_relax_time_sec": safe_float(r.get("rl_relax_time_sec")),
            "baseline_relax_time_sec": safe_float(r.get("baseline_relax_time_sec")),
            "rl_solve_time_sec": safe_float(r.get("rl_solve_time_sec")),
            "baseline_solve_time_sec": safe_float(r.get("baseline_solve_time_sec")),
            "rl_pipeline_time_sec": safe_float(r.get("rl_pipeline_time_sec")),
            "baseline_pipeline_time_sec": safe_float(r.get("baseline_pipeline_time_sec")),
            "rl_added_vars": safe_float(r.get("rl_added_vars")),
            "baseline_added_vars": safe_float(r.get("baseline_added_vars")),
            "rl_added_cons": safe_float(r.get("rl_added_cons")),
            "baseline_added_cons": safe_float(r.get("baseline_added_cons")),
            "rl_added_nnz": safe_float(r.get("rl_added_nnz")),
            "baseline_added_nnz": safe_float(r.get("baseline_added_nnz")),
            "rl_added_psd_size": safe_float(r.get("rl_added_psd_size")),
            "baseline_added_psd_size": safe_float(r.get("baseline_added_psd_size")),
        })

    # 排序
    if args.sort == "pct":
        rows.sort(key=lambda x: (x["lb_improve_pct"] is None, -(x["lb_improve_pct"] or -1e100)))
    elif args.sort == "abs":
        rows.sort(key=lambda x: (x["lb_improve"] is None, -(x["lb_improve"] or -1e100)))
    else:
        rows.sort(key=lambda x: str(x["name"]))

    print(f"[INFO] loaded json rows: {len(data)} | usable: {len(rows)} | bad_row: {bad}")
    print(f"[INFO] source json: {json_path}")

    # 逐题打印
    if has_baseline:
        header_rl_vs_base = (
            "idx  "
            + "name".ljust(32) + "  "
            + "base_lb".rjust(12) + "  "
            + "rl_lb".rjust(12) + "  "
            + "rl-base".rjust(12) + "  "
            + "rl-base_%".rjust(12)
        )
        print("\n[TABLE] Baseline vs RL")
        print(header_rl_vs_base)
        print("-" * len(header_rl_vs_base))
        for i, row in enumerate(rows, start=1):
            line = (
                f"{i:>3d}  "
                f"{str(row['name'])[:32].ljust(32)}  "
                f"{fmt(row.get('baseline_lb'))}  "
                f"{fmt(row['rl_lb'])}  "
                f"{fmt(row.get('rl_minus_baseline'))}  "
                f"{fmt(row.get('rl_minus_baseline_pct'))}"
            )
            print(line)

        header_base_vs_solver = (
            "idx  "
            + "name".ljust(32) + "  "
            + "root_lb".rjust(12) + "  "
            + "scip_lb".rjust(12) + "  "
            + "base_lb".rjust(12) + "  "
            + "base-gurobi".rjust(12) + "  "
            + "base-g_%".rjust(12) + "  "
            + "base-scip".rjust(12) + "  "
            + "base-s_%".rjust(12)
        )
        print("\n[TABLE] Baseline vs Gurobi/SCIP")
        print(header_base_vs_solver)
        print("-" * len(header_base_vs_solver))
        for i, row in enumerate(rows, start=1):
            line = (
                f"{i:>3d}  "
                f"{str(row['name'])[:32].ljust(32)}  "
                f"{fmt(row['gurobi_root_bound'])}  "
                f"{fmt(row.get('scip_root_bound'))}  "
                f"{fmt(row.get('baseline_lb'))}  "
                f"{fmt(row.get('lb_improve_baseline'))}  "
                f"{fmt(row.get('lb_improve_baseline_pct'))}  "
                f"{fmt(row.get('lb_improve_baseline_scip'))}  "
                f"{fmt(row.get('lb_improve_baseline_scip_pct'))}"
            )
            print(line)
    else:
        header = (
            "idx  "
            + "name".ljust(32) + "  "
            + "root_lb".rjust(12) + "  "
            + "scip_lb".rjust(12) + "  "
            + "rl_lb".rjust(12) + "  "
            + "improve".rjust(12) + "  "
            + "improve_%".rjust(12) + "  "
            + "scip_impr".rjust(12) + "  "
            + "scip_%".rjust(12)
        )
        print(header)
        print("-" * len(header))
        for i, row in enumerate(rows, start=1):
            line = (
                f"{i:>3d}  "
                f"{str(row['name'])[:32].ljust(32)}  "
                f"{fmt(row['gurobi_root_bound'])}  "
                f"{fmt(row.get('scip_root_bound'))}  "
                f"{fmt(row['rl_lb'])}  "
                f"{fmt(row['lb_improve'])}  "
                f"{fmt(row['lb_improve_pct'])}  "
                f"{fmt(row.get('lb_improve_scip'))}  "
                f"{fmt(row.get('lb_improve_scip_pct'))}"
            )
            print(line)

    # 统计：只在同时有 root_lb 和 rl_lb 的样本上统计百分比
    pct_list = [r["lb_improve_pct"] for r in rows if r["lb_improve_pct"] is not None]
    abs_list = [r["lb_improve"] for r in rows if r["lb_improve"] is not None]
    scip_pct_list = [r["lb_improve_scip_pct"] for r in rows if r["lb_improve_scip_pct"] is not None]
    scip_abs_list = [r["lb_improve_scip"] for r in rows if r["lb_improve_scip"] is not None]
    base_pct_list = [r["lb_improve_baseline_pct"] for r in rows if r.get("lb_improve_baseline_pct") is not None]
    base_abs_list = [r["lb_improve_baseline"] for r in rows if r.get("lb_improve_baseline") is not None]
    base_scip_pct_list = [r["lb_improve_baseline_scip_pct"] for r in rows if r.get("lb_improve_baseline_scip_pct") is not None]
    base_scip_abs_list = [r["lb_improve_baseline_scip"] for r in rows if r.get("lb_improve_baseline_scip") is not None]
    rl_minus_base_list = [r["rl_minus_baseline"] for r in rows if r.get("rl_minus_baseline") is not None]
    rl_minus_base_pct_list = [r["rl_minus_baseline_pct"] for r in rows if r.get("rl_minus_baseline_pct") is not None]
    rl_relax_time_list = [r["rl_relax_time_sec"] for r in rows if r.get("rl_relax_time_sec") is not None]
    base_relax_time_list = [r["baseline_relax_time_sec"] for r in rows if r.get("baseline_relax_time_sec") is not None]
    rl_solve_time_list = [r["rl_solve_time_sec"] for r in rows if r.get("rl_solve_time_sec") is not None]
    base_solve_time_list = [r["baseline_solve_time_sec"] for r in rows if r.get("baseline_solve_time_sec") is not None]
    rl_pipeline_time_list = [r["rl_pipeline_time_sec"] for r in rows if r.get("rl_pipeline_time_sec") is not None]
    base_pipeline_time_list = [r["baseline_pipeline_time_sec"] for r in rows if r.get("baseline_pipeline_time_sec") is not None]
    rl_added_vars_list = [r["rl_added_vars"] for r in rows if r.get("rl_added_vars") is not None]
    base_added_vars_list = [r["baseline_added_vars"] for r in rows if r.get("baseline_added_vars") is not None]
    rl_added_cons_list = [r["rl_added_cons"] for r in rows if r.get("rl_added_cons") is not None]
    base_added_cons_list = [r["baseline_added_cons"] for r in rows if r.get("baseline_added_cons") is not None]
    rl_added_nnz_list = [r["rl_added_nnz"] for r in rows if r.get("rl_added_nnz") is not None]
    base_added_nnz_list = [r["baseline_added_nnz"] for r in rows if r.get("baseline_added_nnz") is not None]
    rl_added_psd_size_list = [r["rl_added_psd_size"] for r in rows if r.get("rl_added_psd_size") is not None]
    base_added_psd_size_list = [r["baseline_added_psd_size"] for r in rows if r.get("baseline_added_psd_size") is not None]
    
    # -------------------- 分桶统计：improve / same / worse --------------------
    eps = float(args.eps)

    valid_rows = [r for r in rows if (r["lb_improve"] is not None)]
    n_valid = len(valid_rows)

    n_improve = sum(1 for r in valid_rows if r["lb_improve"] >  eps)
    n_same    = sum(1 for r in valid_rows if abs(r["lb_improve"]) <= eps)
    n_worse   = sum(1 for r in valid_rows if r["lb_improve"] < -eps)

    def _pct(n, d):
        return 0.0 if d == 0 else (n / d * 100.0)

    print(
        f"[BIN] using eps={eps:g} on lb_improve; valid={n_valid}/{len(rows)} | "
        f"improve={n_improve} ({_pct(n_improve, n_valid):.2f}%) | "
        f"same={n_same} ({_pct(n_same, n_valid):.2f}%) | "
        f"worse={n_worse} ({_pct(n_worse, n_valid):.2f}%)"
    )

    valid_rows_scip = [r for r in rows if (r["lb_improve_scip"] is not None)]
    n_valid_scip = len(valid_rows_scip)

    n_improve_scip = sum(1 for r in valid_rows_scip if r["lb_improve_scip"] >  eps)
    n_same_scip    = sum(1 for r in valid_rows_scip if abs(r["lb_improve_scip"]) <= eps)
    n_worse_scip   = sum(1 for r in valid_rows_scip if r["lb_improve_scip"] < -eps)

    print(
        f"[BIN-SCIP] using eps={eps:g} on lb_improve_scip; valid={n_valid_scip}/{len(rows)} | "
        f"improve={n_improve_scip} ({_pct(n_improve_scip, n_valid_scip):.2f}%) | "
        f"same={n_same_scip} ({_pct(n_same_scip, n_valid_scip):.2f}%) | "
        f"worse={n_worse_scip} ({_pct(n_worse_scip, n_valid_scip):.2f}%)"
    )

    # 可选：把“最差/最好”的样本也打印出来（非常有用）
    if n_valid > 0:
        best = max(valid_rows, key=lambda r: r["lb_improve"])
        worst = min(valid_rows, key=lambda r: r["lb_improve"])
        print(f"[BIN] best  : {best['name']} | lb_improve={best['lb_improve']:.6g} | pct={best['lb_improve_pct']}")
        print(f"[BIN] worst : {worst['name']} | lb_improve={worst['lb_improve']:.6g} | pct={worst['lb_improve_pct']}")

    if n_valid_scip > 0:
        best_scip = max(valid_rows_scip, key=lambda r: r["lb_improve_scip"])
        worst_scip = min(valid_rows_scip, key=lambda r: r["lb_improve_scip"])
        print(f"[BIN-SCIP] best  : {best_scip['name']} | lb_improve_scip={best_scip['lb_improve_scip']:.6g} | pct={best_scip['lb_improve_scip_pct']}")
        print(f"[BIN-SCIP] worst : {worst_scip['name']} | lb_improve_scip={worst_scip['lb_improve_scip']:.6g} | pct={worst_scip['lb_improve_scip_pct']}")


    def stat_block(xs, name):
        if not xs:
            print(f"[STAT] {name}: no valid samples")
            return
        print(
            f"[STAT] {name}: n={len(xs)} | "
            f"mean={mean(xs):.6g} | median={median(xs):.6g} | "
            f"min={min(xs):.6g} | max={max(xs):.6g}"
        )

    stat_block(abs_list, "lb_improve (absolute)")
    stat_block(pct_list, "lb_improve_pct (%)")
    stat_block(scip_abs_list, "lb_improve_scip (absolute)")
    stat_block(scip_pct_list, "lb_improve_scip_pct (%)")
    if has_baseline:
        stat_block(base_abs_list, "baseline lb_improve (absolute)")
        stat_block(base_pct_list, "baseline lb_improve_pct (%)")
        stat_block(base_scip_abs_list, "baseline lb_improve_scip (absolute)")
        stat_block(base_scip_pct_list, "baseline lb_improve_scip_pct (%)")
        stat_block(rl_minus_base_list, "rl_minus_baseline")
        stat_block(rl_minus_base_pct_list, "rl_minus_baseline_pct (%)")
        stat_block(rl_relax_time_list, "rl_relax_time_sec")
        stat_block(base_relax_time_list, "baseline_relax_time_sec")
        stat_block(rl_solve_time_list, "rl_solve_time_sec")
        stat_block(base_solve_time_list, "baseline_solve_time_sec")
        stat_block(rl_pipeline_time_list, "rl_pipeline_time_sec")
        stat_block(base_pipeline_time_list, "baseline_pipeline_time_sec")
        stat_block(rl_added_vars_list, "rl_added_vars")
        stat_block(base_added_vars_list, "baseline_added_vars")
        stat_block(rl_added_cons_list, "rl_added_cons")
        stat_block(base_added_cons_list, "baseline_added_cons")
        stat_block(rl_added_nnz_list, "rl_added_nnz")
        stat_block(base_added_nnz_list, "baseline_added_nnz")
        stat_block(rl_added_psd_size_list, "rl_added_psd_size")
        stat_block(base_added_psd_size_list, "baseline_added_psd_size")

    # 输出 csv
    out_csv = args.out_csv or os.path.join(save_dir, "summary.csv")
    fieldnames = list(rows[0].keys()) if rows else [
        "name", "gurobi_root_bound", "scip_root_bound", "rl_lb", "lb_improve", "lb_improve_pct",
        "lb_improve_scip", "lb_improve_scip_pct",
        "rl_cost",
        "baseline_lb", "lb_improve_baseline", "lb_improve_baseline_pct",
        "lb_improve_baseline_scip", "lb_improve_baseline_scip_pct",
        "baseline_mccormick_lb", "baseline_sdp_lb",
        "baseline_mccormick_cost", "baseline_sdp_cost",
        "rl_minus_baseline", "rl_minus_baseline_pct",
        "tight_gain", "gap", "gurobi_obj", "gurobi_bound", "rl_lb_tight"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[INFO] wrote csv -> {out_csv}")

if __name__ == "__main__":
    main()
