from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_pct(num: pd.Series, denom: pd.Series, eps: float) -> pd.Series:
    return 100.0 * num / (denom.abs() + eps)


def load_overall_metric_map(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["mean"]))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    wsum = float(weights.sum())
    if wsum <= 0:
        return float("nan")
    return float((values * weights).sum() / wsum)


def build_lb_pct_tables(log_dir: Path, out_dir: Path, eps: float, exclude_seed: int | None) -> None:
    lb_base_seed = pd.read_csv(log_dir / "table_lb_vs_baselines_by_seed.csv")
    lb_root_seed = pd.read_csv(log_dir / "table_lb_vs_gurobi_scip_by_seed.csv")
    lb_base_overall = pd.read_csv(log_dir / "table_lb_vs_baselines_overall.csv")
    lb_root_overall = pd.read_csv(log_dir / "table_lb_vs_gurobi_scip_overall.csv")

    if exclude_seed is not None:
        lb_base_seed = lb_base_seed[lb_base_seed["seed"].astype(int) != int(exclude_seed)].copy()
        lb_root_seed = lb_root_seed[lb_root_seed["seed"].astype(int) != int(exclude_seed)].copy()

    seed = lb_base_seed.merge(
        lb_root_seed[["seed", "n_cases", "lb_improve_mean", "lb_improve_median"]],
        on=["seed", "n_cases"],
        how="inner",
    )

    seed["mccormick_lb_over_root_mean"] = seed["lb_improve_mean"] - seed["rl_minus_mccormick_lb_mean"]
    seed["sdp_lb_over_root_mean"] = seed["lb_improve_mean"] - seed["rl_minus_sdp_lb_mean"]
    seed["rl_vs_mccormick_lb_pct_mean"] = add_pct(
        seed["rl_minus_mccormick_lb_mean"], seed["mccormick_lb_over_root_mean"], eps
    )
    seed["rl_vs_sdp_lb_pct_mean"] = add_pct(seed["rl_minus_sdp_lb_mean"], seed["sdp_lb_over_root_mean"], eps)

    seed["mccormick_lb_over_root_median"] = seed["lb_improve_median"] - seed["rl_minus_mccormick_lb_median"]
    seed["sdp_lb_over_root_median"] = seed["lb_improve_median"] - seed["rl_minus_sdp_lb_median"]
    seed["rl_vs_mccormick_lb_pct_median"] = add_pct(
        seed["rl_minus_mccormick_lb_median"], seed["mccormick_lb_over_root_median"], eps
    )
    seed["rl_vs_sdp_lb_pct_median"] = add_pct(seed["rl_minus_sdp_lb_median"], seed["sdp_lb_over_root_median"], eps)
    seed["pct_formula"] = "100*(RL-BASE)/(abs(BASE)+eps)"
    seed["eps"] = eps

    if exclude_seed is None:
        overall = lb_base_overall.copy()
        overall["lb_improve_mean"] = lb_root_overall.loc[0, "lb_improve_mean"]
        overall["lb_improve_median"] = lb_root_overall.loc[0, "lb_improve_median"]

        overall["mccormick_lb_over_root_mean"] = overall["lb_improve_mean"] - overall["rl_minus_mccormick_lb_mean"]
        overall["sdp_lb_over_root_mean"] = overall["lb_improve_mean"] - overall["rl_minus_sdp_lb_mean"]
        overall["rl_vs_mccormick_lb_pct_mean"] = add_pct(
            overall["rl_minus_mccormick_lb_mean"], overall["mccormick_lb_over_root_mean"], eps
        )
        overall["rl_vs_sdp_lb_pct_mean"] = add_pct(
            overall["rl_minus_sdp_lb_mean"], overall["sdp_lb_over_root_mean"], eps
        )

        overall["mccormick_lb_over_root_median"] = (
            overall["lb_improve_median"] - overall["rl_minus_mccormick_lb_median"]
        )
        overall["sdp_lb_over_root_median"] = overall["lb_improve_median"] - overall["rl_minus_sdp_lb_median"]
        overall["rl_vs_mccormick_lb_pct_median"] = add_pct(
            overall["rl_minus_mccormick_lb_median"], overall["mccormick_lb_over_root_median"], eps
        )
        overall["rl_vs_sdp_lb_pct_median"] = add_pct(
            overall["rl_minus_sdp_lb_median"], overall["sdp_lb_over_root_median"], eps
        )
        overall["pct_formula"] = "100*(RL-BASE)/(abs(BASE)+eps)"
        overall["eps"] = eps
    else:
        weights = seed["n_cases"]
        overall = pd.DataFrame(
            [
                {
                    "scope": f"overall_excluding_{exclude_seed}",
                    "n_seed_problem_pairs": int(weights.sum()),
                    "rl_minus_mccormick_lb_mean": weighted_mean(seed["rl_minus_mccormick_lb_mean"], weights),
                    "rl_minus_sdp_lb_mean": weighted_mean(seed["rl_minus_sdp_lb_mean"], weights),
                    "lb_improve_mean": weighted_mean(seed["lb_improve_mean"], weights),
                    "mccormick_lb_over_root_mean": weighted_mean(seed["mccormick_lb_over_root_mean"], weights),
                    "sdp_lb_over_root_mean": weighted_mean(seed["sdp_lb_over_root_mean"], weights),
                    "rl_vs_mccormick_lb_pct_mean": weighted_mean(seed["rl_vs_mccormick_lb_pct_mean"], weights),
                    "rl_vs_sdp_lb_pct_mean": weighted_mean(seed["rl_vs_sdp_lb_pct_mean"], weights),
                    "pct_formula": "100*(RL-BASE)/(abs(BASE)+eps)",
                    "eps": eps,
                    "note": "computed as weighted mean over filtered seeds",
                }
            ]
        )

    suffix = "" if exclude_seed is None else f"_excluding_{exclude_seed}"
    seed.to_csv(out_dir / f"table_lb_vs_baselines_by_seed_pct{suffix}.csv", index=False)
    overall.to_csv(out_dir / f"table_lb_vs_baselines_overall_pct{suffix}.csv", index=False)


def build_cost_pct_tables(log_dir: Path, out_dir: Path, eps: float, exclude_seed: int | None) -> None:
    cost_seed = pd.read_csv(log_dir / "table_cost_vs_baselines_by_seed.csv")
    cost_overall = pd.read_csv(log_dir / "table_cost_vs_baselines_overall.csv")
    lb_root_seed = pd.read_csv(log_dir / "table_lb_vs_gurobi_scip_by_seed.csv")
    lb_root_overall = pd.read_csv(log_dir / "table_lb_vs_gurobi_scip_overall.csv")
    seed_summary = pd.read_csv(log_dir / "multiseed_seed_summary.csv")
    overall_map = load_overall_metric_map(log_dir / "multiseed_overall_summary.csv")

    if exclude_seed is not None:
        cost_seed = cost_seed[cost_seed["seed"].astype(int) != int(exclude_seed)].copy()
        lb_root_seed = lb_root_seed[lb_root_seed["seed"].astype(int) != int(exclude_seed)].copy()
        seed_summary = seed_summary[seed_summary["seed"].astype(int) != int(exclude_seed)].copy()

    seed = (
        cost_seed
        .merge(lb_root_seed[["seed", "n_cases", "rl_pipeline_time_sec_mean"]], on=["seed", "n_cases"], how="inner")
        .merge(
            seed_summary[["seed", "n_cases", "rl_added_vars_mean", "rl_added_cons_mean", "rl_added_nnz_mean"]],
            on=["seed", "n_cases"],
            how="inner",
        )
    )

    seed["mccormick_time_mean"] = seed["rl_pipeline_time_sec_mean"] - seed["rl_minus_mccormick_time_mean"]
    seed["sdp_time_mean"] = seed["rl_pipeline_time_sec_mean"] - seed["rl_minus_sdp_time_mean"]
    seed["rl_vs_mccormick_time_pct_mean"] = add_pct(
        seed["rl_minus_mccormick_time_mean"], seed["mccormick_time_mean"], eps
    )
    seed["rl_vs_sdp_time_pct_mean"] = add_pct(seed["rl_minus_sdp_time_mean"], seed["sdp_time_mean"], eps)

    for obj in ["added_vars", "added_cons", "added_nnz"]:
        rl_col = {
            "added_vars": "rl_added_vars_mean",
            "added_cons": "rl_added_cons_mean",
            "added_nnz": "rl_added_nnz_mean",
        }[obj]
        d_m = f"rl_minus_mccormick_{obj}_mean"
        d_s = f"rl_minus_sdp_{obj}_mean"
        m_base = f"mccormick_{obj}_mean"
        s_base = f"sdp_{obj}_mean"

        seed[m_base] = seed[rl_col] - seed[d_m]
        seed[s_base] = seed[rl_col] - seed[d_s]
        seed[f"rl_vs_mccormick_{obj}_pct_mean"] = add_pct(seed[d_m], seed[m_base], eps)
        seed[f"rl_vs_sdp_{obj}_pct_mean"] = add_pct(seed[d_s], seed[s_base], eps)

    seed["pct_formula"] = "100*(RL-BASE)/(abs(BASE)+eps)"
    seed["eps"] = eps

    if exclude_seed is None:
        overall = cost_overall.copy()
        rl_time = float(lb_root_overall.loc[0, "rl_pipeline_time_sec_mean"])
        rl_vars = float(overall_map["rl_added_vars"])
        rl_cons = float(overall_map["rl_added_cons"])
        rl_nnz = float(overall_map["rl_added_nnz"])

        overall["rl_pipeline_time_sec_mean"] = rl_time
        overall["rl_added_vars_mean"] = rl_vars
        overall["rl_added_cons_mean"] = rl_cons
        overall["rl_added_nnz_mean"] = rl_nnz

        overall["mccormick_time_mean"] = overall["rl_pipeline_time_sec_mean"] - overall["rl_minus_mccormick_time_mean"]
        overall["sdp_time_mean"] = overall["rl_pipeline_time_sec_mean"] - overall["rl_minus_sdp_time_mean"]
        overall["rl_vs_mccormick_time_pct_mean"] = add_pct(
            overall["rl_minus_mccormick_time_mean"], overall["mccormick_time_mean"], eps
        )
        overall["rl_vs_sdp_time_pct_mean"] = add_pct(
            overall["rl_minus_sdp_time_mean"], overall["sdp_time_mean"], eps
        )

        for obj in ["added_vars", "added_cons", "added_nnz"]:
            rl_col = f"rl_{obj}_mean"
            d_m = f"rl_minus_mccormick_{obj}_mean"
            d_s = f"rl_minus_sdp_{obj}_mean"
            m_base = f"mccormick_{obj}_mean"
            s_base = f"sdp_{obj}_mean"

            overall[m_base] = overall[rl_col] - overall[d_m]
            overall[s_base] = overall[rl_col] - overall[d_s]
            overall[f"rl_vs_mccormick_{obj}_pct_mean"] = add_pct(overall[d_m], overall[m_base], eps)
            overall[f"rl_vs_sdp_{obj}_pct_mean"] = add_pct(overall[d_s], overall[s_base], eps)

        overall["pct_formula"] = "100*(RL-BASE)/(abs(BASE)+eps)"
        overall["eps"] = eps
    else:
        weights = seed["n_cases"]
        overall = pd.DataFrame(
            [
                {
                    "scope": f"overall_excluding_{exclude_seed}",
                    "n_seed_problem_pairs": int(weights.sum()),
                    "rl_minus_mccormick_time_mean": weighted_mean(seed["rl_minus_mccormick_time_mean"], weights),
                    "rl_minus_sdp_time_mean": weighted_mean(seed["rl_minus_sdp_time_mean"], weights),
                    "rl_pipeline_time_sec_mean": weighted_mean(seed["rl_pipeline_time_sec_mean"], weights),
                    "mccormick_time_mean": weighted_mean(seed["mccormick_time_mean"], weights),
                    "sdp_time_mean": weighted_mean(seed["sdp_time_mean"], weights),
                    "rl_vs_mccormick_time_pct_mean": weighted_mean(seed["rl_vs_mccormick_time_pct_mean"], weights),
                    "rl_vs_sdp_time_pct_mean": weighted_mean(seed["rl_vs_sdp_time_pct_mean"], weights),
                    "rl_minus_mccormick_added_vars_mean": weighted_mean(
                        seed["rl_minus_mccormick_added_vars_mean"], weights
                    ),
                    "rl_minus_mccormick_added_cons_mean": weighted_mean(
                        seed["rl_minus_mccormick_added_cons_mean"], weights
                    ),
                    "rl_minus_mccormick_added_nnz_mean": weighted_mean(
                        seed["rl_minus_mccormick_added_nnz_mean"], weights
                    ),
                    "rl_minus_sdp_added_vars_mean": weighted_mean(seed["rl_minus_sdp_added_vars_mean"], weights),
                    "rl_minus_sdp_added_cons_mean": weighted_mean(seed["rl_minus_sdp_added_cons_mean"], weights),
                    "rl_minus_sdp_added_nnz_mean": weighted_mean(seed["rl_minus_sdp_added_nnz_mean"], weights),
                    "mccormick_added_vars_mean": weighted_mean(seed["mccormick_added_vars_mean"], weights),
                    "mccormick_added_cons_mean": weighted_mean(seed["mccormick_added_cons_mean"], weights),
                    "mccormick_added_nnz_mean": weighted_mean(seed["mccormick_added_nnz_mean"], weights),
                    "sdp_added_vars_mean": weighted_mean(seed["sdp_added_vars_mean"], weights),
                    "sdp_added_cons_mean": weighted_mean(seed["sdp_added_cons_mean"], weights),
                    "sdp_added_nnz_mean": weighted_mean(seed["sdp_added_nnz_mean"], weights),
                    "rl_vs_mccormick_added_vars_pct_mean": weighted_mean(
                        seed["rl_vs_mccormick_added_vars_pct_mean"], weights
                    ),
                    "rl_vs_mccormick_added_cons_pct_mean": weighted_mean(
                        seed["rl_vs_mccormick_added_cons_pct_mean"], weights
                    ),
                    "rl_vs_mccormick_added_nnz_pct_mean": weighted_mean(
                        seed["rl_vs_mccormick_added_nnz_pct_mean"], weights
                    ),
                    "rl_vs_sdp_added_vars_pct_mean": weighted_mean(seed["rl_vs_sdp_added_vars_pct_mean"], weights),
                    "rl_vs_sdp_added_cons_pct_mean": weighted_mean(seed["rl_vs_sdp_added_cons_pct_mean"], weights),
                    "rl_vs_sdp_added_nnz_pct_mean": weighted_mean(seed["rl_vs_sdp_added_nnz_pct_mean"], weights),
                    "pct_formula": "100*(RL-BASE)/(abs(BASE)+eps)",
                    "eps": eps,
                    "note": "computed as weighted mean over filtered seeds",
                }
            ]
        )

    suffix = "" if exclude_seed is None else f"_excluding_{exclude_seed}"
    seed.to_csv(out_dir / f"table_cost_vs_baselines_by_seed_pct{suffix}.csv", index=False)
    overall.to_csv(out_dir / f"table_cost_vs_baselines_overall_pct{suffix}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--out-dir", default="figures/meeting_report")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--exclude-seed", type=int, default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_lb_pct_tables(log_dir=log_dir, out_dir=out_dir, eps=args.eps, exclude_seed=args.exclude_seed)
    build_cost_pct_tables(log_dir=log_dir, out_dir=out_dir, eps=args.eps, exclude_seed=args.exclude_seed)

    suffix = "" if args.exclude_seed is None else f"_excluding_{args.exclude_seed}"
    print(f"Saved pct tables with suffix '{suffix}' to: {out_dir}")


if __name__ == "__main__":
    main()
