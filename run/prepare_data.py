#!/usr/bin/env python3
"""Generate datasets and precompute solver metadata."""

from _bootstrap import dispatch


if __name__ == "__main__":
    dispatch(
        {
            "generate": "autoconvexrelax.problems.finetune_fraction",
            "hard-fraction": "autoconvexrelax.problems.generate_hard_fraction",
            "feasibility": "autoconvexrelax.tools.cache_feasibility",
            "gurobi-root": "autoconvexrelax.tools.cache_gurobi",
            "scip-root": "autoconvexrelax.tools.cache_scip",
        },
        __doc__ or "Data preparation",
    )
