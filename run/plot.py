#!/usr/bin/env python3
"""Generate training, evaluation, and manuscript figures."""

from _bootstrap import dispatch


if __name__ == "__main__":
    dispatch(
        {
            "training": "autoconvexrelax.analysis.plot_training",
            "main": "autoconvexrelax.analysis.plot_main",
            "paper": "autoconvexrelax.analysis.plot_paper",
            "summary": "autoconvexrelax.analysis.plot_summary",
        },
        __doc__ or "Plotting",
    )
