#!/usr/bin/env python3
"""Aggregate, filter, and summarize evaluation outputs."""

from _bootstrap import dispatch


if __name__ == "__main__":
    dispatch(
        {
            "checkpoints": "autoconvexrelax.analysis.checkpoints",
            "summarize": "autoconvexrelax.analysis.summarize",
            "multiseed": "autoconvexrelax.analysis.summarize_multiseed",
            "fraction": "autoconvexrelax.analysis.summarize_fraction",
            "filter-fraction": "autoconvexrelax.analysis.filter_fraction",
            "fraction-subset": "autoconvexrelax.analysis.make_fraction_subset",
            "percent": "autoconvexrelax.analysis.convert_percent",
        },
        __doc__ or "Result analysis",
    )
