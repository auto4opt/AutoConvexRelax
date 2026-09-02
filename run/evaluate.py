#!/usr/bin/env python3
"""Run the paper comparison or the real-application workflow."""

import sys

from _bootstrap import run_module


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "real-applications":
        run_module(
            "autoconvexrelax.evaluation.real_applications.run_all", args[1:]
        )
    else:
        run_module("autoconvexrelax.evaluation.compare", args)
