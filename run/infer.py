#!/usr/bin/env python3
"""Run a trained AutoConvexRelax policy on saved QCQP instances."""

import sys

from _bootstrap import run_module


if __name__ == "__main__":
    run_module("autoconvexrelax.inference", sys.argv[1:])
