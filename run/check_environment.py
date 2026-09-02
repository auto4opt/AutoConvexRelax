#!/usr/bin/env python3
"""Check Python dependencies and optional solver bindings."""

import sys

from _bootstrap import run_module


if __name__ == "__main__":
    run_module("autoconvexrelax.tools.check_environment", sys.argv[1:])
