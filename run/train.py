#!/usr/bin/env python3
"""Unified entry point for the two training stages."""

import argparse
import sys

from _bootstrap import run_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args, remaining = parser.parse_known_args()
    module = f"autoconvexrelax.training.stage{args.stage}"
    run_module(module, remaining)


if __name__ == "__main__":
    main()
