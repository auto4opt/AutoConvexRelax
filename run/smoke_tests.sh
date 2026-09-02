#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" tests/test_qcr_cases.py
"$PYTHON_BIN" -m pytest -q \
  tests/test_remove_fraction.py \
  tests/test_global_actions.py \
  tests/test_baseline_heuristics.py \
  tests/test_filter_fraction.py \
  tests/test_generate_hard_fraction.py \
  tests/test_summarize_fraction.py
