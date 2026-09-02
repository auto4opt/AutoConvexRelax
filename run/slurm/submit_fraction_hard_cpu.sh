#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# sbatch reads SBATCH_* variables before the batch script starts. Clear common
# resource overrides here so a stale GPU/QoS login environment cannot affect
# this CPU submission.
unset SBATCH_QOS
unset SBATCH_ACCOUNT
unset SBATCH_PARTITION
unset SBATCH_GRES

exec sbatch "$SCRIPT_DIR/sbatch_eval_fraction_hard_full.sh" "$@"
