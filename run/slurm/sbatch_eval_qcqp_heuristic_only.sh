#!/bin/bash
#SBATCH -J APA-qcqp-h1-only
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# Run from the repository root on a SLURM cluster:
#   sbatch run/slurm/sbatch_eval_qcqp_heuristic_only.sh
#
# Purpose:
#   Reuse the saved learned-model results from an existing compare-all JSON and
#   compute only the H1 structure heuristic baseline. This does not run random
#   and does not recompute the learned policy trajectory.
#
# Defaults:
#   source result cache:
#     outputs/logs/eval_seed_42_minimal/eval_compare_all.json
#   output:
#     outputs/logs/eval_seed_42_minimal_h1_strict_k2_tau0p25/eval_compare_all.json
#   H1 thresholds:
#     k_min=2, tau_density=0.25
#   cache:
#     outputs/logs/qcqp_h1_strict_baseline_cache.json
#
# By default this script overwrites only the output JSON with the saved learned
# results before running. That prevents stale baseline_structure_* fields from
# being reused when H1 code or thresholds change. Set APA_REUSE_EXISTING_OUTPUT=1
# only if you intentionally want to resume an incomplete strict-H1 run.

set -euo pipefail

PROJ="${APA_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJ"
mkdir -p logs

: "${MOSEKLM_LICENSE_FILE:?Set MOSEKLM_LICENSE_FILE to a license file outside the repository}"
export MOSEKLM_LICENSE_FILE
[[ -f "" ]] || { echo "[FATAL] MOSEK license file not found"; exit 3; }

module load anaconda/24.1.2
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh || true
eval "$(conda shell.bash hook)" || true
conda activate APA-gpu || { echo "[FATAL] conda activate APA-gpu failed"; exit 2; }

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

SEED="${APA_SEED:-42}"
STAGE2_SUBDIR="${APA_STAGE2_SUBDIR:-stage2_finetune_1200_costaware}"
CKPT_TAG="${APA_CKPT_TAG:-checkpoint_150.pt}"
CKPT="${APA_CKPT:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/${CKPT_TAG}}"

DATA="${APA_EVAL_DATA:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
SPLIT_JSON="${APA_SPLIT_JSON:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/${DATASET_TAG}_split_indices.json}"

SOURCE_RESULT_JSON="${APA_SOURCE_RESULT_JSON:-$PROJ/outputs/logs/eval_seed_${SEED}_minimal/eval_compare_all.json}"
STRUCTURE_K_MIN="${APA_STRUCTURE_K_MIN:-2}"
STRUCTURE_TAU_DENSITY="${APA_STRUCTURE_TAU_DENSITY:-0.25}"
STRUCTURE_TAU_TAG="${STRUCTURE_TAU_DENSITY//./p}"
H1_TAG="h1_strict_k${STRUCTURE_K_MIN}_tau${STRUCTURE_TAU_TAG}"
SAVE_DIR="${APA_SAVE_DIR:-$PROJ/outputs/logs/eval_seed_${SEED}_minimal_${H1_TAG}}"
OUT_JSON_NAME="${APA_OUT_JSON_NAME:-eval_compare_all.json}"
OUT_JSON="$SAVE_DIR/$OUT_JSON_NAME"
REUSE_EXISTING_OUTPUT="${APA_REUSE_EXISTING_OUTPUT:-0}"

N_PROBLEMS="${APA_N_PROBLEMS:--1}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"

ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/qcqp_h1_strict_baseline_cache.json}"

echo "[INFO] ================================================"
echo "[INFO] QCQP H1-only eval using saved learned results"
echo "[INFO] SEED=$SEED"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] DATA=$DATA"
echo "[INFO] SPLIT_JSON=$SPLIT_JSON"
echo "[INFO] SOURCE_RESULT_JSON=$SOURCE_RESULT_JSON"
echo "[INFO] SAVE_DIR=$SAVE_DIR"
echo "[INFO] OUT_JSON_NAME=$OUT_JSON_NAME"
echo "[INFO] H1 strict: k_min=$STRUCTURE_K_MIN tau_density=$STRUCTURE_TAU_DENSITY"
echo "[INFO] BASELINE_CACHE_JSON=$BASELINE_CACHE_JSON"
echo "[INFO] REUSE_EXISTING_OUTPUT=$REUSE_EXISTING_OUTPUT"

[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }
[[ -f "$DATA" ]] || { echo "[FATAL] data not found: $DATA"; exit 11; }
[[ -f "$SPLIT_JSON" ]] || { echo "[FATAL] split json not found: $SPLIT_JSON"; exit 12; }
[[ -f "$SOURCE_RESULT_JSON" ]] || { echo "[FATAL] source result json not found: $SOURCE_RESULT_JSON"; exit 13; }

mkdir -p "$SAVE_DIR"
if [[ "$REUSE_EXISTING_OUTPUT" != "1" ]]; then
  echo "[INFO] Seeding output from saved learned results; stale H1 fields will not be reused"
  cp "$SOURCE_RESULT_JSON" "$OUT_JSON"
elif [[ ! -f "$OUT_JSON" ]]; then
  echo "[INFO] Existing-output resume requested but output is absent; seeding from $SOURCE_RESULT_JSON"
  cp "$SOURCE_RESULT_JSON" "$OUT_JSON"
else
  echo "[INFO] Reusing existing output JSON as result resume cache: $OUT_JSON"
fi

RUNNER_EXTRA_ARGS=(--no_case_pkl --baseline_cache_json "$BASELINE_CACHE_JSON")
if [[ -f "$ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--root_lb_cache "$ROOT_CACHE")
else
  echo "[WARN] root cache not found, runner will proceed without it: $ROOT_CACHE"
fi

if [[ -f "$SCIP_ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--scip_root_cache "$SCIP_ROOT_CACHE")
else
  echo "[WARN] SCIP root cache not found, runner will proceed without it: $SCIP_ROOT_CACHE"
fi

python run/evaluate.py \
  --ckpt "$CKPT" \
  --data "$DATA" \
  --dataset_tag "$DATASET_TAG" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --sample "$SAMPLE" \
  --time_limit "$TIME_LIMIT" \
  --scip_time_limit "$SCIP_TIME_LIMIT" \
  --split_json "$SPLIT_JSON" \
  --save_dir "$SAVE_DIR" \
  --out_json_name "$OUT_JSON_NAME" \
  --baseline_modes structure \
  --structure_k_min "$STRUCTURE_K_MIN" \
  --structure_tau_density "$STRUCTURE_TAU_DENSITY" \
  "${RUNNER_EXTRA_ARGS[@]}"

echo "[DONE] wrote $OUT_JSON"
