#!/bin/bash
#SBATCH -J APA-frac-subset
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
#   sbatch run/slurm/sbatch_eval_fraction_subset.sh
#
# Output:
#   candidate pool:
#     outputs/logs/fraction_subset_candidates_120_seed_42_checkpoint_800/eval_compare_fraction_candidates.json
#   valid 50 rows:
#     outputs/logs/fraction_subset_candidates_120_seed_42_checkpoint_800/eval_compare_fraction_valid50.json
#
# The output JSON uses the same per-instance metric fields as QCQP compare-all:
#   rl_lb, baseline_mccormick_lb, baseline_sdp_lb,
#   baseline_structure_lb, baseline_random_lb,
#   rl_minus_*, baseline_*_added_vars, baseline_*_added_cons, baseline_*_added_nnz.

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
CKPT="${APA_CKPT:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/checkpoint_800.pt}"

FULL_DATA="${APA_FRACTION_FULL_DATA:-$PROJ/outputs/data/vector_finetune_fraction.pkl}"
TARGET_VALID_N="${APA_FRACTION_TARGET_VALID_N:-50}"
CANDIDATE_N="${APA_FRACTION_CANDIDATE_N:-120}"
SUBSET_SEED="${APA_FRACTION_SUBSET_SEED:-42}"
SUBSET_DATA="${APA_FRACTION_SUBSET_DATA:-$PROJ/outputs/data/vector_fraction_eval_candidates_${CANDIDATE_N}_seed${SUBSET_SEED}.pkl}"

DATASET_TAG="${APA_DATASET_TAG:-FRACTION_CANDIDATES_${CANDIDATE_N}_SEED${SUBSET_SEED}}"
SAVE_DIR="${APA_SAVE_DIR:-$PROJ/outputs/logs/fraction_subset_candidates_${CANDIDATE_N}_seed_${SEED}_checkpoint_800}"
OUT_JSON_NAME="${APA_OUT_JSON_NAME:-eval_compare_fraction_candidates.json}"
VALID_JSON_NAME="${APA_VALID_JSON_NAME:-eval_compare_fraction_valid${TARGET_VALID_N}.json}"
VALID_SUMMARY_NAME="${APA_VALID_SUMMARY_NAME:-eval_compare_fraction_valid${TARGET_VALID_N}_summary.json}"

N_PROBLEMS="${APA_N_PROBLEMS:--1}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"
STRUCTURE_K_MIN="${APA_STRUCTURE_K_MIN:-4}"
STRUCTURE_TAU_DENSITY="${APA_STRUCTURE_TAU_DENSITY:-0.75}"
RANDOM_ROLLOUTS="${APA_RANDOM_ROLLOUTS:-1}"

ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_fraction_candidates_${CANDIDATE_N}_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_fraction_candidates_${CANDIDATE_N}_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/fraction_subset_baseline_cache.json}"

echo "[INFO] ================================================"
echo "[INFO] Fraction subset eval with learned policy + baselines"
echo "[INFO] SEED=$SEED"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] FULL_DATA=$FULL_DATA"
echo "[INFO] SUBSET_DATA=$SUBSET_DATA"
echo "[INFO] CANDIDATE_N=$CANDIDATE_N TARGET_VALID_N=$TARGET_VALID_N SUBSET_SEED=$SUBSET_SEED"
echo "[INFO] SAVE_DIR=$SAVE_DIR"
echo "[INFO] RANDOM_ROLLOUTS=$RANDOM_ROLLOUTS"
echo "[INFO] H1: k_min=$STRUCTURE_K_MIN tau_density=$STRUCTURE_TAU_DENSITY"

[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }

if [[ ! -f "$FULL_DATA" ]]; then
  echo "[INFO] Fraction data not found; generating with run/prepare_data.py generate fraction"
  python run/prepare_data.py generate fraction
fi
[[ -f "$FULL_DATA" ]] || { echo "[FATAL] fraction data not found after generation: $FULL_DATA"; exit 11; }

if [[ ! -f "$SUBSET_DATA" ]]; then
  echo "[INFO] Fraction subset not found; sampling true fractional problems"
  python run/analyze.py fraction-subset \
    --input "$FULL_DATA" \
    --output "$SUBSET_DATA" \
    --n "$CANDIDATE_N" \
    --seed "$SUBSET_SEED"
fi
[[ -f "$SUBSET_DATA" ]] || { echo "[FATAL] fraction subset not found after sampling: $SUBSET_DATA"; exit 12; }

RUNNER_EXTRA_ARGS=(--no_split --no_case_pkl --baseline_cache_json "$BASELINE_CACHE_JSON")
if [[ -f "$ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--root_lb_cache "$ROOT_CACHE")
else
  echo "[WARN] fraction Gurobi root cache not found, runner will proceed without it: $ROOT_CACHE"
fi

if [[ -f "$SCIP_ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--scip_root_cache "$SCIP_ROOT_CACHE")
else
  echo "[WARN] fraction SCIP root cache not found, runner will proceed without it: $SCIP_ROOT_CACHE"
fi

python run/evaluate.py \
  --ckpt "$CKPT" \
  --data "$SUBSET_DATA" \
  --dataset_tag "$DATASET_TAG" \
  --seed "$SEED" \
  --n_problems "$N_PROBLEMS" \
  --sample "$SAMPLE" \
  --time_limit "$TIME_LIMIT" \
  --scip_time_limit "$SCIP_TIME_LIMIT" \
  --save_dir "$SAVE_DIR" \
  --out_json_name "$OUT_JSON_NAME" \
  --baseline_modes mccormick sdp structure random \
  --structure_k_min "$STRUCTURE_K_MIN" \
  --structure_tau_density "$STRUCTURE_TAU_DENSITY" \
  --random_rollouts "$RANDOM_ROLLOUTS" \
  --skip_failed_baselines \
  --skip_failed_instances \
  "${RUNNER_EXTRA_ARGS[@]}"

echo "[DONE] wrote $SAVE_DIR/$OUT_JSON_NAME"

python run/analyze.py filter-fraction \
  --input "$SAVE_DIR/$OUT_JSON_NAME" \
  --output "$SAVE_DIR/$VALID_JSON_NAME" \
  --summary "$SAVE_DIR/$VALID_SUMMARY_NAME" \
  --target "$TARGET_VALID_N" \
  --required-baselines mccormick sdp structure random

echo "[DONE] wrote $SAVE_DIR/$VALID_JSON_NAME"
