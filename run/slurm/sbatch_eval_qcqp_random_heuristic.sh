#!/bin/bash
#SBATCH -J APA-qcqp-rand-h1
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
#   sbatch run/slurm/sbatch_eval_qcqp_random_heuristic.sh
#
# Output:
#   outputs/logs/qcqp_random_h1_seed_42_checkpoint_800/eval_compare_all.json
#
# Caching:
#   - Existing rows in the output JSON are reused first; complete rows skip
#     model inference and all baseline solves.
#   - Baseline solves use the shared cache below, so old McCormick/SDP results
#     from previous compare-all runs can be reused.
#
# The JSON keeps the same per-instance metric shape as McCormick/SDP:
#   lower bounds:
#     baseline_mccormick_lb, baseline_sdp_lb,
#     baseline_structure_lb, baseline_random_lb
#   RL improvements:
#     rl_minus_mccormick, rl_minus_sdp,
#     rl_minus_structure, rl_minus_random
#   size growth:
#     baseline_*_added_vars, baseline_*_added_cons, baseline_*_added_nnz
#
# Default controls are intentionally weak/simple:
#   random: one rollout only
#   H1: conservative SDP trigger, k_min=4 and tau_density=0.75

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
DATA="${APA_EVAL_DATA:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
SPLIT_JSON="${APA_SPLIT_JSON:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/${DATASET_TAG}_split_indices.json}"
SAVE_DIR="${APA_SAVE_DIR:-$PROJ/outputs/logs/qcqp_random_h1_seed_${SEED}_checkpoint_800}"
OUT_JSON_NAME="${APA_OUT_JSON_NAME:-eval_compare_all.json}"

N_PROBLEMS="${APA_N_PROBLEMS:--1}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"
STRUCTURE_K_MIN="${APA_STRUCTURE_K_MIN:-4}"
STRUCTURE_TAU_DENSITY="${APA_STRUCTURE_TAU_DENSITY:-0.75}"
RANDOM_ROLLOUTS="${APA_RANDOM_ROLLOUTS:-1}"

ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/multiseed_eval/baseline_cache.json}"

echo "[INFO] ================================================"
echo "[INFO] QCQP single-seed eval with random + H1 baselines"
echo "[INFO] SEED=$SEED"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] DATA=$DATA"
echo "[INFO] SPLIT_JSON=$SPLIT_JSON"
echo "[INFO] SAVE_DIR=$SAVE_DIR"
echo "[INFO] RANDOM_ROLLOUTS=$RANDOM_ROLLOUTS"
echo "[INFO] H1: k_min=$STRUCTURE_K_MIN tau_density=$STRUCTURE_TAU_DENSITY"

[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }
[[ -f "$DATA" ]] || { echo "[FATAL] data not found: $DATA"; exit 11; }
[[ -f "$SPLIT_JSON" ]] || { echo "[FATAL] split json not found: $SPLIT_JSON"; exit 12; }

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
  --baseline_modes mccormick sdp structure random \
  --structure_k_min "$STRUCTURE_K_MIN" \
  --structure_tau_density "$STRUCTURE_TAU_DENSITY" \
  --random_rollouts "$RANDOM_ROLLOUTS" \
  "${RUNNER_EXTRA_ARGS[@]}"

echo "[DONE] wrote $SAVE_DIR/$OUT_JSON_NAME"
