#!/bin/bash
#SBATCH -J APA-frac-hard-cpu
#SBATCH -p cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#
# One-command hard fractional-QCQP pipeline for the paper demo.
#
# Run from the repository root on a SLURM cluster:
#   sbatch run/slurm/sbatch_eval_fraction_hard_full.sh
#
# Outputs:
#   generated candidates:
#     vector_fraction_hard_candidates_60_seed42.pkl
#   full compare-all candidate rows:
#     outputs/logs/fraction_hard_candidates_60_seed_42_checkpoint_150_random_v2_cpu/eval_compare_fraction_hard_candidates.json
#   selected valid demo rows:
#     outputs/logs/fraction_hard_candidates_60_seed_42_checkpoint_150_random_v2_cpu/eval_compare_fraction_hard_valid15.json
#   metric summary:
#     outputs/logs/fraction_hard_candidates_60_seed_42_checkpoint_150_random_v2_cpu/eval_compare_fraction_hard_valid15_metrics.json
#   LaTeX table rows:
#     outputs/logs/fraction_hard_candidates_60_seed_42_checkpoint_150_random_v2_cpu/eval_compare_fraction_hard_valid15_table_rows.tex

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
CKPT="${APA_CKPT:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/checkpoint_150.pt}"

HARD_REPEAT="${APA_FRACTION_HARD_REPEAT:-12}"
CANDIDATE_N="${APA_FRACTION_CANDIDATE_N:-60}"
TARGET_VALID_N="${APA_FRACTION_TARGET_VALID_N:-15}"
GEN_SEED="${APA_FRACTION_GEN_SEED:-42}"
HARD_DATA="${APA_FRACTION_HARD_DATA:-$PROJ/outputs/data/vector_fraction_hard_candidates_${CANDIDATE_N}_seed${GEN_SEED}.pkl}"

DATASET_TAG="${APA_DATASET_TAG:-FRACTION_HARD_CANDIDATES_${CANDIDATE_N}_SEED${GEN_SEED}}"
SAVE_DIR="${APA_SAVE_DIR:-$PROJ/outputs/logs/fraction_hard_candidates_${CANDIDATE_N}_seed_${SEED}_checkpoint_150_random_v2_cpu}"
OUT_JSON_NAME="${APA_OUT_JSON_NAME:-eval_compare_fraction_hard_candidates.json}"
RESUME_FROM_JSON="${APA_RESUME_FROM_JSON:-$PROJ/outputs/logs/fraction_hard_candidates_${CANDIDATE_N}_seed_${SEED}_checkpoint_150/eval_compare_fraction_hard_candidates.json}"
VALID_JSON_NAME="${APA_VALID_JSON_NAME:-eval_compare_fraction_hard_valid${TARGET_VALID_N}.json}"
VALID_SUMMARY_NAME="${APA_VALID_SUMMARY_NAME:-eval_compare_fraction_hard_valid${TARGET_VALID_N}_summary.json}"
METRICS_JSON_NAME="${APA_METRICS_JSON_NAME:-eval_compare_fraction_hard_valid${TARGET_VALID_N}_metrics.json}"
TABLE_ROWS_NAME="${APA_TABLE_ROWS_NAME:-eval_compare_fraction_hard_valid${TARGET_VALID_N}_table_rows.tex}"

N_PROBLEMS="${APA_N_PROBLEMS:--1}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"
STRUCTURE_K_MIN="${APA_STRUCTURE_K_MIN:-2}"
STRUCTURE_TAU_DENSITY="${APA_STRUCTURE_TAU_DENSITY:-0.25}"
RANDOM_ROLLOUTS="${APA_RANDOM_ROLLOUTS:-1}"

ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_fraction_hard_${CANDIDATE_N}_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_fraction_hard_${CANDIDATE_N}_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/fraction_hard_baseline_cache_random_v2_cpu.json}"

echo "[INFO] ================================================"
echo "[INFO] Hard fraction full pipeline"
echo "[INFO] SEED=$SEED"
echo "[INFO] GEN_SEED=$GEN_SEED"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] HARD_REPEAT=$HARD_REPEAT CANDIDATE_N=$CANDIDATE_N TARGET_VALID_N=$TARGET_VALID_N"
echo "[INFO] HARD_DATA=$HARD_DATA"
echo "[INFO] SAVE_DIR=$SAVE_DIR"
echo "[INFO] RANDOM_ROLLOUTS=$RANDOM_ROLLOUTS"
echo "[INFO] Heuristic: k_min=$STRUCTURE_K_MIN tau_density=$STRUCTURE_TAU_DENSITY"
mkdir -p "$SAVE_DIR"

if [[ ! -f "$SAVE_DIR/$OUT_JSON_NAME" && -f "$RESUME_FROM_JSON" ]]; then
  echo "[INFO] Seeding resume file from previous results: $RESUME_FROM_JSON"
  cp "$RESUME_FROM_JSON" "$SAVE_DIR/$OUT_JSON_NAME"
fi

[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }

EXPECTED_N=$((HARD_REPEAT * 5))
if [[ "$EXPECTED_N" -ne "$CANDIDATE_N" ]]; then
  echo "[WARN] CANDIDATE_N=$CANDIDATE_N but HARD_REPEAT*5=$EXPECTED_N; using generated problem count from HARD_REPEAT"
fi

echo "[INFO] Generating hard fraction candidates"
python -u run/prepare_data.py hard-fraction \
  --output "$HARD_DATA" \
  --num-repeat "$HARD_REPEAT" \
  --seed "$GEN_SEED"
[[ -f "$HARD_DATA" ]] || { echo "[FATAL] hard fraction data not found after generation: $HARD_DATA"; exit 11; }
echo "[INFO] Starting runner_compare_all at $(date)"

RUNNER_EXTRA_ARGS=(--no_split --no_case_pkl --baseline_cache_json "$BASELINE_CACHE_JSON")
if [[ -f "$ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--root_lb_cache "$ROOT_CACHE")
else
  echo "[WARN] hard fraction Gurobi root cache not found, runner will proceed without it: $ROOT_CACHE"
fi

if [[ -f "$SCIP_ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--scip_root_cache "$SCIP_ROOT_CACHE")
else
  echo "[WARN] hard fraction SCIP root cache not found, runner will proceed without it: $SCIP_ROOT_CACHE"
fi

python run/evaluate.py \
  --ckpt "$CKPT" \
  --data "$HARD_DATA" \
  --dataset_tag "$DATASET_TAG" \
  --seed "$SEED" \
  --device cpu \
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
  --required-baselines mccormick sdp structure random \
  --prefer-rl-better-than mccormick sdp structure random

echo "[DONE] wrote $SAVE_DIR/$VALID_JSON_NAME"
echo "[DONE] wrote $SAVE_DIR/$VALID_SUMMARY_NAME"

python run/analyze.py fraction \
  --input "$SAVE_DIR/$VALID_JSON_NAME" \
  --summary-json "$SAVE_DIR/$METRICS_JSON_NAME" \
  --latex "$SAVE_DIR/$TABLE_ROWS_NAME" \
  --baselines mccormick sdp structure random

echo "[DONE] wrote $SAVE_DIR/$METRICS_JSON_NAME"
echo "[DONE] wrote $SAVE_DIR/$TABLE_ROWS_NAME"
echo "[INFO] Hard fraction full pipeline complete"
