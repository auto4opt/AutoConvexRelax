#!/bin/bash
#SBATCH -J APA-eval-one
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

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

python - <<'PY'
import torch, sys
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("GPU OK:", torch.cuda.is_available(), "num:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available inside job"
PY

SEED="${APA_SEED:-42}"
STAGE2_SUBDIR="${APA_STAGE2_SUBDIR:-stage2_finetune_1200_costaware}"
CKPT_TAG="${APA_CKPT_TAG:-checkpoint_100.pt}"
DATA="${APA_EVAL_DATA:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
N_PROBLEMS="${APA_N_PROBLEMS:--1}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"

CKPT="${APA_CKPT:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/${CKPT_TAG}}"
SPLIT_JSON="${APA_SPLIT_JSON:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/${DATASET_TAG}_split_indices.json}"
SAVE_DIR="${APA_SAVE_DIR:-$PROJ/outputs/logs/eval_seed_${SEED}_minimal}"
OUT_JSON_NAME="${APA_OUT_JSON_NAME:-eval_compare_all.json}"

ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/multiseed_eval/baseline_cache.json}"

echo "[INFO] ================================================"
echo "[INFO] Single-seed minimal eval"
echo "[INFO] SEED=$SEED"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] DATA=$DATA"
echo "[INFO] SPLIT_JSON=$SPLIT_JSON"
echo "[INFO] SAVE_DIR=$SAVE_DIR"
echo "[INFO] OUT_JSON_NAME=$OUT_JSON_NAME"
echo "[INFO] N_PROBLEMS=$N_PROBLEMS SAMPLE=$SAMPLE"
echo "[INFO] BASELINE_CACHE_JSON=$BASELINE_CACHE_JSON"

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
  "${RUNNER_EXTRA_ARGS[@]}"

echo "[DONE] wrote $SAVE_DIR/$OUT_JSON_NAME"
