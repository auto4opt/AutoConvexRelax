#!/bin/bash
#SBATCH -J APA-ckpt-sweep
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch, sys
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("GPU OK:", torch.cuda.is_available(), "num:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available inside job"
PY

# ===== sweep controls =====
# Quick diagnostic example:
#   sbatch --export=ALL,APA_SWEEP_SEEDS="42",APA_CKPT_TAGS="checkpoint_100.pt checkpoint_300.pt checkpoint_600.pt latest.pt",APA_N_PROBLEMS=80 sbatch_eval_checkpoint_sweep.sh
#
# Stage-1 checkpoint on the same stage-2 evaluation split:
#   sbatch --export=ALL,APA_SWEEP_SEEDS="42",APA_CKPT_SUBDIR="stage1_no_lb_1600",APA_SPLIT_SUBDIR="stage2_finetune_1200",APA_CKPT_TAGS="checkpoint_50.pt checkpoint_100.pt checkpoint_150.pt checkpoint_200.pt latest.pt",APA_SWEEP_OUT_ROOT="$PROJ/outputs/logs/checkpoint_sweep_stage1",APA_N_PROBLEMS=80 sbatch_eval_checkpoint_sweep.sh
#
# Full-ish one-seed diagnostic:
#   sbatch --export=ALL,APA_SWEEP_SEEDS="42",APA_N_PROBLEMS=-1 sbatch_eval_checkpoint_sweep.sh
#
# Multi-seed after trend is clear:
#   sbatch --export=ALL,APA_SWEEP_SEEDS="42 52 62 72 82",APA_N_PROBLEMS=-1 sbatch_eval_checkpoint_sweep.sh

SWEEP_SEEDS_STR="${APA_SWEEP_SEEDS:-42}"
read -r -a SWEEP_SEEDS <<< "$SWEEP_SEEDS_STR"

CKPT_TAGS_STR="${APA_CKPT_TAGS:-checkpoint_100.pt checkpoint_300.pt checkpoint_600.pt latest.pt}"
read -r -a CKPT_TAGS <<< "$CKPT_TAGS_STR"

CKPT_SUBDIR="${APA_CKPT_SUBDIR:-${APA_STAGE2_SUBDIR:-stage2_finetune_1200}}"
SPLIT_SUBDIR="${APA_SPLIT_SUBDIR:-stage2_finetune_1200}"
DATA="${APA_EVAL_DATA:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_scip_root_only.json}"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$PROJ/outputs/logs/multiseed_eval/baseline_cache.json}"

N_PROBLEMS="${APA_N_PROBLEMS:-80}"
SAMPLE="${APA_SAMPLE:-head}"
TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"
SORT_KEY="${APA_SUMMARIZE_SORT:-pct}"

OUT_ROOT="${APA_SWEEP_OUT_ROOT:-$PROJ/outputs/logs/checkpoint_sweep}"
mkdir -p "$OUT_ROOT"

RUNNER_EXTRA_ARGS=()
if [[ -f "$ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--root_lb_cache "$ROOT_CACHE")
else
  echo "[WARN] root cache not found: $ROOT_CACHE"
fi

if [[ -f "$SCIP_ROOT_CACHE" ]]; then
  RUNNER_EXTRA_ARGS+=(--scip_root_cache "$SCIP_ROOT_CACHE")
else
  echo "[WARN] SCIP root cache not found: $SCIP_ROOT_CACHE"
fi

echo "[INFO] ================================================"
echo "[INFO] Checkpoint sweep"
echo "[INFO] SWEEP_SEEDS=${SWEEP_SEEDS[*]}"
echo "[INFO] CKPT_TAGS=${CKPT_TAGS[*]}"
echo "[INFO] CKPT_SUBDIR=$CKPT_SUBDIR"
echo "[INFO] SPLIT_SUBDIR=$SPLIT_SUBDIR"
echo "[INFO] N_PROBLEMS=$N_PROBLEMS SAMPLE=$SAMPLE"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] BASELINE_CACHE_JSON=$BASELINE_CACHE_JSON"

for SEED in "${SWEEP_SEEDS[@]}"; do
  SPLIT_JSON="$PROJ/outputs/train_runs/seed_${SEED}/${SPLIT_SUBDIR}/${DATASET_TAG}_split_indices.json"
  [[ -f "$SPLIT_JSON" ]] || { echo "[FATAL] split json not found: $SPLIT_JSON"; exit 11; }

  for CKPT_TAG in "${CKPT_TAGS[@]}"; do
    CKPT="$PROJ/outputs/train_runs/seed_${SEED}/${CKPT_SUBDIR}/checkpoints/${CKPT_TAG}"
    CKPT_LABEL="${CKPT_TAG%.pt}"
    SEED_CKPT_OUT_DIR="$OUT_ROOT/seed_${SEED}/${CKPT_LABEL}"

    echo "[INFO] ================================================"
    echo "[INFO] seed=$SEED ckpt=$CKPT_TAG"
    echo "[INFO] CKPT=$CKPT"
    echo "[INFO] SPLIT_JSON=$SPLIT_JSON"
    echo "[INFO] SAVE_DIR=$SEED_CKPT_OUT_DIR"

    if [[ ! -f "$CKPT" ]]; then
      echo "[WARN] checkpoint not found, skip: $CKPT"
      continue
    fi

    mkdir -p "$SEED_CKPT_OUT_DIR"

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
      --out_json_name "eval_compare_all.json" \
      --baseline_cache_json "$BASELINE_CACHE_JSON" \
      --save_dir "$SEED_CKPT_OUT_DIR" \
      "${RUNNER_EXTRA_ARGS[@]}" \
      2>&1 | tee "$SEED_CKPT_OUT_DIR/runner_stdout.log"

    python run/analyze.py summarize \
      --dir "$SEED_CKPT_OUT_DIR" \
      --json "$SEED_CKPT_OUT_DIR/eval_compare_all.json" \
      --out_csv "$SEED_CKPT_OUT_DIR/summary.csv" \
      --sort "$SORT_KEY" | tee "$SEED_CKPT_OUT_DIR/summary.txt"

    cat > "$SEED_CKPT_OUT_DIR/meta.txt" <<EOF
mode=checkpoint_sweep
seed=${SEED}
ckpt_tag=${CKPT_TAG}
ckpt=${CKPT}
ckpt_subdir=${CKPT_SUBDIR}
split_subdir=${SPLIT_SUBDIR}
split_json=${SPLIT_JSON}
data=${DATA}
dataset_tag=${DATASET_TAG}
n_problems=${N_PROBLEMS}
sample=${SAMPLE}
time_limit=${TIME_LIMIT}
scip_time_limit=${SCIP_TIME_LIMIT}
summary_sort=${SORT_KEY}
json=eval_compare_all.json
baseline_cache_json=${BASELINE_CACHE_JSON}
EOF
  done
done

python run/analyze.py checkpoints --root "$OUT_ROOT" --out "$OUT_ROOT/checkpoint_sweep_summary.csv"

echo "[INFO] Finished. Sweep outputs are under $OUT_ROOT"
echo "[INFO] Summary: $OUT_ROOT/checkpoint_sweep_summary.csv"
