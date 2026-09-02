#!/bin/bash
#SBATCH -J APA-eval-multiseed
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

EVAL_SEEDS_STR="${APA_EVAL_SEEDS:-42 52 62 72 82 92 102 112}"
read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_STR"

STAGE2_SUBDIR="${APA_STAGE2_SUBDIR:-stage2_finetune_1200}"

DATA="${APA_EVAL_DATA:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
ROOT_CACHE="${APA_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
SCIP_ROOT_CACHE="${APA_SCIP_ROOT_CACHE:-$PROJ/outputs/data/root_lb_cache_scip_root_only.json}"

TIME_LIMIT="${APA_EVAL_TIME_LIMIT:-60}"
SCIP_TIME_LIMIT="${APA_SCIP_TIME_LIMIT:-60}"
SORT_KEY="${APA_SUMMARIZE_SORT:-pct}"

OUT_ROOT="${APA_EVAL_OUT_ROOT:-$PROJ/outputs/logs/multiseed_eval}"
RUNNER_ROOT="$OUT_ROOT/compare_all_basic_baselines"
BASELINE_CACHE_JSON="${APA_BASELINE_CACHE_JSON:-$OUT_ROOT/baseline_cache.json}"

mkdir -p "$RUNNER_ROOT"

echo "[INFO] ================================================"
echo "[INFO] Multi-seed compare-all eval (RL once + mccormick + sdp)"
echo "[INFO] EVAL_SEEDS=${EVAL_SEEDS[*]}"
echo "[INFO] DATA=$DATA"
echo "[INFO] DATASET_TAG=$DATASET_TAG"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] BASELINE_CACHE_JSON=$BASELINE_CACHE_JSON"

RUNNER_EXTRA_ARGS=()
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

for SEED in "${EVAL_SEEDS[@]}"; do
  CKPT="$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/latest.pt"
  SPLIT_JSON="$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/${DATASET_TAG}_split_indices.json"
  SEED_OUT_DIR="$RUNNER_ROOT/seed_${SEED}"

  echo "[INFO] ================================================"
  echo "[INFO] Running compare-all runner for seed=$SEED"
  echo "[INFO] CKPT=$CKPT"
  echo "[INFO] SPLIT_JSON=$SPLIT_JSON"
  echo "[INFO] SAVE_DIR=$SEED_OUT_DIR"

  [[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }
  [[ -f "$SPLIT_JSON" ]] || { echo "[FATAL] split json not found: $SPLIT_JSON"; exit 11; }

  mkdir -p "$SEED_OUT_DIR"

  python run/evaluate.py \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --dataset_tag "$DATASET_TAG" \
    --seed "$SEED" \
    --n_problems -1 \
    --sample head \
    --time_limit "$TIME_LIMIT" \
    --scip_time_limit "$SCIP_TIME_LIMIT" \
    --split_json "$SPLIT_JSON" \
    --out_json_name "eval_compare_all.json" \
    --baseline_cache_json "$BASELINE_CACHE_JSON" \
    --save_dir "$SEED_OUT_DIR" \
    "${RUNNER_EXTRA_ARGS[@]}"

  python run/analyze.py summarize \
    --dir "$SEED_OUT_DIR" \
    --json "$SEED_OUT_DIR/eval_compare_all.json" \
    --out_csv "$SEED_OUT_DIR/summary.csv" \
    --sort "$SORT_KEY" | tee "$SEED_OUT_DIR/summary.txt"

  cat > "$SEED_OUT_DIR/meta.txt" <<EOF
mode=runner
seed=${SEED}
ckpt=${CKPT}
split_json=${SPLIT_JSON}
data=${DATA}
dataset_tag=${DATASET_TAG}
time_limit=${TIME_LIMIT}
scip_time_limit=${SCIP_TIME_LIMIT}
summary_sort=${SORT_KEY}
json=eval_compare_all.json
compare=rl_vs_gurobi_scip_vs_mccormick_sdp
baseline_cache_json=${BASELINE_CACHE_JSON}
EOF
done

echo "[INFO] ================================================"
echo "[INFO] Aggregating multi-seed outputs under $RUNNER_ROOT"

python run/analyze.py multiseed \
  --root_dir "$RUNNER_ROOT" \
  --out_dir "$RUNNER_ROOT"

echo "[INFO] Finished. Outputs are under $OUT_ROOT"
