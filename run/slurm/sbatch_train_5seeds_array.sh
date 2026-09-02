#!/bin/bash
#SBATCH -J APA-2stage-5seed
#SBATCH -p v100
#SBATCH --qos=dcgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=0-4%5
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

PROJ="${APA_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJ"
mkdir -p logs

# ===== MOSEK license =====
: "${MOSEKLM_LICENSE_FILE:?Set MOSEKLM_LICENSE_FILE to a license file outside the repository}"
export MOSEKLM_LICENSE_FILE
[[ -f "" ]] || { echo "[FATAL] MOSEK license file not found"; exit 3; }

# ===== env =====
module load anaconda/24.1.2
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh || true
eval "$(conda shell.bash hook)" || true
conda activate APA-gpu || { echo "[FATAL] conda activate APA-gpu failed"; exit 2; }

echo "[INFO] nvidia-smi:"
nvidia-smi

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

# ===== run controls (overridable at sbatch submit time) =====
# Example:
# sbatch --export=ALL,APA_FORCE_FRESH=1,APA_STAGE1_UPDATES=200,APA_STAGE2_UPDATES=800 sbatch_train_5seeds_array.sh
APA_FORCE_FRESH="${APA_FORCE_FRESH:-0}"
APA_STAGE1_UPDATES="${APA_STAGE1_UPDATES:-200}"
APA_STAGE2_UPDATES="${APA_STAGE2_UPDATES:-800}"
APA_STAGE2_SOLVER_PROB="${APA_STAGE2_SOLVER_PROB:-0.2}"

# ===== seed setup =====
SEEDS=(42 52 62 72 82)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="$PROJ/outputs/train_runs/seed_${SEED}"
STAGE1_DIR="$RUN_ROOT/stage1_no_lb_1600"
STAGE2_DIR="$RUN_ROOT/stage2_finetune_1200"
mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"

echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID seed=$SEED"
echo "[INFO] RUN_ROOT=$RUN_ROOT"

# =========================================================
# Stage 1: run/train.py --stage 1 on vector_all_mix_1600.pkl
# =========================================================
if [[ "$APA_FORCE_FRESH" == "1" ]]; then
  export APA_RESUME=0
else
  if [[ -f "$STAGE1_DIR/checkpoints/latest.pt" ]]; then
    export APA_RESUME=1
  else
    export APA_RESUME=0
  fi
fi
echo "[INFO] Stage1 APA_RESUME=$APA_RESUME"

export APA_TRAIN_SEED="$SEED"
export APA_FINETUNE=0
export APA_DATASET_SPLIT_SEED=42
export APA_FINETUNE_FILE="$PROJ/outputs/data/vector_all_mix_1600.pkl"
export APA_DATASET_TAG="HYBRID_MIX_1600"
export APA_GROUP_UPDATES="$APA_STAGE1_UPDATES"
export APA_LOG_DIR="$STAGE1_DIR"
export APA_CKPT_SAVE_DIR="$STAGE1_DIR/checkpoints"
export APA_SOLVER_REWARD_PROB=0.0

python run/train.py --stage 1

STAGE1_CKPT="$STAGE1_DIR/checkpoints/latest.pt"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[FATAL] stage1 checkpoint not found: $STAGE1_CKPT"
  exit 10
fi

# =========================================================
# Stage 2: train.py on vector_finetune_HYBRID_MIX_1200.pkl
# =========================================================
if [[ "$APA_FORCE_FRESH" == "1" ]]; then
  export APA_RESUME=0
else
  if [[ -f "$STAGE2_DIR/checkpoints/latest.pt" ]]; then
    export APA_RESUME=1
  else
    export APA_RESUME=0
  fi
fi
echo "[INFO] Stage2 APA_RESUME=$APA_RESUME"

export APA_TRAIN_SEED="$SEED"
export APA_FINETUNE=1
export APA_DATASET_SPLIT_SEED=42
export APA_FINETUNE_FILE="$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl"
export APA_DATASET_TAG="HYBRID_MIX_1200"
export APA_ROOT_LB_CACHE_PATH="$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json"
export APA_GROUP_UPDATES="$APA_STAGE2_UPDATES"
export APA_LOG_DIR="$STAGE2_DIR"
export APA_CKPT_SAVE_DIR="$STAGE2_DIR/checkpoints"
export APA_LOAD_WEIGHTS_FROM="$STAGE1_CKPT"
# 推荐先降比例加速；如追求最终质量可设回 1.0
export APA_SOLVER_REWARD_PROB="$APA_STAGE2_SOLVER_PROB"

python run/train.py --stage 2

echo "[DONE] seed=$SEED finished."
