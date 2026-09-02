#!/bin/bash
#SBATCH -J APA-stage2-v100
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

# ===== single-seed stage-2 controls =====
SEED="${APA_SEED:-42}"
APA_STAGE2_UPDATES="${APA_STAGE2_UPDATES:-200}"
APA_STAGE2_SOLVER_PROB="${APA_STAGE2_SOLVER_PROB:-0.2}"
APA_FORCE_FRESH="${APA_FORCE_FRESH:-1}"

RUN_ROOT="$PROJ/outputs/train_runs/seed_${SEED}"
STAGE1_DIR="${APA_STAGE1_DIR:-$RUN_ROOT/stage1_no_lb_1600}"
STAGE2_SUBDIR="${APA_STAGE2_SUBDIR:-stage2_finetune_1200_costaware}"
STAGE2_DIR="${APA_STAGE2_DIR:-$RUN_ROOT/$STAGE2_SUBDIR}"
STAGE1_CKPT="${APA_STAGE1_CKPT:-$STAGE1_DIR/checkpoints/latest.pt}"

if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "[FATAL] stage1 checkpoint not found: $STAGE1_CKPT"
  echo "        This script is stage2-only and will not modify/train stage1."
  exit 10
fi

mkdir -p "$STAGE2_DIR"

echo "[INFO] seed=$SEED"
echo "[INFO] stage1 checkpoint=$STAGE1_CKPT"
echo "[INFO] stage2 dir=$STAGE2_DIR"
echo "[INFO] stage2 updates=$APA_STAGE2_UPDATES solver_prob=$APA_STAGE2_SOLVER_PROB"

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
export APA_FINETUNE_FILE="${APA_FINETUNE_FILE:-$PROJ/outputs/data/vector_finetune_HYBRID_MIX_1200.pkl}"
export APA_DATASET_TAG="${APA_DATASET_TAG:-HYBRID_MIX_1200}"
export APA_ROOT_LB_CACHE_PATH="${APA_ROOT_LB_CACHE_PATH:-$PROJ/outputs/data/root_lb_cache_HYBRID_MIX_1200_root_only.json}"
export APA_GROUP_UPDATES="$APA_STAGE2_UPDATES"
export APA_LOG_DIR="$STAGE2_DIR"
export APA_CKPT_SAVE_DIR="$STAGE2_DIR/checkpoints"
export APA_LOAD_WEIGHTS_FROM="$STAGE1_CKPT"

# Solver reward and cost-aware lower-bound shaping.
export APA_SOLVER_REWARD_PROB="$APA_STAGE2_SOLVER_PROB"
export APA_SOLVER_REWARD_COMPLEXITY_DECAY="${APA_SOLVER_REWARD_COMPLEXITY_DECAY:-1.0}"
export APA_SOLVER_REWARD_SIZE_PENALTY="${APA_SOLVER_REWARD_SIZE_PENALTY:-0.25}"
export APA_SOLVER_REWARD_SIZE_PREMIUM="${APA_SOLVER_REWARD_SIZE_PREMIUM:-0.08}"

python run/train.py --stage 2

echo "[DONE] stage2-only seed=$SEED finished."
