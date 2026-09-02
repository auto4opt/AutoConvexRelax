#!/bin/bash
#SBATCH -J APA-real-app
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

module load anaconda/24.1.2
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh || true
eval "$(conda shell.bash hook)" || true
conda activate APA-gpu || { echo "[FATAL] conda activate APA-gpu failed"; exit 2; }

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

SEED="${APA_REAL_SEED:-42}"
STAGE2_SUBDIR="${APA_STAGE2_SUBDIR:-stage2_finetune_1200}"
CKPT="${APA_REAL_CKPT:-$PROJ/outputs/train_runs/seed_${SEED}/${STAGE2_SUBDIR}/checkpoints/latest.pt}"
OUT_DIR="${APA_REAL_OUT_DIR:-$PROJ/outputs/real_applications/runs/seed_${SEED}}"

echo "[INFO] CKPT=$CKPT"
echo "[INFO] OUT_DIR=$OUT_DIR"
[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 10; }

python run/evaluate.py real-applications \
  --ckpt "$CKPT" \
  --device cuda \
  --exact_time_limit 60 \
  --relax_time_limit 30 \
  --out_dir "$OUT_DIR"
