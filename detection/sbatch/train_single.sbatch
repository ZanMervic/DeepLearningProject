#!/bin/bash
#SBATCH --job-name=yolo26_train
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[01-10]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/d/hpc/projects/FRI/zm3587/dl"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/deeplearning/bin/python"

# Positional args with defaults.
#   1 MODEL
#   2 IMGSZ
#   3 DATA
#   4 BATCH
#   5 NAME
#   6 EPOCHS
#   7 PATIENCE
#   8 FREEZE
MODEL="${1:-yolo26n.pt}"
IMGSZ="${2:-640}"
DATA="${3:-$PROJECT_ROOT/datasets/crowdhuman/data.yaml}"
BATCH="${4:-16}"
NAME="${5:-train_$(basename "$MODEL" .pt)_$(basename "$(dirname "$DATA")")_${IMGSZ}}"
EPOCHS="${6:-50}"
PATIENCE="${7:-10}"
FREEZE="${8:-0}"

PROJECT="$PROJECT_ROOT/runs/person_detection"

export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$PROJECT_ROOT/slurm"
mkdir -p "$PROJECT"

cd "$PROJECT_ROOT"

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Image size: $IMGSZ"
echo "Batch: $BATCH"
echo "Name: $NAME"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Freeze: $FREEZE"
echo "Project: $PROJECT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON_BIN" - <<'PY'
import os
import sys
import torch
import ultralytics

print("Python:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)
PY

"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/hpc_yolo_train.py" \
  --model "$MODEL" \
  --data "$DATA" \
  --name "$NAME" \
  --project "$PROJECT" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device 0 \
  --workers 8 \
  --patience "$PATIENCE" \
  --freeze "$FREEZE"

echo "Job finished on $(date)"


# How to run (with all overrides):
# Note: these are positional parameters, so if we want to change the
# n-th one we need to provide all n-1 parameters before it!
# sbatch -J train_yolo26s_ch960 sbatch/train_single_param.sbatch \
#   yolo26s.pt \
#   960 \
#   /d/hpc/projects/FRI/zm3587/dl/datasets/crowdhuman/data.yaml \
#   16 \
#   yolo26s_crowdhuman_960 \
#   50 \
#   10 \
#   0