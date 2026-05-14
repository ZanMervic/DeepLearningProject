#!/bin/bash
#SBATCH --job-name=yolo26_eval
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="/d/hpc/projects/FRI/zm3587/dl"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/dl_v100/bin/python"

# Positional args with defaults
MODEL="${1:-$PROJECT_ROOT/runs/person_detection/yolo26s_crowdhuman_960/weights/best.pt}"
IMGSZ="${2:-960}"
DATA="${3:-$PROJECT_ROOT/datasets/crowdhuman/data.yaml}"
BATCH="${4:-16}"
NAME="${5:-eval_$(basename "$(dirname "$(dirname "$MODEL")")")_on_$(basename "$(dirname "$DATA")")_${IMGSZ}}"

PROJECT="$PROJECT_ROOT/runs/person_detection_eval"
OUT="$PROJECT_ROOT/results/${NAME}.csv"
JSON_OUT="$PROJECT_ROOT/results/${NAME}.json"

export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$PROJECT_ROOT/slurm"
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT"

cd "$PROJECT_ROOT"

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Image size: $IMGSZ"
echo "Batch: $BATCH"
echo "Name: $NAME"
echo "Output CSV: $OUT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON_BIN" - <<'PY'
import os
import torch
import ultralytics

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Ultralytics:", ultralytics.__version__)
PY

"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/hpc_yolo_eval.py" \
  --model "$MODEL" \
  --data "$DATA" \
  --name "$NAME" \
  --project "$PROJECT" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device 0 \
  --workers 8 \
  --out "$OUT" \
  --json-out "$JSON_OUT" \
  --plots

echo "Job finished on $(date)"


# How to run this:
# sbatch -J eval_default sbatch/eval_single.sbatch \
#   /d/hpc/projects/FRI/zm3587/dl/runs/person_detection/yolo26n_crowdhuman_960/weights/best.pt \ <- override model
#   960 \ <- image size
#   /d/hpc/projects/FRI/zm3587/dl/datasets/crowdhuman/data.yaml \ <- dataset
#   16 \ <- batch
#   eval_yolo26n_960_crowdhuman_on_crowdhuman <- eval name