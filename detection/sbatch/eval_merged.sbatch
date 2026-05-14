#!/bin/bash
#SBATCH --job-name=yolo_eval_all
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-all-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-all-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="/d/hpc/projects/FRI/zm3587/dl"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/dl_v100/bin/python"

# Positional args with defaults.
# Keep the first arguments analogous to eval_single.sbatch:
#   1 MODEL
#   2 IMGSZ
#   3 DATASET_GROUP     all | main | hard | ch | wp | coco | och | ch_wp | ch_wp_coco | ch_wp_coco_och
#   4 BATCH
#   5 NAME_PREFIX
MODEL="${1:-$PROJECT_ROOT/runs/person_detection/yolo26s_crowdhuman_960/weights/best.pt}"
IMGSZ="${2:-960}"
DATASET_GROUP="${3:-all}"
BATCH="${4:-16}"
NAME_PREFIX="${5:-eval_$(basename "$(dirname "$(dirname "$MODEL")")")_${IMGSZ}_${DATASET_GROUP}}"

PROJECT="$PROJECT_ROOT/runs/person_detection_eval"
OUT="$PROJECT_ROOT/results/${NAME_PREFIX}.csv"
JSON_OUT="$PROJECT_ROOT/results/${NAME_PREFIX}.json"
MD_OUT="$PROJECT_ROOT/results/${NAME_PREFIX}.md"

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
echo "Image size: $IMGSZ"
echo "Dataset group: $DATASET_GROUP"
echo "Batch: $BATCH"
echo "Name prefix: $NAME_PREFIX"
echo "Output CSV: $OUT"
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

"$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/hpc_yolo_eval_all.py" \
  --model "$MODEL" \
  --imgsz "$IMGSZ" \
  --dataset-group "$DATASET_GROUP" \
  --batch "$BATCH" \
  --name-prefix "$NAME_PREFIX" \
  --project-root "$PROJECT_ROOT" \
  --project "$PROJECT" \
  --device 0 \
  --workers 8 \
  --out "$OUT" \
  --json-out "$JSON_OUT" \
  --md-out "$MD_OUT" \
  --plots

echo "Job finished on $(date)"

# Examples:
# sbatch -J eval_yolo26n_ch_960_all sbatch/eval_merged.sbatch /d/hpc/projects/FRI/zm3587/dl/runs/person_detection/yolo26n_crowdhuman_960/weights/best.pt 960 all 16 eval_yolo26n_crowdhuman_960_all

# sbatch -J eval_yolo26n_ch_wp_960_all sbatch/eval_merged.sbatch /d/hpc/projects/FRI/zm3587/dl/runs/person_detection/yolo26n_ch_wp_960/weights/best.pt 960 all 16 eval_yolo26n_ch_wp_960_all

# sbatch -J eval_yolo26n_ch_wp_coco_960_all sbatch/eval_merged.sbatch /d/hpc/projects/FRI/zm3587/dl/runs/person_detection/yolo26n_ch_wp_coco_960/weights/best.pt 960 all 16 eval_yolo26n_ch_wp_coco_960_all

# sbatch -J eval_yolo26n_ch_wp_coco_och_960_all sbatch/eval_merged.sbatch /d/hpc/projects/FRI/zm3587/dl/runs/person_detection/yolo26n_ch_wp_coco_och_960/weights/best.pt 960 all 16 eval_yolo26n_ch_wp_coco_och_960_all
