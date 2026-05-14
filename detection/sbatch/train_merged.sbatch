#!/bin/bash
#SBATCH --job-name=yolo_train_merged
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-merged-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-merged-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/d/hpc/projects/FRI/zm3587/dl"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/dl_v100/bin/python"

# Positional args with defaults.
# Keep ordering aligned with eval style:
#   1 MODEL
#   2 IMGSZ
#   3 DATASET_KEY       crowdhuman | ch_wp | ch_wp_coco | ch_wp_coco_och | ch_ochuman | custom_yaml_path
#   4 BATCH
#   5 NAME
# Train-only:
#   6 EPOCHS
#   7 PATIENCE
#   8 FREEZE
MODEL="${1:-yolo26s.pt}"
IMGSZ="${2:-960}"
DATASET_KEY="${3:-ch_wp_coco}"
BATCH="${4:-16}"
NAME="${5:-train_$(basename "$MODEL" .pt)_${DATASET_KEY}_${IMGSZ}}"
EPOCHS="${6:-50}"
PATIENCE="${7:-10}"
FREEZE="${8:-0}"

case "$DATASET_KEY" in
  crowdhuman|ch)
    DATA="$PROJECT_ROOT/datasets/crowdhuman/data.yaml"
    ;;
  widerperson|wp)
    DATA="$PROJECT_ROOT/datasets/widerperson/data.yaml"
    ;;
  coco|coco2017)
    DATA="$PROJECT_ROOT/datasets/coco2017/data.yaml"
    ;;
  ochuman|och)
    DATA="$PROJECT_ROOT/datasets/ochuman/data.yaml"
    ;;
  ch_wp)
    DATA="$PROJECT_ROOT/datasets/ch_wp.yaml"
    ;;
  ch_wp_coco)
    DATA="$PROJECT_ROOT/datasets/ch_wp_coco.yaml"
    ;;
  ch_wp_coco_och|all)
    DATA="$PROJECT_ROOT/datasets/ch_wp_coco_och.yaml"
    ;;
  ch_ochuman|ch_och)
    DATA="$PROJECT_ROOT/datasets/ch_ochuman.yaml"
    ;;
  /*)
    DATA="$DATASET_KEY"
    ;;
  *)
    if [[ -f "$PROJECT_ROOT/$DATASET_KEY" ]]; then
      DATA="$PROJECT_ROOT/$DATASET_KEY"
    else
      echo "Unknown DATASET_KEY or YAML path: $DATASET_KEY" >&2
      echo "Known keys: crowdhuman, ch_wp, ch_wp_coco, ch_wp_coco_och, ch_ochuman" >&2
      exit 2
    fi
    ;;
esac

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
echo "Image size: $IMGSZ"
echo "Dataset key: $DATASET_KEY"
echo "Data YAML: $DATA"
echo "Batch: $BATCH"
echo "Name: $NAME"
echo "Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "Freeze: $FREEZE"
echo "Project: $PROJECT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ ! -f "$DATA" ]]; then
  echo "data.yaml not found: $DATA" >&2
  exit 3
fi

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

# Examples:
# sbatch -J train_s_chwp960 sbatch/train_merged.sbatch yolo26s.pt 960 ch_wp 16 yolo26s_ch_wp_960 50 10 0
# sbatch -J train_s_chwpcoco960 sbatch/train_merged.sbatch yolo26s.pt 960 ch_wp_coco 16 yolo26s_ch_wp_coco_960 50 10 0
# sbatch -J train_n_bestmix960 sbatch/train_merged.sbatch yolo26n.pt 960 ch_wp_coco 16 yolo26n_ch_wp_coco_960 50 10 0
