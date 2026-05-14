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

DL_ROOT="/d/hpc/projects/FRI/zm3587/dl"
SCRIPT_ROOT="$DL_ROOT/scripts/pipelines"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/dl_v100/bin/python"

derive_model_name() {
  local model_path="$1"
  local base_name
  base_name="$(basename "$model_path")"

  if [[ "$base_name" == "best.pt" || "$base_name" == "last.pt" ]]; then
    if [[ "$(basename "$(dirname "$model_path")")" == "weights" ]]; then
      local run_name
      run_name="$(basename "$(dirname "$(dirname "$model_path")")")"
      if [[ "$base_name" == "last.pt" ]]; then
        echo "${run_name}_last"
      else
        echo "$run_name"
      fi
      return
    fi
  fi

  echo "${base_name%.pt}"
}

dataset_tag_from_key() {
  if [[ "$1" == */* || "$1" == *.yaml ]]; then
    echo "$(basename "$1" .yaml)"
    return
  fi

  case "$1" in
    crowdhuman|ch) echo "chv" ;;
    widerperson|wp) echo "wp" ;;
    coco|coco2017) echo "coco" ;;
    ochuman|och) echo "och" ;;
    ch_wp) echo "chv_wp" ;;
    ch_wp_coco) echo "chv_wp_coco" ;;
    ch_wp_coco_och|all) echo "chv_wp_coco_och" ;;
    ch_ochuman|ch_och) echo "chv_och" ;;
    *) echo "$1" ;;
  esac
}

# Positional args with defaults.
# Keep the historical ordering to avoid breaking existing calls:
#   1 MODEL
#   2 IMGSZ
#   3 DATASET_KEY       crowdhuman | ch_wp | ch_wp_coco | ch_wp_coco_och | ch_ochuman | custom_yaml_path
#   4 BATCH
#   5 NAME
#   6 EPOCHS
#   7 PATIENCE
#   8 FREEZE
MODEL="${1:-yolo26n.pt}"
IMGSZ="${2:-640}"
DATASET_KEY="${3:-ch_wp_coco_och}"
BATCH="${4:-16}"
MODEL_NAME="$(derive_model_name "$MODEL")"
DATASET_TAG="$(dataset_tag_from_key "$DATASET_KEY")"
NAME="${5:-${MODEL_NAME}_${DATASET_TAG}_${IMGSZ}}"
EPOCHS="${6:-50}"
PATIENCE="${7:-10}"
FREEZE="${8:-0}"

case "$DATASET_KEY" in
  crowdhuman|ch)
    DATA="$DL_ROOT/datasets/crowdhuman/data.yaml"
    ;;
  widerperson|wp)
    DATA="$DL_ROOT/datasets/widerperson/data.yaml"
    ;;
  coco|coco2017)
    DATA="$DL_ROOT/datasets/coco2017/data.yaml"
    ;;
  ochuman|och)
    DATA="$DL_ROOT/datasets/ochuman/data.yaml"
    ;;
  ch_wp)
    DATA="$DL_ROOT/datasets/ch_wp.yaml"
    ;;
  ch_wp_coco)
    DATA="$DL_ROOT/datasets/ch_wp_coco.yaml"
    ;;
  ch_wp_coco_och|all)
    DATA="$DL_ROOT/datasets/ch_wp_coco_och.yaml"
    ;;
  ch_ochuman|ch_och)
    DATA="$DL_ROOT/datasets/ch_ochuman.yaml"
    ;;
  /*)
    DATA="$DATASET_KEY"
    ;;
  *)
    if [[ -f "$DL_ROOT/$DATASET_KEY" ]]; then
      DATA="$DL_ROOT/$DATASET_KEY"
    else
      echo "Unknown DATASET_KEY or YAML path: $DATASET_KEY" >&2
      echo "Known keys: crowdhuman, ch_wp, ch_wp_coco, ch_wp_coco_och, ch_ochuman" >&2
      exit 2
    fi
    ;;
esac

PROJECT="$DL_ROOT/outputs/training"
RUN_DIR="$PROJECT/$NAME"

export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$DL_ROOT/.ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$DL_ROOT/slurm"
mkdir -p "$RUN_DIR"

# Keep a copy of the full job log inside the training run directory.
exec > >(tee -a "$RUN_DIR/slurm_train.out") 2>&1

cd "$DL_ROOT"

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "DL root: $DL_ROOT"
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

"$PYTHON_BIN" -u "$SCRIPT_ROOT/hpc_yolo_train.py" \
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
# sbatch -J train_n_best640 sbatch/train_merged.sbatch yolo26n.pt 640 ch_wp_coco_och 16 yolo26n_chv_wp_coco_och_640 50 10 0
# sbatch -J train_n_best960 sbatch/train_merged.sbatch yolo26n.pt 960 ch_wp_coco_och 16 yolo26n_chv_wp_coco_och_960 50 10 0
