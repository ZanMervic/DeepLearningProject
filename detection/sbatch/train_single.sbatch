#!/bin/bash
#SBATCH --job-name=yolo26_train
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-single-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-train-single-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[01-10]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

DL_ROOT="/d/hpc/projects/FRI/zm3587/dl"
SCRIPT_ROOT="$DL_ROOT/scripts/pipelines"
PYTHON_BIN="/d/hpc/home/zm3587/.conda/envs/deeplearning/bin/python"

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

dataset_tag_from_data() {
  local data_path="$1"
  local data_file
  data_file="$(basename "$data_path")"

  if [[ "$data_file" == "data.yaml" ]]; then
    case "$(basename "$(dirname "$data_path")")" in
      crowdhuman) echo "chv" ;;
      widerperson) echo "wp" ;;
      coco2017) echo "coco" ;;
      ochuman) echo "och" ;;
      *) echo "$(basename "$(dirname "$data_path")")" ;;
    esac
  else
    echo "${data_file%.yaml}"
  fi
}

# Positional args with defaults.
# Keep the historical ordering to avoid breaking existing calls:
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
DATA="${3:-$DL_ROOT/datasets/crowdhuman/data.yaml}"
BATCH="${4:-16}"
MODEL_NAME="$(derive_model_name "$MODEL")"
DATA_TAG="$(dataset_tag_from_data "$DATA")"
NAME="${5:-${MODEL_NAME}_${DATA_TAG}_${IMGSZ}}"
EPOCHS="${6:-50}"
PATIENCE="${7:-10}"
FREEZE="${8:-0}"

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

# Example:
# sbatch -J train_n_chv640 sbatch/train_single.sbatch \
#   yolo26n.pt \
#   640 \
#   /d/hpc/projects/FRI/zm3587/dl/datasets/crowdhuman/data.yaml \
#   16 \
#   yolo26n_chv_640 \
#   50 \
#   10 \
#   0
