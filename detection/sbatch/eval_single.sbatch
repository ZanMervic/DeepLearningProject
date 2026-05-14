#!/bin/bash
#SBATCH --job-name=yolo_eval_single
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-single-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-eval-single-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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

derive_dataset_name() {
  local data_path="$1"
  local data_file
  data_file="$(basename "$data_path")"
  if [[ "$data_file" == "data.yaml" ]]; then
    echo "$(basename "$(dirname "$data_path")")"
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
#   5 RUN_NAME
MODEL="${1:-$DL_ROOT/outputs/training/yolo26n_chv_960/weights/best.pt}"
IMGSZ="${2:-960}"
DATA="${3:-$DL_ROOT/datasets/crowdhuman/data.yaml}"
BATCH="${4:-16}"
DATASET_NAME="$(derive_dataset_name "$DATA")"
RUN_NAME="${5:-$(derive_model_name "$MODEL")__on_${DATASET_NAME}}"

RUN_DIR="$DL_ROOT/outputs/evaluation/$RUN_NAME"
SUMMARY_CSV="$RUN_DIR/summary.csv"
SUMMARY_JSON="$RUN_DIR/summary.json"
SUMMARY_MD="$RUN_DIR/summary.md"

export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$DL_ROOT/.ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$DL_ROOT/slurm"
mkdir -p "$RUN_DIR"

# Keep a copy of the full job log inside the evaluation run directory.
exec > >(tee -a "$RUN_DIR/slurm_eval.out") 2>&1

cd "$DL_ROOT"

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Image size: $IMGSZ"
echo "Batch: $BATCH"
echo "Run name: $RUN_NAME"
echo "Run directory: $RUN_DIR"
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

"$PYTHON_BIN" -u "$SCRIPT_ROOT/hpc_yolo_eval_all.py" \
  --model "$MODEL" \
  --imgsz "$IMGSZ" \
  --data "$DATA" \
  --batch "$BATCH" \
  --name-prefix "$DATASET_NAME" \
  --project-root "$DL_ROOT" \
  --project "$RUN_DIR" \
  --device 0 \
  --workers 8 \
  --out "$SUMMARY_CSV" \
  --json-out "$SUMMARY_JSON" \
  --md-out "$SUMMARY_MD" \
  --plots

generated_dir="$RUN_DIR/${DATASET_NAME}__on_${DATASET_NAME}"
if [[ -d "$generated_dir" ]]; then
  mv "$generated_dir" "$RUN_DIR/$DATASET_NAME"
fi

echo "Job finished on $(date)"

# Example:
# sbatch -J eval_n_chv960 sbatch/eval_single.sbatch \
#   /d/hpc/projects/FRI/zm3587/dl/outputs/training/yolo26n_chv_960/weights/best.pt \
#   960 \
#   /d/hpc/projects/FRI/zm3587/dl/datasets/crowdhuman/data.yaml \
#   16 \
#   yolo26n_chv_960__on_crowdhuman
