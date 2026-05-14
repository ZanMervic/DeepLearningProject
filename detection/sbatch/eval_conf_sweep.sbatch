#!/bin/bash
#SBATCH --job-name=yolo_conf_sweep
#SBATCH --output=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-conf-sweep-%j.out
#SBATCH --error=/d/hpc/projects/FRI/zm3587/dl/slurm/slurm-yolo-conf-sweep-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

# Positional args.
#   1 MODEL
#   2 IMGSZ
#   3 DATASET_GROUP   all | main | hard | ch | wp | coco | och | ch_wp | ch_wp_coco | ch_wp_coco_och
#   4 BATCH
#   5 CONF_VALUES     comma-separated list, e.g. 0.10,0.15,0.20,0.25,0.30,0.40
# Optional:
#   6 IOU
#   7 MAX_DET
#   8 RUN_NAME        use "auto" (or leave empty) to derive <model_name>_conf_sweep
MODEL="${1:-$DL_ROOT/outputs/training/yolo26n_chv_wp_coco_och_640/weights/best.pt}"
IMGSZ="${2:-640}"
DATASET_GROUP="${3:-all}"
BATCH="${4:-16}"
CONF_VALUES="${5:-0.10,0.15,0.20,0.25,0.30,0.40}"
IOU="${6:-}"
MAX_DET="${7:-}"
RUN_NAME="${8:-auto}"

MODEL_NAME="$(derive_model_name "$MODEL")"
if [[ -z "$RUN_NAME" || "$RUN_NAME" == "auto" ]]; then
  RUN_NAME="${MODEL_NAME}_conf_sweep"
fi

RUN_DIR="$DL_ROOT/outputs/evaluation/$RUN_NAME"

export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$DL_ROOT/.ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$DL_ROOT/slurm"
mkdir -p "$RUN_DIR"

# Keep a copy of the full job log inside the evaluation run directory.
exec > >(tee -a "$RUN_DIR/slurm_eval_conf_sweep.out") 2>&1

cd "$DL_ROOT"

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Run name: $RUN_NAME"
echo "Model: $MODEL"
echo "Image size: $IMGSZ"
echo "Dataset group: $DATASET_GROUP"
echo "Batch: $BATCH"
echo "Confidence values: $CONF_VALUES"
echo "Run directory: $RUN_DIR"
if [[ -n "$IOU" ]]; then
  echo "NMS IoU override: $IOU"
fi
if [[ -n "$MAX_DET" ]]; then
  echo "max_det override: $MAX_DET"
fi
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

CMD=(
  "$PYTHON_BIN" -u "$SCRIPT_ROOT/hpc_yolo_eval_conf_sweep.py"
  --model "$MODEL"
  --imgsz "$IMGSZ"
  --dataset-group "$DATASET_GROUP"
  --batch "$BATCH"
  --run-dir "$RUN_DIR"
  --project-root "$DL_ROOT"
  --device 0
  --workers 8
  --conf-values "$CONF_VALUES"
)

if [[ -n "$IOU" ]]; then
  CMD+=(--iou "$IOU")
fi
if [[ -n "$MAX_DET" ]]; then
  CMD+=(--max-det "$MAX_DET")
fi

"${CMD[@]}"

echo "Job finished on $(date)"

# Examples:
# sbatch -J sweep_n_all640 sbatch/eval_conf_sweep.sbatch /d/hpc/projects/FRI/zm3587/dl/outputs/training/yolo26n_chv_wp_coco_och_640/weights/best.pt 640 all 16 0.10,0.15,0.20,0.25,0.30,0.40
#
# sbatch -J sweep_n_all960 sbatch/eval_conf_sweep.sbatch \
#   /d/hpc/projects/FRI/zm3587/dl/outputs/training/yolo26n_chv_wp_coco_och_960/weights/best.pt \
#   960 \
#   all \
#   16 \
#   0.10,0.15,0.20,0.25,0.30,0.40 \
#   0.70
