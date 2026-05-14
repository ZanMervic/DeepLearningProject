#!/usr/bin/env python3
"""Train/fine-tune an Ultralytics YOLO detector on an HPC node.

Example:
    python scripts/hpc_yolo_train.py \
      --model yolo26s.pt \
      --data /path/to/datasets/CrowdHuman/data.yaml \
      --name yolo26s_crowdhuman_vbox_640 \
      --imgsz 640 --batch 16 --epochs 50 --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an Ultralytics YOLO model.")
    parser.add_argument("--model", required=True, help="Pretrained model or checkpoint, e.g. yolo26s.pt or path/to/best.pt")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml")
    parser.add_argument("--name", required=True, help="Experiment/run name")
    parser.add_argument("--project", default="runs/person_detection", help="Output project directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device index, comma list, 'cpu', or leave per Ultralytics conventions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers. 0 = no freeze")
    parser.add_argument("--cache", default=None, choices=[None, "ram", "disk", "False", "false"], help="Ultralytics cache option")
    parser.add_argument("--amp", type=str2bool, default=True, help="Use automatic mixed precision if supported")
    parser.add_argument("--cos-lr", type=str2bool, default=False, help="Use cosine LR schedule")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in final N epochs")
    parser.add_argument("--resume", type=str2bool, default=False, help="Resume training from checkpoint")
    parser.add_argument("--exist-ok", type=str2bool, default=False, help="Allow overwriting/resuming same output name")
    parser.add_argument("--extra-json", default=None, help="Optional JSON file with extra Ultralytics train kwargs")
    return parser.parse_args()


def print_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = repr(exc)

    try:
        import ultralytics

        env["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        env["ultralytics_error"] = repr(exc)

    print("\n=== Environment ===")
    print(json.dumps(env, indent=2))
    print("===================\n")
    return env


def main() -> None:
    args = parse_args()
    env = print_environment()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    extra_kwargs: Dict[str, Any] = {}
    if args.extra_json:
        with open(args.extra_json, "r", encoding="utf-8") as f:
            extra_kwargs = json.load(f)

    from ultralytics import YOLO

    model = YOLO(args.model)

    cache_value: bool | str | None
    if args.cache in {"False", "false"}:
        cache_value = False
    else:
        cache_value = args.cache

    train_kwargs: Dict[str, Any] = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        freeze=args.freeze,
        cache=cache_value,
        amp=args.amp,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )
    train_kwargs.update(extra_kwargs)

    print("\n=== Train kwargs ===")
    print(json.dumps(train_kwargs, indent=2, default=str))
    print("====================\n")

    results = model.train(**train_kwargs)

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "train_kwargs": train_kwargs,
        "environment": env,
        "result_type": str(type(results)),
    }
    with open(run_dir / "hpc_train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Training finished. Run directory: {run_dir}")
    print(f"Expected best checkpoint: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
