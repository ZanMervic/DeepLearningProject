#!/usr/bin/env python3
"""Evaluate an Ultralytics YOLO detector on one or more data.yaml files.

Example:
    python scripts/hpc_yolo_eval.py \
      --model runs/person_detection/yolo26s_crowdhuman_vbox_640/weights/best.pt \
      --data datasets/CrowdHuman/data.yaml datasets/OCHuman/data.yaml \
      --imgsz 640 --device 0 \
      --out results/eval_crowdhuman_model.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on one or more datasets.")
    parser.add_argument("--model", required=True, help="Path to checkpoint, e.g. best.pt")
    parser.add_argument("--data", required=True, nargs="+", help="One or more data.yaml files")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", default="runs/person_detection_eval")
    parser.add_argument("--name", default=None, help="Optional Ultralytics eval run name")
    parser.add_argument("--out", required=True, help="CSV file where metrics will be appended/written")
    parser.add_argument("--json-out", default=None, help="Optional JSON output file")
    parser.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO-style JSON predictions where supported")
    parser.add_argument("--plots", action="store_true", help="Ask Ultralytics to save validation plots")
    return parser.parse_args()


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
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

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        import ultralytics

        info["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        info["ultralytics_error"] = repr(exc)
    return info


def flatten_metrics(metrics_obj: Any) -> Dict[str, Any]:
    """Extract a stable flat dict from Ultralytics metrics object."""
    row: Dict[str, Any] = {}

    # Most Ultralytics versions expose a dict here.
    results_dict = getattr(metrics_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        row.update(results_dict)

    # Add common convenience fields if available.
    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for attr in ["map", "map50", "map75", "mp", "mr"]:
            if hasattr(box, attr):
                try:
                    row[f"box_{attr}"] = float(getattr(box, attr))
                except Exception:  # noqa: BLE001
                    row[f"box_{attr}"] = str(getattr(box, attr))

    speed = getattr(metrics_obj, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            row[f"speed_{key}"] = value

    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = YOLO(args.model)
    rows: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []

    for data_yaml in args.data:
        data_path = Path(data_yaml)
        if not data_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

        eval_name = args.name or f"eval_{model_path.stem}_{data_path.parent.name}_{args.imgsz}"
        print(f"\n=== Evaluating {args.model} on {data_yaml} @ imgsz={args.imgsz} ===")
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            split=args.split,
            project=args.project,
            name=eval_name,
            save_json=args.save_json,
            plots=args.plots,
        )

        flat = flatten_metrics(metrics)
        flat.update(
            {
                "model": str(args.model),
                "data": str(data_path),
                "dataset_name": data_path.parent.name,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "split": args.split,
            }
        )
        rows.append(flat)
        all_details.append({"row": flat, "metrics_repr": repr(metrics)})

    write_csv(Path(args.out), rows)
    print(f"Wrote metrics CSV: {args.out}")

    if args.json_out:
        out = {"environment": env_info(), "evaluations": all_details}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Wrote metrics JSON: {args.json_out}")


if __name__ == "__main__":
    main()
