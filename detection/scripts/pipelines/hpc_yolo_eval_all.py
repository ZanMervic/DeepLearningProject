#!/usr/bin/env python3
"""Evaluate one Ultralytics YOLO detector on multiple validation datasets.

This is intended for systematic cross-dataset evaluation, e.g.:
  - CrowdHuman val
  - WiderPerson val
  - COCO_Person val
  - OCHuman val

It writes:
  1. one CSV row per dataset
  2. optional JSON with environment + raw metric reprs
  3. optional Markdown summary table
  4. normal Ultralytics validation artifacts per dataset under --project
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_PROJECT_ROOT = Path("/d/hpc/projects/FRI/zm3587/dl")

DEFAULT_DATASETS: Dict[str, str] = {
    "crowdhuman": "datasets/crowdhuman/data.yaml",
    "widerperson": "datasets/widerperson/data.yaml",
    "coco2017": "datasets/coco2017/data.yaml",
    "ochuman": "datasets/ochuman/data.yaml",
}

DATASET_GROUPS: Dict[str, List[str]] = {
    "all": ["crowdhuman", "widerperson", "coco2017", "ochuman"],
    "main": ["crowdhuman", "widerperson", "coco2017"],
    "hard": ["crowdhuman", "ochuman"],
    "ch": ["crowdhuman"],
    "wp": ["widerperson"],
    "coco": ["coco2017"],
    "och": ["ochuman"],
    "ch_wp": ["crowdhuman", "widerperson"],
    "ch_wp_coco": ["crowdhuman", "widerperson", "coco2017"],
    "ch_wp_coco_och": ["crowdhuman", "widerperson", "coco2017", "ochuman"],
}

PREFERRED_COLUMNS = [
    "dataset_name",
    "model_name",
    "model",
    "data",
    "imgsz",
    "batch",
    "split",
    "box_map50",
    "box_map",
    "box_map75",
    "box_mp",
    "box_mr",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
    "speed_preprocess",
    "speed_inference",
    "speed_loss",
    "speed_postprocess",
    "save_dir",
]


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one YOLO model on multiple validation datasets."
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint, e.g. weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--project", default=None, help="Ultralytics output project. Default: <project-root>/runs/person_detection_eval")
    parser.add_argument("--name-prefix", default=None, help="Prefix for per-dataset eval run names")
    parser.add_argument(
        "--dataset-group",
        default="all",
        help=(
            "Dataset group key or comma-separated dataset keys. "
            f"Known groups: {', '.join(sorted(DATASET_GROUPS))}. "
            f"Known datasets: {', '.join(sorted(DEFAULT_DATASETS))}."
        ),
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Explicit data.yaml paths. If provided, overrides --dataset-group.",
    )
    parser.add_argument("--out", required=True, help="CSV summary path")
    parser.add_argument("--json-out", default=None, help="Optional JSON details path")
    parser.add_argument("--md-out", default=None, help="Optional Markdown summary table path")
    parser.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO-style JSON predictions where supported")
    parser.add_argument("--plots", action="store_true", help="Ask Ultralytics to save validation plots")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse existing eval run dirs")
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
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            try:
                info["arch_list"] = torch.cuda.get_arch_list()
            except Exception as exc:  # noqa: BLE001
                info["arch_list_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        import ultralytics

        info["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        info["ultralytics_error"] = repr(exc)
    return info


def flatten_metrics(metrics_obj: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    results_dict = getattr(metrics_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        row.update(results_dict)

    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for attr in ["map", "map50", "map75", "mp", "mr"]:
            if hasattr(box, attr):
                value = getattr(box, attr)
                try:
                    row[f"box_{attr}"] = float(value)
                except Exception:  # noqa: BLE001
                    row[f"box_{attr}"] = str(value)

    speed = getattr(metrics_obj, "speed", None)
    if isinstance(speed, dict):
        for key, value in speed.items():
            row[f"speed_{key}"] = value

    save_dir = getattr(metrics_obj, "save_dir", None)
    if save_dir is not None:
        row["save_dir"] = str(save_dir)

    return row


def ordered_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    keys = {k for row in rows for k in row.keys()}
    ordered = [k for k in PREFERRED_COLUMNS if k in keys]
    remaining = sorted(keys - set(ordered))
    return ordered + remaining


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def md_value(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            value = row[key]
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
    return "—"


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Dataset", "mAP50", "mAP50-95", "Precision", "Recall", "Inference ms", "Save dir"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = [
            row.get("dataset_name", "—"),
            md_value(row, ["box_map50", "metrics/mAP50(B)"]),
            md_value(row, ["box_map", "metrics/mAP50-95(B)"]),
            md_value(row, ["box_mp", "metrics/precision(B)"]),
            md_value(row, ["box_mr", "metrics/recall(B)"]),
            md_value(row, ["speed_inference"]),
            row.get("save_dir", "—"),
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_dataset_group(group_expr: str) -> List[str]:
    group_expr = group_expr.strip()
    if group_expr in DATASET_GROUPS:
        return DATASET_GROUPS[group_expr]
    keys = [x.strip() for x in group_expr.split(",") if x.strip()]
    resolved: List[str] = []
    for key in keys:
        if key in DATASET_GROUPS:
            resolved.extend(DATASET_GROUPS[key])
        elif key in DEFAULT_DATASETS:
            resolved.append(key)
        else:
            raise KeyError(
                f"Unknown dataset/group key '{key}'. Known groups={sorted(DATASET_GROUPS)}; "
                f"known datasets={sorted(DEFAULT_DATASETS)}"
            )
    # Preserve order but remove duplicates.
    deduped: List[str] = []
    seen = set()
    for key in resolved:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def resolve_data_paths(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    project_root = Path(args.project_root)
    if args.data:
        out: List[Tuple[str, Path]] = []
        for p in args.data:
            path = Path(p)
            if not path.is_absolute():
                path = project_root / path
            out.append((path.parent.name, path))
        return out

    keys = resolve_dataset_group(args.dataset_group)
    return [(key, project_root / DEFAULT_DATASETS[key]) for key in keys]


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    project_root = Path(args.project_root)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    project = Path(args.project) if args.project else project_root / "runs/person_detection_eval"
    if not project.is_absolute():
        project = project_root / project
    project.mkdir(parents=True, exist_ok=True)

    data_items = resolve_data_paths(args)
    for _, data_path in data_items:
        if not data_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(str(model_path))
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    model_name = model_path.stem
    if model_path.name == "best.pt" and model_path.parent.name == "weights":
        model_name = model_path.parent.parent.name
    elif model_path.name == "last.pt" and model_path.parent.name == "weights":
        model_name = model_path.parent.parent.name + "_last"

    prefix = args.name_prefix or f"eval_{model_name}_{args.imgsz}"
    prefix = slugify(prefix)

    print("\n=== Evaluation plan ===")
    print(f"Model: {model_path}")
    print(f"Datasets: {[str(p) for _, p in data_items]}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Project: {project}")
    print("=======================\n")

    for dataset_name, data_path in data_items:
        dataset_name = slugify(dataset_name)
        eval_name = f"{prefix}__on_{dataset_name}"
        print(f"\n=== Evaluating {model_path} on {data_path} @ imgsz={args.imgsz} ===")
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            split=args.split,
            project=str(project),
            name=eval_name,
            save_json=args.save_json,
            plots=args.plots,
            exist_ok=args.exist_ok,
        )

        flat = flatten_metrics(metrics)
        flat.update(
            {
                "model": str(model_path),
                "model_name": model_name,
                "data": str(data_path),
                "dataset_name": dataset_name,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "split": args.split,
            }
        )
        rows.append(flat)
        details.append({"row": flat, "metrics_repr": repr(metrics)})

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    write_csv(out_path, rows)
    print(f"Wrote metrics CSV: {out_path}")

    if args.md_out:
        md_path = Path(args.md_out)
        if not md_path.is_absolute():
            md_path = project_root / md_path
        write_markdown(md_path, rows)
        print(f"Wrote Markdown table: {md_path}")

    if args.json_out:
        json_path = Path(args.json_out)
        if not json_path.is_absolute():
            json_path = project_root / json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"environment": env_info(), "evaluations": details}, f, indent=2, default=str)
        print(f"Wrote metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
