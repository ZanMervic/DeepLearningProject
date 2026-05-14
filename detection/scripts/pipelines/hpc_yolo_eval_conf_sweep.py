#!/usr/bin/env python3
"""Sweep confidence thresholds for one YOLO checkpoint across validation datasets.

This is evaluation-only. It does not retrain or modify the checkpoint.

The script writes one run directory containing:
  - summary.csv: one row per threshold and dataset
  - threshold_summary.csv: one macro-average row per threshold
  - summary.json: arguments, environment, detailed rows, macro rows
  - summary.md: readable macro + per-dataset tables
  - per-threshold Ultralytics artifacts under <run-dir>/conf_<value>/<dataset>/
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

DETAIL_COLUMNS = [
    "conf",
    "dataset_name",
    "model_name",
    "model",
    "data",
    "imgsz",
    "batch",
    "split",
    "nms_iou",
    "max_det",
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

THRESHOLD_COLUMNS = [
    "rank_by_macro_recall",
    "conf",
    "datasets_evaluated",
    "macro_precision",
    "macro_recall",
    "macro_map50",
    "macro_map50_95",
    "avg_inference_ms",
    "nms_iou",
    "max_det",
]


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep confidence thresholds for one YOLO checkpoint."
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint, e.g. weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Used only when --run-dir is not provided.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output directory. Default: <project-root>/outputs/evaluation/<derived-run-name>",
    )
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
    parser.add_argument(
        "--conf-values",
        default="0.10,0.15,0.20,0.25,0.30,0.40",
        help="Comma or whitespace separated confidence thresholds to evaluate.",
    )
    parser.add_argument("--iou", type=float, default=None, help="Optional NMS IoU override.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional max_det override.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style predictions where supported.")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots.")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse an existing run directory instead of raising an error.",
    )
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
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
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


def parse_conf_values(text: str) -> List[float]:
    values: List[float] = []
    for chunk in re.split(r"[\s,]+", text.strip()):
        if not chunk:
            continue
        value = float(chunk)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {value}")
        values.append(value)

    deduped: List[float] = []
    seen = set()
    for value in values:
        key = round(value, 8)
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    if not deduped:
        raise ValueError("At least one confidence threshold is required.")
    return deduped


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

    deduped: List[str] = []
    seen = set()
    for key in resolved:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def dataset_name_from_path(path: Path) -> str:
    if path.name == "data.yaml":
        return path.parent.name
    return path.stem


def resolve_data_paths(args: argparse.Namespace, project_root: Path) -> List[Tuple[str, Path]]:
    if args.data:
        out: List[Tuple[str, Path]] = []
        for raw_path in args.data:
            path = Path(raw_path)
            if not path.is_absolute():
                path = project_root / path
            out.append((slugify(dataset_name_from_path(path)), path))
        return out

    keys = resolve_dataset_group(args.dataset_group)
    return [(key, project_root / DEFAULT_DATASETS[key]) for key in keys]


def derive_model_name(model_path: Path) -> str:
    if model_path.name == "best.pt" and model_path.parent.name == "weights":
        return model_path.parent.parent.name
    if model_path.name == "last.pt" and model_path.parent.name == "weights":
        return f"{model_path.parent.parent.name}_last"
    return model_path.stem


def resolve_run_dir(args: argparse.Namespace, project_root: Path, model_name: str) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
        return run_dir

    run_name = slugify(args.run_name or f"{model_name}_conf_sweep")
    return project_root / "outputs" / "evaluation" / run_name


def conf_slug(conf: float) -> str:
    return f"conf_{conf:.2f}".replace(".", "p")


def ordered_columns(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    keys = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in keys]
    remaining = sorted(keys - set(ordered))
    return ordered + remaining


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows, preferred)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row: Dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key not in row or row[key] in (None, ""):
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def mean_or_none(values: Sequence[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def build_threshold_summary(detail_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[float, List[Dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(float(row["conf"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for conf, rows in grouped.items():
        summary_rows.append(
            {
                "conf": conf,
                "datasets_evaluated": len(rows),
                "macro_precision": mean_or_none(
                    [metric_value(row, ["box_mp", "metrics/precision(B)"]) for row in rows]
                ),
                "macro_recall": mean_or_none(
                    [metric_value(row, ["box_mr", "metrics/recall(B)"]) for row in rows]
                ),
                "macro_map50": mean_or_none(
                    [metric_value(row, ["box_map50", "metrics/mAP50(B)"]) for row in rows]
                ),
                "macro_map50_95": mean_or_none(
                    [metric_value(row, ["box_map", "metrics/mAP50-95(B)"]) for row in rows]
                ),
                "avg_inference_ms": mean_or_none(
                    [metric_value(row, ["speed_inference"]) for row in rows]
                ),
                "nms_iou": rows[0].get("nms_iou"),
                "max_det": rows[0].get("max_det"),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -(row["macro_recall"] if row["macro_recall"] is not None else -1.0),
            -(row["macro_precision"] if row["macro_precision"] is not None else -1.0),
            row["conf"],
        )
    )
    for index, row in enumerate(summary_rows, start=1):
        row["rank_by_macro_recall"] = index
    return summary_rows


def format_md_value(value: Any, decimals: int = 4) -> str:
    if value in (None, ""):
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def write_markdown(
    path: Path,
    model_path: Path,
    run_dir: Path,
    data_items: Sequence[Tuple[str, Path]],
    conf_values: Sequence[float],
    detail_rows: Sequence[Dict[str, Any]],
    threshold_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Confidence Threshold Sweep",
        "",
        f"- Model: `{model_path}`",
        f"- Run directory: `{run_dir}`",
        f"- Thresholds: `{', '.join(f'{value:.2f}' for value in conf_values)}`",
        f"- Datasets: `{', '.join(name for name, _ in data_items)}`",
        "- Ranking: thresholds sorted by macro-average recall across the evaluated datasets.",
        "",
        "## Threshold Summary",
        "",
        "| Rank | Conf | Macro Recall | Macro Precision | Macro mAP50 | Macro mAP50-95 | Avg Inference ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in threshold_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank_by_macro_recall"]),
                    format_md_value(row["conf"], decimals=2),
                    format_md_value(row["macro_recall"]),
                    format_md_value(row["macro_precision"]),
                    format_md_value(row["macro_map50"]),
                    format_md_value(row["macro_map50_95"]),
                    format_md_value(row["avg_inference_ms"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Dataset Results",
            "",
            "| Conf | Dataset | Recall | Precision | mAP50 | mAP50-95 | Inference ms | Save dir |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in detail_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_md_value(row["conf"], decimals=2),
                    str(row["dataset_name"]),
                    format_md_value(metric_value(row, ["box_mr", "metrics/recall(B)"])),
                    format_md_value(metric_value(row, ["box_mp", "metrics/precision(B)"])),
                    format_md_value(metric_value(row, ["box_map50", "metrics/mAP50(B)"])),
                    format_md_value(metric_value(row, ["box_map", "metrics/mAP50-95(B)"])),
                    format_md_value(metric_value(row, ["speed_inference"])),
                    str(row.get("save_dir", "—")),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    project_root = Path(args.project_root)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model_name = derive_model_name(model_path)
    run_dir = resolve_run_dir(args, project_root, model_name)
    if run_dir.exists() and not args.exist_ok:
        allowed_preexisting = {"slurm_eval_conf_sweep.out", "slurm_eval_conf_sweep.err"}
        meaningful_entries = [
            entry.name for entry in run_dir.iterdir() if entry.name not in allowed_preexisting
        ]
        if meaningful_entries:
            raise FileExistsError(
                f"Run directory already contains results: {run_dir}. "
                "Pass --exist-ok or choose a different run name."
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    data_items = resolve_data_paths(args, project_root)
    for _, data_path in data_items:
        if not data_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

    conf_values = parse_conf_values(args.conf_values)
    conf_order = {value: index for index, value in enumerate(conf_values)}
    dataset_order = {name: index for index, (name, _) in enumerate(data_items)}

    model = YOLO(str(model_path))
    detail_rows: List[Dict[str, Any]] = []
    details_json: List[Dict[str, Any]] = []

    print("\n=== Confidence threshold sweep ===")
    print(f"Model: {model_path}")
    print(f"Run directory: {run_dir}")
    print(f"Datasets: {[str(path) for _, path in data_items]}")
    print(f"Thresholds: {conf_values}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    if args.iou is not None:
        print(f"NMS IoU override: {args.iou}")
    if args.max_det is not None:
        print(f"max_det override: {args.max_det}")
    print("===============================\n")

    for conf in conf_values:
        threshold_dir = run_dir / conf_slug(conf)
        threshold_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, data_path in data_items:
            print(
                f"\n=== Evaluating {model_path} on {data_path} "
                f"@ imgsz={args.imgsz}, conf={conf:.2f} ==="
            )
            val_kwargs: Dict[str, Any] = {
                "data": str(data_path),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "split": args.split,
                "project": str(threshold_dir),
                "name": dataset_name,
                "conf": conf,
                "save_json": args.save_json,
                "plots": args.plots,
                # Keep predictable dataset directory names after the run-dir guard above.
                "exist_ok": True,
            }
            if args.iou is not None:
                val_kwargs["iou"] = args.iou
            if args.max_det is not None:
                val_kwargs["max_det"] = args.max_det

            metrics = model.val(**val_kwargs)
            flat = flatten_metrics(metrics)
            flat.update(
                {
                    "conf": conf,
                    "model": str(model_path),
                    "model_name": model_name,
                    "data": str(data_path),
                    "dataset_name": dataset_name,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "split": args.split,
                    "nms_iou": args.iou,
                    "max_det": args.max_det,
                }
            )
            detail_rows.append(flat)
            details_json.append({"row": flat, "metrics_repr": repr(metrics)})

    detail_rows.sort(
        key=lambda row: (
            conf_order[float(row["conf"])],
            dataset_order[str(row["dataset_name"])],
        )
    )
    threshold_rows = build_threshold_summary(detail_rows)

    summary_csv = run_dir / "summary.csv"
    threshold_csv = run_dir / "threshold_summary.csv"
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    write_csv(summary_csv, detail_rows, DETAIL_COLUMNS)
    write_csv(threshold_csv, threshold_rows, THRESHOLD_COLUMNS)
    write_markdown(summary_md, model_path, run_dir, data_items, conf_values, detail_rows, threshold_rows)

    payload = {
        "arguments": vars(args),
        "resolved": {
            "model": str(model_path),
            "run_dir": str(run_dir),
            "thresholds": conf_values,
            "datasets": [{"name": name, "data": str(path)} for name, path in data_items],
        },
        "environment": env_info(),
        "threshold_summary": threshold_rows,
        "evaluations": details_json,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(f"Wrote detailed summary CSV: {summary_csv}")
    print(f"Wrote threshold summary CSV: {threshold_csv}")
    print(f"Wrote summary Markdown: {summary_md}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
