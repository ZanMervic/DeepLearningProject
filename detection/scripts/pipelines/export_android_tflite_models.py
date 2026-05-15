#!/usr/bin/env python3
"""Export selected YOLO checkpoints to Android app TFLite assets.

This helper copies exported `.tflite` files into the Android evaluation app and
rewrites the model manifest the app uses for its dropdown.

Example:
    python detection/scripts/pipelines/export_android_tflite_models.py \
      --root C:/Users/zanme/MyStuff/Faks/DL/Project \
      --model yolo26n_all_640 detection/outputs/training/yolo26n_chv_wp_coco_och_640/weights/best.pt 640 "YOLO26n_all_640" \
      --model yolo26n_all_960 detection/outputs/training/yolo26n_chv_wp_coco_och_960/weights/best.pt 960 "YOLO26n_all_960"
      --model yolo26s_chv_960 detection/outputs/training/yolo26s_chv_960/weights/best.pt 960 "YOLO26s_chv_960"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO checkpoints into Android TFLite assets."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root that contains android_model_eval/.",
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("KEY", "PT_PATH", "INPUT_SIZE", "LABEL"),
        required=True,
        help="Model entry: key, checkpoint path, input size, display label.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Request INT8 export. If not set, standard TFLite export is used.",
    )
    return parser.parse_args()


def export_model(pt_path: Path, output_path: Path, input_size: int, int8: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    exported = model.export(format="tflite", imgsz=input_size, int8=int8)
    exported_path = Path(exported)
    if not exported_path.exists():
        raise FileNotFoundError(
            f"Ultralytics reported export path, but it does not exist: {exported_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(exported_path.read_bytes())


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    assets_dir = (
        root / "android_model_eval" / "app" / "src" / "main" / "assets" / "models"
    )
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    for key, pt_path_raw, input_size_raw, label in args.model:
        pt_path = Path(pt_path_raw).resolve()
        if not pt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

        input_size = int(input_size_raw)
        asset_name = f"{key}.tflite"
        output_path = assets_dir / asset_name

        print(f"Exporting {pt_path} -> {output_path}")
        export_model(
            pt_path=pt_path,
            output_path=output_path,
            input_size=input_size,
            int8=args.int8,
        )

        manifest.append(
            {
                "key": key,
                "label": label,
                "assetFile": f"models/{asset_name}",
                "inputSize": input_size,
                "defaultThreshold": 0.20,
            },
        )

    manifest_path = assets_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
