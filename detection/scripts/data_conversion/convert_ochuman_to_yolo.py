#!/usr/bin/env python3
"""
Convert OCHuman annotations to Ultralytics YOLO detection format.

Supports two cases:
  1) COCO-format OCHuman JSONs:
       ochuman_coco_format_val_range_0.00_1.00.json
       ochuman_coco_format_test_range_0.00_1.00.json
     COCO bbox is [x, y, width, height].

  2) Original OCHuman JSON:
       ochuman.json
     The original OCHuman bbox is commonly [x1, y1, x2, y2].

Important:
  --prefer-segmentation derives boxes only for instances with masks. OCHuman has
  13,360 bbox annotations but only 8,110 mask annotations. If your goal is detection,
  use bbox mode first. Use segmentation-derived boxes only if you explicitly want
  tight visible-mask boxes and accept fewer annotated people.

Examples:
  # Recommended first: use all bbox annotations from original OCHuman json.
  python convert_ochuman_to_yolo_v2.py \
      --json OCHuman/ochuman.json \
      --images OCHuman/images \
      --out datasets/ochuman_bbox \
      --split train \
      --format original \
      --box-source bbox

  # COCO-format val split, bbox mode.
  python convert_ochuman_to_yolo_v2.py \
      --json OCHuman/ochuman_coco_format_val_range_0.00_1.00.json \
      --images OCHuman/images \
      --out datasets/ochuman_coco \
      --split val \
      --format coco \
      --box-source bbox

  # Segmentation-derived boxes where masks exist; falls back to bbox unless disabled.
  python convert_ochuman_to_yolo_v2.py \
      --json OCHuman/ochuman_coco_format_val_range_0.00_1.00.json \
      --images OCHuman/images \
      --out datasets/ochuman_segbox \
      --split val \
      --format coco \
      --box-source segmentation \
      --fallback-to-bbox
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_image(images_dir: Path, file_name_or_id: Any) -> Optional[Path]:
    raw = str(file_name_or_id)
    candidates = [images_dir / raw]
    stem = Path(raw).stem
    if raw.isdigit():
        candidates.extend([
            images_dir / f"{int(raw):06d}.jpg",
            images_dir / f"{raw}.jpg",
        ])
    for ext in IMG_EXTS:
        candidates.append(images_dir / f"{stem}{ext}")
    for p in candidates:
        if p.exists():
            return p
    return None


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    x1 = max(0.0, min(float(w), float(x1)))
    y1 = max(0.0, min(float(h), float(y1)))
    x2 = max(0.0, min(float(w), float(x2)))
    y2 = max(0.0, min(float(h), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"0 {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def bbox_to_xyxy(bbox: Any, bbox_format: str):
    if bbox is None or len(bbox) < 4:
        return None
    a, b, c, d = map(float, bbox[:4])
    if bbox_format == "xywh":
        if c <= 0 or d <= 0:
            return None
        return a, b, a + c, b + d
    if bbox_format == "xyxy":
        if c <= a or d <= b:
            return None
        return a, b, c, d
    raise ValueError(f"Unsupported bbox_format: {bbox_format}")


def polygon_segmentation_to_bbox(seg: Any):
    if not isinstance(seg, list) or not seg:
        return None
    xs, ys = [], []
    for poly in seg:
        if not isinstance(poly, (list, tuple)) or len(poly) < 6:
            continue
        try:
            arr = np.asarray(poly, dtype=float).reshape(-1, 2)
        except Exception:
            continue
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def rle_segmentation_to_bbox(seg: Any):
    if mask_utils is None or not isinstance(seg, dict):
        return None
    try:
        x, y, w, h = mask_utils.toBbox(seg).tolist()
        if w <= 0 or h <= 0:
            return None
        return float(x), float(y), float(x + w), float(y + h)
    except Exception:
        return None


def segmentation_to_bbox(seg: Any):
    if isinstance(seg, list):
        return polygon_segmentation_to_bbox(seg)
    if isinstance(seg, dict):
        return rle_segmentation_to_bbox(seg)
    return None


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    if symlink:
        if not dst.exists():
            dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def normalize_image_record(img: dict[str, Any], fmt: str) -> tuple[Any, str, Optional[int], Optional[int]]:
    if fmt == "coco":
        image_id = img.get("id")
        file_name = img.get("file_name", f"{int(image_id):06d}.jpg" if str(image_id).isdigit() else str(image_id))
        return image_id, file_name, img.get("width"), img.get("height")

    # Original OCHuman frequently stores image_id/file_name fields differently across mirrors.
    image_id = img.get("id", img.get("image_id", img.get("file_name", img.get("imgname"))))
    file_name = img.get("file_name", img.get("imgname", img.get("img_name", None)))
    if file_name is None and image_id is not None:
        file_name = f"{int(image_id):06d}.jpg" if str(image_id).isdigit() else str(image_id)
    return image_id, file_name, img.get("width"), img.get("height")


def collect_original_annotations(data: dict[str, Any]):
    """Return image records and a mapping from image id/name to list of annotation dicts.

    Original OCHuman mirrors have appeared in at least two shapes:
      A) COCO-like top-level images + annotations
      B) image records containing an 'annotations' list
    This function handles both.
    """
    images = []
    anns_by_key: dict[Any, list[dict[str, Any]]] = defaultdict(list)

    if isinstance(data.get("images"), list):
        images = data["images"]
        # Top-level annotations, if present.
        for ann in data.get("annotations", []):
            key = ann.get("image_id", ann.get("id", ann.get("file_name")))
            anns_by_key[key].append(ann)
        # Nested annotations inside each image record.
        for img in images:
            image_id, file_name, _, _ = normalize_image_record(img, "original")
            nested = img.get("annotations", img.get("annos", img.get("objects", [])))
            if isinstance(nested, list):
                anns_by_key[image_id].extend(nested)
                anns_by_key[file_name].extend(nested)
        return images, anns_by_key

    # Some JSONs are a list of image records directly.
    if isinstance(data, list):
        images = data
        for img in images:
            image_id, file_name, _, _ = normalize_image_record(img, "original")
            nested = img.get("annotations", img.get("annos", img.get("objects", [])))
            if isinstance(nested, list):
                anns_by_key[image_id].extend(nested)
                anns_by_key[file_name].extend(nested)
        return images, anns_by_key

    raise ValueError("Unsupported original OCHuman JSON structure. Inspect top-level keys.")


def convert_coco_like(data: dict[str, Any], images_dir: Path, out: Path, split: str, args):
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if args.category_ids and int(ann.get("category_id", -999)) not in args.category_ids:
            continue
        if not args.keep_iscrowd and int(ann.get("iscrowd", 0) or 0) == 1:
            continue
        anns_by_image[ann.get("image_id")].append(ann)

    return write_yolo(images.values(), anns_by_image, images_dir, out, split, args, fmt="coco")


def convert_original(data: dict[str, Any], images_dir: Path, out: Path, split: str, args):
    images, anns_by_key = collect_original_annotations(data)
    return write_yolo(images, anns_by_key, images_dir, out, split, args, fmt="original")


def get_ann_bbox(ann: dict[str, Any], args, fmt: str):
    xyxy = None

    # Try segmentation first if requested.
    if args.box_source == "segmentation":
        seg = ann.get("segmentation", ann.get("segms", ann.get("mask", None)))
        if seg not in (None, [], {}):
            xyxy = segmentation_to_bbox(seg)
        if xyxy is None and not args.fallback_to_bbox:
            return None

    if xyxy is None:
        bbox = ann.get("bbox", ann.get("box", None))
        if bbox is None:
            return None
        # COCO-format JSON bbox is xywh. Original OCHuman bbox is usually xyxy.
        bbox_fmt = "xywh" if fmt == "coco" else args.original_bbox_format
        xyxy = bbox_to_xyxy(bbox, bbox_fmt)

    return xyxy


def write_yolo(image_records, anns_by_key, images_dir: Path, out: Path, split: str, args, fmt: str):
    out_img_dir = out / "images" / split
    out_lbl_dir = out / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    converted_images = 0
    total_boxes = 0
    missing = 0
    images_with_zero_boxes = 0

    for img in image_records:
        image_id, file_name, img_w, img_h = normalize_image_record(img, fmt)
        if file_name is None:
            print(f"[WARN] image record has no file name/id: {img}")
            missing += 1
            continue

        img_path = find_image(images_dir, file_name)
        if img_path is None:
            img_path = find_image(images_dir, image_id)
        if img_path is None:
            print(f"[WARN] Missing image: image_id={image_id}, file_name={file_name}")
            missing += 1
            continue

        if not img_w or not img_h:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        img_w, img_h = int(img_w), int(img_h)

        # COCO keys by numeric image_id; original may need image_id or file_name.
        anns = []
        anns.extend(anns_by_key.get(image_id, []))
        if file_name != image_id:
            anns.extend(anns_by_key.get(file_name, []))
        # Deduplicate by object id or bbox.
        seen = set()
        unique_anns = []
        for ann in anns:
            marker = ann.get("id", None)
            if marker is None:
                marker = str(ann.get("bbox", ann.get("box", ann)))
            if marker in seen:
                continue
            seen.add(marker)
            unique_anns.append(ann)

        yolo_lines = []
        for ann in unique_anns:
            if ann.get("ignore", 0) == 1 and not args.keep_ignore:
                continue
            xyxy = get_ann_bbox(ann, args, fmt)
            if xyxy is None:
                continue
            clipped = clip_xyxy(*xyxy, img_w, img_h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            if (x2 - x1) < args.min_box_size or (y2 - y1) < args.min_box_size:
                continue
            yolo_lines.append(xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h))

        out_img_path = out_img_dir / img_path.name
        copy_or_link(img_path, out_img_path, args.symlink_images)
        (out_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        converted_images += 1
        total_boxes += len(yolo_lines)
        if not yolo_lines:
            images_with_zero_boxes += 1

    return converted_images, total_boxes, missing, images_with_zero_boxes


def write_yaml(out: Path) -> None:
    yaml_text = f"""# OCHuman YOLO\npath: {out.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: person\n"""
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--format", choices=["auto", "coco", "original"], default="auto")
    p.add_argument("--box-source", choices=["bbox", "segmentation"], default="bbox")
    p.add_argument("--fallback-to-bbox", action="store_true")
    p.add_argument("--original-bbox-format", choices=["xyxy", "xywh"], default="xyxy")
    p.add_argument("--category-ids", nargs="*", type=int, default=None)
    p.add_argument("--keep-iscrowd", action="store_true")
    p.add_argument("--keep-ignore", action="store_true")
    p.add_argument("--min-box-size", type=float, default=2.0)
    p.add_argument("--symlink-images", action="store_true")
    args = p.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    fmt = args.format
    if fmt == "auto":
        if isinstance(data, dict) and "annotations" in data and "images" in data and "categories" in data:
            fmt = "coco"
        else:
            fmt = "original"

    args.out.mkdir(parents=True, exist_ok=True)
    if fmt == "coco":
        n_img, n_box, n_missing, n_zero = convert_coco_like(data, args.images, args.out, args.split, args)
    else:
        n_img, n_box, n_missing, n_zero = convert_original(data, args.images, args.out, args.split, args)

    write_yaml(args.out)
    print(f"format={fmt} box_source={args.box_source} fallback_to_bbox={args.fallback_to_bbox}")
    print(f"[{args.split}] images={n_img}, boxes={n_box}, missing_images={n_missing}, images_with_zero_boxes={n_zero}")
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
