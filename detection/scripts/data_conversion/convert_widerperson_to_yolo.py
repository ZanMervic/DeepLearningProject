#!/usr/bin/env python3
"""
Convert WiderPerson annotations to Ultralytics/YOLO detection format.

Expected WiderPerson structure, usually:
  WiderPerson/
    Images/000001.jpg
    Annotations/000001.jpg.txt
    train.txt
    val.txt
    test.txt

WiderPerson annotation rows are:
  class_label x1 y1 x2 y2
where:
  1 = pedestrian
  2 = rider
  3 = partially-visible person
  4 = ignore region
  5 = crowd

For this project, default behavior keeps classes 1 and 3 as `person`, optionally 2.
Ignore/crowd regions are skipped because YOLO label files do not support ignore boxes directly.

Usage:
  python convert_widerperson_to_yolo.py \
    --root /path/to/WiderPerson \
    --out /path/to/out/widerperson_yolo \
    --splits train val \
    --include-riders
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_image(images_dir: Path, image_stem_or_name: str) -> Optional[Path]:
    raw = image_stem_or_name.strip()
    p = images_dir / raw
    if p.exists():
        return p
    # split files sometimes contain stems, sometimes names
    stem = Path(raw).stem
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_annotation(ann_dir: Path, image_stem_or_name: str) -> Optional[Path]:
    raw = image_stem_or_name.strip()
    stem = Path(raw).stem
    candidates = [
        ann_dir / f"{raw}.txt",
        ann_dir / f"{stem}.txt",
        ann_dir / f"{stem}.jpg.txt",
        ann_dir / f"{stem}.jpeg.txt",
        ann_dir / f"{stem}.png.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"0 {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def read_split(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation_file(ann_path: Path) -> Iterable[tuple[int, float, float, float, float]]:
    lines = [line.strip() for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    # First line is number of annotations; tolerate files without it.
    start = 1 if len(lines[0].split()) == 1 else 0
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])
        rows.append((cls, x1, y1, x2, y2))
    return rows


def convert_split(
    root: Path,
    out: Path,
    split: str,
    include_riders: bool,
    min_box_size: float,
    copy_images: bool,
) -> tuple[int, int, int]:
    images_dir = root / "Images"
    ann_dir = root / "Annotations"
    split_file = root / f"{split}.txt"

    out_img_dir = out / "images" / split
    out_lbl_dir = out / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    keep_classes = {1, 3}
    if include_riders:
        keep_classes.add(2)

    converted_images = 0
    total_boxes = 0
    missing = 0

    for name in read_split(split_file):
        img_path = find_image(images_dir, name)
        ann_path = find_annotation(ann_dir, name)
        if img_path is None or ann_path is None:
            print(f"[WARN] Missing image or annotation for {name}: image={img_path}, ann={ann_path}")
            missing += 1
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        for cls, x1, y1, x2, y2 in parse_annotation_file(ann_path):
            if cls not in keep_classes:
                continue
            clipped = clip_xyxy(x1, y1, x2, y2, img_w, img_h)
            if clipped is None:
                continue
            cx1, cy1, cx2, cy2 = clipped
            if (cx2 - cx1) < min_box_size or (cy2 - cy1) < min_box_size:
                continue
            yolo_lines.append(xyxy_to_yolo(cx1, cy1, cx2, cy2, img_w, img_h))

        out_img_path = out_img_dir / img_path.name
        out_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"

        if copy_images:
            shutil.copy2(img_path, out_img_path)
        else:
            if not out_img_path.exists():
                out_img_path.symlink_to(img_path.resolve())

        out_lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        converted_images += 1
        total_boxes += len(yolo_lines)

    return converted_images, total_boxes, missing


def write_yaml(out: Path, dataset_name: str = "WiderPerson YOLO") -> None:
    yaml_text = f"""# {dataset_name}\npath: {out.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: person\n"""
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Path to WiderPerson root directory")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert, e.g. train val")
    parser.add_argument("--include-riders", action="store_true", help="Map class 2 riders to person as well")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Skip boxes smaller than this in pixels")
    parser.add_argument("--symlink-images", action="store_true", help="Symlink instead of copying images")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    copy_images = not args.symlink_images

    for split in args.splits:
        n_img, n_box, n_missing = convert_split(
            root=args.root,
            out=args.out,
            split=split,
            include_riders=args.include_riders,
            min_box_size=args.min_box_size,
            copy_images=copy_images,
        )
        print(f"[{split}] images={n_img}, boxes={n_box}, missing={n_missing}")

    write_yaml(args.out)
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
